from __future__ import annotations

import cv2
import torch 
import os
import numpy as np
from pathlib import Path
from typing import List, Sequence, Tuple, Union
from PIL import Image

from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
# from bitsandbytes import BitsAndBytesConfig
from bitsandbytes.nn import Linear4bit




from utils import count_frames, get_video_chunk

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # enable 4-bit weights
    bnb_4bit_quant_type="nf4",              # NF4 quantization
    bnb_4bit_compute_dtype=torch.float16,    # compute in fp16
    bnb_4bit_use_double_quant=True           # slightly better accuracy
)


def _pick_model_id(size: str) -> str:
    size = size.lower().replace("-", "")
    if size in {"0.5b", "0_5b", "0.5"}:
        return "llava-hf/llava-interleave-qwen-0.5b-hf"
    elif size in {"7b", "7"}:
        return "llava-hf/llava-interleave-qwen-7b-hf"
    else:
        raise ValueError(f"Un-recognised Qwen size: {size!r}")


def _resize(frame: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize a H×W×3 uint8 frame to (H_out, W_out)."""
    h_out, w_out = target_hw
    return cv2.resize(frame, (w_out, h_out), interpolation=cv2.INTER_LINEAR)


def _sample_indices(start: int, end: int, n: int) -> List[int]:
    """Evenly sample n indices in [start, end)."""
    if n >= (end - start):
        return list(range(start, end))
    step = (end - start) / n
    return [int(start + i * step) for i in range(n)]

def _format_timestamp(seconds: float) -> str:
    """
    Convert seconds (can be fractional) to HH:MM:SS format,
    rounding to the nearest second.
    """
    td = timedelta(seconds=round(seconds))
    # str(td) gives "H:MM:SS" or "D days, H:MM:SS"
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class VideoChatBot:
    """
    Summarises a video chunk-by-chunk with LLaVA-Qwen.

    Parameters
    ----------
    size              – "0.5B" or "7B"
    quant_4bit        – load in 4-bit NF4 weights if True, fp16 otherwise
    target_hw         – (H, W) working resolution for analysis
    frames_per_chunk  – number of frames to feed the model for every chunk
    gen_kwargs        – extra kwargs forwarded to `model.generate`
    """

    def __init__(
        self,
        size: str = "0.5B",
        quant_4bit: bool = False,
        target_hw: Tuple[int, int] = (360, 480),
        frames_per_chunk: int = 12,
        fps: float = 30.0,
        **gen_kwargs,
    ):
        model_id = _pick_model_id(size)

        model_kwargs = dict(device_map="auto", trust_remote_code=True)
        if quant_4bit:
            # ←—— new: build a bnb-config instead of load_in_4bit flag
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,                      # enable 4-bit
                bnb_4bit_quant_type="nf4",              # NF4
                bnb_4bit_compute_dtype=torch.float16,   # compute in fp16
                bnb_4bit_use_double_quant=True          # optional double-quant
            )
            model_kwargs["quantization_config"] = bnb_config
        else:
            # keep fp16 as before
            model_kwargs["torch_dtype"] = torch.float16

        self.model = (
            LlavaForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.target_hw = target_hw
        self.frames_per_chunk = frames_per_chunk
        # sensible defaults, can still be overridden per-call
        self.gen_defaults = dict(
            max_new_tokens=600, do_sample=True, temperature=0.7, top_p=0.9
        )
        self.gen_defaults.update(gen_kwargs)

        has_4bit = any(isinstance(m, Linear4bit) for m in self.model.modules())
        print("4-bit layers present:", has_4bit)

    # --------------------------------------------------------------------- #
    #  PUBLIC API                                                           #
    # --------------------------------------------------------------------- #
    def __call__(
        self, video_path: Union[str, Path], user_prompt: str, **gen_overrides
    ) -> str:
        """
        Describe the whole video with the user-supplied prompt.

        Returns a single concatenated text string (one paragraph per chunk).
        """
        chunks = list(self._iter_chunks(video_path))
        # descriptions: List[str] = []

        total_frames = count_frames(video_path)

        lines: List[str] = []
        for idx, frames in enumerate(chunks):
            # 1) get the raw description
            desc = self._describe_chunk(frames, idx, user_prompt, **gen_overrides).strip()
            print(idx, desc)

            # 2) compute frame range → seconds → HH:MM:SS
            start_frame = idx * self.frames_per_chunk
            end_frame = min((idx + 1) * self.frames_per_chunk, total_frames)

            start_sec = start_frame / self.fps
            end_sec = end_frame / self.fps

            start_ts = _format_timestamp(start_sec)
            end_ts = _format_timestamp(end_sec)

            # 3) prepend timestamp
            lines.append(f"{start_ts} --> {end_ts} {desc}")

        # join into a single text blob
        return "\n".join(lines)

    # --------------------------------------------------------------------- #
    #  INTERNALS                                                            #
    # --------------------------------------------------------------------- #
    def _describe_chunk(
        self,
        frames: Sequence[np.ndarray],
        chunk_idx: int,
        user_prompt: str,
        **gen_overrides,
    ) -> str:
        """
        Feed a list of H×W×3 uint8 frames to the vision-language model
        and return the decoded response string.
        """

        pil_frames = [Image.fromarray(f) for f in frames]

        # Build the chat message for this chunk
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in pil_frames],
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]

        enc = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            num_frames=len(frames),
            video_load_backend="opencv",
        ).to(self.model.device, torch.float16)

        out = self.model.generate(
            **enc,
            **{**self.gen_defaults, **gen_overrides},  # allow run-time overrides
        )

        decoded = self.processor.decode(out[0][2:])
        return decoded.split("<|im_start|>assistant")[-1].strip()

    def _iter_chunks(
        self,
        video_path: Union[str, Path]
    ) -> Generator[List[np.ndarray], None, None]:
        """
        Yield exactly `frames_per_chunk` resized frames per iteration,
        by calling THE user-provided get_video_chunk.
        """
        # 1) get total frame count
        total = count_frames(video_path)
    
        # 2) walk in steps of `frames_per_chunk`
        start = 0
        while start < total:
            end = min(start + self.frames_per_chunk, total)
    
            # 3) pull raw frames using the user’s function
            raw = get_video_chunk(video_path, start, end)  # shape: (F, H, W, 3)
    
            # 4) resize each frame to self.target_hw
            #    target_hw is (H, W), so we flip for cv2.resize
            frames = [
                cv2.resize(f, self.target_hw[::-1], interpolation=cv2.INTER_LINEAR)
                for f in raw
            ]
    
            yield frames
            start = end


























