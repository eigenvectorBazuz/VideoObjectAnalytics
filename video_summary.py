from __future__ import annotations

import cv2
import torch 
import os
import numpy as np
from pathlib import Path
from typing import List, Sequence, Tuple, Union
from PIL import Image

from transformers import AutoProcessor, LlavaForConditionalGeneration




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


class VideoChatBot:
    """
    Summarises a video chunk-by-chunk with LLaVA-Qwen.

    Parameters
    ----------
    size              – "0.5B" or "7B"
    quant_4bit        – load in 4-bit NF4 weights if True, fp16 otherwise
    chunk_seconds     – length of each analysed segment
    target_hw         – (H, W) working resolution for analysis
    frames_per_chunk  – number of frames to feed the model for every chunk
    gen_kwargs        – extra kwargs forwarded to `model.generate`
    """

    def __init__(
        self,
        size: str = "0.5B",
        quant_4bit: bool = False,
        chunk_seconds: int = 5,
        target_hw: Tuple[int, int] = (360, 480),
        frames_per_chunk: int = 12,
        **gen_kwargs,
    ):
        model_id = _pick_model_id(size)

        model_kwargs = dict(device_map="auto", trust_remote_code=True)
        if quant_4bit:
            model_kwargs["load_in_4bit"] = True
        else:
            model_kwargs["torch_dtype"] = torch.float16

        self.model = (
            LlavaForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.chunk_seconds = chunk_seconds
        self.target_hw = target_hw
        self.frames_per_chunk = frames_per_chunk
        # sensible defaults, can still be overridden per-call
        self.gen_defaults = dict(
            max_new_tokens=600, do_sample=True, temperature=0.7, top_p=0.9
        )
        self.gen_defaults.update(gen_kwargs)

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
        descriptions: List[str] = []

        for idx, frames in enumerate(chunks):
            desc = self._describe_chunk(frames, idx, user_prompt, **gen_overrides)
            descriptions.append(desc.strip())

        return "\n\n".join(descriptions)

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
        # Build the chat message for this chunk
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in frames],
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
        self, video_path: Union[str, Path]
    ) -> "Generator[List[np.ndarray], None, None]":
        """
        Yield resized frames for each chunk using the user-supplied helpers:

          • get_video_chunk(video_path, start, end)
          • resize_video(video_path, new_size)  ⟵ optional, see below
        """
        # ── 1.  quick metadata pass (OpenCV is simplest here) ──────────
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        chunk_len = int(self.chunk_seconds * fps)
        start = 0
        while start < total_frames:
            end = min(start + chunk_len, total_frames)

            # ── 2.  pull the raw frames with YOUR function ─────────────
            raw_chunk = get_video_chunk(video_path, start, end)  # shape: (F, H, W, 3)

            # ── 3.  pick a uniform subset to feed LLaVA ───────────────
            keep = _sample_indices(0, raw_chunk.shape[0], self.frames_per_chunk)
            sampled = raw_chunk[keep]

            # ── 4.  resize each frame to the working resolution ────────
            # (You already have `resize_video`, but per-frame resize is cheaper
            #  than re-reading the video, so we do it inline.)
            frames = [_resize(f, self.target_hw) for f in sampled]

            yield frames
            start = end





























