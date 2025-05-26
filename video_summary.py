

import cv2, torch, math, os
from PIL import Image

!pip install bitsandbytes

def sample_frames(path, n=12):
    """Return n evenly-spaced RGB PIL frames from an mp4 (uses OpenCV)."""
    cap  = cv2.VideoCapture(path)
    tot  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = [round(i*tot/n) for i in range(n)]
    frames = []
    for i in range(tot):
        ret,f = cap.read()
        if not ret:
            break
        if i in idxs:
            frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames

from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"
model_id = "llava-hf/llava-interleave-qwen-7b-hf"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,   # fp16 weights
    # load_in_4bit=True,       # ← 4-bit quantisation (NF4)
    device_map="auto",       # first GPU gets everything
    trust_remote_code=True,  # LLaVA custom layers
)
processor = AutoProcessor.from_pretrained(model_id)

video_path = clip_file   # ← your file
frames     = sample_frames(video_path, n=12)

instructions = """You are an intelligence analyst working on a UAV surveillance video.
Identify key events in the video (e.g., object appearances, interactions, unusual movements, bridges ... ) \
Generate a natural language summary of the video content \
Include temporal and spatial information in your summary \
Highlight the most significant events based on your analysis \
"""

messages = [
    {
      "role": "user",
      "content": [
          *[{"type": "image", "image": img} for img in frames],
          {"type": "text",  "text": instructions}
      ],
    }
]

enc = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,      # insert assistant-turn marker
        tokenize=True,                   # ← MUST be True
        return_tensors="pt",             # ← ask for PyTorch tensors
        return_dict=True,                # BatchEncoding with dict-like access
        video_load_backend="opencv",     # only needed if you give a video path
        num_frames=len(frames)
      ).to(model.device, torch.float16)  # now .to() works

out = model.generate(
        **enc,
        max_new_tokens=600,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
      )
print('---------------')
full = processor.decode(out[0][2:])
# full = processor.decode(out[0][2:], skip_special_tokens=True)
answer_text = full.split("<|im_start|>assistant")[-1].strip()
print(answer_text)

