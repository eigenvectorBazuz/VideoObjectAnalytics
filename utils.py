import cv2
import imageio.v3 as iio
import numpy as np

def count_frames(video):
  cap = cv2.VideoCapture(video)
  n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  cap.release()
  return n_frames


def resize_video(video, new_sz):
  reader = iio.imiter(video, plugin="FFMPEG")  
  
  # Iterate frame by frame, resize, and collect
  resized_frames = []
  for frame in reader:
      # frame is H×W×3 (uint8). Resize to 384(height)×640(width):
      frame_resized = cv2.resize(frame, new_sz[::-1], interpolation=cv2.INTER_LINEAR)
      resized_frames.append(frame_resized)
  
  reader.close()

  return np.stack(resized_frames)

# using the python convention, frames from start to end-1.
def get_video_chunk(video, start, end):
  frames = []
  reader = iio.imiter(
      video,
      plugin='FFMPEG',
      # no input_params here, since we're not seeking by time
      output_params=[
          '-vf',      f'select=between(n\\,{start}\\,{end})',  # only keep frames n∈[100,200]
          '-vsync',   '0',                                     # disable frame-rate correction
      ]
  )
  for frame_number, frame in zip(range(start, end), reader):
    frames.append(frame)
  return np.stack(frames)
