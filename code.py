import torch
import imageio.v3 as iio
import cv2
import numpy as np

from ultralytics import YOLO, YOLOE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Offline CoTracker:
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)


# # Load a model
# model = YOLO("yolo11x.pt")  # pretrained YOLO11n model
# model = YOLOE("yoloe-11l-seg-pf.pt")
# Initialize a YOLOE model
# model = YOLOE("yoloe-11l-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes
# names = ["person", "vehicle", "bridge", "sign"]
# model.set_classes(names, model.get_text_pe(names))

# clip_file = '/content/drive/MyDrive/BR/seg0.mp4'
# track_results = model.track(clip_file, agnostic_nms=True, persist=True, tracker="/content/drive/MyDrive/BR/botsort_modified.yaml")  # Tracking with default tracker

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
  for frame_number, frame in zip(range(start, end+1), reader):
    frames.append(frame)
  return np.stack(frames)

# use the BB's
# TBD - use the masks
def get_raw_YOLO_detections(video, yolo_model):
  results = yolo_model.predict(video, agnostic_nms=True)
  return results

# TBD - use the query mechanism of cotracker to continue the same tie points across chunks
# TBD - resize to save time and resize back the results.
def create_tie_points(video, video_chunk_size=100, overlap=20, grid_size=20):
  """
  Split `frames` into overlapping chunks, run offline cotracker on each,
  and collect the per-chunk tie points.

  Returns:
    List of dicts, each with keys:
      - 'frame_ids': list of global frame indices in that chunk
      - 'tracks': tensor of predicted tracks for that chunk
      - 'visible': tensor of predicted visibility for that chunk
  """
  tie_points = []
  step = video_chunk_size - overlap
  num_frames = count_frames(video)

  for start in range(0, num_frames, step):
      end = min(start + video_chunk_size, num_frames)
      frame_ids = list(range(start, end))
      print(f'processing {start} to {end}')

      # build one-chunk tensor [1, T, C, H, W]
      chunk_tensor = get_video_chunk(video, start, end)

      # offline cotracker call 
      pred_tracks, pred_visibility = cotracker(
          video=chunk_tensor,
          grid_size=grid_size
      )

      tie_points.append({
          "frame_ids": frame_ids,
          "tracks": pred_tracks,
          "visible": pred_visibility,
      })

      # stop once we've covered the last frame
      if end == num_frames:
          break

  return tie_points


  
  
  

# video is an mp4 file or some compatible source/iterator
# TBD - print a list of supported models and make a switch case
def discover_objects_in_video(video, yolo_model_name):
  yolo_model = YOLO(yolo_model_name) # unless it's a YOLOE model....
  pass


