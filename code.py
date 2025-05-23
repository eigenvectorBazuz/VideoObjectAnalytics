import torch
import imageio.v3 as iio
import cv2

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

# use the BB's
# TBD - use the masks
def get_raw_YOLO_detections(video, yolo_model):
  results = yolo_model.predict(video, agnostic_nms=True)
  return results

# frames - has the entire (resized) frame set of the video
# TBD - use the query mechanism of cotracker to continue the same tie points across chunks
# The input is frames rather than video because of reasons 
# (the offline cotracker does not work with raw mp4 and preprocessing is better done in bulk in advance). Still TBD better software design.
def create_tie_points(frames, video_chunk_size=100, overlap=20, grid_size=20):
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
  num_frames = len(frames)

  for start in range(0, num_frames, step):
      end = min(start + video_chunk_size, num_frames)
      frame_ids = list(range(start, end))

      # build one-chunk tensor [1, T, C, H, W]
      chunk_tensor = (
          torch.tensor(frames[frame_ids])
               .permute(0, 3, 1, 2)[None]
               .float()
               .to(device)
      )

      # offline cotracker call (no is_first_step / query args)
      pred_tracks, pred_visibility = cotracker(
          video_chunk=chunk_tensor,
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


