import torch
import imageio.v3 as iio
import cv2
import numpy as np
import networkx as nx


from ultralytics import YOLO, YOLOE

from utils import count_frames, get_video_chunk
from utils import display_track_on_frames
from utils import make_yolo_data, find_matching_bbox

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

      # 1) grab raw frames (T, H, W, C)
      np_frames = get_video_chunk(video, start, end)

      # 2) build the [B=1, T, C, H, W] tensor
      chunk_tensor = (
          torch.from_numpy(np_frames)
               .permute(0, 3, 1, 2)   # (T, C, H, W)
               [None]                 # (1, T, C, H, W)
               .float()
               .to(device)
      )

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

# yolo_data is the output from make_yolo_data()
def get_tracks_by_nodegroups(yolo_data, G, node_groups):
    tracks = []
    for comp in node_groups:
        # build a sorted, enriched list of dicts
        seq = []
        for (f, d) in sorted(comp, key=lambda x: x[0]):
            det = dict(yolo_data[f][d])      # copy bbox/conf/cls
            det['frame'] = f
            det['vertex'] = d
            seq.append(det)
        tracks.append(seq)
    return tracks

def build_tie_graph(yolo_data, tie_point_bunches):
    """
    Graph‐based grouping. Returns:
      [ [ {'frame':…, 'bbox':…, 'conf':…, 'cls':…}, … ], … ]
    """
    G = nx.Graph()

    # 1) Add every detection as a node
    for frame_idx, dets in yolo_data.items():
        for det_idx in range(len(dets)):
            G.add_node((frame_idx, det_idx))

    print(G)

    # 2) Link detections via tie‐points
    for c, chunk in enumerate(tie_point_bunches):
        frames = chunk['frame_ids']
        tracks = chunk['tracks'][0,:]      # shape (T,N,2)
        vis    = chunk['visible'][0,:]     # shape (T,N)
        T, N, _ = tracks.shape

        for t in range(T):
            f1 = frames[t]
            for n in range(N):
                if not vis[t,n]:
                    continue
                det1 = find_matching_bbox(f1, tracks[t,n], yolo_data)
                if det1 is None:
                    continue
                # connect to any later appearance of same tie‐point
                for t2 in range(t+1, T):
                    if not vis[t2,n]:
                        continue
                    f2 = frames[t2]
                    det2 = find_matching_bbox(f2, tracks[t2,n], yolo_data)
                    if det2 is None:
                        continue
                    # add edge with metadata
                    G.add_edge((f1, det1), (f2, det2), tie_point_index=n, coord1=tuple(tracks[t, n]), coord2=tuple(tracks[t2, n]))
        print(c, G)

    return G

  

# video is an mp4 file or some compatible source/iterator
# TBD - print a list of supported models and make a switch case
def discover_objects_in_video(video, yolo_model_name):
  yolo_model = YOLO(yolo_model_name) # unless it's a YOLOE model....
  pass


