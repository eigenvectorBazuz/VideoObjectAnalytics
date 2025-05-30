import torch
import imageio.v3 as iio
import cv2
import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

from ultralytics import YOLO, YOLOE

from utils import display_track_on_frames
from utils import make_yolo_data

from tracking import create_tie_points, get_tracks_by_nodegroups, build_tie_graph_nextsight, split_track, merge_tracks, split_track_reid
from reid import Encode, CompareTrackList


# # Load a model
# model = YOLO("yolo11x.pt")  # pretrained YOLO11n model
# model = YOLOE("yoloe-11l-seg-pf.pt")
# Initialize a YOLOE model
# model = YOLOE("yoloe-11l-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes
# names = ["person", "vehicle", "bridge", "sign"]
# model.set_classes(names, model.get_text_pe(names))

# clip_file = '/content/drive/MyDrive/BR/seg0.mp4'
# track_results = model.track(clip_file, agnostic_nms=True, persist=True, tracker="/content/drive/MyDrive/BR/botsort_modified.yaml")  # Tracking with default tracker


# TBD - merge several models, either YOLO/YOLOE or DETR ones and return a unified detections list in YOLO format. 
# the input in this case will be either a string or a list of strings.
# TBD - perhaps set a confidence threshold?
def get_raw_YOLO_detections(video, yolo_model):
  results = yolo_model.predict(video, agnostic_nms=True)
  return results

def merge_tracks_by_reid(tracks, G, mean_dst_th=0.25, include_final_split=True):
  tracks = Encode(tracks)
  dst = CompareTrackList(tracks)
  adj = (dst < 0.25)
  np.fill_diagonal(adj, False)
  H = nx.from_numpy_array(adj)
  commH = greedy_modularity_communities(H)
  print(H)
  merged_tracks = [merge_tracks([tracks[i] for i in c], G) if len(c) > 1 else tracks[next(iter(c))] for c in commH]

  if include_final_split:
    final_tracks = [child for t in merged_tracks for child in split_track_reid(t)]
    return final_tracks
  else:
    return merged_tracks

  
  

# video is an mp4 file or some compatible source/iterator
# TBD - print a list of supported models and make a switch case
def discover_objects_in_video(video, yolo_model_name, tie_point_params: dict | None = None, include_final_split=True, return_data=False):
  """
    Discover objects in a video using YOLO + tie-point tracking.

    Parameters:
    -----------
    video :
        Your video source (path or array).
    yolo_model_name : str
        Name or path of the YOLO model to load.
    tie_point_params : dict, optional
        Dict may contain any of:
          - video_chunk_size: int (default 250)
          - overlap:          int (default 10)
          - grid_size:        int (default 20)
    return_data : bool, default=False
        If True, return raw detections and tie-points.
  """
  defaults = {
        "video_chunk_size": 250,
        "overlap": 10,
        "grid_size": 20,
    }
  # Safely overlay any user-provided values
  params = {**defaults, **(tie_point_params or {})}
  
  yolo_model = YOLO(yolo_model_name) # unless it's a YOLOE model....
  yolo_preds = get_raw_YOLO_detections(video, yolo_model)
  yd = make_yolo_data(yolo_preds)
  
  ties = create_tie_points(video, video_chunk_size=params["video_chunk_size"], overlap=params["overlap"], grid_size=params["grid_size"])
  G = build_tie_graph_nextsight(yd, ties)
  communities = greedy_modularity_communities(G) # TBD - use Louvain
  comms = [c for c in communities if len(c)>1]

  raw_tracks = get_tracks_by_nodegroups(yd, G, comms)
  tracks = [child for t in raw_tracks for child in split_track(t)]
  
  # now merge track/tracklets by ReID
  final_tracks = merge_tracks_by_reid(tracks, G, mean_dst_th=0.25, include_final_split=include_final_split)
  tracks_sorted = sorted(final_tracks , key=lambda t: min(item['frame'] for item in t['track']))
  
  if return_data:
    return tracks_sorted, {'yolo_preds':yolo_preds, 'ties':ties, 'G':G, 'raw_tracks':raw_tracks, 'split_tracklets':tracks}
  else:
    return tracks_sorted


  


