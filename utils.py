import cv2
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import pulp


def has_duplicates(lst):
    return len(lst) != len(set(lst))

def get_repeats(lst):
    return [item for item, cnt in Counter(lst).items() if cnt > 1]

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


# Example usage:
# display_track_on_frames(example_track, "my_video.mp4")
def display_track_on_frames(track, video_path, stop=None):
    """
    Draws bounding boxes from a track on their respective frames and displays them.
    
    Args:
        track: List of dicts with keys 'frame', 'bbox' (x1,y1,x2,y2), etc.
        video_path: Path to the source video file.
    """
    cap = cv2.VideoCapture(video_path)
    
    i = 0
    for det in track:
        frame_idx = det['frame']
        x1, y1, x2, y2 = map(int, det['bbox'])
        
        # Seek to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame {frame_idx}")
            continue
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Convert BGR to RGB and display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 6))
        plt.imshow(rgb)
        plt.title(f"Frame {frame_idx}")
        plt.axis('off')
        i += 1
        if stop:
            if i >= stop:
                break
        plt.show()
    
    cap.release()

def make_yolo_data(yolo_preds):
    """
    Turn ultralytics Results into:
      { frame_idx: [ {'bbox':(x1,y1,x2,y2), 'conf':float, 'cls':int}, … ] }
    """
    yolo_data = {}
    for frame_idx, res in enumerate(yolo_preds):
        dets = []
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss  = res.boxes.cls.cpu().numpy()
        for box, c, cl in zip(boxes, confs, clss):
            dets.append({
                'bbox': tuple(box.tolist()),
                'conf': float(c),
                'cls':  int(cl)
            })
        yolo_data[frame_idx] = dets
    return yolo_data

def find_matching_bbox(frame_idx, point, yolo_data):
    """
    Return index of the box in yolo_data[frame_idx] containing point, or None.
    """
    x, y = point
    for idx, det in enumerate(yolo_data.get(frame_idx, [])):
        x1,y1,x2,y2 = det['bbox']
        if x1 <= x <= x2 and y1 <= y <= y2:
            return idx
    return None

import pulp
import networkx as nx

def multicut_ilp_pulp(G, pairs):
    nodes = list(G.nodes())
    edges = list(G.edges())
    # Problem definition
    prob = pulp.LpProblem("multicut", pulp.LpMinimize)
    # Edge cut variables
    x = {e: pulp.LpVariable(f"x_{e}", cat="Binary") for e in edges}
    # Label variables
    y = {}
    for idx, (s, t) in enumerate(pairs):
        for v in nodes:
            y[(idx, v)] = pulp.LpVariable(f"y_{idx}_{v}", cat="Binary")
        # s=0, t=1
        prob += y[(idx, s)] == 0
        prob += y[(idx, t)] == 1
        # cut constraints
        for u, v in edges:
            prob += y[(idx, u)] - y[(idx, v)] <= x[(u, v)]
            prob += y[(idx, v)] - y[(idx, u)] <= x[(u, v)]

    # Objective
    prob += pulp.lpSum(x[e] for e in edges)

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    cut_edges = {e for e in edges if pulp.value(x[e]) > 0.5}
    return cut_edges

# # Test
# G = nx.path_graph(6)
# pairs = [(0, 5), (1, 4)]
# cutset = multicut_ilp_pulp(G, pairs)
# print("Cut edges:", cutset)
# H = G.copy()
# H.remove_edges_from(cutset)
# print("Components:", list(nx.connected_components(H)))

# # Build a balanced binary tree of height 3 (15 nodes)
# G = nx.balanced_tree(2, 3)

# # Choose some leaf-pairs to separate
# pairs = [(8, 9), (10, 11), (12, 13)]

# cutset = multicut_ilp_pulp(G, pairs)
# print(cutset)
# # print(components)
# print('sdfsfd')

# nx.draw(G, with_labels=True)

