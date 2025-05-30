import torch
import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from itertools import chain

from torchreid.reid.metrics.distance import compute_distance_matrix

from utils import count_frames, get_video_chunk
from utils import has_duplicates, get_repeats #ironic
from utils import build_separation_pairs, multicut, stack_track_feats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Offline CoTracker:
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)

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
            det['box_id'] = d
            seq.append(det)
        tracks.append({'track':seq, 'nodes':comp, 'subgraph':G.subgraph(comp)})
    return tracks

def build_tie_graph_nextsight(yolo_data, tie_point_bunches):
    G = nx.Graph()
    nodes = [(f,i) for f, dets in yolo_data.items() for i in range(len(dets))]
    G.add_nodes_from(nodes)

    bboxes = {
        f: np.array([det['bbox'] for det in dets], dtype=float)
        for f, dets in yolo_data.items()
    }

    def _match_point(frame, point):
        arr = bboxes.get(frame)
        if arr is None or arr.size == 0:
            return None
        x, y = point
        if hasattr(x, 'item'):
            x = x.item(); y = y.item()
        xs, ys, xe, ye = arr.T
        mask = (xs <= x) & (x <= xe) & (ys <= y) & (y <= ye)
        idxs = np.nonzero(mask)[0]
        return int(idxs[0]) if idxs.size else None

    for c, chunk in enumerate(tie_point_bunches):
        frames = chunk['frame_ids']
        tracks = chunk['tracks'][0]
        vis    = chunk['visible'][0]
        T, N = vis.shape

        for n in range(N):
            matched = {}
            for t in range(T):
                if not vis[t, n]:
                    continue
                f = frames[t]
                pt = tuple(tracks[t, n])
                det = _match_point(f, pt)
                if det is not None:
                    matched[t] = (f, det, pt)

            times = sorted(matched)
            for i in range(len(times)-1):
                t1, t2 = times[i], times[i+1]
                f1, d1, pt1 = matched[t1]
                f2, d2, pt2 = matched[t2]
                G.add_edge((f1,d1), (f2,d2),
                           tie_point_index=n,
                           coord1=pt1,
                           coord2=pt2)
        print(c, G)

    return G


def check_track(t):
    track_frames = [app['frame'] for app in t]
    # print(track_frames)
    if has_duplicates(track_frames):
        return False, get_repeats(track_frames)
    return True, None

def split_track(t):
    flag, error_frames = check_track(t['track'])
    if flag:
      return [t]
    
    pairs = build_separation_pairs(t['nodes'])
    # S = G.subgraph(t['nodes'])
    S = t['subgraph']
    # cutset = multicut_ilp_pulp(S, pairs)
    cutset = multicut(S, pairs, method='auto', node_threshold=100, verbose=True)
    S_cut = S.edge_subgraph(set(S.edges()) - set(cutset)).copy()


    comps = [c for c in nx.connected_components(S_cut) if len(c)>1]
    # t_split = get_tracks_by_nodegroups(yolo_data, S, comps)
    t_split = []
    for c in comps:
      piece = [app for app in t['track'] if (app['frame'], app['box_id']) in c]
      subtrack = {'track':piece, 'nodes':c, 'subgraph':S.subgraph(c)}
      t_split.append(subtrack)
    return t_split

def split_track_reid(t):
    flag, error_frames = check_track(t['track'])
    if flag:
      return [t]
    st = stack_track_feats(t['track'])
    m = compute_distance_matrix(st, st, 'cosine')
    adj = (m.cpu().numpy() < 0.25)
    np.fill_diagonal(adj, False)
    H = nx.from_numpy_array(adj)
    # commH = greedy_modularity_communities(H)
    # print(H)
    comps = [c for c in nx.connected_components(H) if len(c)>1]

    S = t['subgraph']
    t_split = []
    for c in comps:
      print(c)
      piece = [app for i, app in enumerate(t['track']) if i in c]
      print(piece)
      subtrack = {'track':piece, 'nodes':c, 'subgraph':S.subgraph(c)}
      t_split.append(subtrack)
    return t_split

    

def merge_tracks(tracks, G):
    """
    tracks: List[dict], each with keys 'track' (list) and 'nodes' (iterable)
    G:      your NetworkX graph (to build subgraph later)
    """
    merged_track = list(chain.from_iterable(t.get('track', []) for t in tracks))
    merged_nodes = list(chain.from_iterable(t.get('nodes', []) for t in tracks))

    merged_subgraph = G.subgraph(merged_nodes).copy()

    return {
        'track':    merged_track,
        'nodes':    merged_nodes,
        'subgraph': merged_subgraph
    }


    
    


  

