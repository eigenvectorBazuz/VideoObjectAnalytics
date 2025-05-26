from single_movie_analysis import discover_objects_in_video
from reid import Encode, CompareTwoTrackLists
from utils import annotate_video_with_tracks

import numpy as np

def unify_track_ids(track_lists, threshold=0.25, mode='mean'):
    """
    Given a list of clips (each a list of track‐dicts),
    assigns consistent IDs across clips by nearest‐neighbor matching
    under a distance threshold.

    Uses:
      Encode(tracks)                    # mutates each track to add 'feat'
      CompareTwoTrackLists(tracks1,     # returns (len(tracks1)×len(tracks2)) dists
                           tracks2,
                           mode)

    Returns:
        A list of ID‐lists (one per clip).
    """
    all_ids_per_clip = []
    all_tracks = []    # flat list of track dicts with 'feat'
    all_ids    = []    # parallel list of assigned IDs
    next_id    = 0

    for clip_idx, clip in enumerate(track_lists):
        # 1) compute features in place
        Encode(clip)

        if clip_idx == 0:
            # first clip: give IDs 0..n0-1
            n0 = len(clip)
            ids0 = list(range(n0))
            all_ids_per_clip.append(ids0)

            all_tracks = clip.copy()
            all_ids    = ids0.copy()
            next_id    = n0

        else:
            # compute cross‐distances prev vs new
            D = CompareTwoTrackLists(all_tracks, clip, mode=mode)
            prev_n, new_n = D.shape

            ids_new = []
            for j in range(new_n):
                i_min = int(np.argmin(D[:, j]))
                if D[i_min, j] < threshold:
                    ids_new.append(all_ids[i_min])
                else:
                    ids_new.append(next_id)
                    next_id += 1

            all_ids_per_clip.append(ids_new)

            # add to the pool
            all_tracks.extend(clip)
            all_ids.extend(ids_new)

    return all_ids_per_clip


def analyze_and_annotate_videos(videos_list):
  results = []
  for video in videos_list:
    tracks = discover_objects_in_video(video, 'yolo12x.pt', tie_point_params={'video_chunk_size':50, 'overlap':5, 'grid_size':20}, 
                               include_final_split=False, return_data=False)
    results.append(tracks)


