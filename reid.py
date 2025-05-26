import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchreid.reid.utils import FeatureExtractor
from torchreid.reid.metrics.distance import compute_distance_matrix

extractor = FeatureExtractor(
    model_name = 'osnet_ain_x1_0',  # strong vehicle‐ReID backbone
    device = str(device)            # or 'cpu'
)

# the idea is to use reid features to see if two tracks refer to the same objects 
# - either within the same video or across different videos.
# TBD - batch the calls.
# TBD - use several feature extractors and combine the results somehow (how?)
# another possible option - compare the means of the vectors.
# or maybe return the median.
# or the Hausdorff distance in embedding space.
def CompareTracks(t1, t2, mode='mean'):
    crops1 = [app['crop'] for app in t1['track']]
    crops2 = [app['crop'] for app in t2['track']]

    feat1 = extractor(crops1)
    feat2 = extractor(crops2)

    mat = compute_distance_matrix(feat1, feat2, 'cosine')

    if mode == 'mean':
        return torch.mean(mat)
    elif mode == 'median':
        return torch.median(mat)

def Encode(tracks, backbone=None):
    for t in tracks:
        crops = [app['crop'] for app in t['track']]
        feat = extractor(crops)
        t['feat'] = feat
    return tracks

def CompareTrackList(tracks, mode='mean'):
    n = len(tracks)
    dist_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            mat = compute_distance_matrix(tracks[i]['feat'], tracks[j]['feat'], 'cosine')
            if mode == 'mean':
                d = torch.mean(mat)
            elif mode == 'median':
                d = torch.median(mat)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
        print(i)

    return dist_matrix

def CompareTwoTrackLists(tracks1, tracks2, mode='mean'):
    """
    Given two lists of track‐dicts (each with a 'feat' array),
    returns an (n1 × n2) matrix of distances between every pair.
    """
    n1 = len(tracks1)
    n2 = len(tracks2)
    dist_matrix = np.zeros((n1, n2), dtype=float)

    for i in range(n1):
        for j in range(n2):
            mat = compute_distance_matrix(
                tracks1[i]['feat'],
                tracks2[j]['feat'],
                'cosine'
            )
            if mode == 'mean':
                d = torch.mean(mat)
            elif mode == 'median':
                d = torch.median(mat)
            else:
                raise ValueError(f"Unknown mode: {mode!r}")
            dist_matrix[i, j] = float(d)
        print(f"Compared row {i}/{n1}")
    return dist_matrix

    

    
