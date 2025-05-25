import torch

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
def CompareTracks(t1, t2):
    crops1 = [app['crop'] for app in t1['track']]
    crops2 = [app['crop'] for app in t1['track']]

    feat1 = extractor(crops1)
    feat2 = extractor(crops2)

    mat = compute_distance_matrix(feat1, feat2, 'cosine')

    return torch.mean(mat)
