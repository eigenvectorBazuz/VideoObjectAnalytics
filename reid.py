import torch
import torchreid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchreid.utils import FeatureExtractor

extractor = FeatureExtractor(
    model_name = 'osnet_ain_x1_0',  # strong vehicle‐ReID backbone
    device = device            # or 'cpu'
)

# the idea is to use reid features to see if two tracks refer to the same objects - either within the same video or across different videos.
def CompareGalleries(t1, t2):
    pass
