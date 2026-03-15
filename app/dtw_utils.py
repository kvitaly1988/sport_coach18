from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np

def stack_features(series, keys):
    return np.stack([series[k] for k in keys], axis=1)

def align_by_dtw(a, b):
    _, path = fastdtw(a, b, dist=euclidean)
    return None, None, path
