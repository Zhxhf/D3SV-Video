import torch
import numpy as np
from models.kts_src.kts_utils import cpd_auto, l2_normalize_np_array

class VideoSegmentor:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        
    def __call__(self, video_features, video_length):
        K = l2_normalize_np_array(video_features)
        K = np.dot(K, K.T)
        clip_num = K.shape[0]
        print("clip_num",clip_num)
        max_seg_num = clip_num // self.alpha
        print("max_seg_num",max_seg_num)
        # cps, _ = cpd_auto(K, max_seg_num - 1, 0)
        cps, _ = cpd_auto(K, max_seg_num - 1, vmax=1)
        seg_num = len(cps) + 1

        print("seg_num ",seg_num)

        seg_points = [x * self.beta for x in cps]
        seg_points = np.insert(seg_points, 0, 0)
        seg_points = np.append(seg_points, video_length)
        print("seg_points",seg_points)
    
        seg_windows = [[int(seg_points[i]), int(seg_points[i+1])] for i in range(seg_num)]
        return seg_windows
