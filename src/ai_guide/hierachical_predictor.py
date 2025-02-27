import ultralytics
from ai_guide import model_utils
from ai_guide import models
import torch
import numpy as np
import open3d as o3d
import time
import scipy
from ai_guide import path_utils
from ai_guide import pcd_utils
from ai_guide import path_utils

class HierachicalPredictor(object):
    def __init__(self, pcd_model_file1, pcd_model_file2):
        self.pcd_model1 = models.PointNetEx(input_size=6)
        self.pcd_model1.load_state_dict(torch.load(pcd_model_file1))
        self.pcd_model2 = models.PointNetEx(input_size=6)
        self.pcd_model2.load_state_dict(torch.load(pcd_model_file2))

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            pass

        self.pcd_model1.to(self.device)
        self.pcd_model1.eval()
        self.pcd_model2.to(self.device)
        self.pcd_model2.eval()
        pass

    def predict(self, pcd):
        select_index = pcd_utils.select_points(pcd)
        pcd = pcd.select_by_index(select_index)
        pcd_down, _, trace = pcd.voxel_down_sample_and_trace(voxel_size=50, min_bound=pcd.get_min_bound(), max_bound=pcd.get_max_bound())

        index, proba = model_utils.predict_pcd(pcd_down, self.pcd_model1, self.device, 0.5)
        # index = np.argsort(proba)[::-1]
        # index = index[:35]
        # index = index.astype(int)
        print(len(index))
        
        predict_indices = []
        points = np.asarray(pcd.points)
        points_down = np.asarray(pcd_down.points)
        tree = scipy.spatial.cKDTree(points)
        subpcds = []
        for i, idx in enumerate(index):
            trace_idx = trace[idx]
            if len(trace_idx) < 10000:
                mean = np.mean(points[trace_idx], axis=0)
                qindex = tree.query(mean, k=10000)[1]
                idx_set = set(trace_idx)
                for j in qindex:
                    if j not in idx_set:
                        trace_idx.append(j)
                        pass
                    if len(trace_idx) >= 10000:
                        break
                    pass
                pass
            else:
                pass
            trace_idx = np.unique(trace_idx)
            subpcd = pcd.select_by_index(trace_idx)
            subpcds.append(subpcd)
            trace[idx] = trace_idx
            pass
        
        start = time.time()
        subpcd_indices, _ = model_utils.predict_pcd_batch(subpcds, self.pcd_model2, self.device)
        print("Time taken for predict batch: ", time.time() - start)
        for i, subpcd_index in enumerate(subpcd_indices):
            trace_idx = np.asarray(trace[index[i]])
            predict_indices.append(trace_idx[subpcd_index])
            pass
        
        predict_indices = np.unique(np.concatenate(predict_indices))
        
        predict_xyz = points[predict_indices]
        start = time.time()
        path_index = path_utils.find_path(predict_xyz)
        print("Time taken for path: ", time.time() - start)
        predict_indices = predict_indices[path_index]

        return select_index[predict_indices]