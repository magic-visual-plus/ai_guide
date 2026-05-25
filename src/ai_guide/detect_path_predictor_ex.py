import ultralytics
from ai_guide import model_utils
from ai_guide.models import ex as models
import torch
import numpy as np
import open3d as o3d
import time
from ai_guide import path_utils
from hq_det.models import rtdetr

class Predictor(object):
    def __init__(self, det_model_file, pcd_model_file):
        self.det_model = rtdetr.HQRTDETR(model=det_model_file)
        self.pcd_model = models.PointNetEx(input_size=6)
        self.pcd_model.load_state_dict(torch.load(pcd_model_file))

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            pass
        
        self.det_model.to(self.device)
        self.pcd_model.to(self.device)
        self.det_model.eval()
        pass

    def predict(self, img, pcd, return_2d_points=False):
        start = time.time()
        box_result = self.det_model.predict([img], bgr=True, confidence=0.5)[0]
        cost_det = time.time() - start

        print(f"Time taken for detection: {cost_det}")

        num_boxes = 0
        indices = []
        normals = []
        original_indices = []
        for ibox, box in enumerate(box_result.bboxes):
            num_boxes += 1
            x1, y1, x2, y2 = box.astype(int).flatten()
            # enlarge box

            pad = 20
            x1 = int(max(0, x1 - pad))
            x2 = int(min(img.shape[1], x2 + pad))
            y1 = int(max(0, y1 - pad))
            y2 = int(min(img.shape[0], y2 + pad))

            subframe = img[y1:y2, x1:x2]
            
            point_indices = []
            for y in range(y1, y2):
                xrange = np.arange(x1, x2)
                p_indices = xrange + y * img.shape[1]
                point_indices.append(p_indices)
                pass
            point_indices = np.concatenate(point_indices)
            subpcd = pcd.select_by_index(point_indices)
            
            subpcd, _, trace = subpcd.voxel_down_sample_and_trace(
                voxel_size=0.5,min_bound=subpcd.get_min_bound(), max_bound=subpcd.get_max_bound())
            
            subpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))
            start = time.time()
            subpcd_index, _ = model_utils.predict_pcd(subpcd, self.pcd_model, self.device, threshold=0.2)

            if len(subpcd_index) == 0:
                continue
            elif len(subpcd_index) > 1:
                subpcd_index_in_original = []
                for si in subpcd_index:
                    if trace[si][-1] < 0:
                        continue
                    point_indices_per_box = trace[si][-1]
                    subpcd_index_in_original.append(point_indices[point_indices_per_box])
                    pass

                # selected_index = path_utils.find_path(np.asarray(subpcd.points)[subpcd_index])
                # subpcd_index = subpcd_index[selected_index]
                cost_pcd = time.time() - start
                print("Time taken for predict: ", cost_pcd)
                subpcd_normals = np.asarray(subpcd.normals)
                indices.append(
                    subpcd_index_in_original)
                normals.append(subpcd_normals[subpcd_index])
                original_indices.append(p_indices)
            else:
                indices.append(point_indices[subpcd_index])
                normals.append(np.asarray(subpcd.normals)[subpcd_index])
                original_indices.append(p_indices)
            pass

        indices_groups_before_concat = []
        for idx in range(len(indices)):
            indices_groups_before_concat.append([idx] * len(indices[idx]))
            pass
        max_groups = len(indices)
        indices = np.concatenate(indices)
        normals = np.concatenate(normals)
        points = np.asarray(pcd.points)
        indices_groups_before_concat = np.concatenate(indices_groups_before_concat)

        path = path_utils.find_path(points[indices])

        indices_groups = [[] for _ in range(max_groups)]
        normals_groups = [[] for _ in range(max_groups)]
        for idx in range(len(path)):
            indices_groups[indices_groups_before_concat[path[idx]]].append(indices[path[idx]])
            normals_groups[indices_groups_before_concat[path[idx]]].append(normals[path[idx]])
            pass

        indices_groups_ = []
        normals_groups_ = []
        original_indices_ = []
        for idx in range(len(indices_groups)):
            if len(indices_groups[idx]) > 0:
                indices_groups_.append(indices_groups[idx])
                normals_groups_.append(normals_groups[idx])
                original_indices_.append(original_indices[idx])
                pass
            pass

        if return_2d_points:
            return indices_groups_, normals_groups_, original_indices_
        else:
            return indices_groups_, normals_groups_
    
        # indices = indices[path]
        # normals = normals[path]

        # indices_groups = [[indices[0]]]
        # normals_groups = [[normals[0]]]
        # for idx in range(1, len(indices)):
        #     p0 = points[indices[idx - 1]]
        #     p1 = points[indices[idx]]
        #     if np.linalg.norm(p0 - p1) > 10:
        #         indices_groups.append([indices[idx]])
        #         normals_groups.append([normals[idx]])
        #         pass
        #     else:
        #         indices_groups[-1].append(indices[idx])
        #         normals_groups[-1].append(normals[idx])
        #         pass
        #     pass

        # return indices_groups, normals_groups