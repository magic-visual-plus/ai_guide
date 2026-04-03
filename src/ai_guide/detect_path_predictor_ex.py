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

    def predict(self, img, pcd):
        start = time.time()
        box_result = self.det_model.predict([img], bgr=True, confidence=0.5)[0]
        cost_det = time.time() - start

        num_boxes = 0
        indices = []
        normals = []
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
                point_indices.append(xrange + y * img.shape[1])
                pass
            point_indices = np.concatenate(point_indices)
            subpcd = pcd.select_by_index(point_indices)
            
            subpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))
            start = time.time()
            subpcd_index, _ = model_utils.predict_pcd(subpcd, self.pcd_model, self.device, threshold=0.2)
            if len(subpcd_index) == 0:
                continue
            elif len(subpcd_index) > 1:
                cost_pcd = time.time() - start
                print("Time taken for predict: ", cost_pcd)
                subpcd_normals = np.asarray(subpcd.normals)
                indices.append(
                    point_indices[subpcd_index])
                normals.append(subpcd_normals[subpcd_index])
            else:
                indices.append(point_indices[subpcd_index])
                normals.append(np.asarray(subpcd.normals)[subpcd_index])
            pass

        indices = np.concatenate(indices)
        normals = np.concatenate(normals)
        points = np.asarray(pcd.points)

        path = path_utils.find_path(points[indices])

        indices = indices[path]
        normals = normals[path]

        indices_groups = [[indices[0]]]
        normals_groups = [[normals[0]]]
        for idx in range(1, len(indices)):
            p0 = points[indices[idx - 1]]
            p1 = points[indices[idx]]
            if np.linalg.norm(p0 - p1) > 10:
                indices_groups.append([indices[idx]])
                normals_groups.append([normals[idx]])
                pass
            else:
                indices_groups[-1].append(indices[idx])
                normals_groups[-1].append(normals[idx])
                pass
            pass

        return indices_groups, normals_groups