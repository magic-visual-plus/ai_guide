import ultralytics
from ai_guide import model_utils
from ai_guide.models import pt as models
import torch
import numpy as np
import open3d as o3d
import time
from ai_guide import path_utils

class Predictor(object):
    def __init__(self, det_model_file, pcd_model_file):
        self.det_model = ultralytics.YOLO(det_model_file)
        self.pcd_model = models.PointTransformerPointcept(input_size=6)
        self.pcd_model.load_state_dict(torch.load(pcd_model_file))

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            pass

        self.pcd_model.to(self.device)
        pass

    def predict(self, img, pcd):
        start = time.time()
        box_result = self.det_model(img)
        cost_det = time.time() - start

        num_boxes = 0
        indices = []
        normals = []
        for r in box_result:
            for ibox, box in enumerate(r.boxes):
                if box.conf > 0.3:
                    num_boxes += 1
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy().astype(int).flatten()
                    # enlarge box
                    box_width = x2 - x1
                    box_height = y2 - y1
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
                    subpcd_index, _ = model_utils.predict_pcd_pt(subpcd, self.pcd_model, self.device)
                    subpcd_points = np.asarray(subpcd.points)
                    subpcd_path = path_utils.find_path(subpcd_points[subpcd_index])
                    subpcd_index = subpcd_index[subpcd_path]
                    cost_pcd = time.time() - start
                    print("Time taken for predict: ", cost_pcd)
                    subpcd_normals = np.asarray(subpcd.normals)
                    indices.append(
                        point_indices[subpcd_index])
                    normals.append(subpcd_normals[subpcd_index])
                    pass
                pass
            pass
        return indices, normals