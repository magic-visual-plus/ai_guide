import ultralytics
from ai_guide import pcd_utils, models
import torch
import numpy as np
import open3d as o3d
import time
from ai_guide import path_utils, model_utils
import hq_seg.predictor
import cv2


class Predictor(object):
    def __init__(self, seg_model_file, pcd_model_file):
        self.seg_model = hq_seg.predictor.Predictor(seg_model_file, (512, 512))
        self.pcd_model = models.PointNetEx(input_size=6)
        self.pcd_model.load_state_dict(torch.load(pcd_model_file))

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            pass

        self.pcd_model.to(self.device)
        pass


    def extract_mask(self, img):

        mask = self.seg_model.predict(img)

        mask = mask * 255

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = max(contours, key = cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [contours], -1, (255,), 3)
        
        cv2.GaussianBlur(mask, (63, 63), 0, mask)

        return mask > 0
    
    def predict(self, img, pcd):
        start = time.time()
        mask = self.extract_mask(img)
        
        pcd = pcd_utils.replace_nan(pcd, 0)

        select_index = np.where(mask.flatten())[0]
        print(len(select_index))
        pcd = pcd.select_by_index(select_index)

        index, _ = model_utils.predict_pcd(pcd, self.pcd_model, self.device, 0.1)
        print(len(index))
        # print(index)
        path = path_utils.find_path(np.asarray(pcd.points)[index])
        # path = np.arange(len(index))

        return select_index[index[path]]

class PredictorPt(object):
    def __init__(self, seg_model_file, pcd_model_file):
        self.seg_model = hq_seg.predictor.Predictor(seg_model_file, (512, 512))
        self.pcd_model = models.PointTransformerPointcept(input_size=6)
        self.pcd_model.load_state_dict(torch.load(pcd_model_file))

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            pass

        self.pcd_model.to(self.device)
        pass


    def extract_mask(self, img):

        mask = self.seg_model.predict(img)

        mask = mask * 255

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = max(contours, key = cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [contours], -1, (255,), 3)
        
        cv2.GaussianBlur(mask, (63, 63), 0, mask)

        return mask > 0
    
    def predict(self, img, pcd):
        start = time.time()
        mask = self.extract_mask(img)
        
        pcd = pcd_utils.replace_nan(pcd, 0)

        select_index = np.where(mask.flatten())[0]
        print(len(select_index))
        pcd = pcd.select_by_index(select_index)

        index, _ = model_utils.predict_pcd_pt(pcd, self.pcd_model, self.device, 0.05)
        print(len(index))
        # print(index)
        path = path_utils.find_path(np.asarray(pcd.points)[index])
        # path = np.arange(len(index))

        return select_index[index[path]]