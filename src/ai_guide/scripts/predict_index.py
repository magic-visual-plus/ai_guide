import ultralytics
import sys
import cv2
import os
import open3d as o3d
import numpy as np
import torch
from ai_guide import model_utils
from ai_guide import models
from ai_guide import detect_path_predictor
import time

def run_infer(yolo_model_file, pcd_model_file, input_img_file, input_pcd_file, output_file, output_normal_file):
    index_predictor = detect_path_predictor.Predictor(yolo_model_file, pcd_model_file)

    frame = cv2.imread(input_img_file)
    pcd = o3d.io.read_point_cloud(input_pcd_file)
    
    start = time.time()
    indices, normals = index_predictor.predict(frame, pcd)
    print("Time taken: ", time.time() - start)
    # indices = np.concatenate(indices, axis=0)

    with open(output_file, "w") as f, open(output_normal_file, "w") as nf:
        for indices_per_box, normal_per_box in zip(indices, normals):
            f.write(",".join([str(i) for i in indices_per_box]))
            f.write("\n")

            for normal in normal_per_box:
                nf.write(" ".join([str(n) for n in normal]))
                nf.write(",")
                pass
            nf.write("\n")
            pass
        pass
    pass
    return pcd


if __name__ == "__main__":
    
    yolo_model_file = sys.argv[1]
    pcd_model_file = sys.argv[2]
    input_img_file = sys.argv[3]
    input_pcd_file = sys.argv[4]
    output_file = sys.argv[5]
    output_normal_file = sys.argv[6]

    run_infer(yolo_model_file, pcd_model_file, input_img_file, input_pcd_file, output_file, output_normal_file)