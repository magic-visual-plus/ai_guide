import sys
from ai_guide import segment_path_predictor
import open3d as o3d
import cv2


if __name__ == '__main__':
    seg_model_file = sys.argv[1]
    pcd_model_file = sys.argv[2]
    input_pcd_file = sys.argv[4]
    input_img_file = sys.argv[3]
    output_file = sys.argv[5]

    predictor = segment_path_predictor.PredictorPt(seg_model_file, pcd_model_file)
    pcd = o3d.io.read_point_cloud(input_pcd_file)
    img = cv2.imread(input_img_file)
    index = predictor.predict(img, pcd)

    with open(output_file, 'w') as f:
        for idx in index:
            f.write(f'{idx}\n')
            pass
        pass

    pass