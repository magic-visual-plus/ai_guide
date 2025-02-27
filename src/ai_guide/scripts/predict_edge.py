import sys
from ai_guide import hierachical_predictor
import open3d as o3d


if __name__ == '__main__':
    input_model_file1 = sys.argv[1]
    input_model_file2 = sys.argv[2]
    input_pcd_file = sys.argv[3]
    output_file = sys.argv[4]

    predictor = hierachical_predictor.HierachicalPredictor(input_model_file1, input_model_file2)
    pcd = o3d.io.read_point_cloud(input_pcd_file)
    index = predictor.predict(pcd)

    with open(output_file, 'w') as f:
        for idx in index:
            f.write(f'{idx}\n')
            pass
        pass

    pass