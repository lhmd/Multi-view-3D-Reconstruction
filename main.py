import argparse
from lib.dataset.sun import SUNDataset
from lib.dataset.scannet import SCANNETDataset
from lib.dataset.tum import TUMDataset
from lib.sample.sample import sample_all
from lib.fusion.fusion import do_fusion
from lib.visualize.wis3d import Visualizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', choices=['sun', 'tum', 'scannet'], help='Type of dataset', default='scannet')
    parser.add_argument('--root_dir', type=str, help='Root directory of dataset')
    args = parser.parse_args()
    
    print("Loading dataset...")
    if args.data_type == 'sun':
        dataset = SUNDataset(args.root_dir)
    elif args.data_type == 'tum':
        dataset = TUMDataset(args.root_dir)
    else:
        dataset = SCANNETDataset(args.root_dir)
    print(f"Dataset size: {len(dataset)}")
    
    print("Sampling...")
    filtered_points = sample_all(dataset)
    
    vis = Visualizer("point")
    point_clouds = []
    import numpy as np
    for points in filtered_points:
        point_clouds.append(np.array([point[0] for point in points]))
    vis.add_points(point_clouds)
    
    print("Sample point fusion...")
    fused_points = do_fusion(dataset, filtered_points)
    
    wis3d = Visualizer('final')
    wis3d.add_point_cloud(fused_points)
    