import os
import argparse
from lib.dataset.sun import SUNDataset
from lib.dataset.scannet import SCANNETDataset
from lib.dataset.tum import TUMDataset
from lib.sample.sample import sample_all
from lib.fusion.fusion import do_fusion
from lib.visualize.wis3d import Visualizer
from lib.utils.read_save import save_points, read_points
from lib.extract.poisson import poisson_reconstruction

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', choices=['sun', 'tum', 'scannet'], help='Type of dataset', default='scannet')
    parser.add_argument('--root_dir', type=str, help='Root directory of dataset')
    parser.add_argument('--output_dir', type=str, help='Output directory', default='output')
    parser.add_argument('--exp_name', type=str, help='Experiment name', default='points100')
    parser.add_argument('--num_frames', type=int, help='Number of frames to sample', default=100)
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.path.exists(os.path.join(args.output_dir, args.exp_name)):
        os.system(f"rm -r {os.path.join(args.output_dir, args.exp_name)}")
    os.makedirs(os.path.join(args.output_dir, args.exp_name))
        
    
    print("Loading dataset...")
    if args.data_type == 'sun':
        dataset = SUNDataset(args.root_dir)
    elif args.data_type == 'tum':
        dataset = TUMDataset(args.root_dir)
    else:
        dataset = SCANNETDataset(args.root_dir, num_frames=args.num_frames)
    print(f"Dataset size: {len(dataset)}")
    
    print("Sampling...")
    filtered_points = sample_all(dataset, os.path.join(args.output_dir, args.exp_name))
    
    save_points(filtered_points, os.path.join(args.output_dir, args.exp_name))
    vis_point = Visualizer(args.exp_name + "_point")
    point_clouds = []
    import numpy as np
    for points in filtered_points:
        point_clouds.append(np.array([point[0] for point in points]))
    vis_point.add_points(point_clouds)
    
    filtered_points = read_points(os.path.join(args.output_dir, args.exp_name))
    print("Sample point fusion...")
    fused_points = do_fusion(dataset, filtered_points)
    
    vis_fused = Visualizer(args.exp_name + "_fused")
    vis_fused.add_point_cloud(fused_points)
    
    mesh = poisson_reconstruction(fused_points, os.path.join(args.output_dir, args.exp_name, 'mesh.ply'))
    
    vis_mesh = Visualizer(args.exp_name + "_mesh")
    vis_mesh.add_mesh(mesh)