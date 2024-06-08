import argparse

from lib.dataset.sun import SUNDataset
from lib.sample.sample import depth_sampling


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', choices=['sun', 'tum'], help='Type of dataset', default='sun')
    parser.add_argument('--root_dir', type=str, help='Root directory of dataset')
    args = parser.parse_args()
    
    dataset = SUNDataset(args.root_dir)
    
    images, depth_maps, camera_params = zip(*dataset)
    sampled_points = depth_sampling(images, depth_maps, camera_params)
    