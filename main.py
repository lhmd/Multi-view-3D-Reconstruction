import argparse
from lib.dataset.sun import SUNDataset
from lib.sample.sample import sample_all
from lib.fusion.fusion import do_fusion

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', choices=['sun', 'tum'], help='Type of dataset', default='sun')
    parser.add_argument('--root_dir', type=str, help='Root directory of dataset')
    args = parser.parse_args()
    
    print("Loading dataset...")
    dataset = SUNDataset(args.root_dir)
    print(f"Dataset size: {len(dataset)}")
    
    print("Sampling...")
    filtered_points = sample_all(dataset)
    
    print("Sample point fusion...")
    fused_points = do_fusion(dataset, filtered_points)
    
    