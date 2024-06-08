import argparse
from lib.dataset.sun import SUNDataset
from lib.sample.sample import sample_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', choices=['sun', 'tum'], help='Type of dataset', default='sun')
    parser.add_argument('--root_dir', type=str, help='Root directory of dataset')
    args = parser.parse_args()
    
    dataset = SUNDataset(args.root_dir)
    print(f"Dataset size: {len(dataset)}")
    
    sample_all(dataset)
    
    