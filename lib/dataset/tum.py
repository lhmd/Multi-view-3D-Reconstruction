import os
import sys
import cv2
import numpy as np

sys.path.insert(0, sys.path[0]+"/../../")
from lib.utils.associate import read_file_list, associate
sys.path.pop(0)

class TUMDataset:
    def __init__(self, root_dir, rgb_dir='rgb', depth_dir='depth'):
        self.root_dir = root_dir
        self.rgb_dir = os.path.join(root_dir, rgb_dir)
        self.depth_dir = os.path.join(root_dir, depth_dir)
        first_list = read_file_list(os.path.join(root_dir, 'rgb.txt'))
        second_list = read_file_list(os.path.join(root_dir, 'depth.txt'))
        matches = associate(first_list, second_list, 0, 0.02)
        
        self.rgb_files = [first_list[match[0]][0] for match in matches]
        self.depth_files = [second_list[match[1]][0] for match in matches]
        
        self.rgb_files = [os.path.join(self.root_dir, file) for file in self.rgb_files]
        self.depth_files = [os.path.join(self.root_dir, file) for file in self.depth_files]
        
        assert len(self.rgb_files) == len(self.depth_files), "RGB and depth image counts do not match"
    
    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, idx):
        if idx >= len(self) or idx < 0:
            raise IndexError("Index out of bounds")
        
        rgb_path = os.path.join(self.rgb_dir, self.rgb_files[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        
        rgb_image = cv2.imread(rgb_path)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        if rgb_image is None:
            raise FileNotFoundError(f"RGB image at {rgb_path} not found")
        if depth_image is None:
            raise FileNotFoundError(f"Depth image at {depth_path} not found")
        
        return rgb_image, depth_image

if __name__ == '__main__':
    root_directory = '/data/wangweijie/MV3D_Recon/data/TUM/rgbd_dataset_freiburg1_teddy'  # 修改为TUM数据集的根目录
    dataset = TUMDataset(root_directory)

    rgb_image, depth_image = dataset[0]

    cv2.imshow('RGB Image', rgb_image)
    cv2.imshow('Depth Image', depth_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
