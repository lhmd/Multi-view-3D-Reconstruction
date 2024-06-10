import os
import sys
import cv2
import numpy as np
from torch.utils.data import Dataset

class SCANNETDataset(Dataset):
    def __init__(self, root_dir, rgb_dir='color', depth_dir='depth', intrinsics_dir='intrinsic', poses_dir='pose', num_frames=100):
        self.root_dir = root_dir
        self.rgb_dir = os.path.join(root_dir, rgb_dir)
        self.depth_dir = os.path.join(root_dir, depth_dir)
        self.poses_dir = os.path.join(root_dir, poses_dir)
        # 文件名按阿拉伯数字顺序排序，不以默认顺序排
        self.rgb_files = sorted([os.path.join(self.rgb_dir, file) for file in os.listdir(self.rgb_dir)], key=lambda x: int(x.split('/')[-1].split('.')[0]))
        self.depth_files = sorted([os.path.join(self.depth_dir, file) for file in os.listdir(self.depth_dir)], key=lambda x: int(x.split('/')[-1].split('.')[0]))
        self.pose_files = sorted([os.path.join(self.poses_dir, file) for file in os.listdir(self.poses_dir)], key=lambda x: int(x.split('/')[-1].split('.')[0]))
        # self.rgb_files = sorted([os.path.join(self.rgb_dir, file) for file in os.listdir(self.rgb_dir)])
        # self.depth_files = sorted([os.path.join(self.depth_dir, file) for file in os.listdir(self.depth_dir)])
        # self.pose_files = sorted([os.path.join(self.poses_dir, file) for file in os.listdir(self.poses_dir)])
        # print(self.rgb_files)
        
        assert len(self.rgb_files) == len(self.depth_files), "RGB and depth image counts do not match"
        assert len(self.rgb_files) == len(self.pose_files), "RGB and pose counts do not match"
        
        self.intrinsics_dir = os.path.join(root_dir, intrinsics_dir)
        self.intrinsics_color_file = os.path.join(self.intrinsics_dir, 'intrinsic_color.txt')
        self.intrinsics_depth_file = os.path.join(self.intrinsics_dir, 'intrinsic_depth.txt')
        
        self.camera_params = []
        
        with open(self.intrinsics_color_file, 'r') as f:
            lines = f.readlines()
            self.K_color = np.array([[float(x) for x in line.split()[:3]] for line in lines[:3]], dtype=np.float32)
            
        with open(self.intrinsics_depth_file, 'r') as f:
            lines = f.readlines()
            self.K_depth = np.array([[float(x) for x in line.split()[:3]] for line in lines[:3]], dtype=np.float32)
            
        for i in range(len(self.pose_files)):
            with open(self.pose_files[i], 'r') as f:
                lines = f.readlines()
                R = np.array([[float(x) for x in line.split()[:3]] for line in lines[:3]], dtype=np.float32)
                T = np.array([float(line.split()[-1]) for line in lines[:3]], dtype=np.float32)
                
            self.camera_params.append((self.K_depth, R, T))
        
        if num_frames < len(self):
            # 间隔采样
            self.rgb_files = self.rgb_files[::len(self) // num_frames]
            self.depth_files = self.depth_files[::len(self) // num_frames]
            self.camera_params = self.camera_params[::len(self) // num_frames]
        print(self.rgb_files)
        
        
    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, idx):
        if idx >= len(self) or idx < 0:
            raise IndexError("Index out of bounds")
        
        rgb_path = self.rgb_files[idx]
        depth_path = self.depth_files[idx]
        
        rgb_image = cv2.imread(rgb_path)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_image = depth_image.astype(np.float32) / 1000.0
        
        if rgb_image is None:
            raise FileNotFoundError(f"RGB image at {rgb_path} not found")
        if depth_image is None:
            raise FileNotFoundError(f"Depth image at {depth_path} not found")
        
        return rgb_image, depth_image, self.camera_params[idx]
    
if __name__ == '__main__':
    root_directory = '/data/wangweijie/MV3D_Recon/data/SCANNET/scene0707_00'
    dataset = SCANNETDataset(root_directory)
    rgb_image0, depth_image0, camera_param0 = dataset[0]
    # 输出depth_image0的最大值
    print(depth_image0)
    print(camera_param0)