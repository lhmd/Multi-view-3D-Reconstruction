import os
import sys
import cv2
import numpy as np
from torch.utils.data import Dataset

class SCANNETDataset(Dataset):
    def __init__(self, root_dir, rgb_dir='color', depth_dir='depth', intrinsics_dir='intrinsic', poses_dir='pose', num_frames=100):
        root_dir = root_dir
        rgb_dir = os.path.join(root_dir, rgb_dir)
        depth_dir = os.path.join(root_dir, depth_dir)
        poses_dir = os.path.join(root_dir, poses_dir)
        # 文件名按阿拉伯数字顺序排序，不以默认顺序排
        rgb_files = sorted([os.path.join(rgb_dir, file) for file in os.listdir(rgb_dir)], key=lambda x: int(x.split('/')[-1].split('.')[0]))
        depth_files = sorted([os.path.join(depth_dir, file) for file in os.listdir(depth_dir)], key=lambda x: int(x.split('/')[-1].split('.')[0]))
        pose_files = sorted([os.path.join(poses_dir, file) for file in os.listdir(poses_dir)], key=lambda x: int(x.split('/')[-1].split('.')[0]))
        
        # 总帧数num_frames
        step = len(rgb_files) // num_frames
        print(f"Step: {step}")
        rgb_files = rgb_files[::step]
        depth_files = depth_files[::step]
        pose_files = pose_files[::step]
        
        assert len(rgb_files) == len(depth_files), "RGB and depth image counts do not match"
        assert len(rgb_files) == len(pose_files), "RGB and pose counts do not match"
        
        intrinsics_dir = os.path.join(root_dir, intrinsics_dir)
        intrinsics_color_file = os.path.join(intrinsics_dir, 'intrinsic_color.txt')
        intrinsics_depth_file = os.path.join(intrinsics_dir, 'intrinsic_depth.txt')
        
        camera_params = []
        rgb_params = []
        K_color = None
        K_depth = None
        
        with open(intrinsics_color_file, 'r') as f:
            lines = f.readlines()
            K_color = np.array([[float(x) for x in line.split()[:3]] for line in lines[:3]], dtype=np.float32)
            
        with open(intrinsics_depth_file, 'r') as f:
            lines = f.readlines()
            K_depth = np.array([[float(x) for x in line.split()[:3]] for line in lines[:3]], dtype=np.float32)
            
        for i in range(len(pose_files)):
            with open(pose_files[i], 'r') as f:
                lines = f.readlines()
                R = np.array([[float(x) for x in line.split()[:3]] for line in lines[:3]], dtype=np.float32)
                T = np.array([float(line.split()[-1]) for line in lines[:3]], dtype=np.float32)
                
            camera_params.append((K_depth, R, T))
            rgb_params.append((K_color, R, T))
        
        print(f"Loaded {len(rgb_files)} frames")
        self.rgb_files = rgb_files
        self.depth_files = depth_files
        self.camera_params = camera_params
        self.rgb_params = rgb_params
        # print(self.rgb_files)
        
        # self.rgb_files = self.rgb_files[:num_frames]
        # self.depth_files = self.depth_files[:num_frames]
        # self.camera_params = self.camera_params[:num_frames]
        
        
    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, idx):
        if idx >= len(self) or idx < 0:
            raise IndexError("Index out of bounds")
        
        rgb_path = self.rgb_files[idx]
        depth_path = self.depth_files[idx]
        # print(depth_path)
        
        rgb_image = cv2.imread(rgb_path)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_image = depth_image.astype(np.float32) / 1000.0
        
        if rgb_image is None:
            raise FileNotFoundError(f"RGB image at {rgb_path} not found")
        if depth_image is None:
            raise FileNotFoundError(f"Depth image at {depth_path} not found")
        
        return rgb_image, depth_image, self.camera_params[idx], self.rgb_params[idx]
    
if __name__ == '__main__':
    root_directory = '/data/wangweijie/MV3D_Recon/data/SCANNET/scene0707_00'
    dataset = SCANNETDataset(root_directory)
    rgb_image0, depth_image0, camera_param0 = dataset[0]
    # 输出depth_image0的最大值
    print(depth_image0)
    print(camera_param0)