import os
import sys
import cv2
from torch.utils.data import Dataset

class SUNDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        self.dir_paths = [os.path.join(self.root_dir, dir) for dir in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, dir))]
        self.rgb_files = []
        self.depth_files = []
        self.camera_params = []
        
        for dir_path in self.dir_paths:
            basename = os.path.basename(dir_path)
            img_path = os.path.join(dir_path, 'image', os.listdir(os.path.join(dir_path, 'image'))[0])
            depth_path = os.path.join(dir_path, 'depth', os.listdir(os.path.join(dir_path, 'depth'))[0])
            self.rgb_files.append(img_path)
            self.depth_files.append(depth_path)
            
            intrinsic_path = os.path.join(dir_path, 'intrinsics.txt')
            extrinsic_path = os.path.join(dir_path, 'extrinsics', os.listdir(os.path.join(dir_path, 'extrinsics'))[0])
            
            with open(intrinsic_path, 'r') as f:
                lines = f.readlines()
                K = [[float(x) for x in line.split()] for line in lines]
            
            with open(extrinsic_path, 'r') as f:
                lines = f.readlines()
                # 0.994043 -0.108925 -0.003710 0.000000
                # 0.108925 0.991733 0.067837 0.000000
                # -0.003710 -0.067837 0.997690 0.000000
                R = [[float(x) for x in line.split()[:3]] for line in lines]
                T = [float(line.split()[-1]) for line in lines]
                
            self.camera_params.append((K, R, T))
            
        assert len(self.rgb_files) == len(self.depth_files), "RGB and depth image counts do not match"
        
    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, idx):
        if idx >= len(self) or idx < 0:
            raise IndexError("Index out of bounds")
        
        rgb_path = self.rgb_files[idx]
        depth_path = self.depth_files[idx]
        
        rgb_image = cv2.imread(rgb_path)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        if rgb_image is None:
            raise FileNotFoundError(f"RGB image at {rgb_path} not found")
        if depth_image is None:
            raise FileNotFoundError(f"Depth image at {depth_path} not found")
        
        return rgb_image, depth_image, self.camera_params[idx]
    

if __name__ == '__main__':
    root_directory = '/data/wangweijie/MV3D_Recon/data/SUNRGBD/xtion/sun3ddata/mit_lab_16/lab_16_nov_2_2012_scan1_erika'
    dataset = SUNDataset(root_directory)
    rgb_image0, depth_image0, camera_param0 = dataset[0]
    print(camera_param0)