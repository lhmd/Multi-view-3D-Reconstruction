import sys
import cv2
import torch
import open3d as o3d
import numpy as np
from wis3d.wis3d import Wis3D

sys.path.insert(0, sys.path[0]+"/../../")
from lib.dataset.sun import SUNDataset
sys.path.pop(0)

def non_uniform_sampling(depth_map, K, Rt, Tt, delta_d=8e-8, r_max=8):
    # Move everything to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    depth_map = depth_map.to(device)
    K = K.to(device)
    Rt = Rt.to(device)
    Tt = Tt.to(device)
    
    H, W = depth_map.shape
    sampled_points = []
    valid_mask = depth_map > 0
    sample_map = cv2.resize(valid_mask.cpu().numpy().astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    
    def reproject_pixel_to_3d(u, v, depth):
        depth = 1 / depth
        u = (u - K[0, 2]) * depth / K[0, 0]
        v = (v - K[1, 2]) * depth / K[1, 1]
        point_3d = torch.tensor([u, v, depth], dtype=torch.float32, device=device)
        point_3d = torch.matmul(torch.transpose(Rt, 0, 1), (point_3d - Tt))
        return point_3d
    
    def compute_plane_normal(points):
        # Center the points
        centroid = torch.mean(points, dim=0)
        centered_points = points - centroid

        # Perform SVD
        _, _, vh = torch.linalg.svd(centered_points)

        # The normal of the plane is the last column of vh
        normal = vh[-1]

        return normal / torch.norm(normal)  # Return a unit normal vector

    
    def is_in_plane(point, normal, centroid):
        # print(torch.abs(torch.matmul(normal, point - centroid)) < delta_d, torch.abs(torch.matmul(normal, point - centroid)))
        return torch.abs(torch.matmul(normal, point - centroid)) < delta_d
    
    while valid_mask.any():
        x_t, y_t = torch.where(valid_mask)[0][0].item(), torch.where(valid_mask)[1][0].item()
        sample_map[x_t, y_t] = 255
        
        # print(f"Sampling point at ({x_t}, {y_t})")  
        
        depth_t = depth_map[x_t, y_t]
        P_xt = reproject_pixel_to_3d(x_t, y_t, depth_t)
        # 算相邻点的P
        local_points = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x_t + dx, y_t + dy
                if 0 <= nx < H and 0 <= ny < W and valid_mask[nx, ny] and depth_map[nx, ny] > 0:
                    depth_n = depth_map[nx, ny]
                    P_n = reproject_pixel_to_3d(nx, ny, depth_n)
                    local_points.append(P_n)
        # print("Points:", local_points)
        normal = compute_plane_normal(torch.stack(local_points))
        r_xt = 1
        
        while r_xt <= r_max:
            cnt = 0
            for dx in range(-r_xt, r_xt + 1):
                for dy in [-r_xt, r_xt]:
                    nx, ny = x_t + dx, y_t + dy
                    if 0 <= nx < H and 0 <= ny < W and valid_mask[nx, ny] and depth_map[nx, ny] > 0:
                        depth_n = depth_map[nx, ny]
                        P_n = reproject_pixel_to_3d(nx, ny, depth_n)
                        if is_in_plane(P_n, normal, P_xt):
                            valid_mask[nx, ny] = False
                        else:
                            cnt += 1
            for dy in range(-r_xt, r_xt + 1):
                for dx in [-r_xt, r_xt]:
                    nx, ny = x_t + dx, y_t + dy
                    if 0 <= nx < H and 0 <= ny < W and valid_mask[nx, ny] and depth_map[nx, ny] > 0:
                        depth_n = depth_map[nx, ny]
                        P_n = reproject_pixel_to_3d(nx, ny, depth_n)
                        if is_in_plane(P_n, normal, P_xt):
                            valid_mask[nx, ny] = False
                        else:
                            cnt += 1
            
            if cnt > 0.75 * (2 * r_xt + 1) ** 2:
                print(f"Sampled {cnt} points at radius {r_xt} ", 0.75 * (2 * r_xt + 1) ** 2)
                break
            r_xt += 1
        
        # print(f"Sampled point at ({x_t}, {y_t}) with radius {r_xt}")
        sampled_points.append((P_xt, r_xt))
        # print(f"Sampled {len(sampled_points)} points")
        valid_mask[x_t, y_t] = False
        # points = torch.stack(plane_points)
        # color = torch.tensor([0, 255, 0], dtype=torch.uint8, device=device)
        # wis3d.add_point_cloud(points, color)
        # valid_mask中值为True的个数
        # print(f"Remaining valid pixels: {valid_mask.sum()}")
    cv2.imwrite("output/sample_map.png", sample_map)
    
    return sampled_points

def save_point_cloud(sampled_points, filename='output_point_cloud.ply'):
    # 将采样点转换为numpy数组
    points = torch.stack([point for point, _ in sampled_points]).cpu().numpy()
    
    # 创建Open3D点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    # 为点云添加颜色（可选）
    colors = np.array([[0, 1, 0] for _ in range(points.shape[0])])  # 绿色
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    # 保存点云为PLY文件
    o3d.io.write_point_cloud(filename, point_cloud)
    print(f"Point cloud saved to {filename}")

if __name__ == '__main__':
    root_directory = '/data/wangweijie/MV3D_Recon/data/SUNRGBD/xtion/sun3ddata/mit_lab_16/lab_16_nov_2_2012_scan1_erika'
    dataset = SUNDataset(root_directory)
    rgb_image0, depth_image0, camera_param0 = dataset[0]
    
    cv2.imwrite("output/1.png", depth_image0)
    Dt = torch.tensor(depth_image0, dtype=torch.float32)
    H, W = 10, 10
    depth_map = torch.rand(H, W)
    Kt = torch.tensor(camera_param0[0], dtype=torch.float32)
    Rt = torch.tensor(camera_param0[1], dtype=torch.float32)
    Tt = torch.tensor(camera_param0[2], dtype=torch.float32)

    sampled_points = non_uniform_sampling(Dt, Kt, Rt, Tt)
    
    # save_point_cloud(sampled_points, 'output_point_cloud.ply')
    
    # 可视化
    
    wis3d = Wis3D("output/vis", 'point', xyz_pattern=('x', 'y', 'z'))
    points = torch.stack([point for point, _ in sampled_points])
    print(len(points))
    # points归一化
    points = points / points.max()
    color = torch.tensor([0, 255, 0], dtype=torch.uint8)
    wis3d.add_point_cloud(points, color)