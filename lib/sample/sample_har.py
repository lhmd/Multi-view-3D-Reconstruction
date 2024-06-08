import sys
import cv2
import random
import numpy as np
from wis3d.wis3d import Wis3D
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, sys.path[0]+"/../../")
from lib.dataset.sun import SUNDataset
sys.path.pop(0)

def non_uniform_sampling(depth_map, K, Rt, Tt, delta_d=8e-8, r_max=8, num_samples=1000):
    H, W = depth_map.shape
    sampled_points = []
    valid_mask = depth_map > 0
    sample_map = cv2.resize(valid_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

    def reproject_pixel_to_3d(u, v, depth):
        depth = 1 / depth
        u = (u - K[0, 2]) * depth / K[0, 0]
        v = (v - K[1, 2]) * depth / K[1, 1]
        point_3d = np.array([u, v, depth], dtype=np.float32)
        point_3d = np.dot(Rt.T, (point_3d - Tt))
        return point_3d

    def compute_plane_normal(points):
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        _, _, vh = np.linalg.svd(centered_points)
        normal = vh[-1]
        return normal / np.linalg.norm(normal)

    def is_in_plane(point, normal, centroid):
        return np.abs(np.dot(normal, point - centroid)) < delta_d

    def process_point(x_t, y_t):
        sample_map[x_t, y_t] = 255

        depth_t = depth_map[x_t, y_t]
        P_xt = reproject_pixel_to_3d(x_t, y_t, depth_t)
        local_points = []

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x_t + dx, y_t + dy
                if 0 <= nx < H and 0 <= ny < W and valid_mask[nx, ny] and depth_map[nx, ny] > 0:
                    depth_n = depth_map[nx, ny]
                    P_n = reproject_pixel_to_3d(nx, ny, depth_n)
                    local_points.append(P_n)

        if not local_points:
            return None

        normal = compute_plane_normal(np.array(local_points))
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
                break
            r_xt += 1

        sampled_points.append((P_xt, r_xt))
        valid_mask[x_t, y_t] = False
        return P_xt, r_xt

    while np.any(valid_mask):
        sum_valid = valid_mask.sum()
        print(sum_valid)
        num_samples = max(sum_valid // 500, 1)
        valid_indices = np.argwhere(valid_mask)
        # 从valid_indices中随机选取num_samples个点
        indices_to_process = random.sample(list(valid_indices), min(num_samples, len(valid_indices)))
        
        with ThreadPoolExecutor(max_workers=num_samples) as executor:
            futures = [executor.submit(process_point, x_t, y_t) for x_t, y_t in indices_to_process]
            results = [f.result() for f in futures if f.result()]

        if results:
            for P_xt, r_xt in results:
                sampled_points.append((P_xt, r_xt))

        # Update the valid_mask for the processed points
        for x_t, y_t in indices_to_process:
            valid_mask[x_t, y_t] = False

    cv2.imwrite("output/sample_map.png", sample_map)
    
    return sampled_points

if __name__ == '__main__':
    root_directory = '/data/wangweijie/MV3D_Recon/data/SUNRGBD/xtion/sun3ddata/mit_lab_16/lab_16_nov_2_2012_scan1_erika'
    dataset = SUNDataset(root_directory)
    rgb_image0, depth_image0, camera_param0 = dataset[0]
    
    cv2.imwrite("output/1.png", depth_image0)
    Dt = depth_image0.astype(np.float32)
    H, W = 10, 10
    depth_map = np.random.rand(H, W)
    Kt = np.array(camera_param0[0], dtype=np.float32)
    Rt = np.array(camera_param0[1], dtype=np.float32)
    Tt = np.array(camera_param0[2], dtype=np.float32)

    sampled_points = non_uniform_sampling(Dt, Kt, Rt, Tt)
    
    wis3d = Wis3D("output/vis", 'point', xyz_pattern=('x', 'y', 'z'))
    points = np.array([point for point, _ in sampled_points])
    print(len(points))
    points /= points.max()
    color = np.array([255, 0, 0], dtype=np.uint8)
    wis3d.add_point_cloud(points, color)
