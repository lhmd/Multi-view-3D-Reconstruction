import os
import sys
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, sys.path[0] + "/../../")
from lib.dataset.sun import SUNDataset
from lib.dataset.scannet import SCANNETDataset
from lib.visualize.wis3d import Visualizer
sys.path.pop(0)

def non_uniform_sampling(depth_map, K, Rt, Tt, idx, output_path, delta_d=7e-3, r_max=8):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    def reproject_image_to_3d(image):
        u = np.tile(np.arange(W), (H, 1))
        v = np.tile(np.arange(H).reshape(-1, 1), (1, W))
        
        u_transformed = (u - K[0, 2]) * image / K[0, 0]
        v_transformed = (v - K[1, 2]) * image / K[1, 1]
        
        points_3d = np.stack([u_transformed, v_transformed, image], axis=-1)
        
        points_3d = points_3d.reshape(-1, 3)
        points_3d = np.dot(Rt, points_3d.T).T + Tt.T
        
        points_3d = points_3d.reshape(H, W, 3)
        
        return points_3d

    def reproject_pixel_to_3d(u, v, depth):
        u = (u - K[0, 2]) * depth / K[0, 0]
        v = (v - K[1, 2]) * depth / K[1, 1]
        point_3d = np.array([u, v, depth], dtype=np.float32)
        point_3d = np.dot(Rt, point_3d.T)
        return point_3d

    def compute_plane_normal(points):
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        _, _, vh = np.linalg.svd(centered_points)
        normal = vh[-1]
        return normal / np.linalg.norm(normal)

    def is_in_plane(points, normal, centroid):
        return np.abs(np.dot(points - centroid, normal)) < delta_d
    
    H, W = depth_map.shape
    sampled_points = []
    valid_mask = depth_map > 0
    sample_map = cv2.resize(valid_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

    point_3d_map = reproject_image_to_3d(depth_map)
    
    # print(point_3d_map.shape)

    while np.any(valid_mask):
        x_t, y_t = np.where(valid_mask)[0][0], np.where(valid_mask)[1][0]
        sample_map[x_t, y_t] = 255

        # depth_t = depth_map[x_t, y_t]
        # P_xt = reproject_pixel_to_3d(x_t, y_t, depth_t)
        P_xt = point_3d_map[x_t, y_t]
                    
        offsets = np.array([[dx, dy] for dx in range(-1, 2) for dy in range(-1, 2)])
        nx = x_t + offsets[:, 0]
        ny = y_t + offsets[:, 1]
        valid_indices = (0 <= nx) & (nx < H) & (0 <= ny) & (ny < W)
        nx = nx[valid_indices]
        ny = ny[valid_indices]
        valid_points_mask = valid_mask[nx, ny] & (depth_map[nx, ny] > 0)
        local_points = point_3d_map[nx[valid_points_mask], ny[valid_points_mask]]
                    
        normal = compute_plane_normal(np.array(local_points))
        r_xt = 1

        while r_xt <= r_max:
            cnt = 0
            x_range = np.arange(-r_xt, r_xt + 1)
            y_range = np.arange(-r_xt, r_xt + 1)
            dx, dy = np.meshgrid(x_range, y_range)
            nx = np.clip(x_t + dx, 0, H - 1)
            ny = np.clip(y_t + dy, 0, W - 1)
            mask = valid_mask[nx, ny] & (depth_map[nx, ny] > 0)
            P_n = point_3d_map[nx[mask], ny[mask]]
            in_plane = is_in_plane(P_n, normal, P_xt)
            valid_mask[nx[mask], ny[mask]] = ~in_plane
            cnt += np.count_nonzero(~in_plane)
            
            if cnt > 0.75 * (2 * r_xt + 1) ** 2:
                break
            r_xt += 1
        
        sampled_points.append((P_xt, r_xt, x_t, y_t))
        valid_mask[x_t, y_t] = False
        
        if len(sampled_points) % 1000 == 0:
            print(f"Remaining valid pixels: {valid_mask.sum()}")
        
    sample_map_path = os.path.join(output_path, f"{idx}.png")
    # print(f"Saving sample map to {sample_map_path}")
    cv2.imwrite(sample_map_path, sample_map)
    
    return sampled_points

def calculate_depth_error(sampled_points, depth_maps, camera_params, lambda_d):
    def project_3d_to_2d(point_3d, K, Rt, Tt):
        point_3d = np.dot(Rt.T, point_3d - Tt)
        u = point_3d[0] * K[0, 0] / point_3d[2] + K[0, 2]
        v = point_3d[1] * K[1, 1] / point_3d[2] + K[1, 2]
        # print(u, v)
        return int(u), int(v)

    def depth_error(P_xt, neighbors, camera_params, depth_t):
        errors = []
        for t_prime in neighbors:
            u, v = project_3d_to_2d(P_xt, camera_params[t_prime][0], camera_params[t_prime][1], camera_params[t_prime][2])
            if 0 <= u < len(depth_maps[t_prime][0]) and 0 <= v < len(depth_maps[t_prime]):
                depth_actual = depth_maps[t_prime][v, u]
                error = (depth_t - depth_actual) ** 2
                errors.append(error)
        # 如果没有errors，返回极大值
        return np.mean(errors) if errors else np.inf

    filtered_points = []
    for i, points in enumerate(sampled_points):
        valid_points = []
        neighbors = list(range(max(0, i - 20), min(len(depth_maps), i + 20), 2))
        error_limit = lambda_d * (depth_maps[i].max() - depth_maps[i].min())
        for P_xt, r_xt, x_t, y_t in points:
            error = depth_error(P_xt, neighbors, camera_params, depth_maps[i][x_t, y_t])
            # print(error)
            if error <= error_limit:
                valid_points.append((P_xt, r_xt, x_t, y_t))
        filtered_points.append(valid_points)
    return filtered_points

def sample_all(dataset, output_path='output', lambda_d=0.03):
    futures = []
    depth_images = []
    camera_params = []
    for i in range(len(dataset)):
        depth_images.append(dataset[i][1])
        camera_params.append(dataset[i][2])
    with ProcessPoolExecutor() as executor:
        for i in range(len(dataset)):
            Dt = depth_images[i].astype(np.float32)
            Kt = np.array(camera_params[i][0], dtype=np.float32)
            Rt = np.array(camera_params[i][1], dtype=np.float32)
            Tt = np.array(camera_params[i][2], dtype=np.float32)
            future = executor.submit(non_uniform_sampling, Dt, Kt, Rt, Tt, i, os.path.join(output_path, 'sample_map'))
            futures.append(future)
    
    sampled_points = []
    for future in as_completed(futures):
        sampled_points.append(future.result())
    
    filtered_points = calculate_depth_error(sampled_points, depth_images, camera_params, lambda_d)
    
    return filtered_points

# def sample_all_single(dataset, output_path='output', lambda_d=0.03):
#     sampled_points = []
#     depth_images = []
#     camera_params = []
#     for i in range(len(dataset)):
#         depth_images.append(dataset[i][1])
#         camera_params.append(dataset[i][2])
#     for i in range(len(dataset)):
#         rgb_image, depth_image, camera_param = dataset[i]
#         Dt = depth_image.astype(np.float32)
#         Kt = np.array(camera_param[0], dtype=np.float32)
#         Rt = np.array(camera_param[1], dtype=np.float32)
#         Tt = np.array(camera_param[2], dtype=np.float32)

#         sampled_points.append(non_uniform_sampling(Dt, Kt, Rt, Tt, i, os.path.join(output_path, 'sample_map')))
    
#     filtered_points = calculate_depth_error(sampled_points, depth_images, camera_params, lambda_d)
    
#     return filtered_points


if __name__ == '__main__':
    # root_directory = '/data/wangweijie/MV3D_Recon/data/SUNRGBD/xtion/sun3ddata/brown_bm_2/brown_bm_2'
    # root_directory = '/data/wangweijie/MV3D_Recon/data/SUNRGBD/xtion/sun3ddata/mit_lab_16/lab_16_nov_2_2012_scan1_erika'
    root_directory = '/data/wangweijie/MV3D_Recon/data/SCANNET/scene0707_00'
    
    print("Loading dataset...")
    dataset = SCANNETDataset(root_directory, num_frames=100)
    
    print("Sampling...")
    filtered_points = sample_all(dataset)
    # filtered_points = sample_all_single(dataset)
    
    
    vis = Visualizer("point")
    point_clouds = []
    for points in filtered_points:
        point_clouds.append(np.array([point[0] for point in points]))
    vis.add_points(point_clouds)
    

# if __name__ == '__main__':
#     root_directory = '/data/wangweijie/MV3D_Recon/data/SCANNET/scene0707_00'
#     dataset = SCANNETDataset(root_directory)
#     rgb_image0, depth_image0, camera_param0 = dataset[0]
    
#     cv2.imwrite("output/1.png", depth_image0)
#     Dt = depth_image0.astype(np.float32)
#     # print(Dt)
#     H, W = 10, 10
#     # depth_map从0到100，按顺序排列
#     depth_map = np.arange(100).reshape(H, W)
#     Kt = np.array(camera_param0[0], dtype=np.float32)
#     Rt = np.array(camera_param0[1], dtype=np.float32)
#     Tt = np.array(camera_param0[2], dtype=np.float32)

#     sampled_points = non_uniform_sampling(Dt, Kt, Rt, Tt)
    
#     vis = Visualizer("test")
#     point_cloud = np.array([point[0] for point in sampled_points])
#     vis.add_point_cloud(point_cloud)