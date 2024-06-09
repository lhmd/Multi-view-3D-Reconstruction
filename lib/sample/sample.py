import sys
import cv2
import numpy as np
from wis3d.wis3d import Wis3D
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, sys.path[0] + "/../../")
from lib.dataset.sun import SUNDataset
sys.path.pop(0)

def non_uniform_sampling(depth_map, K, Rt, Tt, delta_d=7e-3, r_max=16):

    def reproject_image_to_3d(image):
        height, width = image.shape
        
        u = np.tile(np.arange(width), (height, 1))
        v = np.tile(np.arange(height).reshape(-1, 1), (1, width))
        
        u_transformed = (u - K[0, 2]) * image / K[0, 0]
        v_transformed = (v - K[1, 2]) * image / K[1, 1]
        
        points_3d = np.stack([u_transformed, v_transformed, image], axis=-1)
        
        points_3d = points_3d.reshape(-1, 3)
        points_3d = (points_3d - Tt).dot(Rt.T)
        
        points_3d = points_3d.reshape(height, width, 3)
        
        return points_3d

    def reproject_pixel_to_3d(u, v, depth):
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
        # local_points = []

        # for dx in range(-1, 2):
        #     for dy in range(-1, 2):
        #         nx, ny = x_t + dx, y_t + dy
        #         if 0 <= nx < H and 0 <= ny < W and valid_mask[nx, ny] and depth_map[nx, ny] > 0:
        #             # depth_n = depth_map[nx, ny]
        #             # P_n = reproject_pixel_to_3d(nx, ny, depth_n)
        #             P_n = point_3d_map[nx, ny]
        #             local_points.append(P_n)
                    
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
            # for dx in range(-r_xt, r_xt + 1):
            #     for dy in [-r_xt, r_xt]:
            #         nx, ny = x_t + dx, y_t + dy
            #         if 0 <= nx < H and 0 <= ny < W and valid_mask[nx, ny] and depth_map[nx, ny] > 0:
            #             # depth_n = depth_map[nx, ny]
            #             # P_n1 = reproject_pixel_to_3d(nx, ny, depth_n)
            #             P_n = point_3d_map[nx, ny]
            #             if is_in_plane(P_n, normal, P_xt):
            #                 valid_mask[nx, ny] = False
            #             else:
            #                 cnt += 1
            # for dy in range(-r_xt, r_xt + 1):
            #     for dx in [-r_xt, r_xt]:
            #         nx, ny = x_t + dx, y_t + dy
            #         if 0 <= nx < H and 0 <= ny < W and valid_mask[nx, ny] and depth_map[nx, ny] > 0:
            #             # depth_n = depth_map[nx, ny]
            #             # P_n = reproject_pixel_to_3d(nx, ny, depth_n)
            #             P_n = point_3d_map[nx, ny]
            #             if is_in_plane(P_n, normal, P_xt):
            #                 valid_mask[nx, ny] = False
            #             else:
            #                 cnt += 1
            
            if cnt > 0.75 * (2 * r_xt + 1) ** 2:
                break
            r_xt += 1
        
        sampled_points.append((P_xt, r_xt, x_t, y_t))
        valid_mask[x_t, y_t] = False
        
        # if len(sampled_points) % 500 == 0:
        #     print(f"Remaining valid pixels: {valid_mask.sum()}")
        
    cv2.imwrite("output/sample_map.png", sample_map)
    
    return sampled_points

def calculate_depth_error(sampled_points, depth_maps, camera_params, lambda_d):
    def project_3d_to_2d(point_3d, K, Rt, Tt):
        point_3d = np.dot(Rt, point_3d) + Tt
        u = point_3d[0] * K[0, 0] / point_3d[2] + K[0, 2]
        v = point_3d[1] * K[1, 1] / point_3d[2] + K[1, 2]
        print(u, v)
        return int(u), int(v)

    def depth_error(P_xt, neighbors, camera_params, depth_t):
        errors = []
        for t_prime in neighbors:
            u, v = project_3d_to_2d(P_xt, camera_params[t_prime][0], camera_params[t_prime][1], camera_params[t_prime][2])
            if 0 <= u < depth_maps[t_prime].shape[1] and 0 <= v < depth_maps[t_prime].shape[0]:
                depth_actual = depth_maps[t_prime][v, u]
                error = (depth_t - depth_actual) ** 2
                errors.append(error)
        return np.mean(errors)

    filtered_points = []
    for i, points in enumerate(sampled_points):
        valid_points = []
        neighbors = list(range(max(0, i - 20), min(len(depth_maps), i + 20), 2))
        for P_xt, r_xt, x_t, y_t in points:
            error = depth_error(P_xt, neighbors, camera_params, depth_maps[i][x_t, y_t])
            if error <= lambda_d:
                valid_points.append((P_xt, r_xt))
        filtered_points.append(valid_points)
    return filtered_points

def sample_all(dataset, lambda_d=1.2):
    images, depth_maps, camera_params = zip(*dataset)
    
    futures = []
    with ThreadPoolExecutor() as executor:
        for i, depth_map in enumerate(depth_maps):
            future = executor.submit(non_uniform_sampling, depth_map.astype(np.float32),
                                     np.array(camera_params[i][0], dtype=np.float32),
                                     np.array(camera_params[i][1], dtype=np.float32),
                                     np.array(camera_params[i][2], dtype=np.float32))
            futures.append(future)
    
    sampled_points = []
    for future in as_completed(futures):
        sampled_points.append(future.result())
    
    filtered_points = calculate_depth_error(sampled_points, depth_maps, camera_params, lambda_d)
    
    return filtered_points

def visualize(all_points):
    wis3d = Wis3D("output/vis", 'test', xyz_pattern=('x', 'y', 'z'))
    
    
    for points in all_points:
        point_cloud = np.array([point[0] for point in points])
        # print(len(point_cloud))
        point_cloud /= point_cloud.max()
        color = np.array([255, 0, 0], dtype=np.uint8)
        wis3d.add_point_cloud(point_cloud, color)
        print("Point number: ", len(point_cloud))

if __name__ == '__main__':
    root_directory = '/data/wangweijie/MV3D_Recon/data/SUNRGBD/xtion/sun3ddata/brown_bm_2/brown_bm_2'
    # root_directory = '/data/wangweijie/MV3D_Recon/data/SUNRGBD/xtion/sun3ddata/mit_lab_16/lab_16_nov_2_2012_scan1_erika'
    dataset = SUNDataset(root_directory)
    
    filtered_points = sample_all(dataset)
    
    visualize(filtered_points)

# if __name__ == '__main__':
#     root_directory = '/data/wangweijie/MV3D_Recon/data/SUNRGBD/xtion/sun3ddata/mit_lab_16/lab_16_nov_2_2012_scan1_erika'
#     dataset = SUNDataset(root_directory)
#     rgb_image0, depth_image0, camera_param0 = dataset[0]
    
#     cv2.imwrite("output/1.png", depth_image0)
#     Dt = depth_image0.astype(np.float32)
#     # print(Dt)
#     H, W = 10, 10
#     depth_map = np.random.rand(H, W)
#     Kt = np.array(camera_param0[0], dtype=np.float32)
#     Rt = np.array(camera_param0[1], dtype=np.float32)
#     Tt = np.array(camera_param0[2], dtype=np.float32)

#     sampled_points = non_uniform_sampling(Dt, Kt, Rt, Tt)
    
#     visualize([sampled_points])