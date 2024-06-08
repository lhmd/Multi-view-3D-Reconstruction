import sys
import cv2
import numpy as np
from wis3d.wis3d import Wis3D
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, sys.path[0]+"/../../")
from lib.dataset.sun import SUNDataset
sys.path.pop(0)

def non_uniform_sampling(depth_map, K, Rt, Tt, delta_d=8e-8, r_max=8):
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

    # 选择一个点进行采样
    while np.any(valid_mask):
        x_t, y_t = np.where(valid_mask)[0][0], np.where(valid_mask)[1][0]
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
        print(f"Remaining valid pixels: {valid_mask.sum()}")
        
    cv2.imwrite("output/sample_map.png", sample_map)
    
    return sampled_points

def sample_all(dataset):
    images, depth_maps, camera_params = zip(*dataset)
    
    futures = []
    with ThreadPoolExecutor() as executor:
        for i, depth_map in enumerate(depth_maps):
            future = executor.submit(non_uniform_sampling, depth_map.astype(np.float32),
                                     np.array(camera_params[i][0], dtype=np.float32),
                                     np.array(camera_params[i][1], dtype=np.float32),
                                     np.array(camera_params[i][2], dtype=np.float32))
            futures.append(future)
    
    results = []
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
        results.append(future.result())
    
def visualize(results):
    wis3d = Wis3D("output/vis", 'test', xyz_pattern=('x', 'y', 'z'))
    
    for i in range(len(results)):
        points = np.array([point for point, _ in results[i]])
        print(len(points))
        points /= points.max()
        color = np.array([255, 0, 0], dtype=np.uint8)
        wis3d.add_point_cloud(points, color)

def compute_depth_error(depth_map_t, depth_map_tp, points_t, transformation_matrix):
    errors = []
    for point in points_t:
        # Project the point onto the next frame using the transformation matrix
        projected_point = transformation_matrix @ np.append(point, 1)
        projected_point /= projected_point[3]
        projected_point = projected_point[:3]
        
        # Calculate the depth at the projected point in the next frame
        depth_t = depth_map_t[int(point[1]), int(point[0])]
        depth_tp = depth_map_tp[int(projected_point[1]), int(projected_point[0])]
        
        # Calculate the depth error
        error = np.linalg.norm(depth_t - depth_tp)
        errors.append(error)
    return np.array(errors)

def remove_high_error_points(depth_maps, points_sets, threshold_lambda):
    filtered_points_sets = []
    for t, points_t in enumerate(points_sets):
        if t == 0 or t == len(points_sets) - 1:
            filtered_points_sets.append(points_t)
            continue
        
        depth_errors = []
        for t_prime in range(max(0, t-20), min(len(points_sets), t+20), 2):
            if t_prime != t:
                transformation_matrix = np.eye(4)  # This should be replaced by the actual transformation matrix
                depth_error = compute_depth_error(depth_maps[t], depth_maps[t_prime], points_t, transformation_matrix)
                depth_errors.append(depth_error)
        
        depth_errors = np.mean(depth_errors, axis=0)
        
        # Thresholding based on lambda_d
        lambda_d = 0.03 * (depth_maps[t].max() - depth_maps[t].min())
        filtered_points_t = points_t[depth_errors <= lambda_d]
        filtered_points_sets.append(filtered_points_t)
    
    return filtered_points_sets

# Example usage
depth_maps = [np.random.rand(100, 100) for _ in range(10)]  # Replace with actual depth maps
points_sets = [np.random.rand(1000, 3) for _ in range(10)]  # Replace with actual 3D points

filtered_points_sets = remove_high_error_points(depth_maps, points_sets, threshold_lambda=0.03)


if __name__ == '__main__':
    root_directory = '/data/wangweijie/MV3D_Recon/data/SUNRGBD/xtion/sun3ddata/mit_lab_16/lab_16_nov_2_2012_scan1_erika'
    dataset = SUNDataset(root_directory)
    rgb_image0, depth_image0, camera_param0 = dataset[0]
    
    cv2.imwrite("output/1.png", depth_image0)
    Dt = depth_image0.astype(np.float32)
    # print(Dt)
    H, W = 10, 10
    depth_map = np.random.rand(H, W)
    Kt = np.array(camera_param0[0], dtype=np.float32)
    Rt = np.array(camera_param0[1], dtype=np.float32)
    Tt = np.array(camera_param0[2], dtype=np.float32)

    sampled_points = non_uniform_sampling(Dt, Kt, Rt, Tt)
    
    visualize([sampled_points])