
import sys
import heapq

import numpy as np
from collections import deque

sys.path.insert(0, sys.path[0] + "/../../")
from lib.dataset.sun import SUNDataset
from lib.dataset.scannet import SCANNETDataset
from lib.visualize.wis3d import Visualizer
sys.path.pop(0)

def do_fusion(dataset, points, delta_f=0.86):
    
    def reproject_image_to_3d(image, K, Rt, Tt):
        # H, W = image.shape
        H = len(image)
        W = len(image[0])
        u = np.tile(np.arange(W), (H, 1))
        v = np.tile(np.arange(H).reshape(-1, 1), (1, W))
        
        u_transformed = (u - K[0, 2]) * image / K[0, 0]
        v_transformed = (v - K[1, 2]) * image / K[1, 1]
        
        points_3d = np.stack([u_transformed, v_transformed, image], axis=-1)
        
        points_3d = points_3d.reshape(-1, 3)
        points_3d = np.dot(Rt, points_3d.T).T + Tt.T
        
        points_3d = points_3d.reshape(H, W, 3)
        
        return points_3d
    
    def reproject_points_2d_to_3d(depth, u, v, K, Rt, Tt):
        u = (u - K[0, 2]) * depth / K[0, 0]
        v = (v - K[1, 2]) * depth / K[1, 1]
        points_3d = np.stack([u, v, depth], axis=-1)
        
        points_3d = points_3d.reshape(-1, 3)
        points_3d = np.dot(Rt, points_3d.T).T + Tt.T
        
        points_3d = points_3d.reshape(-1, 3)
        
        return points_3d
    
    
    def project_3d_to_image(point_3d, K, Rt, Tt):
        point_3d = point_3d.reshape(-1, 3)
        point_3d = np.dot(Rt.T, (point_3d - Tt.T).T)
        u = point_3d[0] * K[0, 0] / point_3d[2] + K[0, 2]
        v = point_3d[1] * K[1, 1] / point_3d[2] + K[1, 2]
        u = u.astype(np.int64)
        v = v.astype(np.int64)
        return u, v
    
    def confidence(point_cloud, rgb_image, depth_image, camera_param, idx, delta=0.01):
        """
        对每个三维采样点计算其深度置信度, 并对当前帧所有的三维采样点按照置信度由高到低排序得到有序队列Qt
        """
        conf_que = deque()
        neighbors = list(range(max(0, idx - 20), min(len(depth_images), idx + 20), 2))
        if idx in neighbors:
            neighbors.remove(idx)
        for P, r, u, v in point_cloud:
            x, y, z = P
            delta_d = delta * (depth_images[idx].max() - depth_images[idx].min())
            n_depth = [depth_images[idx][u, v] + k * delta_d for k in range(-2, 3)]
            u_s = np.full(5, u)
            v_s = np.full(5, v)
            current_color = rgb_image[u - 1: u + 2, v - 1: v + 2]
            reprojected_points = reproject_points_2d_to_3d(n_depth, u_s, v_s, camera_param[0], camera_param[1], camera_param[2])
            conf_neihbors = []
            for k in neighbors:
                x_coords, y_coords = project_3d_to_image(reprojected_points, camera_params[k][0], camera_params[k][1], camera_params[k][2])
                # u_coords, v_coords = project_3d_to_image(reprojected_points, camera_param[0], camera_param[1], camera_param[2])
                # print(sum(u_coords - u_s), sum(v_coords - v_s))
                now_error = float('inf')
                errors = []
                for h, (a, b) in enumerate(zip(x_coords, y_coords)):
                    neighbour_color = rgb_images[k][a - 1: a + 2, b - 1: b + 2]
                    # print(neighbour_color)
                    if neighbour_color.shape != (3, 3, 3) or current_color.shape != (3, 3, 3):
                        continue
                    if h == 2:
                        now_error = np.sum((current_color - neighbour_color)**2)
                    else:
                        errors.append(np.sum((current_color - neighbour_color)**2))
                if now_error == float('inf') or now_error == 0 or not errors:
                    continue
                if now_error == 0:
                    now_error = 1e-6
                min_error = min(errors)
                conf_neihbors.append(min_error / now_error)
            if not conf_neihbors:
                conf_neihbors = [0]
            conf_point = sum(conf_neihbors) / len(conf_neihbors)
            # if conf_point > 1.0:
            #     print("Confidence: ", conf_point)
            conf_que.append((-conf_point, x, y, z, u, v, r, idx))
        conf_que = deque(sorted(conf_que))
        return conf_que
            
        # que = [-conf, x, y, z, u, v, r, id]
        pass
    
    rgb_images = []
    depth_images = []
    camera_params = []
    confidence_queue = []
    for i in range(len(dataset)):
        rgb_images.append(dataset[i][0])
        depth_images.append(dataset[i][1])
        camera_params.append(dataset[i][2])
        
    H = len(depth_images[0])
    W = len(depth_images[0][0])
    # print("The origin number of points is: ", sum([len(p) for p in points]))
    for i, point_cloud in enumerate(points):
        que = confidence(point_cloud, rgb_images[i], depth_images[i], camera_params[i], i)
        confidence_queue.append(que)
    
    # 取出queue中每个三维点队列最顶端的三维点, 创建一个置信度最大堆 H, 如果是空值则跳过
    conf_h = []
    for q in confidence_queue:
        if q:
            conf_h.append(q[0]) 
    # conf_h = [q.popleft() for q in confidence_queue]
    heapq.heapify(conf_h)
    print("Confidence queue created, the number of points is: ", sum([len(q) for q in confidence_queue]))
    
    for i, q in enumerate(confidence_queue):
        print(f"Frame {i} has {len([x for x in q if x[0] <= -1.0])} points with confidence greater than 1.0")
        
    confs = [val[0] for q in confidence_queue for val in q]
    confs = np.array(confs)
    print(confs)
    from collections import Counter
    intervals = [(int(x * 100) / 100) for x in confs]
    interval_counts = Counter(intervals)
    for interval, count in sorted(interval_counts.items()):
        print(f"Interval [{interval:.2f}, {interval + 0.01:.2f}): {count}")
    
    
    # 创建一个按照置信度由高到低排序的空队列 M
    merged_queue = []
    while conf_h:
        conf, x, y, z, u, v, r, idx = heapq.heappop(conf_h)
        print("Now Confidence: ", -conf)
        if -conf < delta_f:
            break
        # print(f"Processing point {x}, {y}, {z} at frame {idx}")
        merged_queue.append((x, y, z))
        square = [(x, y) for x in range(u - r, u + r) for y in range(v - r, v + r)]
        projected_points = reproject_image_to_3d(square, camera_params[idx][0], camera_params[idx][1], camera_params[idx][2])
        # 对于该点所属帧 t 之外其余的每帧
        for j in range(len(points)):
            if j == idx:
                continue
            x_other_coords, y_other_coords = project_3d_to_image(projected_points, camera_params[j][0], camera_params[j][1], camera_params[j][2])
            for x_other, y_other in zip(x_other_coords, y_other_coords):
                if x_other < 0 or x_other >= H or y_other < 0 or y_other >= W:
                    continue
                # print(depth_images[j][x_other, y_other] - depth_images[idx][u, v])
                if abs(depth_images[j][x_other, y_other] - depth_images[idx][u, v]) < 0.03:
                    # print("Remove point")
                    q = [q for q in confidence_queue[j] if q[4] == x_other and q[5] == y_other]
                    if q:
                        confidence_queue[j].remove(q[0])
                    # confidence_queue[j] = deque([q for q in confidence_queue[j] if not (q[4] == x_other and q[5] == y_other)])
                    # for q in confidence_queue[j]:
                    #     if q[4] == x_other and q[5] == y_other:
                    #         confidence_queue[j].remove(q)
                            
        if confidence_queue[idx]:
            # print("New Confidence: ", confidence_queue[idx][0][0])
            heapq.heappush(conf_h, confidence_queue[idx].popleft())
            
    return np.array(merged_queue)