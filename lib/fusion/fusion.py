
import sys
import heapq

import numpy as np
from collections import deque

sys.path.insert(0, sys.path[0] + "/../../")
from lib.dataset.sun import SUNDataset
from lib.dataset.scannet import SCANNETDataset
from lib.visualize.wis3d import Visualizer
sys.path.pop(0)

def do_fusion(dataset, points, delta_f=0.8):
    
    def reproject_image_to_3d(image, K, Rt, Tt):
        H, W = image.shape
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
        u = u.astype(np.int)
        v = v.astype(np.int)
        return u, v
    
    def confidence(point_cloud, rgb_image, depth_image, camera_param, idx, delta=0.01):
        """
        对每个三维采样点计算其深度置信度，并对当前帧所有的三维采样点按照置信度由高到低排序得到有序队列Qt
        """
        conf_que = deque()
        neighbors = list(range(max(0, idx - 20), min(len(depth_images), idx + 20), 2))
        for P, r, u, v in point_cloud:
            x, y, z = P
            confidences = []
            delta_d = delta * (depth_images[idx].max() - depth_images[idx].min())
            n_depth = [depth_images[idx][u, v] + k * delta_d for k in range(-2, 3)]
            u_s = np.full(5, u)
            v_s = np.full(5, v)
            current_color = rgb_image[u - 1: u + 2, v - 1: v + 2]
            reprojected_points = reproject_points_2d_to_3d(n_depth, u_s, v_s, camera_param[0], camera_param[1], camera_param[2])
            conf_neihbors = []
            for i in neighbors:
                project_other_points = project_3d_to_image(reprojected_points, camera_params[i][0], camera_params[i][1], camera_params[i][2])
                now_error = 0
                errors = []
                for i, a, b in enumerate(project_other_points):
                    neighbour_color = rgb_images[i][a - 1: a + 2, b - 1: b + 2]
                    if i == 2:
                        now_error = np.sum((current_color - neighbour_color)**2)
                    else:
                        errors.append(np.sum((current_color - neighbour_color)**2))
                min_error = min(errors)
                conf_neihbors.append(min_error / now_error)
            conf_point = conf_neihbors.sum() / len(conf_neihbors)
            conf_que.append([-conf_point, x, y, z, u, v, r, idx])
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
    for i, point_cloud in enumerate(points):
        que = confidence(point_cloud, rgb_images[i], depth_images[i], camera_params[i], i)
        confidence_queue.append(que)
    # 取出queue中每个三维点队列最顶端的三维点, 创建一个置信度最大堆 H
    conf_h = [q.popleft() for q in confidence_queue]
    heapq.heapify(conf_h)
    # 创建一个按照置信度由高到低排序的空队列 M
    merged_queue = []
    while conf_h:
        conf, x, y, z, u, v, r, idx = heapq.heappop(conf_h)
        if conf <= delta_f:
            break
        merged_queue.append((x, y, z))
        square = [(x, y) for x in range(u - r, u + r) for y in range(v - r, v + r)]
        projected_points = reproject_image_to_3d(square, camera_params[idx][0], camera_params[idx][1], camera_params[idx][2])
        # 对于该点所属帧 t 之外其余的每帧
        for j in range(len(points)):
            if j == idx:
                continue
            projected_square = project_3d_to_image(projected_points, camera_params[j][0], camera_params[j][1], camera_params[j][2])
            for x, y in projected_square:
                if abs(depth_images[j][x, y] - depth_images[idx][u, v]) < 0.1:
                    for q in confidence_queue[j]:
                        if q[4] == x and q[5] == y:
                            confidence_queue[j].remove(q)
                            
        if confidence_queue[idx]:
            heapq.heappush(conf_h, confidence_queue[idx].popleft())
            
    return merged_queue