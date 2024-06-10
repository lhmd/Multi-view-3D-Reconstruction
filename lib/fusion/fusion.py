
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
    
    
    def project_3d_to_image(point_3d, K, Rt, Tt):
        point_3d = point_3d.reshape(-1, 3)
        point_3d = np.dot(Rt.T, (point_3d - Tt.T).T)
        u = point_3d[0] * K[0, 0] / point_3d[2] + K[0, 2]
        v = point_3d[1] * K[1, 1] / point_3d[2] + K[1, 2]
        u = u.astype(np.int)
        v = v.astype(np.int)
        return u, v
    
    def confidence(point_cloud, rgb_image, depth_image, camera_param, idx):
        """
        对每个三维采样点计算其深度置信度，并对当前帧所有的三维采样点按照置信 度由高到低排序得到有序队列Qt
        """
        neighbors = list(range(max(0, idx - 20), min(len(depth_images), idx + 20), 2))
        for x, y, z, r, u, v in point_cloud:
            confidences = []
            
            
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
            
    
                    

            