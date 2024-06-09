import numpy as np
from wis3d.wis3d import Wis3D

class Visualizer:
    def __init__(self, output_dir):
        self.wis3d = Wis3D('output/vis', output_dir, xyz_pattern=('x', 'y', 'z'))

    def add_points(self, all_points):
        for points in all_points:
            # point_cloud = np.array([point[0] for point in points])
            # print(len(point_cloud))
            # point_cloud /= point_cloud.max()
            color = np.array([255, 0, 0], dtype=np.uint8)
            self.wis3d.add_point_cloud(points, color)
            print("Point number: ", len(points))

        
    def add_point_cloud(self, point_cloud):
        print("Point number: ", len(point_cloud))
        # print(point_cloud[0])
        color = np.array([255, 0, 0], dtype=np.uint8)
        self.wis3d.add_point_cloud(point_cloud, color)