import os
import numpy as np

def save_point_cloud(point_cloud, file_path):
    """
    Save point cloud to file
    :param point_cloud: Point cloud [(points, r, x, y)]
    :param file_path: File path
    """
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'w') as f:
        for point, r, x, y in point_cloud:
            f.write(f"{point[0]} {point[1]} {point[2]} {r} {x} {y}\n")
    
def save_points(points, folder_path):
    """
    Save points to folder
    :param points: Points
    :param folder_path: Folder path
    """
    if os.path.exists(folder_path):
        os.system(f'rm -r {folder_path}')
    os.makedirs(folder_path)
    for i, point in enumerate(points):
        save_point_cloud(point, folder_path + f'/{i}.txt')
        
def read_point_cloud(file_path):
    """
    Read point cloud from file
    :param file_path: File path
    :return: Point cloud
    """
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            x, y, z, r, u, v = line.strip().split(' ')
            points.append((np.array([float(x), float(y), float(z)]), int(r), int(u), int(v)))
    return points

def read_points(folder_path):
    """
    Read points from folder
    :param folder_path: Folder path
    :return: Points
    """
    points = []
    i = 0
    while True:
        try:
            points.append(read_point_cloud(folder_path + f'/{i}.txt'))
            i += 1
        except FileNotFoundError:
            break
    return points
    