import numpy as np
import open3d as o3d

from trimesh import Trimesh

def poisson_reconstruction(points, output_mesh_path=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(100)
    # 泊松表面重建
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    
    # 可选：根据密度信息过滤掉一些面
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    if output_mesh_path is not None:
        o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    
    # 转换为trimesh格式
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    mesh = Trimesh(vertices=vertices, faces=faces)
    
    return mesh
    
