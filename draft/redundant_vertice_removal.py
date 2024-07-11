import open3d as o3d
import numpy as np
import os


def remove_redundant_vertices(mesh):
    # Get the vertices and triangles
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Identify unique vertex indices used in the triangles
    unique_vertex_indices = np.unique(triangles)
    
    # Create a mapping from old vertex indices to new ones
    old_to_new_indices = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)}
    
    # Create a new list of vertices that are only the used ones
    new_vertices = vertices[unique_vertex_indices]
    
    # Update triangle indices to the new vertex indices
    new_triangles = np.array([[old_to_new_indices[old_idx] for old_idx in triangle] for triangle in triangles])
    
    # Create a new mesh with the cleaned vertices and updated triangles
    cleaned_mesh = o3d.geometry.TriangleMesh()
    cleaned_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    cleaned_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    
    # Copy colors, normals, etc. if they exist
    if mesh.has_vertex_colors():
        cleaned_mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors)[unique_vertex_indices])
    if mesh.has_vertex_normals():
        cleaned_mesh.vertex_normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals)[unique_vertex_indices])
    
    return cleaned_mesh

def process_all_files(input_dir, output_dir):
    for file in os.listdir(input_dir):
        if file.endswith('.ply'):
            # Load the mesh
            print(f"process file: {file}")
            mesh = o3d.io.read_triangle_mesh(os.path.join(input_dir, file))
            print(f"before: v:{np.array(mesh.vertices).shape}, f:{np.array(mesh.triangles).shape}")

            # Remove redundant vertices
            intraoral_mesh = remove_redundant_vertices(mesh)
            print(f"after: v:{np.array(intraoral_mesh.vertices).shape}, f:{np.array(intraoral_mesh.triangles).shape}")

            # Save the cleaned mesh
            o3d.io.write_triangle_mesh(os.path.join(output_dir, file), intraoral_mesh)


if __name__ == '__main__':
    input_dir = r'D:\sunny\Codes\DPS\data_teethseg_v0\label'
    output_dir = r'D:\sunny\Codes\DPS\data_teethseg\label'
    process_all_files(input_dir, output_dir)
