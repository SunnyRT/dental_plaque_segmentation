import open3d as o3d
import numpy as np
import os
import shutil


"""-------------------------PCA normalization-------------------------"""
def align_mesh_principal_axes(mesh):
    # Compute the covariance matrix of the vertices
    vertices = np.asarray(mesh.vertices)
    cov = np.cov(vertices.T)
    
    # Perform eigen decomposition to get the principal axes
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort the eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Align the principal axes to the global axes
    # Primary (longest) to X, Secondary to Z, Tertiary to Y
    target_axes = np.array([[1, 0, 0],  # X-axis
                            [0, 0, 1],  # Z-axis
                            [0, 1, 0]]) # Y-axis
    
    rotation_matrix = np.dot(eigenvectors, target_axes)

    # Check for reflection/mirroring, ensure normals are not inverted
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, 2] *= -1

    mesh.rotate(rotation_matrix, center=(0, 0, 0))
    
    return mesh, rotation_matrix


def center_mesh_bb(mesh):
    # Compute the center of the bounding box
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox_center = bbox.get_center()
    
    # Translate the mesh to the origin
    mesh.translate(-bbox_center)
    
    return mesh, -bbox_center # translation in x,y,z 

def compute_centre(mesh):
    vertices = np.asarray(mesh.vertices)
    centre = np.mean(vertices, axis=0)
    # print(f"vertices mean: {centre}")
    return centre

def align_mesh_orientation(mesh): # input mesh must be centered with its bounding box (bb)

    centre = compute_centre(mesh)
    rotation_matrix = np.eye(3)

    if centre[1] < 0:
        if centre[2] < 0: 
            # rotate 180 degrees around x
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz([np.pi, 0, 0])
            mesh.rotate(rotation_matrix, center=(0, 0, 0))
        else: 
            # rotate 180 degrees around z
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, np.pi])
            mesh.rotate(rotation_matrix, center=(0, 0, 0))
    
    else:
        if centre[2] < 0:
            # rotate 180 degrees around y
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz([0, np.pi, 0])
            mesh.rotate(rotation_matrix, center=(0, 0, 0))
        else:
            pass
    
    # centre = compute_centre(mesh)

    return mesh, rotation_matrix






"""-------------------------iterative region growth-------------------------"""
def find_seed_point(mesh, seg_labels): # grow from end boundary of gum (bottom of mesh)
    # Get the vertices as a numpy array
    vertices = np.asarray(mesh.vertices)
    # Find the index of the vertex with min y-coordinate 

    unlabeled_indices = np.where(seg_labels == -1)[0]
    seed_index = unlabeled_indices[np.argmin(vertices[unlabeled_indices, 1])]
    
    return seed_index

def create_adjacency_list(mesh):
    adjacency_list = {i: set() for i in range(len(mesh.vertices))}
    triangles = np.asarray(mesh.triangles)
    for triangle in triangles:
        for i, j in zip(triangle, triangle[[1, 2, 0]]):
            adjacency_list[i].add(j)
            adjacency_list[j].add(i)
    return adjacency_list

def region_growing_segmentation(mesh, adjacency_list, seed_index, seg_labels, y_threshold=0.02, normal_threshold=0.9, color_threshold=0.1):
    """
    Perform region growing segmentation on a mesh starting from a seed index using only z-axis distance.

    Parameters:
    - mesh: open3d.geometry.TriangleMesh, the input mesh
    - adjacency_list: dict, adjacency list of vertices
    - seed_index: int, the index of the seed vertex (start from gum)
    - y_threshold: float, y-axis distance threshold for region growing
    - normal_threshold: float, normal dot product threshold for region growing
    - color_threshold: float, color difference threshold for region growing

    Returns:
    - seg_labels: np.ndarray, an array of seg_labels for each vertex in the mesh
    
    Labels:
    - 1: within the gum region
    - 0: gum-boundary region
    - -1: unlabeled, outside the gum region (i.e. teeth region)
    """

    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    colors = np.asarray(mesh.vertex_colors)

    
    region = [seed_index]
    seg_labels[seed_index] = 1

    while region:
        current_index = region.pop()
        current_vertex = vertices[current_index]
        current_normal = normals[current_index]
        current_color = colors[current_index]
        
        for neighbor_index in adjacency_list[current_index]:
            if seg_labels[neighbor_index] == -1:
                neighbor_vertex = vertices[neighbor_index]
                neighbor_normal = normals[neighbor_index]
                neighbor_color = colors[neighbor_index]
                
                y_distance = abs(current_vertex[2] - neighbor_vertex[2])
                normal_dot = np.dot(current_normal, neighbor_normal)
                color_diff = np.linalg.norm(current_color - neighbor_color)
                
                if y_distance < y_threshold and normal_dot > normal_threshold and color_diff < color_threshold:
                    seg_labels[neighbor_index] = 0
                    region.append(neighbor_index)
                else:
                    seg_labels[neighbor_index] = 1
    
    return seg_labels



def display_region_growth_outcome(input_file_path, output_file_path, seg_labels):
    # Duplicate the original mesh
    # Update the vertex colors based on labels for visualiation

    shutil.copyfile(input_file_path, output_file_path)
    segmented_mesh = o3d.io.read_triangle_mesh(output_file_path)

    colors = np.asarray(segmented_mesh.vertex_colors)
    for i in range(len(seg_labels)):
        if seg_labels[i] == 1:
            colors[i] = [0, 1, 0]  # Green for label 1
        elif seg_labels[i] == 0:
            colors[i] = [1, 0, 0]  # Red for label 0
        # Else keep the original color for label -1

    segmented_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_triangle_mesh(output_file_path, segmented_mesh)




def process_all_files(input_dir, output_dir, input_dir_label, output_dir_label, y_threshold, normal_threshold, color_threshold):
    for file_name in os.listdir(input_dir):
        # split by "_" to get the base name
        base_name  = file_name.split("_")[0]

        print(f"Processing {base_name}")
        if file_name.endswith(".ply"):
            input_file_path = os.path.join(input_dir, file_name)
            output_file_path = os.path.join(output_dir, file_name)
            input_file_path_label = os.path.join(input_dir_label, f"{base_name}.ply")
            output_file_path_label = os.path.join(output_dir_label, f"{base_name}.ply")

            # Load the mesh
            mesh = o3d.io.read_triangle_mesh(input_file_path)
            
            """-----------Normalize the PCA axes and bb center the mesh-----------"""
            """ Track the transformation matrices, which will be applied for the label mesh later"""
            mesh, rot1 = align_mesh_principal_axes(mesh)
            mesh, trans2 = center_mesh_bb(mesh)
            mesh, rot3 = align_mesh_orientation(mesh)

            # """-----------Perform teeth-gum separation with region growth-----------"""
            # mesh.compute_vertex_normals()
            # vertices = np.asarray(mesh.vertices)
            # triangles = np.asarray(mesh.triangles)   

            # # Create adjacency list for vertices
            # adjacency_list = create_adjacency_list(mesh)

            # # Initialize seg_labels array
            # seg_labels = np.full(len(vertices), -1, dtype=int)        
            # n_teeth_triangles = np.count_nonzero(seg_labels == -1)
            # n_itr = 0    

            # while n_teeth_triangles > 22000 and n_itr < 20: # face numbers reduced by half
            #     n_itr += 1
            #     seed_index = find_seed_point(mesh, seg_labels)
            #     seg_labels = region_growing_segmentation(mesh, adjacency_list, seed_index, seg_labels, y_threshold, normal_threshold, color_threshold)
            #     n_teeth_triangles = np.count_nonzero(seg_labels == -1)
            #     # print(f"Iteration {n_itr}: Number of teeth triangles: {n_teeth_triangles}")

            # print(f"Itr {n_itr} => Number of teeth, gum, boundary triangles: {n_teeth_triangles}, {np.count_nonzero(seg_labels == 1)}, {np.count_nonzero(seg_labels == 0)}")




            # """-----------Extract and save the segmented mesh-----------"""
            # teeth_triangles = triangles[np.all(seg_labels[triangles] == -1, axis=1)]
            # if len(teeth_triangles) == 0:
            #     raise ValueError("No teeth region found.")

            # # shutil.copyfile(input_file_path, output_file_path)
            # mesh.triangles = o3d.utility.Vector3iVector(teeth_triangles)

            # # Remove disconnected small component pieces => preserve the largest connected component
            # comps_label = np.array(mesh.cluster_connected_triangles()[0])
            # assert len(comps_label) == len(teeth_triangles), "Mismatch between number of triangles and size of comps_label array."
            # maxcomp = np.argmax(np.bincount(comps_label)) # largest connected component
            # mask_maxcomp = (comps_label == maxcomp)
            # teeth_triangles = teeth_triangles[mask_maxcomp]
            # mesh.triangles = o3d.utility.Vector3iVector(teeth_triangles)
            
            # Correct face normals
            mesh.orient_triangles()
            # mesh.compute_vertex_normals()

            o3d.io.write_triangle_mesh(output_file_path, mesh)


            """-----------Segment the corresponding label mesh-----------"""
            
            # Read the label mesh
            if not os.path.exists(input_file_path_label):
                print(f"Label file {input_file_path_label} does not exist.")
                continue
            else:
                mesh_label = o3d.io.read_triangle_mesh(input_file_path_label)
                # reorientate the label mesh with the same transformation matrices as original mesh 
                mesh_label.rotate(rot1, center = (0, 0, 0))
                mesh_label.translate(trans2)
                mesh_label.rotate(rot3, center = (0, 0, 0))
                # mesh_label.triangles = o3d.utility.Vector3iVector(teeth_triangles)

                # Correct face normals
                mesh_label.orient_triangles()
                # mesh_label.compute_vertex_normals()
                o3d.io.write_triangle_mesh(output_file_path_label, mesh_label)



            





"""-------------------------Set parameters-------------------------"""
if __name__ == "__main__":
    input_dir = "D:\sunny\Codes\DPS\data\Origin"
    output_dir = "D:\sunny\Codes\DPS\data_withgum_orient\origin"
    input_dir_label = "D:\sunny\Codes\DPS\data\Label"
    output_dir_label = "D:\sunny\Codes\DPS\data_withgum_orient\label"

    # y_threshold = 11.0
    # normal_threshold = 0.986 
    # color_threshold = 0.05
    y_threshold = 100
    normal_threshold = 0
    color_threshold = 1

    process_all_files(input_dir, output_dir, input_dir_label, output_dir_label, y_threshold, normal_threshold, color_threshold)

