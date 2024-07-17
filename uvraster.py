import open3d as o3d
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
import os




def process_all_files(origin_dir, label_dir, save_dir):
    for file in os.listdir(origin_dir):
        if file.endswith(".ply"):
            mesh_name = file.split("_")[0]
            print(f"Processing {mesh_name}...")
            origin_file_path = f"{origin_dir}/{mesh_name}_origin.ply"
            label_file_path = f"{label_dir}/{mesh_name}.ply"
            intraoral_mesh = o3d.io.read_triangle_mesh(origin_file_path)
            label_mesh = o3d.io.read_triangle_mesh(label_file_path)
            process_single_mesh(intraoral_mesh, label_mesh, save_dir, mesh_name)


def process_single_mesh(intraoral_mesh, label_mesh, save_dir, mesh_name):
    # Recenter the meshes
    intraoral_mesh = intraoral_mesh.translate(-intraoral_mesh.get_center())
    label_mesh = label_mesh.translate(-label_mesh.get_center())


    """"""""""""""""1. Create Circle and Extrude to Cylinders as Projection Screens"""""""""""""""

    # Calculate the radii
    distances = np.linalg.norm(np.asarray(intraoral_mesh.vertices)[:, [0, 2]], axis=1)
    outer_radius = np.max(distances)
    inner_radius = np.min(distances)
    n_points = 100

    # Calculate the extrusion length
    y_min = np.min(np.asarray(intraoral_mesh.vertices)[:, 1])
    y_max = np.max(np.asarray(intraoral_mesh.vertices)[:, 1])
    extrusion_length_pos = y_max
    extrusion_length_neg = y_min


    """""""""""""""""""""""""""""""""" 2. Mesh Separation (Inward and Outward)"""""""""""""""""""""""""""""""""""
    # Calculate vertex normals
    vertices = np.asarray(intraoral_mesh.vertices)
    triangles = np.asarray(intraoral_mesh.triangles)
    intraoral_mesh.compute_vertex_normals()
    vertex_normals = np.asarray(intraoral_mesh.vertex_normals)
    vertex_colors = np.asarray(intraoral_mesh.vertex_colors)

    # Classify each vertex as outward-facing or inward-facing
    outward_vertex_mask = np.array([np.dot(vertex_normal, vertex) > 0 for vertex_normal, vertex in zip(vertex_normals, vertices)])
    inward_vertex_mask = ~outward_vertex_mask

    # Convert vertex mask into face mask
    outward_face_mask = np.any(outward_vertex_mask[triangles], axis=1)
    inward_face_mask = ~outward_face_mask

    # Clean mesh separation
    mesh_in, mesh_out = clean_io_separation(intraoral_mesh, outward_face_mask, inward_face_mask)
    label_mesh_in, label_mesh_out = clean_io_separation(label_mesh, outward_face_mask, inward_face_mask)


    """"""""""""""""""""""""""" 3. Project 3D Meshes onto Cylinders as Projection Screens"""""""""""""""""""""""""""
    # Extract the vertices, triangles, vertex colors and face colors
    vertices_out = np.asarray(mesh_out.vertices)
    triangles_out = np.asarray(mesh_out.triangles)
    vertex_colors_out = np.asarray(mesh_out.vertex_colors)
    face_colors_out = get_face_colors(triangles_out, vertex_colors_out)
    vertex_labels_out = np.asarray(label_mesh_out.vertex_colors)
    face_labels_out = get_face_colors(triangles_out, vertex_labels_out)

    vertices_in = np.asarray(mesh_in.vertices)
    triangles_in = np.asarray(mesh_in.triangles)
    vertex_colors_in = np.asarray(mesh_in.vertex_colors)
    face_colors_in = get_face_colors(triangles_in, vertex_colors_in)
    vertex_labels_in = np.asarray(label_mesh_in.vertex_colors)
    face_labels_in = get_face_colors(triangles_in, vertex_labels_in)

    # Flip face_labels colors (white (1,1,1) to black (0,0,0), black to white)
    face_labels_out = 1 - face_labels_out
    face_labels_in = 1 - face_labels_in

    # Projection
    projected_vertices_out, vert_depth_out = project_onto_cylinder(vertices_out, outer_radius)
    projected_vertices_in, vert_depth_in = project_onto_cylinder(vertices_in, inner_radius)

    # Flatten the projected vertices onto a 2D plane screen
    uv_out = flatten_2d_uv(projected_vertices_out)
    uv_in = flatten_2d_uv(projected_vertices_in)

    # Normalize the uv coordinates into [0, 1]
    uv_norm_out = normalize_uv_coordinates(uv_out, is_outward=True)
    uv_norm_in = normalize_uv_coordinates(uv_in, is_outward=False)



    """"""""""""""""""""""""""" 4. Generate 2D Images via Rasterization """""""""""""""""""""""""""
    px_h = 256 # y axis resolution
    px_w = px_h * 8 # theta axis resolution
    
    # Convert uv cooridnates to pixel coordinates, scaling theta to [0,2048) range
    uv_pixel_out = np.copy(uv_norm_out)
    uv_pixel_out[:, 0] = (uv_pixel_out[:, 0] * px_w-1).astype(np.int32)
    uv_pixel_out[:, 1] = (uv_pixel_out[:, 1] * px_h-1).astype(np.int32)

    uv_pixel_in = np.copy(uv_norm_in)
    uv_pixel_in[:, 0] = (uv_pixel_in[:, 0] * px_w-1).astype(np.int32) 
    uv_pixel_in[:, 1] = (uv_pixel_in[:, 1] * px_h-1).astype(np.int32)


    # Calculate mean projection depth for each face
    face_depth_out = np.mean(vert_depth_out[triangles_out], axis=1)
    face_depth_in = np.mean(vert_depth_in[triangles_in], axis=1)

    # Sort the faces by depth (ascending)
    sorted_indices_out = np.argsort(face_depth_out)
    sorted_triangles_out = triangles_out[sorted_indices_out]
    sorted_face_colors_out = face_colors_out[sorted_indices_out]
    sorted_face_colors_out = (sorted_face_colors_out * 255).astype(np.uint8)
    sorted_face_labels_out = face_labels_out[sorted_indices_out]
    sorted_face_labels_out = (sorted_face_labels_out * 255).astype(np.uint8)


    sorted_indices_in = np.argsort(face_depth_in)
    sorted_triangles_in = triangles_in[sorted_indices_in]
    sorted_face_colors_in = face_colors_in[sorted_indices_in]
    sorted_face_colors_in = (sorted_face_colors_in * 255).astype(np.uint8)
    sorted_face_labels_in = face_labels_in[sorted_indices_in]
    sorted_face_labels_in = (sorted_face_labels_in * 255).astype(np.uint8)

    # Initialize the large 2D image (1024x8192) for the entire dataset
    entire_origin_image_out = np.zeros((px_h, px_w, 3), dtype=np.uint8)
    entire_origin_image_in = np.zeros((px_h, px_w, 3), dtype=np.uint8)
    entire_label_image_out = np.zeros((px_h, px_w, 3), dtype=np.uint8)
    entire_label_image_in = np.zeros((px_h, px_w, 3), dtype=np.uint8)

    # Rasterize the triangles onto the large image
    # Crop the large image into 8 sections, each with size 1024x1024x3
    def rasterize(sorted_triangles, sorted_face_colors, uv_pixel, large_image):
        """ Return section_images as numpy arrays, with shape (8, 1024, 1024, 3)"""
        for i, triangle in enumerate(sorted_triangles):
            pts = uv_pixel[triangle].reshape((-1, 1, 2)).astype(np.int32)
            face_color = tuple(int(c) for c in sorted_face_colors[i])  # Convert to tuple for cv2.fillPoly
            cv2.fillPoly(large_image, [pts], face_color)

        section_images = np.array([large_image[:, i*px_h:(i+1)*px_h] for i in range(8)])
        return section_images, large_image

    origin_images_out, entire_origin_image_out = rasterize(sorted_triangles_out, sorted_face_colors_out, uv_pixel_out, entire_origin_image_out)
    origin_images_in, entire_origin_image_in = rasterize(sorted_triangles_in, sorted_face_colors_in, uv_pixel_in, entire_origin_image_in)
    origin_images = np.array([origin_images_out, origin_images_in])
    # flatten into (16, 256, 256, 3)
    origin_images = origin_images.reshape(-1, px_h, px_h, 3)

    label_images_out, entire_label_image_out = rasterize(sorted_triangles_out, sorted_face_labels_out, uv_pixel_out, entire_label_image_out)
    label_images_in, entire_label_image_in = rasterize(sorted_triangles_in, sorted_face_labels_in, uv_pixel_in, entire_label_image_in)
    label_images = np.array([label_images_out, label_images_in])
    # flatten into (16, 256, 256, 3)
    label_images = label_images.reshape(-1, px_h, px_h, 3)
    label_images_binary = convert_to_binary(label_images)

    
    """"""""""""""""""""""""""" 5. Visualize and Save """""""""""""""""""""""""""
    # Save the .npy image files in square sections
    save_origin_path = os.path.join(save_dir, "origin", f'{mesh_name}.npy')
    save_label_path = os.path.join(save_dir, "label", f'{mesh_name}.npy')
    # Save the section images as a single npy file
    np.save(save_origin_path, origin_images)
    np.save(save_label_path, label_images_binary)

    print(f"Saved {mesh_name}.npy")

    # # Display and Save the entire images as jpg
    # fig, ax = plt.subplots(4,1, figsize=(15, 10))

    # ax[0].imshow(entire_origin_image_out)
    # ax[0].set_title("Entire Origin Image Outward")
    # ax[0].axis('off')

    # ax[1].imshow(entire_label_image_out)
    # ax[1].set_title("Entire Label Image Outward")
    # ax[1].axis('off')

    # ax[2].imshow(entire_origin_image_in)
    # ax[2].set_title("Entire Origin Image Inward")
    # ax[2].axis('off')

    # ax[3].imshow(entire_label_image_in)
    # ax[3].set_title("Entire Label Image Inward")
    # ax[3].axis('off')

    # plt.savefig(os.path.join(save_dir, "jpg", f"{mesh_name}.jpg"))
    # # plt.show()

    """"""""""""""""""""""""""" 6. Test the Saved .npy Files """""""""""""""""""""""""""
    # tensor_origin = np.load(save_origin_path)
    # tensor_label = np.load(save_label_path)
    # origin_image_test = tensor_origin[1, 4, :, :, :]
    # label_image_test = tensor_label[1, 4, :, :, :]

    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # ax[0].imshow(origin_image_test)
    # ax[0].axis("off")
    # ax[1].imshow(label_image_test)
    # ax[1].axis("off")
    # # plt.show()
    
















""" 1. Create Circle and Extrude to Cylinders as Projection Screens"""
# Function to create a circle in the x-z plane
def create_circle(radius, n_points=100):
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = radius * np.cos(theta)
    z = radius * np.sin(theta)
    points = np.vstack((x, np.zeros(n_points), z)).T
    lines = np.column_stack((np.arange(n_points), np.roll(np.arange(n_points), -1)))
    return points, lines

# Function to extrude the circles in the y direction to create cylinders
def extrude_circle(points, extrusion_length_pos, extrusion_length_neg):
    extruded_points = []
    for y in [extrusion_length_neg, extrusion_length_pos]:
        extruded_points.append(points + np.array([0, y, 0]))
    extruded_points = np.vstack(extruded_points)
    
    # Create triangular faces for the extrusion
    n_points = len(points)
    faces = []
    for i in range(n_points):
        next_i = (i + 1) % n_points  # Ensure circular connection
        # Create two triangles for each face of the cylinder
        faces.append([i, next_i, next_i + n_points])
        faces.append([i, next_i + n_points, i + n_points])
    
    return extruded_points, faces





""" 2. Mesh Separation (Inward and Outward)"""
def clean_io_separation(original_mesh, face_mask_out, face_mask_in, threshold=500):
    """ Preserve only the connected components with number of faces larger than the threshold 
        within each mask. Remove and add the rest components to the opposite mask.
        Return the updated face masks for outward-facing and inward-facing faces"""
    triangles = np.asarray(original_mesh.triangles)
    vertex_colors = np.asarray(original_mesh.vertex_colors)
    
    mesh_in = copy.deepcopy(original_mesh)
    triangles_in = triangles[face_mask_in]
    mesh_in.triangles = o3d.utility.Vector3iVector(triangles_in)
    triangle_clusters_in, cluster_n_triangles_in, _ = (mesh_in.cluster_connected_triangles())
    triangle_clusters_in = np.array(triangle_clusters_in)
    cluster_n_triangles_in = np.array(cluster_n_triangles_in)
    trimask_in_to_out = cluster_n_triangles_in[triangle_clusters_in] < 500 # mask for triangles to be moved to outward-facing
    triangles_in_to_out = triangles_in[trimask_in_to_out]
    triangles_in_reduced = triangles_in[~trimask_in_to_out]

    mesh_out = copy.deepcopy(original_mesh)
    triangles_out = triangles[face_mask_out]
    mesh_out.triangles = o3d.utility.Vector3iVector(triangles_out)
    triangle_clusters_out, cluster_n_triangles_out, _ = (mesh_out.cluster_connected_triangles())
    triangle_clusters_out = np.array(triangle_clusters_out)
    cluster_n_triangles_out = np.array(cluster_n_triangles_out)
    trimask_out_to_in = cluster_n_triangles_out[triangle_clusters_out] < 500 # mask for triangles to be moved to inward-facing
    triangles_out_to_in = triangles_out[trimask_out_to_in]
    triangles_out_reduced = triangles_out[~trimask_out_to_in]

    triangles_in_updated = np.concatenate((triangles_in_reduced, triangles_out_to_in))
    triangles_out_updated = np.concatenate((triangles_out_reduced, triangles_in_to_out))

    mesh_in.triangles = o3d.utility.Vector3iVector(triangles_in_updated)
    mesh_out.triangles = o3d.utility.Vector3iVector(triangles_out_updated)
    
    # Remove unreferenced vertices
    mesh_in.remove_unreferenced_vertices()
    mesh_out.remove_unreferenced_vertices()

    return mesh_in, mesh_out





""" 3. Project 3D Meshes onto Cylinders as Projection Screens"""
# Project vertices onto the respective cylinders
def project_onto_cylinder(vertices, radius):
    projected_vertices = vertices.copy()
    
    for i in range(vertices.shape[0]):
        x, y, z = vertices[i]
        theta = np.arctan2(z, x)
        projected_vertices[i] = [radius * np.cos(theta), y, radius * np.sin(theta)]
    
    # depth in x-z plane relative to the cylinder
    projected_depth = np.linalg.norm(vertices[:, [0, 2]], axis=1) - radius 
    return projected_vertices, projected_depth


def flatten_2d_uv(vertices):
    """ Flatten the projected vertices onto a 2D plane screen"""
    flattened_vertices = []
    for i in range(vertices.shape[0]):
        x, y, z = vertices[i]
        theta = np.arctan2(z, x)
        if theta < -np.pi/2:
            theta += 2 * np.pi
        flattened_vertices.append([theta, y])
    return np.array(flattened_vertices)

def normalize_uv_coordinates(uv_coords, 
                             theta_range=(-np.pi/2, 3*np.pi/2), 
                             y_range_out=(-9, 7), 
                             y_range_in=(-8, 8), 
                             is_outward=True):
    theta_min, theta_max = theta_range
    if is_outward:
        y_min, y_max = y_range_out
    else:
        y_min, y_max = y_range_in
    uv_coords[:, 0] = (uv_coords[:, 0] - theta_min) / (theta_max - theta_min)
    uv_coords[:, 1] = (uv_coords[:, 1] - y_min) / (y_max - y_min)
    return uv_coords



def get_face_colors(triangles, vertex_colors):
    face_colors = []
    for triangle in triangles:
        colors_3vert = vertex_colors[triangle]
        # Get the minimum color value for each channel
        face_color = np.min(colors_3vert, axis=0)
        face_colors.append(face_color)
    return np.array(face_colors)


def convert_to_binary(label_images):
    binary_images = np.any(label_images != 0, axis=-1).astype(np.uint8)
    return binary_images




if __name__ == "__main__":

    origin_dir = "D:\sunny\Codes\DPS\data_teethseg\origin"
    label_dir = "D:\sunny\Codes\DPS\data_teethseg\label"
    save_dir = "D:\sunny\Codes\DPS\data_npy2d"

    process_all_files(origin_dir, label_dir, save_dir)