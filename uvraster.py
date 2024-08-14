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
            origin_mesh = o3d.io.read_triangle_mesh(origin_file_path)
            label_mesh = o3d.io.read_triangle_mesh(label_file_path)
            process_single_mesh(origin_mesh, label_mesh, save_dir, mesh_name)


def process_single_mesh(origin_mesh, label_mesh, save_dir, mesh_name):
    
    """"""""""""""""1. Recenter """""""""""""""
    # Remove duplicated triangles of the meshes
    origin_mesh.remove_duplicated_triangles()
    label_mesh.remove_duplicated_triangles()

    # Recenter the meshes
    origin_mesh = origin_mesh.translate(-origin_mesh.get_center())
    label_mesh = label_mesh.translate(-label_mesh.get_center())



    """""""""""""""""""""""""""""""""" 2. Mesh Separation (Inward and Outward)"""""""""""""""""""""""""""""""""""
    in_triangles, out_triangles = separate_io_triangles(origin_mesh)

    """"""""""""""""""""""""""" 3. Project 3D Meshes onto Cylinders as Projection Screens"""""""""""""""""""""""""""
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Execution of UV projection (for all vertices in each of 3 cases)

    # Get the vertices of the origin mesh
    vertices = np.asarray(origin_mesh.vertices)

    # UV mapping for all vertices in 3 cases NOTE: all vertices are projected in each case!!!!!!
    uv_out, vert_depth_out = UVmap_cylindrical(vertices, if_inward=False)
    uv_in, vert_depth_in = UVmap_cylindrical(vertices, if_inward=True)

    uv_norm_out = normalize_uv_coords(uv_out, (0, 2*np.pi), (-9, 7), with_nl=True)
    uv_norm_in = normalize_uv_coords(uv_in, (0, 2*np.pi), (-8, 8), with_nl=True)

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Keep track of vertices which belong to each case out of all vertices
    # Find unique vertex indices for each case
    out_vert_idx = np.unique(out_triangles.flatten())
    in_vert_idx = np.unique(in_triangles.flatten())

    # # Print lengths for verification
    # print(out_vert_idx.shape, in_vert_idx.shape)
    # print(vertices.shape)
    # assert out_vert_idx.shape[0] + in_vert_idx.shape[0] == vertices.shape[0], "Vertices are not correctly separated"

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # COLORS!!!!
    vertices_originRGB_raw = np.asarray(origin_mesh.vertex_colors) # r,g,b within (0-1)
    vertices_labelRGB_raw = np.asarray(label_mesh.vertex_colors) # black/grey/non-white >= (0,0,0) for plaque, white=(1,1,1) for non-plaque

    # Convert RGB format into 0-255
    # Flip plaque representation from dark (black or grey) to white
    vertices_originRGB = (vertices_originRGB_raw * 255).astype(np.uint8) # 0-255
    vertices_labelRGB = ((1-vertices_labelRGB_raw) * 255).astype(np.uint8) # 0-255, non-black>(0,0,0) for plaue, black=(0,0,0) for non-plaque



    """"""""""""""""""""""""""" 4. Generate 2D Images via Rasterization """""""""""""""""""""""""""
    img_origin_out, img_label_out, uv_pixel_out = rasterize_uv_to_image(uv_norm_out, vert_depth_out, out_triangles, vertices_originRGB, vertices_labelRGB, px=256, is_binary=False)
    img_origin_in, img_label_in, uv_pixel_in = rasterize_uv_to_image(uv_norm_in, vert_depth_in, in_triangles, vertices_originRGB, vertices_labelRGB, px=256, is_binary=False)
    # TODO: Inward / Outward projected images will be patched into 1x8
    # TODO: Upward projected image will be patched into 2x4

    
    """"""""""""""""""""""""""" 5. Visualize and Save """""""""""""""""""""""""""

    save_origin_path = os.path.join(save_dir, "origin")
    save_label_path = os.path.join(save_dir, "label")


    for idx, (img_origin_xx, img_label_xx) in enumerate(zip([img_origin_in, img_origin_out], [img_label_in, img_label_out])):
        origin_bgr = cv2.cvtColor(img_origin_xx, cv2.COLOR_RGB2BGR)
        label_gray = cv2.cvtColor(img_label_xx, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(os.path.join(save_origin_path, f"{mesh_name}_{idx+1}.png"), origin_bgr)
        cv2.imwrite(os.path.join(save_label_path, f"{mesh_name}_{idx+1}.png"), label_gray)

    # Save UV pixel coordinates and separation of triangles information in npz format
    reconst_info = {"uvpx_in": uv_pixel_in, "uvpx_out": uv_pixel_out,
                    "tri_in": in_triangles, "tri_out": out_triangles}
    np.savez_compressed(os.path.join(save_dir, "info", f"{mesh_name}.npz"), **reconst_info)

    # Visaulize the rasterized images
    # Display the in- and outward large image for visualization 
    # fig, ax = plt.subplots(4,1, figsize=(15, 10))

    # ax[0].imshow(img_origin_out)
    # ax[0].set_title("Outward Facing Projection - Origin")
    # ax[0].axis('off')


    # ax[1].imshow(img_label_out)
    # ax[1].set_title("Outward Facing Projection - Label")
    # ax[1].axis('off')

    # ax[2].imshow(img_origin_in)
    # ax[2].set_title("Inward Facing Projection - Origin")
    # ax[2].axis('off')

    # ax[3].imshow(img_label_in)
    # ax[3].set_title("Inward Facing Projection - Label")
    # ax[3].axis('off')

    # plt.show())

    """"""""""""""""""""""""""" 6. Test map back accuracy """""""""""""""""""""""""""
    pred_img_label_out = img_label_out/255  
    pred_img_label_in = img_label_in/255

    tri_uvpx_in = get_tri_center_uv(in_triangles, uv_pixel_in)
    tri_uvpx_out = get_tri_center_uv(out_triangles, uv_pixel_out)
        

    tri_pred_label_in = get_tri_pred_label(tri_uvpx_in, pred_img_label_in)
    tri_pred_label_out = get_tri_pred_label(tri_uvpx_out, pred_img_label_out)

    # Get the GT label for each triangle face
    tri_GT_labelGRB_in = get_tri_RGB(in_triangles, vertices_labelRGB)/255
    tri_GT_labelGRB_out = get_tri_RGB(out_triangles, vertices_labelRGB)/255

    # Compare the predicted labels with the ground truth labels for all triangle faces
    metrics_in = compute_metrics_tri(tri_GT_labelGRB_in, tri_pred_label_in)
    metrics_out = compute_metrics_tri(tri_GT_labelGRB_out, tri_pred_label_out)

    print(f"Inward: IoU: {metrics_in[0]:.3f}, Dice: {metrics_in[1]:.3f}")
    print(f"Outward: IoU: {metrics_out[0]:.3f}, Dice: {metrics_out[1]:.3f}")

    iou_mean = (metrics_in[0] + metrics_out[0]) / 2
    dice_mean = (metrics_in[1] + metrics_out[1]) / 2
    print(f"Plaque IoU: {iou_mean:.3f}, Plaque Dice: {dice_mean:.3f}")



    metrics_in0 = compute_metrics_tri(tri_GT_labelGRB_in, tri_pred_label_in, is_plaque=False)
    metrics_out0 = compute_metrics_tri(tri_GT_labelGRB_out, tri_pred_label_out, is_plaque=False)

    iou0_mean = (metrics_in0[0] + metrics_out0[0]) / 2
    dice0_mean = (metrics_in0[1] + metrics_out0[1]) / 2
    print(f"Non-Plaque IoU: {iou0_mean:.3f}, Non-Plaque Dice: {dice0_mean:.3f}")



















""" 2. Mesh Separation (Inward and Outward)"""


# Utils function
def set_difference(A, B):
    """ Return the elements in A but not in B"""
    A_view = A.view([('', A.dtype)] * A.shape[1])
    B_view = B.view([('', B.dtype)] * B.shape[1])
    C_view = np.setdiff1d(A_view, B_view)
    C = C_view.view(A.dtype).reshape(-1, A.shape[1])
    # print(f"initial size: {len(A)}, removed size: {len(B)}, final size: {len(C)}")
    return C



# Preserve connected vertices which belong to a big enough / biggest cluster (remove small disconnecting clusters)
def keep_large_component(mesh, max_only=False, min_size=600):
    """ Remove small disconnected components from the mesh
    Args:
        max_only: if True, only keep the biggest component
        min_size: only works if max_only is False, 
            in which case only components with size >= min_size are kept
        
    Return:
        mesh: the mesh with small components removed
        triangles: the preserved triangles of the mesh
        removed_triangles: the removed triangles of the mesh"""
    triangles = np.asarray(mesh.triangles)
    components = np.array(mesh.cluster_connected_triangles()[0])
    comp_sizes = np.bincount(components)
    if max_only: # only keep the biggest component
        max_comp_index = np.argmax(comp_sizes)
        large_comp_mask = components == max_comp_index

    else: # keep components with size >= min_size
        large_comp_indices = np.where(comp_sizes >= min_size)[0]
        large_comp_mask = np.isin(components, large_comp_indices)

    preserved_triangles = triangles[large_comp_mask]
    removed_triangles = triangles[np.logical_not(large_comp_mask)]
    mesh.triangles = o3d.utility.Vector3iVector(preserved_triangles) # update the mesh

    return mesh, preserved_triangles, removed_triangles




""" Inner and Outer Surface Separation"""
# Separate the original mesh vertices into inward-facing and outward-facing vertices relative to the origin
# With original (non re-centered mesh)
def separate_io_triangles(mesh):
    """ Separate the original mesh triangles into 
    inward-facing and outward-facing triangles relative to the origin
    
    All vertices are preserved."""

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    viewpt1 = np.mean(vertices, axis=0)
    vertices_vp1 = vertices - viewpt1
    viewpt2 = viewpt1 / 2
    vertices_vp2 = vertices - viewpt2

    normals = np.asarray(mesh.vertex_normals)

    # analyse the mesh in 3 parts (based on z coords)
    # each part has different viewpt for normal thresholding
    pos_mask1 = vertices[:, 2] > viewpt1[2]
    in_vert_mask1 = np.logical_and(np.sum(normals * vertices_vp1, axis=1) < 0, pos_mask1)
    pos_mask2 = np.logical_and(viewpt2[2] < vertices[:, 2], vertices[:, 2] < viewpt1[2])
    in_vert_mask2 = np.logical_and(np.sum(normals * vertices_vp2, axis=1) < 0, pos_mask2)
    pos_mask3 = vertices[:, 2] < viewpt2[2]
    in_vert_mask3 = np.logical_and(np.sum(normals * vertices, axis=1) < 0, pos_mask3)

    # merge the 3 masks
    in_vert_mask = in_vert_mask1 + in_vert_mask2 + in_vert_mask3

    # separate triangles
    in_triangles_mask = np.all(in_vert_mask[triangles], axis=1)
    in_triangles = triangles[in_triangles_mask]
    in_mesh = copy.deepcopy(mesh)
    in_mesh.triangles = o3d.utility.Vector3iVector(in_triangles)

    out_triangles_mask = ~in_triangles_mask
    out_triangles = triangles[out_triangles_mask]
    out_mesh = copy.deepcopy(mesh)
    out_mesh.triangles = o3d.utility.Vector3iVector(out_triangles)

    assert len(in_triangles) + len(out_triangles) == len(triangles), "Triangles are not correctly separated"

    # clean up
    in_mesh, in_triangles, in_to_out_triangles = keep_large_component(in_mesh, min_size=100)
    out_mesh, out_triangles, out_to_in_triangles = keep_large_component(out_mesh, min_size=100)

    # add out_to_in_triangles to in_triangles
    in_triangles = np.concatenate([in_triangles, out_to_in_triangles])
    out_triangles = np.concatenate([out_triangles, in_to_out_triangles])

    return in_triangles, out_triangles





""" 3. Project 3D Meshes onto Cylinders as Projection Screens"""
# Project vertices onto the respective cylinders
def UVmap_cylindrical(vertices, if_inward=True):
    """ UV mapping from xyz coordinates of inward or outward facing vertices
    Args:
        vertices: xyz coordinates of vertices
        if_inward: if True, the vertices are inward facing, projection depth = xz_norm
                   if False, the vertices are outward facing = max(xz_norm) - xz_norm
        
    Return:
        uv_coords: uv coordinates of vertices (u=theta, v=y) 
        depth_map: projection depth"""

    uv_coords = []
    for i in range(vertices.shape[0]):
        x, y, z = vertices[i]
        theta = np.arctan2(z, x)

        # FIXME: only if mesh is geometrically centered at origin!!!!!!
        theta = theta + np.pi /2
        if theta < 0:
            theta = theta + 2 * np.pi
            
        y = y
        uv_coords.append([theta, y])

    depth_map = np.linalg.norm(vertices[:, [0, 2]], axis=1)
    if if_inward:
        depth_map = np.max(depth_map) - depth_map
    
    return np.array(uv_coords), depth_map



def normalize_uv_coords(uv_coords, u_range, v_range, with_nl=False):
    uv_norm_coords = np.copy(uv_coords)
    u_min, u_max = u_range
    v_min, v_max = v_range
    uv_norm_coords[:, 0] = (uv_coords[:, 0] - u_min) / (u_max - u_min)
    uv_norm_coords[:, 1] = (uv_coords[:, 1] - v_min) / (v_max - v_min)

    return uv_norm_coords



def get_tri_RGB(triangles, vertex_RGB):
    """ Get the RGB of each triangle face from the RGB of its 3 vertices"""
    tri_RGBs = []
    for triangle in triangles:
        colors_3vert = vertex_RGB[triangle]
        # Get the minimum color value for each channel
        tri_rgb = np.mean(colors_3vert, axis=0) # FIXME: max (white [255,255,255] being the plaque) vs mean!!!!
        tri_RGBs.append(tri_rgb)
    return np.array(tri_RGBs)


def rasterize(sorted_tri, sorted_tri_RGB, uv_pixel, img):
    for i, tri in enumerate(sorted_tri):
        pts = uv_pixel[tri].reshape((-1, 1, 2)).astype(np.int32)
        tri_RGB = tuple(int(c) for c in sorted_tri_RGB[i]) # convert to tuple for cv2.fillPoly
        cv2.fillPoly(img, [pts], tri_RGB)
    return img




def rasterize_uv_to_image(uv_norm_coords, vert_depth, xx_triangles, vert_originRGB, vert_labelRGB, px=256, is_binary=True):

    px_h = px # 256
    px_w = px_h*8 # 2048
    
    # Scale uv_norm_coords by width and height to get pixel coordinates
    uv_pixel = np.copy(uv_norm_coords)
    uv_pixel[:, 0] = (uv_norm_coords[:, 0] * px_w-1).astype(np.int32)
    uv_pixel[:, 1] = (uv_norm_coords[:, 1] * px_h-1).astype(np.int32)

    # Calculate projection depth for each triangle face (mean projection depth of 3 vertices)
    tri_depth = np.mean(vert_depth[xx_triangles], axis=1)

    # Sort the triangles by depth (ascending)
    sorted_tri_idx = np.argsort(tri_depth)
    sorted_tri= xx_triangles[sorted_tri_idx]


    """ Get original colors and labels for each triangle face"""
    sorted_tri_originRGB = get_tri_RGB(sorted_tri, vert_originRGB)
    sorted_tri_labelRGB = get_tri_RGB(sorted_tri, vert_labelRGB)
    
    # FIXME: convert all non-zero rgb values to white (255,255,255)!!!! i.e. binary label (plaque or non-plaque)
    nonzero_RGB_con = np.max(sorted_tri_labelRGB, axis=1) > 0
    sorted_tri_labelbiRGB = np.where(nonzero_RGB_con[:, np.newaxis], np.array([255, 255, 255]), np.array([0, 0, 0])) 
    

    """ Rasterize the triangles in sorted order of depth"""
    # Initialize the image with black background
    img_origin = np.zeros((px_h, px_w, 3), dtype=np.uint8)
    img_label = np.zeros((px_h, px_w, 3), dtype=np.uint8)
    # Rasterize
    img_origin = rasterize(sorted_tri, sorted_tri_originRGB, uv_pixel, img_origin)
    img_label = rasterize(sorted_tri, sorted_tri_labelRGB, uv_pixel, img_label)
    # img_label = rasterize(sorted_tri, sorted_tri_labelbiRGB, uv_pixel, img_label) # FIXME: rasterize with binary label?

    # print(f"Image shape: {img_origin.shape}")
    assert img_origin.shape == img_label.shape, "Origin and label images have different shapes"

    """ Simply return the entire complete image as a whole. 
        Use Patchify function to divide the image into 8 patches later before training"""
    
    # FIXME: Not sure whether to convert to binary label for training?????
    if is_binary:
        img_label = np.where(img_label > 0, 1, 0).astype(np.uint8)

    return img_origin, img_label, uv_pixel # img: 3 RGB channel np arrays


""""""""""""""""""""""""""" 6. Test map back accuracy """""""""""""""""""""""""""
def get_tri_center_uv(triangles, uv_pixels):
    tri_center_uv = np.mean(uv_pixels[triangles], axis=1)
    return tri_center_uv

# Extract the color of each triangle face center from the predicted label image
# Need to execute for each direction (outward, inward, upward)
def get_tri_pred_label(tri_uvpx, pred_img_label):
    tri_pred_label = []
    px_h, px_w = pred_img_label.shape[:2]
    for uv in tri_uvpx:
        u, v = uv.astype(np.int32)
        u = np.clip(u, 0, px_w-1)
        v = np.clip(v, 0, px_h-1)
        tri_pred_label.append(pred_img_label[v, u])
    return np.array(tri_pred_label)


def compute_metrics_tri(gt_labels, pred_labels, is_plaque = True):
    """ Compute the IoU and Dice scores for the triangles """
    # Convert RGB to binary scalar
    gt_labels_bi = np.any(gt_labels > 0, axis=1).astype(np.int32)
    pred_labels_bi = np.any(pred_labels > 0, axis=1).astype(np.int32)
    # print(np.sum(gt_labels_bi), np.sum(pred_labels_bi))

    if not is_plaque:
        gt_labels_bi = 1 - gt_labels_bi
        pred_labels_bi = 1 - pred_labels_bi
        # print(np.sum(gt_labels_bi), np.sum(pred_labels_bi))

    if np.sum(pred_labels_bi) <=50: # FIXME: to check
        print("Too little predicted label")
        return 1, 1

    intersection = np.sum(np.logical_and(gt_labels_bi, pred_labels_bi))
    union = np.sum(np.logical_or(gt_labels_bi, pred_labels_bi))
    iou = intersection / union

    intersection_bi = np.sum(np.logical_and(gt_labels_bi, pred_labels_bi))
    dice = 2 * intersection_bi / (np.sum(gt_labels_bi) + np.sum(pred_labels_bi))
    # if non-binary (3 channel)
    # dice = 2/3 * intersection / (np.sum(np.any(gt_labels!=0,axis=1)) + np.sum(np.any(pred_labels!=0,axis=1)))
    return iou, dice









if __name__ == "__main__":

    origin_dir = "D:\sunny\Codes\DPS\data_teethseg\origin"
    label_dir = "D:\sunny\Codes\DPS\data_teethseg\label"
    save_dir = "D:\sunny\Codes\DPS\data_png_io"

    process_all_files(origin_dir, label_dir, save_dir)