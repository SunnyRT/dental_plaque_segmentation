# Given prediction and 2d-3d reconstruction information
# output SOTA_face prediction information for each image


import open3d as o3d
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from dataloader import plydataset
from torch.utils.data import DataLoader
from TSGCNet import TSGCNet
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F



""""""" Load model and checkpoint """""""
dataset = plydataset("xxxxxxxxx")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
model = TSGCNet(in_channels=9, output_channels=2, k=12).cuda()
device = torch.device("cuda:xx") # TODO: change this to the correct device id
model.to(device) 
model.load_state_dict(torch.load("xxxxxxxxx")) # TODO: change this to the correct checkpoint path
model.eval()




""""""" INFO & mesh files """""""
info_dir = "xxxxxxxxx" # TODO: change this to the correct path
origin_dir = "xxxxxxxxx" # TODO: change this to the correct path
save_dir = "xxxxxxxxx" # TODO: change this to the correct path





# iterate over all mesh data
for batch_id, (index, points, label_face, label_face_onehot, name, raw_points_face) in tqdm(enumerate(dataloader), total=len(dataloader), smoothing=0.9):

    """"""" Predict output """""""
    batchsize, num_point, _ = points.size()
    assert batchsize == 1, "Batch size must be 1 for inference"

    mesh_name = name[0] # end with .ply
    data_name = os.path.basename(mesh_name).split(".")[0] # extract base name without extension

    print(f"Processing mesh: {mesh_name}")
    index_face = index[0].numpy()
    print(f"shape of index_face: {index_face.shape}")

    coordinate = points.transpose(2,1).contiguous()
    coordinate = Variable(coordinate.float())
    coordinate = coordinate.to(device)

    with torch.no_grad():
        pred = model(coordinate)
    pred = pred.contiguous().view(-1, 2) # num_classes = 2

    
    # TODO: Convert pred to 1D array of probabilities for class 1
    probs = F.softmax(pred, dim=1) # Apply softmax for both classes
    class_1_probs = probs[:, 1] # Extract the probability of class 1
    class_1_probs = class_1_probs.view(-1) # Convert to a 1D array: [num_faces, 1]

    # Create a dictionary with index_face as key and plaque probs as value
    pred_face_dict_index = dict(zip(index_face, class_1_probs.cpu().numpy())) # IMPORTANT!!!!!


    
    """"""" reconstuction information """""""
    info = np.load(os.path.join(info_dir, f"{data_name}.npz"))
    uvpx_up = info["uvpx_up"]
    uvpx_in = info["uvpx_in"]
    uvpx_out = info["uvpx_out"]
    print(uvpx_up.shape, uvpx_in.shape, uvpx_out.shape)
    tri_up = info["tri_up"]
    tri_in = info["tri_in"]
    tri_out = info["tri_out"]
    print(tri_up.shape, tri_in.shape, tri_out.shape)


    """"""" mesh file to get xyz coordinates of each vertex """""""
    mesh = o3d.io.read_triangle_mesh(os.path.join(origin_dir, mesh_name))
    vertices = np.asarray(mesh.vertices)



    """"""" process triangles for each direction-facing surface: up, in, out """""""
    surface_pred_3dmesh_up, surface_pred_2duv_up = process_surface_triangles(tri_up, vertices, uvpx_up, pred_face_dict_index)
    surface_pred_3dmesh_in, surface_pred_2duv_in = process_surface_triangles(tri_in, vertices, uvpx_in, pred_face_dict_index)
    surface_pred_3dmesh_out, surface_pred_2duv_out = process_surface_triangles(tri_out, vertices, uvpx_out, pred_face_dict_index)


    """"""" rasterize SOTA prediction into 2D image """""""
    img_up = rasterize(surface_pred_2duv_up, is_io=False)
    img_in = rasterize(surface_pred_2duv_in, is_io=True)
    img_out = rasterize(surface_pred_2duv_out, is_io=True)


    """"""" save images & npz file for surface_pred_3dmesh """""""
    cv2.imwrite(os.path.join(save_dir, "SOTA_pred", f"{data_name}_up.png"), img_up)
    cv2.imwrite(os.path.join(save_dir, "SOTA_pred", f"{data_name}_in.png"), img_in)
    cv2.imwrite(os.path.join(save_dir, "SOTA_pred", f"{data_name}_out.png"), img_out)

    SOTA_mesh = {"up": surface_pred_3dmesh_up, "in": surface_pred_3dmesh_in, "out": surface_pred_3dmesh_out}
    np.savez(os.path.join(save_dir, "SOTA_mesh", f"{data_name}.npz"), **SOTA_mesh)









def process_surface_triangles(tri_xx, all_verts, uvpx_xx, pred_face_dict_index):
    surface_pred_3dmesh = np.zeros((len(tri_xx), 10))
    surface_pred_2duv = np.zeros((len(tri_xx), 7))

    for i, face_idx in enumerate(tri_xx): # iterate over all faces which belong to the surface (up, in, out)
        # store the plaque probability in the last column of each row (face)
        face_pred = pred_face_dict_index[face_idx]
        surface_pred_3dmesh[i, 9] = face_pred
        surface_pred_2duv[i, 6] = face_pred
        
        # get 3D coordinates of each of the 3 vertices of the face (x9)
        surface_pred_3dmesh[i, 0:3] = all_verts[face_idx[0]]
        surface_pred_3dmesh[i, 3:6] = all_verts[face_idx[1]]
        surface_pred_3dmesh[i, 6:9] = all_verts[face_idx[2]]


        # get 2D uv coordinates of each vertex (x6)
        surface_pred_2duv[i, 0:2] = uvpx_xx[face_idx[0]]
        surface_pred_2duv[i, 2:4] = uvpx_xx[face_idx[1]]
        surface_pred_2duv[i, 4:6] = uvpx_xx[face_idx[2]]

    return surface_pred_3dmesh, surface_pred_2duv



def rasterize(surface_pred_2duv, is_io):
    px = 256
    if is_io:
        img = np.zeros((px, px*8, 3), dtype=np.uint8)
    else:
        img = np.zeros((px*2, px*3, 3), dtype=np.uint8)

    for tri in surface_pred_2duv:
        tri_uv = tri[:6].reshape((3, 2))
        pts = tri_uv.reshape((-1, 1, 2)).astype(np.int32)
        assert pts.shape == (3, 1, 2), f"Shape of pts is {pts.shape}"
        tri_prob = tri[6]
        tri_RGB = np.array([255, 255, 255]) * tri_prob
        tri_RGB = tuple(tri_RGB.astype(np.int32)) # convert to tuple for cv2.fillPoly
        cv2.fillPoly(img, [pts], tri_RGB)
    return img