import open3d as o3d
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
import os

def load_pred_img(pred2D_path, mesh_name):
    
    # convert to color range [0, 1]
    # pred_img_up = cv2.imread(os.path.join(pred2D_path, f"{mesh_name}_0.png"), cv2.IMREAD_UNCHANGED)/255
    # pred_img_in = cv2.imread(os.path.join(pred2D_path, f"{mesh_name}_1.png"), cv2.IMREAD_UNCHANGED)/255
    # pred_img_out = cv2.imread(os.path.join(pred2D_path, f"{mesh_name}_2.png"), cv2.IMREAD_UNCHANGED)/255

    pred_img_up = (cv2.imread(os.path.join(pred2D_path, f"{mesh_name}_0.png"), cv2.IMREAD_GRAYSCALE) ).astype(np.uint8)/255
    pred_img_in = (cv2.imread(os.path.join(pred2D_path, f"{mesh_name}_1.png"), cv2.IMREAD_GRAYSCALE) ).astype(np.uint8)/255
    pred_img_out = (cv2.imread(os.path.join(pred2D_path, f"{mesh_name}_2.png"), cv2.IMREAD_GRAYSCALE) ).astype(np.uint8)/255

    # print(f"pred_img shapes: {pred_img_up.shape}, {pred_img_in.shape}, {pred_img_out.shape}")

    return pred_img_up, pred_img_in, pred_img_out

def get_tri_RGB(triangles, vertex_RGB):
    """ Get the RGB of each triangle face from the RGB of its 3 vertices"""
    tri_RGBs = []
    for triangle in triangles:
        colors_3vert = vertex_RGB[triangle]
        # Get the minimum color value for each channel
        tri_rgb = np.max(colors_3vert, axis=0) # TODO: max vs mean!!!! (use max to prioritize plaque)
        # print(tri_rgb,"one triangle",float((tri_rgb[0]==1)))
        # print(tri_rgb.shape,(tri_rgb==[1,1,1]).all(axis=0))
        tri_RGBs.append(float((tri_rgb==[1,1,1]).all(axis=0)))
        # tri_RGBs.append(tri_rgb[0])

    return np.array(tri_RGBs)


def get_tri_center_uv(triangles, uv_pixels):
    tri_center_uv = np.mean(uv_pixels[triangles], axis=1)
    return tri_center_uv

def get_tri_pred_label(tri_uvpx, pred_img_label):
    tri_pred_label = []
    px_h, px_w = pred_img_label.shape[:2]
    for uv in tri_uvpx:
        u, v = uv.astype(np.int32)
        u = np.clip(u, 0, px_w-1)
        v = np.clip(v, 0, px_h-1)
        tri_pred_label.append(pred_img_label[v, u])
    return np.array(tri_pred_label)

def compute_metrics_tri(gt_labels, pred_labels, class_id=1):
    """ Compute the IoU and Dice scores for the triangles """
    if class_id ==1: # Class 1 plaque
        # Convert RGB to binary scalar
        gt_labels_bi = (gt_labels > 0.).astype(np.int32)  # TODO: to change threshold 
        pred_labels_bi = (pred_labels > 0.5).astype(np.int32) # TODO: to change threshold 
    # else: # Class 0 non-plaque
    #     gt_labels_bi = (gt_labels == 0.).astype(np.int32)  # TODO: to change threshold 
    #     pred_labels_bi = (pred_labels < 0.5).astype(np.int32)# TODO: to change threshold 
    

    intersection = np.sum(np.logical_and(gt_labels_bi, pred_labels_bi))
    union = np.sum(np.logical_or(gt_labels_bi, pred_labels_bi))
    iou = intersection / union

    intersection_bi = np.sum(np.logical_and(gt_labels_bi, pred_labels_bi))
    dice = 2 * intersection_bi / (np.sum(gt_labels_bi) + np.sum(pred_labels_bi))
    # if non-binary (3 channel)
    # dice = 2/3 * intersection / (np.sum(np.any(gt_labels!=0,axis=1)) + np.sum(np.any(pred_labels!=0,axis=1)))
    return iou, dice

def compute_confusion_matrics(gt_labels_bi, pred_labels_bi, percent=True):
    """ Compute the confusion matrix for the triangles """
    # print('compute_confusion_matrics')
    confusion_matrix = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            confusion_matrix[i, j] = np.sum(np.logical_and(gt_labels_bi == i, pred_labels_bi == j))
    
    if percent:
        TN, FP, FN, TP = confusion_matrix.ravel()
        bg_iou_per = (TN) / (TN + FP + FN) * 100 # BG IoU
        bg_dice_per = TN * 2 / (TN + FP + FN + TN) * 100 # BG Dice

        fg_iou_per = (TP)/ (TP + FP + FN) * 100 # plaque IoU
        fg_dice = 2*TP / (2*TP + FP + FN) * 100

        sen_per = TP / (TP + FN) * 100 # Sensitivity==reall
        acc_per = (TP+TN) / (FP + TP + TN + FN) * 100 # Accuracy

        

    confusion_matrix = np.array([[bg_iou_per, bg_dice_per], [fg_iou_per, fg_dice], [sen_per, acc_per]])

    # print('confusion_matrix', confusion_matrix)

    return confusion_matrix

def update_uv_pred_label(vert_pred_label, uv_pixel, vert_idx, pred_img_label):
    """ Get the predicted label for each vertex UV pixel coordinate 
    located on the respective predicted label image"""
    px_h, px_w = pred_img_label.shape[:2]
    for idx in vert_idx: # idx among all vertices (since uv_pixel is for all vertices)
        u, v = uv_pixel[idx]
        u = np.clip(int(u), 0, px_w-1)
        v = np.clip(int(v), 0, px_h-1)
        
        # if pred_img_label[v, u].any() != 0 and pred_img_label[v,u].any()!=1: # if not black nor white
        #     print(f"pred_img_label: {pred_img_label[v,u]}")
        
        # if vert_pred_label[idx].all() == np.array([-1,-1,-1]).all():
        vert_pred_label[idx] = pred_img_label[v, u]
      
        # else:
            # vert_pred_label[idx] = (pred_img_label[v, u] + vert_pred_label[idx]) /2 # average if already assigned
    return vert_pred_label

# Compare the predicted labels with the ground truth labels for all vertices from the entire mesh
def compute_metrics(gt_labels, pred_labels):
    """ Compute the IoU and Dice scores for the vertices """
    intersection = np.sum(np.logical_and(gt_labels, pred_labels))
    union = np.sum(np.logical_or(gt_labels, pred_labels))
    iou = intersection / union
    dice = 2 * intersection / (np.sum(gt_labels) + np.sum(pred_labels))
    # if non-binary (3 channel)
    # dice = 2/3 * intersection / (np.sum(np.any(gt_labels!=0,axis=1)) + np.sum(np.any(pred_labels!=0,axis=1)))
    return iou, dice

def visualize_pred(GTlabel_mesh, vert_pred_label_binary): # assume vertex_pred_labels is in RGB format (0-1) non-binary
    """ Visualize the predicted labels on the mesh """
    mesh_pred = copy.deepcopy(GTlabel_mesh)
    vertices = np.asarray(mesh_pred.vertices)
    triangles = np.asarray(mesh_pred.triangles)
    colors = np.full((len(vertices), 3), 0.8) # grey: True negative

    
    for i in range(len(vertices)):
        if vert_pred_label_binary[i] == 1: 

            colors[i] = np.array([0, 0, 0]) # Green
                
        else:
            colors[i] = np.array([1, 1, 1]) # Red

        
    mesh_pred.vertex_colors = o3d.utility.Vector3dVector(colors)

    return mesh_pred




pred2D_dir = "/home/Liting/zhanxch/PlaqueIOS/16000/2D_png/val/2Dpred"  #"/home/Liting/zhanxch/oodtest/2Dpred"#"/home/Liting/zhanxch/SAM/2Dpred" 
# pred2D_dir = "/home/Liting/zhanxch/PlaqueIOS/16000/2D_png/test/2Dpred"
info_dir = "/home/Liting/zhanxch/PlaqueIOS/16000/2D_png/info"
label_mesh_dir = "/home/Liting/zhanxch/PlaqueIOS/16000/label"

# pred3D_dir = r"/home/Liting/zhanxch/oodtest/2Dpred/3Dpred"
# pred2D_dir = "/home/Liting/zhanxch/oodtest/2Dpred"
# info_dir = "/home/Liting/zhanxch/oodtest/2D_png/info"
# label_mesh_dir = "/home/Liting/zhanxch/oodtest/label"

iou_list = []
dice_list = []
avg_iou = []
avg_dice = []
iou_3d = []
dice_3d = []
weight = 0.5
bg_iou,bg_dice, plaque_iou, plaque_dice, sen,acc=0,0,0,0,0,0
i = 0
total_I,total_U = 0,0
save_dir = "/home/Liting/zhanxch/PlaqueIOS/16000/2D_png/val/3d_visualizations"  # 3D可视化结果保存目录
os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
with open(r"/home/Liting/zhanxch/PlaqueIOS/val_file_names.txt",'r') as f:
# with open(r"/home/Liting/zhanxch/PlaqueIOS/test_file_names.txt",'r') as f:
    file_names = f.readlines()
for name in file_names:
    if int(name) in [4702]:
        continue
    print(name)

    try:
        # print('i', i)
        i += 1
        mesh_name = name.split()[0]

        pred_img_up, pred_img_in, pred_img_out = load_pred_img(pred2D_dir, mesh_name)
        
        print(pred_img_up.shape,pred_img_in.shape,pred_img_out.shape)
        info = np.load(os.path.join(info_dir, f"{mesh_name}.npz"))
        # print("npz文件中的键：", info.files) #['uvpx_up', 'uvpx_in', 'uvpx_out', 'tri_up', 'tri_in', 'tri_out']
        uvpx_up = info["uvpx_up"]
        uvpx_in = info["uvpx_in"]
        uvpx_out = info["uvpx_out"]

        tri_up = info["tri_up"]
        tri_in = info["tri_in"]
        tri_out = info["tri_out"]

        # load mesh
        mesh = o3d.io.read_triangle_mesh(os.path.join(label_mesh_dir, f"{mesh_name}.ply"))
        vertices = np.asarray(mesh.vertices)
        vert_GT_label = 1 - np.asarray(mesh.vertex_colors) # already in range [0, 1], need to be flipped so that 1 => plaque, 0 => tooth

        tri_uvpx_up = get_tri_center_uv(tri_up, uvpx_up)
        tri_uvpx_in = get_tri_center_uv(tri_in, uvpx_in)
        tri_uvpx_out = get_tri_center_uv(tri_out, uvpx_out)

        tri_pred_label_up = get_tri_pred_label(tri_uvpx_up, pred_img_up)
        tri_pred_label_in = get_tri_pred_label(tri_uvpx_in, pred_img_in)
        # print(tri_uvpx_out.shape,pred_img_out.shape)
        tri_pred_label_out = get_tri_pred_label(tri_uvpx_out, pred_img_out)

        tri_GT_labelGRB_up = get_tri_RGB(tri_up, vert_GT_label)
        tri_GT_labelGRB_in = get_tri_RGB(tri_in, vert_GT_label)
        tri_GT_labelGRB_out = get_tri_RGB(tri_out, vert_GT_label)
        
        
        tri_GT_labelGRB = np.concatenate([tri_GT_labelGRB_up, tri_GT_labelGRB_in, tri_GT_labelGRB_out])
        tri_pred_label = np.concatenate([tri_pred_label_up, tri_pred_label_in, tri_pred_label_out])
        # print('tri_pred_label', tri_pred_label.shape) #(16000,)


        iou_mean, dice_mean = compute_metrics_tri(tri_GT_labelGRB, tri_pred_label, class_id=1)
        print(f"Class 1 IoU: {iou_mean:.3f}, Class 1 Dice: {dice_mean:.3f}")
 
        
        iou_list.append(iou_mean)
        dice_list.append(dice_mean)

        # o3d.io.write_triangle_mesh(f"{save_dir}/{name}_pred.ply", vis_mesh)

        # print(tri_pred_label.shape,pred3d.shape)

        # # print(tri_pred_label.shape,pred3d.shape,avg_pred.shape)



        # print(iou_mean,dice_mean)



        tri_GT_bilabel = (tri_GT_labelGRB > 0).astype(np.int32) # TODO: to change threshold 
        tri_pred_bilabel = (tri_pred_label > 0.5).astype(np.int32)   # TODO: to change threshold 
        # print('777')
        cm= compute_confusion_matrics(tri_GT_bilabel, tri_pred_bilabel)
        # print(f"here is cm: {cm}") #TN FP FN TP
        bg_iou += cm[0,0]
        bg_dice += cm[0,1]
        plaque_iou += cm[1,0]
        plaque_dice += cm[1,1]
        sen += cm[2,0]
        acc += cm[2,1]

    except:
        print("==================")
        print(name)
        print("==================")
print("BG IOU:",bg_iou/i,"BG DICE:",bg_dice/i,"Plaque IOU:",plaque_iou/i,"Plaque DICE:",plaque_dice/i,"sensitivity:",sen/i,"accuracy:",acc/i)
# print(total_I/total_U)
print(sum(iou_list)/len(iou_list))
print(sum(dice_list)/len(dice_list))    

#2d-single
#BG IOU: 81.83945462311245 BG DICE: 89.84900601698655 Plaque IOU: 39.429968836520395 Plaque DICE: 55.77219024258518 sensitivity: 64.35995802455743 accuracy: 84.16988482653564