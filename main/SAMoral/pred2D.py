import os
import numpy as np
import torch
# from PIL import Image
import cv2
from torch.utils.data import DataLoader
# from datasets import Dataset
from torch.utils.data import Dataset
# from .dataloader import load_and_patchify_png_permesh, SAMDataset
# from .config import device
from dataloader import load_and_patchify_png_permesh, SAMDataset
# from config import device
from patchify import unpatchify
# from matplotlib import pyplot as plt
# from transformers import SamModel,SamProcessor
# from .iou import calculate_iou, update_confusion_matrix, calculate_confusion_matrix_percentage
##################### 2D Prediction #######################
# from .model import SASOTA
from model import SASOTA
from segment_anything.build_sam import sam_model_registry
import copy
def load_model(ckpt_path, device):
    """"""""""""""""""""""""""" Load model from saved ckpt(.pth) """""""""""""""""""""""""""""""""""


    # Initialize a new instance of SAM_Decoder (if not already initialized)

    # sam = sam_model_registry["vit_b"](checkpoint="/home/Liting/ruitong/SAMoral/SAMoral/sam_vit_b_01ec64.pth")
    # sam.to(device)
    model = SASOTA(sam_model_registry["vit_b"](checkpoint="/home/Liting/ruitong/SAMoral/SAMoral/sam_vit_b_01ec64.pth").to(device))
    # model = SASOTA(sam_model_registry["vit_b"](checkpoint="/home/Liting/ruitong/SAMoral/SAMoral/sam_vit_b_01ec64.pth", image_size = 256, num_classes = 2).to(device))
    # num_params_excluding_first = count_parameters_excluding_first_n_layers(model, 3,4)
    # print(num_params_excluding_first)
    model.to(device)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    
    return model

def get_triRGB_from_labelmesh(label_mesh, face_order):
    """ sorted with face_order """
    # print(max(face_order)+1)
    sorted_tri_RGBs = np.zeros(max(face_order)+1)
    for (i, order) in enumerate(face_order):
        # tri_label = label_mesh[i][-1]
        # print(i,order,label_mesh[i])
        sorted_tri_RGBs[order] = label_mesh[i]

    return sorted_tri_RGBs

def get_triRGB_from_labelmesh2(label_mesh, face_order):
    """ sorted with face_order """
    # print(max(face_order)+1)
    sorted_tri_RGBs = np.zeros(max(face_order)+1, dtype=np.uint8)
    for (i, order) in enumerate(face_order):
        # tri_label = label_mesh[i][-1]
        sorted_tri_RGBs[order] = label_mesh[i][-1]

    return sorted_tri_RGBs


def remove_patch_dim(list):
    output_list = []
    for patch in list:
        for item in patch:
            output_list.append(item)
    return output_list

def pred_mesh(mesh_name, base_dir, save_dir, model, device):
    test_images, test_masks, patches_idx,test_mesh,label_mesh = load_and_patchify_png_permesh(base_dir, mesh_name)
    # print('test_images', test_images.shape) #(22, 256, 256, 3)
    
    order_dir = os.path.join(base_dir,'order')
    order_npz = os.path.join(order_dir,mesh_name+'.npz')

    # print('order_npz', order_npz) #/home/Liting/zhanxch/PlaqueIOS/16000/2D_png/val/order/011401.npz
    SOTA_mesh_file = np.load(order_npz, allow_pickle=True)      
    label_mesh_up = SOTA_mesh_file['up']
    label_mesh_in = SOTA_mesh_file['in']
    label_mesh_out = SOTA_mesh_file['out']
    # print('label_mesh_up', label_mesh_up.shape) (6,)

    face_order_up = SOTA_mesh_file['face_order_up']
    face_order_in = SOTA_mesh_file['face_order_in']
    face_order_out = SOTA_mesh_file['face_order_out']
    # print('face_order_up', face_order_up.shape) (6,)

    # remove the patch dimension from the label_mesh and face_order 
    label_mesh_up = remove_patch_dim(label_mesh_up)
    label_mesh_in = remove_patch_dim(label_mesh_in)
    label_mesh_out = remove_patch_dim(label_mesh_out)

    face_order_up = remove_patch_dim(face_order_up)
    face_order_in = remove_patch_dim(face_order_in)
    face_order_out = remove_patch_dim(face_order_out)
    # print('label_mesh_up', len(label_mesh_up)) #5983
    # print('label_mesh_up_', len(label_mesh_up[0])) #10
    # print('face_order_up', len(face_order_up)) #5983
    # print('label_mesh_up[0]', label_mesh_up[0])
    # print('face_order_up', face_order_up)


    test_dataset = SAMDataset(test_images, test_masks,test_mesh,label_mesh)


    # Create a DataLoader instance for the training dataset
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False, # DO NOT shuffle and preserve the order
        num_workers=5,
        prefetch_factor=3,
    )

    # Initialize pred_masks arrays
    pred_mask_patches = np.zeros((22, 256, 256), dtype=np.float32)

    pred3d = []
    up_I,up_U = 0,1e-5
    up_I_bg,up_U_bg = 0,1e-5
    in_I,in_U = 0,1e-5
    in_I_bg,in_U_bg = 0,1e-5
    out_I,out_U = 0,1e-5
    out_I_bg,out_U_bg = 0,1e-5
    # up_mask=0
    I,U = 0,1e-5
    I2,U2 = 0,1e-5
    ###################### Generate 2D pred mask patches from model ######################
    with torch.no_grad():
        for i, patch in enumerate(test_dataloader):
            img_patch, mask_patch = patch["pixel_values"],patch["ground_truth_mask"]
            # img_patch = img_patch.to(device)
            mask_patch = mask_patch.cpu().numpy().squeeze()
            # print('mask_patch.shape', mask_patch.shape) #(256, 256)
            # print('img_patch', img_patch.shape) #[1, 3, 256, 256]


            # outputs = model(image=patch["pixel_values"].to(device))
            outputs = model(image=patch["pixel_values"].to(device), mesh = patch["SOTA_mesh"].to(device))
            # print('pred_masks_', outputs["pred_masks"].shape) #[1, 1, 256, 256]
            # print('pred_masks', outputs["pred_masks"][0][0][127]) #[-0.8780, -0.8780, -0.8798,...]
            

            pred_mask = torch.sigmoid(outputs["pred_masks"].squeeze(1))
            # print('pred_mask.shape', pred_mask.shape) #[1, 256, 256]
            # print('pred_mask.shape', pred_mask[0][127]) #[0.2428, 0.2428, 0.2442, 0.24,...]
            # Convert soft mask to hard mask
            pred_mask = pred_mask.cpu().numpy().squeeze()
            # pred_mask_binary = (pred_mask>0.8).astype(np.uint8) # binary mask
            pred_mask_binary = (pred_mask>0.5).astype(np.uint8)
            # print('pred_mask_binary', pred_mask_binary.shape) #[256,256]
            pred_mask_patches[patches_idx[i]] = pred_mask # TIPS: for soft sota pred,current training is binary
            # pred_mask_patches[patches_idx[i]] = pred_mask_binary ##yzq:自己改的
            # print(np.sum(mask_patch==1),np.sum(mask_patch>0))
            if patches_idx[i]<6:
                # up_mask += np.sum(mask_patch==1)
                up_I += np.sum(np.logical_and(pred_mask_binary == 1, mask_patch == 1))
                up_U += np.sum(np.logical_or(pred_mask_binary == 1, mask_patch == 1))
                # up_U += np.sum(mask_patch == 1)
                up_I_bg += np.sum(np.logical_and(pred_mask_binary == 0, mask_patch == 0))
                up_U_bg += np.sum(np.logical_or(pred_mask_binary == 0, mask_patch == 0))
            elif patches_idx[i] <14:
                in_I += np.sum(np.logical_and(pred_mask_binary == 1, mask_patch == 1))
                in_U += np.sum(np.logical_or(pred_mask_binary == 1, mask_patch == 1))
                # in_U += np.sum(mask_patch == 1)
                in_I_bg += np.sum(np.logical_and(pred_mask_binary == 0, mask_patch == 0))
                in_U_bg += np.sum(np.logical_or(pred_mask_binary == 0, mask_patch == 0))
            else:
                out_I += np.sum(np.logical_and(pred_mask_binary == 1, mask_patch == 1))
                out_U += np.sum(np.logical_or(pred_mask_binary == 1, mask_patch == 1))
                # out_U += np.sum(mask_patch == 1)
                out_I_bg += np.sum(np.logical_and(pred_mask_binary == 0, mask_patch == 0))
                out_U_bg += np.sum(np.logical_or(pred_mask_binary == 0, mask_patch == 0))   

    pl_iou = round((up_I+in_I+out_I)/(up_U+in_U+out_U),4)
    # pl_iou = round((up_I+in_I+out_I)/(up_U+in_U+out_U),4)
    pl_dice = round((up_I+in_I+out_I)*2/(up_I+in_I+out_I+up_U+in_U+out_U),4)
    bg_iou =  round((up_I_bg+in_I_bg+out_I_bg)/(up_U_bg+in_U_bg+out_U_bg),4)
    bg_dice =  round((up_I_bg+in_I_bg+out_I_bg)*2/(up_I_bg+in_I_bg+out_I_bg+up_U_bg+in_U_bg+out_U_bg),4)               
    print("|",mesh_name,"|",pl_iou,"|",pl_dice,"|",bg_iou, "|", bg_dice)
    

    # ###################### 2D: Combine patches into 3 large images (up, i/o) ######################
    pred_mask_patches_up = pred_mask_patches[:6].reshape(2,3,256,256)
    pred_masks_patches_in = pred_mask_patches[6:14].reshape(1,8,256,256)
    pred_masks_patches_out = pred_mask_patches[14:].reshape(1,8,256,256)

    pred_mask_up = unpatchify(pred_mask_patches_up, (512, 768))
    pred_mask_in = unpatchify(pred_masks_patches_in, (256, 2048))
    pred_mask_out = unpatchify(pred_masks_patches_out, (256, 2048))
    # print(f"shape: {pred_mask_up.shape}, {pred_mask_in.shape}, {pred_mask_out.shape}")
    
    # save the 3 large images to path (as both greyscale png and npz)
    cv2.imwrite(os.path.join(save_dir, f"{mesh_name}_0.png"), pred_mask_up*255)
    cv2.imwrite(os.path.join(save_dir, f"{mesh_name}_1.png"), pred_mask_in*255)
    cv2.imwrite(os.path.join(save_dir, f"{mesh_name}_2.png"), pred_mask_out*255) # range [0-255]

    np.savez(os.path.join(save_dir, f"{mesh_name}_0.npz"), pred_mask_up)
    np.savez(os.path.join(save_dir, f"{mesh_name}_1.npz"), pred_mask_in)
    np.savez(os.path.join(save_dir, f"{mesh_name}_2.npz"), pred_mask_out) # range [0-1]
    return pl_iou, pl_dice, bg_iou, bg_dice, float(2*I/(U+I)),float(I/U)

def count_parameters_excluding_first_n_layers(model, start,end):
    # 获取模型的所有层
    layers = list(model.children())
    # 计算从第n层开始的参数量
    print(layers[start:end])
    return sum(p.numel() for layer in layers[start:end] for p in layer.parameters() if p.requires_grad)

def main():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    ###################### Load model and trained checkpoint ######################
    # ckpt_path = "/home/Liting/zhanxch/SAM/checkpts/rebuttal_SAM_train/ckpt_ep2.pth" # best is label2d3d 16 #compact_rate18 38 #compact_80decay 36
    # ckpt_path = "/home/Liting/zhanxch/SAM/checkpts/2d-branch-single/ckpt_ep16.pth"  #plaque iou2ds: 0.43015128205128206,non-plaque iou2ds: 0.9567692307692309,overall iou2ds: 0.6934602564102565
    # ckpt_path = "/home/Liting/zhanxch/SAM/checkpts/bmt/ckpt_ep6.pth" #plaque iou2ds: 0.4294717948717947,non-plaque iou2ds: 0.9600179487179487,overall iou2ds: 0.6947448717948717
    # ckpt_path = "/home/Liting/zhanxch/SAM/checkpts/2d_branch_bmt2/ckpt_ep2.pth" #plaque iou2ds: 0.3972743589743589, non-plaque iou2ds: 0.9536897435897438, overall iou2ds: 0.6754820512820514
    # ckpt_path = "/home/Liting/zhanxch/SAM/checkpts/2d_branch_bmt2/ckpt_ep78.pth" #plaque iou2ds: 0.39438717948717944, non-plaque iou2ds: 0.9546820512820514, overall iou2ds: 0.6745346153846155
    # ckpt_path = "/home/Liting/zhanxch/SAM/checkpts/2d_branch_bmt3_fillterface/ckpt_ep20.pth" #plaque iou2ds: 0.4327333333333333, non-plaque iou2ds: 0.9544641025641025
    ckpt_path = "/home/Liting/zhanxch/SAM/checkpts/2d_branch_bmt4/ckpt_ep16.pth" #plaque iou2ds: 0.42751282051282063 non-plaque iou2ds: 0.9599384615384614
    # ckpt_path = "/home/Liting/zhanxch/SAM/checkpts/2d_branch_bmt3_fillterface/ckpt_ep10.pth" #plaque iou2ds: 0.43554615384615386 non-plaque iou2ds: 0.9576692307692307
    base_dir = "/home/Liting/zhanxch/PlaqueIOS/16000/2D_png/val"
    save_dir = "/home/Liting/zhanxch/PlaqueIOS/16000/2D_png/val/2Dpred"
    # base_dir = "/home/Liting/zhanxch/PlaqueIOS/16000/2d_png_filter/val"
    # save_dir = "/home/Liting/zhanxch/PlaqueIOS/16000/2d_png_filter/val/2Dpred"
    # base_dir = "/home/Liting/zhanxch/PlaqueIOS/16000/2D_png/val"
    # save_dir = "/home/Liting/zhanxch/PlaqueIOS/16000/2D_png/val/2Dpred"
    # base_dir = "/home/Liting/zhanxch/oodtest/2D_png"
    # save_dir = "/home/Liting/zhanxch/oodtest/2Dpred"
    print("Loading checkpoints from:",ckpt_path)
    ###################### Load model and test 2D origin png ######################
    model = load_model(ckpt_path, device)
    # model = None
    # For each mesh
    # Load mesh names from text file
    with open("/home/Liting/zhanxch/PlaqueIOS/val_file_names.txt", "r") as file:
    # with open("/home/Liting/zhanxch/PlaqueIOS/splits/val_indices.txt", "r") as file:
    # with open("/home/Liting/zhanxch/PlaqueIOS/test_file_names.txt", "r") as file:

        mesh_names = file.readlines()
        mesh_names = [mesh_name.strip() for mesh_name in mesh_names]
        
    # print('mesh_names', mesh_names) #['011401', '001802', '006402', '015502', '003301',....]
    
    pl_dice2ds = []
    pl_iou2ds = []
    bg_dice2ds = []
    bg_iou2ds = []
    dice3Ds = []
    iou3ds = []
    print("|","mesh_name","|","plaque_iou2d","|","plaque_dice2d", "bg_iou2d", "bg_dice2d")
    for mesh_name in mesh_names:
        if int(mesh_name) in [4702]:
            continue
        # try:
        pl_iou2d, pl_dice2d,bg_iou2d,bg_dice2d,dice3D,iou3d = pred_mesh(mesh_name, base_dir, save_dir, model, device)
        pl_dice2ds.append(pl_dice2d)
        pl_iou2ds.append(pl_iou2d)
        bg_dice2ds.append(bg_dice2d)
        bg_iou2ds.append(bg_iou2d)

        # print(f"Finish processing {mesh_name}")
    print("plaque dice2ds:",sum(pl_dice2ds)/len(pl_dice2ds))
    print("non-plaque dice2ds:",sum(bg_dice2ds)/len(bg_dice2ds))
    print("overall dice2ds:",(sum(bg_dice2ds)/len(bg_dice2ds) + sum(pl_dice2ds)/len(pl_dice2ds))/2)
    print("plaque iou2ds:",sum(pl_iou2ds)/len(pl_iou2ds))
    print("non-plaque iou2ds:",sum(bg_iou2ds)/len(bg_iou2ds))
    print("overall iou2ds:",(sum(bg_iou2ds)/len(bg_iou2ds)+sum(pl_iou2ds)/len(pl_iou2ds))/2)




    





if __name__ == "__main__":
    main()