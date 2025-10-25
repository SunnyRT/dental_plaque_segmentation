import os
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from dataloader import load_and_patchify_png_permesh, SAMDataset
from patchify import unpatchify
from model import SASOTA
from segment_anything.build_sam import sam_model_registry

def calculate_metrics(pred_mask, gt_mask):
    """计算IoU, Dice系数和准确率"""
    pred_mask = pred_mask > 0.5  # 二值化 #0.5
    
    # 计算混淆矩阵
    tp = np.sum(np.logical_and(pred_mask == 1, gt_mask == 1))
    fp = np.sum(np.logical_and(pred_mask == 1, gt_mask == 0))
    fn = np.sum(np.logical_and(pred_mask == 0, gt_mask == 1))
    tn = np.sum(np.logical_and(pred_mask == 0, gt_mask == 0))
    
    # 计算指标
    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-8)
    
    return iou, dice, accuracy

def load_model(ckpt_path, device):
    """加载模型"""
    sam = sam_model_registry["vit_b"](checkpoint="/home/Liting/ruitong/SAMoral/SAMoral/sam_vit_b_01ec64.pth").to(device)
    model = SASOTA(sam)
    model.to(device)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    return model

def pred_mesh(mesh_name, base_dir, save_dir, model, device):
    """预测单个mesh并计算指标"""
    test_images, test_masks, patches_idx, test_mesh, label_mesh = load_and_patchify_png_permesh(base_dir, mesh_name)
    
    # 加载面片顺序信息
    order_dir = os.path.join(base_dir, 'order')
    order_npz = os.path.join(order_dir, mesh_name + '.npz')
    SOTA_mesh_file = np.load(order_npz, allow_pickle=True)      
    
    # 创建数据集和数据加载器
    test_dataset = SAMDataset(test_images, test_masks, test_mesh, label_mesh)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=5,
        prefetch_factor=3,
    )

    # 初始化预测结果存储
    pred_mask_patches = np.zeros((22, 256, 256), dtype=np.float32)
    gt_mask_patches = np.zeros((22, 256, 256), dtype=np.uint8)
    
    # 初始化指标统计
    metrics = {
        'up': {'iou': 0, 'dice': 0, 'accuracy': 0, 'count': 0},
        'in': {'iou': 0, 'dice': 0, 'accuracy': 0, 'count': 0},
        'out': {'iou': 0, 'dice': 0, 'accuracy': 0, 'count': 0},
        'total': {'iou': 0, 'dice': 0, 'accuracy': 0, 'count': 0}
    }

    ###################### 生成预测结果 ######################
    with torch.no_grad():
        for i, patch in enumerate(test_dataloader):
            img_patch, mask_patch = patch["pixel_values"], patch["ground_truth_mask"]
            gt_mask = mask_patch.cpu().numpy().squeeze()
            
            # 模型预测
            outputs = model(image=patch["pixel_values"].to(device), 
                          mesh=patch["SOTA_mesh"].to(device))
            pred_mask = torch.sigmoid(outputs["pred_masks"].squeeze(1))
            pred_mask = pred_mask.cpu().numpy().squeeze()
            
            # 存储预测结果
            pred_mask_patches[patches_idx[i]] = pred_mask
            gt_mask_patches[patches_idx[i]] = gt_mask
            
            # 计算当前patch指标
            iou, dice, accuracy = calculate_metrics(pred_mask, gt_mask)
            
            # 根据patch位置分类统计
            if patches_idx[i] < 6:  # up
                region = 'up'
            elif patches_idx[i] < 14:  # in
                region = 'in'
            else:  # out
                region = 'out'
            
            # 更新指标统计
            metrics[region]['iou'] += iou
            metrics[region]['dice'] += dice
            metrics[region]['accuracy'] += accuracy
            metrics[region]['count'] += 1
            
            metrics['total']['iou'] += iou
            metrics['total']['dice'] += dice
            metrics['total']['accuracy'] += accuracy
            metrics['total']['count'] += 1
    
    # 计算平均指标
    for region in metrics:
        if metrics[region]['count'] > 0:
            metrics[region]['iou'] /= metrics[region]['count']
            metrics[region]['dice'] /= metrics[region]['count']
            metrics[region]['accuracy'] /= metrics[region]['count']
    
    ###################### 合并patch并保存结果 ######################
    # 合并up区域
    pred_mask_patches_up = pred_mask_patches[:6].reshape(2, 3, 256, 256)
    pred_mask_up = unpatchify(pred_mask_patches_up, (512, 768))
    gt_mask_up = unpatchify(gt_mask_patches[:6].reshape(2, 3, 256, 256), (512, 768))
    
    # 合并in区域
    pred_masks_patches_in = pred_mask_patches[6:14].reshape(1, 8, 256, 256)
    pred_mask_in = unpatchify(pred_masks_patches_in, (256, 2048))
    gt_mask_in = unpatchify(gt_mask_patches[6:14].reshape(1, 8, 256, 256), (256, 2048))
    
    # 合并out区域
    pred_masks_patches_out = pred_mask_patches[14:].reshape(1, 8, 256, 256)
    pred_mask_out = unpatchify(pred_masks_patches_out, (256, 2048))
    gt_mask_out = unpatchify(gt_mask_patches[14:].reshape(1, 8, 256, 256), (256, 2048))
    
    # 保存预测结果
    for i, (pred, gt) in enumerate(zip(
        [pred_mask_up, pred_mask_in, pred_mask_out],
        [gt_mask_up, gt_mask_in, gt_mask_out]
    )):
        cv2.imwrite(os.path.join(save_dir, f"{mesh_name}_{i}_pred.png"), pred*255)
        cv2.imwrite(os.path.join(save_dir, f"{mesh_name}_{i}_gt.png"), gt*255)
        np.savez(os.path.join(save_dir, f"{mesh_name}_{i}_pred.npz"), pred)
        np.savez(os.path.join(save_dir, f"{mesh_name}_{i}_gt.npz"), gt)
    
    return metrics

def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    ckpt_path = "/home/Liting/zhanxch/SAM/checkpts/bmt/ckpt_ep6.pth" 
    base_dir = "/home/Liting/zhanxch/PlaqueIOS/16000/2D_png/val"
    save_dir = "/home/Liting/zhanxch/PlaqueIOS/16000/2D_png/val/2Dpred"
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载模型
    print("Loading checkpoints from:", ckpt_path)
    model = load_model(ckpt_path, device)
    
    # 加载mesh列表
    # with open("/home/Liting/zhanxch/PlaqueIOS/test_file_names.txt", "r") as file:
    with open("/home/Liting/zhanxch/PlaqueIOS/val_file_names.txt", "r") as file:
        mesh_names = [line.strip() for line in file if line.strip()]
    
    # 初始化全局指标
    global_metrics = {
        'up': {'iou': 0, 'dice': 0, 'accuracy': 0, 'count': 0},
        'in': {'iou': 0, 'dice': 0, 'accuracy': 0, 'count': 0},
        'out': {'iou': 0, 'dice': 0, 'accuracy': 0, 'count': 0},
        'total': {'iou': 0, 'dice': 0, 'accuracy': 0, 'count': 0}
    }
    
    # 处理每个mesh
    print("| Mesh Name | Region | IoU | Dice | Accuracy |")
    print("|----------|--------|-----|------|----------|")
    for mesh_name in mesh_names:
        if int(mesh_name) in [4702]:  # 跳过特定mesh
            continue
        
        try:
            metrics = pred_mesh(mesh_name, base_dir, save_dir, model, device)
            
            # 更新全局指标
            for region in metrics:
                for metric in ['iou', 'dice', 'accuracy']:
                    global_metrics[region][metric] += metrics[region][metric]
                global_metrics[region]['count'] += 1
            
            # 打印当前mesh结果
            for region in ['up', 'in', 'out', 'total']:
                if metrics[region]['count'] > 0:
                    print(f"| {mesh_name} | {region} | {metrics[region]['iou']:.4f} | {metrics[region]['dice']:.4f} | {metrics[region]['accuracy']:.4f} |")
        
        except Exception as e:
            print(f"Error processing {mesh_name}: {str(e)}")
    
    # 计算并打印全局平均指标
    print("\nGlobal Average Metrics:")
    print("| Region | IoU | Dice | Accuracy |")
    print("|--------|-----|------|----------|")
    for region in ['up', 'in', 'out', 'total']:
        if global_metrics[region]['count'] > 0:
            iou = global_metrics[region]['iou'] / global_metrics[region]['count']
            dice = global_metrics[region]['dice'] / global_metrics[region]['count']
            accuracy = global_metrics[region]['accuracy'] / global_metrics[region]['count']
            print(f"| {region} | {iou:.4f} | {dice:.4f} | {accuracy:.4f} |")

if __name__ == "__main__":
    main()