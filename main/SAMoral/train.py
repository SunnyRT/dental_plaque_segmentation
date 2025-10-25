import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
# from datasets import Dataset # Note difference: from torch.utils.data import Dataset
from torch.utils.data import Dataset
import monai
import os
import re
import sys
from transformers import  SamProcessor
# from .dataloader import SAMDataset,load_and_patchify_png
# from .config import device
from dataloader import SAMDataset,load_and_patchify_png
# from config import device
from tqdm import tqdm
from statistics import mean
# from .model import SASOTA
# from .segment_anything.build_sam import sam_model_registry
from model import SASOTA
from segment_anything.build_sam import sam_model_registry

import torch.nn as nn
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def train_model(num_epochs, start_epoch, train_dataloader, model, optimizer,lr_scheduler, seg_loss, save_dir):
    model.train()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1]).to(device))
    for epoch in range(start_epoch, num_epochs):
        epoch_losses = []
        mesh_losses = []
        for batch in tqdm(train_dataloader):
      
    
            # outputs = model(image=batch["pixel_values"].to(device))
            outputs = model(image=batch["pixel_values"].to(device), mesh = batch["SOTA_mesh"].to(device))

            predicted_masks = outputs["pred_masks"]
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            # print('ground_truth_masks', ground_truth_masks.shape) #[4, 1, 256, 256]

            loss_2d = seg_loss(predicted_masks, ground_truth_masks)


            loss = loss_2d  


            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss_2d.item())


        lr_scheduler.step()
        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')
        # Save the model's state dictionary to a file every 5 epochs
        if epoch % 2 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"ckpt_ep{epoch}.pth"))


def main():
   
    base_dir_train = "/home/Liting/zhanxch/PlaqueIOS/16000/2D_png/train"
    # base_dir_train = "/home/Liting/zhanxch/PlaqueIOS/16000/2d_png_filter/train"
    train_images, train_masks,train_mesh,label_mesh = load_and_patchify_png(base_dir_train)
    # for i, img in enumerate(train_images):
    #     print(f"图像 {i} 尺寸:", img.shape)
    


    # Create an instance of the SAMDataset
    train_dataset = SAMDataset(train_images, train_masks,train_mesh,label_mesh)

    # Create a DataLoader instance for the training dataset
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=False)#,      collate_fn=collater)


    print("-------------------------- Build Model -----------------------")    
    
    # Load the model
    # sam = SamModel.from_pretrained("facebook/sam-vit-base")
    # sam = sam_model_registry["vit_b"](checkpoint="/home/Liting/ruitong/SAMoral/SAMoral/sam_vit_b_01ec64.pth")
    # sam.to(device)

    model = SASOTA(sam_model_registry["vit_b"](checkpoint="/home/Liting/ruitong/SAMoral/SAMoral/sam_vit_b_01ec64.pth").to(device))
    # model = SASOTA(sam_model_registry["vit_b"](checkpoint="/home/Liting/ruitong/SAMoral/SAMoral/sam_vit_b_01ec64.pth", image_size = 256, num_classes = 2).to(device))
    model.to(device)


    # Initialize the optimizer and the loss function
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=0)



    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    # TODO: Try DiceFocalLoss, FocalLoss, DiceCELoss
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    # save_dir_name = input("Save checkpts at dir:")
    # save_dir_name = "2d_branch_fliter_data"
    save_dir_name = "2d_branch_bmt4"
    save_dir = os.path.join("/home/Liting/zhanxch/SAM/checkpts", save_dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
        
    # 查找所有符合格式的权重文件
    weight_files = []
    for file in os.listdir(save_dir):
        if file.startswith("ckpt_ep") and file.endswith(".pth"):
            match = re.search(r"ckpt_ep(\d+)\.pth", file)
            if match:
                epoch = int(match.group(1))
                weight_files.append((epoch, os.path.join(save_dir, file)))

    # 如果找到权重文件，则按轮次排序并加载最新的
    if weight_files:
        # 按轮次降序排序
        weight_files.sort(key=lambda x: x[0], reverse=True)
        latest_epoch, latest_weight = weight_files[0]
        
        try:
            # 加载模型权重
            model.load_state_dict(torch.load(latest_weight))
            start_epoch = latest_epoch + 1
            print(f"已加载轮次 {latest_epoch} 的权重: {latest_weight}")
            print(f"将从第 {start_epoch} 轮开始训练")
            
        except Exception as e:
            print(f"加载权重失败: {e}")
            print("将从头开始训练")
            start_epoch = 0
    else:
        print("未找到权重文件，将从头开始训练")
        start_epoch = 0
        
    train_model(num_epochs=200, start_epoch =  start_epoch, train_dataloader=train_dataloader, model=model, optimizer=optimizer,lr_scheduler= scheduler, seg_loss=seg_loss, save_dir = save_dir)

if __name__ == "__main__":
    main()
