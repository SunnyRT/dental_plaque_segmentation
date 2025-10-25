import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from transformers import SamConfig, SamProcessor, SamModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from iou import calculate_iou, update_confusion_matrix, calculate_confusion_matrix_percentage
from config import device
from dataloader import SAMDataset, load_and_patchify_png
from PIL import Image
from datasets import Dataset

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis('on')
    plt.show()

def show_masks_on_image(raw_image, masks, scores):
    if len(masks.shape) == 4:
        masks = masks.squeeze()
    if scores.shape[0] == 1:
        scores = scores.squeeze()

    nb_predictions = scores.shape[-1]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))

    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask = mask.cpu().detach()
        axes[i].imshow(np.array(raw_image))
        show_mask(mask, axes[i])
        axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
        axes[i].axis("off")
    plt.show()

def calculate_metrics(cm):
    """计算各类评估指标"""
    tn, fp, fn, tp = cm.ravel()
    
    # 基础指标
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Dice系数 (与F1相同)
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'dice': dice,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

def plot_confusion_matrix(cm, title='Confusion Matrix'):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['Negative', 'Positive'],
           yticklabels=['Negative', 'Positive'],
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    # 添加文本注释
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    return fig

def main():
    base_dir_train = "/home/Liting/ruitong/data/data_png/test"
    train_images, train_masks, train_name = load_and_patchify_png(base_dir_train)

    dataset_dict = {
        "image": [Image.fromarray(img) for img in train_images],
        "label": [Image.fromarray(mask) for mask in train_masks], 
        "ID": [n for n in train_name] 
    }
    
    train_dataset_src = Dataset.from_dict(dataset_dict)
    print(f"train_dataset_src: {train_dataset_src}")

    # Initialize the processor
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    # Create an instance of the SAMDataset
    train_dataset = SAMDataset(dataset=train_dataset_src, processor=processor)

    # Create a DataLoader instance for the training dataset
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=False)

    # Load the model
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    model.to(device)
    model.load_state_dict(torch.load("/home/Liting/zhanxch/SAM/checkpts/try/ckpt_ep2.pth"))
    model.eval()

    # Initialize variables to store metrics
    ious_1 = []  # 正类IoU
    ious_0 = []  # 负类IoU
    accuracies = []  # 准确率
    precisions = []  # 精确率
    recalls = []  # 召回率
    f1s = []  # F1分数
    cm = np.zeros((2, 2), dtype=int)  # 混淆矩阵

    # Iterate over all test images
    for idx, batch in enumerate(tqdm(train_dataloader, desc="Evaluating")):
        ground_truth_mask = batch["ground_truth_mask"].cpu().numpy().squeeze()
        
        with torch.no_grad():
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)

        # Apply sigmoid and convert to binary mask
        medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

        # 展平为一维数组用于计算准确率
        y_true = ground_truth_mask.flatten()
        y_pred = medsam_seg.flatten()

        # 计算并存储各项指标
        iou_1 = calculate_iou(medsam_seg, ground_truth_mask, class_label=1)
        ious_1.append(iou_1)
        
        iou_0 = calculate_iou(medsam_seg, ground_truth_mask, class_label=0)
        ious_0.append(iou_0)
        
        # 计算准确率
        accuracy = accuracy_score(y_true, y_pred)
        accuracies.append(accuracy)
        
        # 计算精确率、召回率和F1分数 (只计算正类)
        precision = precision_score(y_true, y_pred, zero_division=0)
        precisions.append(precision)
        
        recall = recall_score(y_true, y_pred, zero_division=0)
        recalls.append(recall)
        
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f1s.append(f1)
        
        # Update confusion matrix
        cm = update_confusion_matrix(cm, medsam_seg, ground_truth_mask)

    # 计算平均指标
    mean_iou_positive = np.mean(ious_1)
    mean_iou_negative = np.mean(ious_0)
    mean_accuracy = np.mean(accuracies)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1s)
    
    # 从混淆矩阵计算额外指标
    metrics = calculate_metrics(cm)

    # 打印评估结果
    print("\n===== 评估结果 =====")
    print(f'Mean IoU (Positive Class): {mean_iou_positive:.4f}')
    print(f'Mean IoU (Negative Class): {mean_iou_negative:.4f}')
    print(f'Mean Accuracy: {mean_accuracy:.4f}')
    print(f'Mean Precision: {mean_precision:.4f}')
    print(f'Mean Recall: {mean_recall:.4f}')
    print(f'Mean F1 Score: {mean_f1:.4f}')
    print("\n===== 混淆矩阵 =====")
    print(f'True Positives (TP): {metrics["tp"]}')
    print(f'False Positives (FP): {metrics["fp"]}')
    print(f'True Negatives (TN): {metrics["tn"]}')
    print(f'False Negatives (FN): {metrics["fn"]}')
    print("\n===== 派生指标 =====")
    print(f'Accuracy: {metrics["accuracy"]:.4f}')
    print(f'Precision: {metrics["precision"]:.4f}')
    print(f'Recall: {metrics["recall"]:.4f}')
    print(f'Dice Coefficient: {metrics["dice"]:.4f}')
    
    # 绘制混淆矩阵
    plot_confusion_matrix(cm)
    plt.show()

if __name__ == "__main__":
    main()