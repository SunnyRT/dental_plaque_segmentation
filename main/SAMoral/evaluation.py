import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from transformers import SamConfig, SamProcessor, SamModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from iou import calculate_iou, update_confusion_matrix, calculate_confusion_matrix_percentage
from config import device
from dataloader import SAMDataset,load_and_patchify_png
# from .iou import calculate_iou, update_confusion_matrix, calculate_confusion_matrix_percentage
# from .config import device
# from .dataloader import load_valid_data, SAMDataset,collater,load_and_patchify_png
from PIL import Image
# from datasets import Dataset
from torch.utils.data import Dataset

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


def main():
    
    base_dir_train = "/home/Liting/ruitong/data/data_png/test"
    train_images, train_masks,train_name = load_and_patchify_png(base_dir_train)

    dataset_dict = {
        "image": [Image.fromarray(img) for img in train_images],
        "label": [Image.fromarray(mask) for mask in train_masks], 
        "ID":   [n for n in train_name] 
    }
    
    train_dataset_src = Dataset.from_dict(dataset_dict)
    print(f"train_dataset_src:{train_dataset_src}")

    # Initialize the processor
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    # Create an instance of the SAMDataset
    train_dataset = SAMDataset(dataset=train_dataset_src, processor=processor)

    # Create a DataLoader instance for the training dataset
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=False)#,      collate_fn=collater)

    # Load the model
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    model.to(device)
    model.load_state_dict(torch.load("/home/Liting/zhanxch/SAM/checkpts/try/ckpt_ep2.pth"))
    model.eval()


# Initialize variables to store IoUs and confusion matrix
    ious_1 = []
    ious_0 = []
    cm = np.zeros((2, 2), dtype=int)

    # Iterate over all test images
    for idx, batch in enumerate(tqdm(train_dataloader, desc="Evaluating")):

        ground_truth_mask = batch["ground_truth_mask"].cpu().numpy().squeeze()
        # Prepare inputs with bounding box as prompt

        with torch.no_grad():
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)

        # Apply sigmoid
        medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        # Convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

        # Calculate IoU
        iou_1 = calculate_iou(medsam_seg, ground_truth_mask, class_label=1)
        ious_1.append(iou_1)

        iou_0 = calculate_iou(medsam_seg, ground_truth_mask, class_label=0)
        ious_0.append(iou_0)

        # Update confusion matrix
        cm = update_confusion_matrix(cm, medsam_seg, ground_truth_mask)

    # Calculate mean IoU for the positive and negative classes
    mean_iou_positive = np.mean(ious_1)
    mean_iou_negative = np.mean(ious_0)
    print(f'Mean IoU (Positive Class): {mean_iou_positive}')
    print(f'Mean IoU (Negative Class): {mean_iou_negative}')

    # Calculate confusion matrix percentage
    # cm_percentage = calculate_confusion_matrix_percentage(cm)
    # print('Confusion Matrix (Percentage):')
    # print(cm_percentage)

    # # Print the meaning of each entry in the confusion matrix
    # tn, fp, fn, tp = cm.ravel()
    # print(f'True Negative (TN): {tn}')
    # print(f'False Positive (FP): {fp}')
    # print(f'False Negative (FN): {fn}')
    # print(f'True Positive (TP): {tp}')

    # # Plot the confusion matrix with text annotations
    # fig, ax = plt.subplots()
    # cax = ax.matshow(cm_percentage, cmap=plt.cm.Blues)
    # fig.colorbar(cax)

    # for (i, j), val in np.ndenumerate(cm_percentage):
    #     ax.text(j, i, f'{val:.2f}%', va='center', ha='center', color='white' if val > 50 else 'black')

    # ax.set_xticks([0, 1])
    # ax.set_yticks([0, 1])
    # ax.set_xticklabels(['Predicted Negative', 'Predicted Positive'])
    # ax.set_yticklabels(['Actual Negative', 'Actual Positive'])

    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.title('Confusion Matrix (Percentage)')
    # plt.show()
    
# if __name__ == "__main__":
#     main()
