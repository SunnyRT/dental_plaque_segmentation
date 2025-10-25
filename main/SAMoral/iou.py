import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix



""""""""""""""""""""""""" For Evaluation: IoU """""""""""""""""""""""""""
# Define Helper Functions
def calculate_iou(pred_mask, true_mask, class_label=1):
    intersection = np.logical_and(pred_mask == class_label, true_mask == class_label)
    union = np.logical_or(pred_mask == class_label, true_mask == class_label)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def update_confusion_matrix(cm, pred_mask, true_mask):
    cm += confusion_matrix(true_mask.flatten(), pred_mask.flatten(), labels=[0, 1])
    return cm

def calculate_confusion_matrix_percentage(cm):
    tn, fp, fn, tp = cm.ravel()
    # print(f'tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}') # debug
    n = tn + fn
    p = fp + tp
    cm_percentage = np.array([[tn/n, fp/p],[fn/n, tp/p]])*100
    return cm_percentage