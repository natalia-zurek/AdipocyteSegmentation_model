# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:29:51 2024

@author: WylezolN
"""
import numpy as np
from sklearn.metrics import confusion_matrix
import json


def write_metrics_to_json(filename, epoch, dice, iou, weighted_iou, mAP, mean_mAP, mean_of_mean_mAP, conf_matrix, class_names):
    # Convert confusion matrix to dictionary
    conf_matrix_dict = {class_name: list(row) for class_name, row in zip(class_names, conf_matrix)}
    
    metrics = {
        'epoch': epoch + 1,
        'DICE': dice,
        'IoU': iou,
        'weighted_IoU': weighted_iou,
        'mAP': mAP,
        'mAP_across_IoU_thr': mean_mAP,
        'mAP_across_classes': mean_of_mean_mAP,
        'confusion_matrix': conf_matrix_dict  # Use dictionary format
    }
    
    with open(filename, 'a') as file:
        json.dump(metrics, file, indent=4)
        file.write('\n')

def compute_confusion_matrix(gt_class_map, pred_class_map, classes):
    """
    Compute the pixel-wise confusion matrix between the ground truth and predicted class maps.

    Parameters:
    - gt_class_map (np.ndarray): Ground truth class map, 2D array of shape (H, W).
    - pred_class_map (np.ndarray): Predicted class map, 2D array of shape (H, W).

    Returns:
    - cm (np.ndarray): Confusion matrix of shape (num_classes, num_classes).
    """
    # Flatten the class maps to create 1D arrays
    gt_flat = gt_class_map.flatten()
    pred_flat = pred_class_map.flatten()
    
    # Compute the confusion matrix
    cm = confusion_matrix(gt_flat, pred_flat, labels=classes)
    
    return cm

def compute_precision_recall_curve(tp_list, fp_list, num_gt_instances):
    precisions = []
    recalls = []

    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)

    for tp, fp in zip(tp_cumsum, fp_cumsum):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / num_gt_instances if num_gt_instances > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls

def compute_dice(gt_instance_map, pred_instance_map):
    """
    Compute the Dice coefficient for binary segmentation maps.

    Parameters:
    - gt_instance_map (np.ndarray): Ground truth instance map, 2D array of shape (H, W).
    - pred_instance_map (np.ndarray): Predicted instance map, 2D array of shape (H, W).

    Returns:
    - dice (float): Dice coefficient, a value between 0 and 1.
    """
    gt_binary_mask = gt_instance_map > 0
    pred_binary_mask = pred_instance_map > 0
    
    intersection = np.sum(gt_binary_mask & pred_binary_mask)
    gt_sum = np.sum(gt_binary_mask)
    pred_sum = np.sum(pred_binary_mask)
    
    dice = (2 * intersection) / (gt_sum + pred_sum) if (gt_sum + pred_sum) > 0 else 1.0
    
    return dice

def compute_iou(gt_mask, pred_mask):
    """
    Compute the Intersection over Union (IoU) for binary masks.

    Parameters:
    - gt_mask (np.ndarray): Ground truth binary mask, 2D array of shape (H, W).
    - pred_mask (np.ndarray): Predicted binary mask, 2D array of shape (H, W).

    Returns:
    - iou (float): Intersection over Union (IoU), a value between 0 and 1.
    """
    intersection = np.sum((gt_mask == 1) & (pred_mask == 1))
    union = np.sum((gt_mask == 1) | (pred_mask == 1))
    iou = intersection / union if union > 0 else 1.0
    return iou

def match_instances_multithresh(gt_instance_map, pred_instance_map, pred_inst_ids, pred_scores, iou_thresholds):
    """
    Match predicted instances to ground truth instances based on multiple IoU thresholds.
    
    Parameters:
    - gt_instance_map (np.ndarray): Ground truth instance map.
    - pred_instance_map (np.ndarray): Predicted instance map.
    - pred_inst_ids (np.ndarray): IDs of predicted instances.
    - pred_scores (np.ndarray): Scores of predicted instances.
    - iou_thresholds (list of float): List of IoU thresholds.
    
    Returns:
    - tp_dict (dict of lists): Dictionary containing TP lists for each IoU threshold.
    - fp_dict (dict of lists): Dictionary containing FP lists for each IoU threshold.
    """
    
    # Sort predicted instances by their scores in descending order
    sorted_indices = np.argsort(pred_scores)[::-1]
    sorted_pred_ids = pred_inst_ids[sorted_indices]

    # Initialize data structures
    tp_dict = {iou_thresh: [] for iou_thresh in iou_thresholds}
    fp_dict = {iou_thresh: [] for iou_thresh in iou_thresholds}
    matched_iou = []

    # Track matched ground truth instances
    matched_gt_ids = set()

    # Iterate over sorted predicted instances
    for pred_id in sorted_pred_ids:
        pred_mask = (pred_instance_map == pred_id)
        
        best_iou = 0
        best_gt_id = None

        # Find the best matching ground truth instance
        for gt_id in np.unique(gt_instance_map):
            if gt_id == 0 or gt_id in matched_gt_ids:  # Skip background and already matched GT
                continue
            
            gt_mask = (gt_instance_map == gt_id)
            iou = compute_iou(gt_mask, pred_mask)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id
            
            # Early exit if perfect match is found
            if best_iou == 1.0:
                break

        # Save the best IoU found for this prediction
        matched_iou.append(best_iou if best_gt_id is not None else float('nan'))
        
        if best_gt_id is not None:
            matched_gt_ids.add(best_gt_id)  # Exclude this matched ground truth from further searches

    # Generate TP and FP lists for each IoU threshold
    for iou_thresh in iou_thresholds:
        for iou in matched_iou:
            if np.isnan(iou):
                tp_dict[iou_thresh].append(0)
                fp_dict[iou_thresh].append(1)
            elif iou >= iou_thresh:
                tp_dict[iou_thresh].append(1)
                fp_dict[iou_thresh].append(0)
            else:
                tp_dict[iou_thresh].append(0)
                fp_dict[iou_thresh].append(1)

    return tp_dict, fp_dict

def compute_average_precision(precisions, recalls):
    """
   Compute the average precision (AP) for a precision-recall curve.

   Parameters:
   - precisions (list of float): List of precision values.
   - recalls (list of float): List of recall values.

   Returns:
   - ap (float): Average precision (AP) score.
   """
    sorted_indices = np.argsort(recalls)
    precisions = np.array(precisions)[sorted_indices]
    recalls = np.array(recalls)[sorted_indices]
    
    ap = 0.0
    for i in range(1, len(precisions)):
        delta_recall = recalls[i] - recalls[i-1]
        ap += precisions[i] * delta_recall
    
    return ap

def match_instances(gt_instance_map, pred_instance_map, iou_thresh):
    """
    Match instances between ground truth and predicted maps based on IoU thresholds.

    Parameters:
    - gt_instance_map (np.ndarray): Ground truth instance map, 2D array of shape (H, W).
    - pred_instance_map (np.ndarray): Predicted instance map, 2D array of shape (H, W).
    - iou_thresholds (list of float): List of IoU thresholds for matching instances.

    Returns:
    - tp_dict (dict): Dictionary of true positives per IoU threshold.
    - fp_dict (dict): Dictionary of false positives per IoU threshold.
    - fn_dict (dict): Dictionary of false negatives per IoU threshold.
    """
    # Unique instance IDs, excluding background (ID 0)
    gt_ids = np.unique(gt_instance_map)[1:]
    pred_ids = np.unique(pred_instance_map)[1:]
    
    # Precompute areas
    gt_areas = {gt_id: np.sum(gt_instance_map == gt_id) for gt_id in gt_ids}
    pred_areas = {pred_id: np.sum(pred_instance_map == pred_id) for pred_id in pred_ids}
    
    tp_list, fp_list = [], []
    matched_gt = set()
    matched_pred = set()
    
    for pred_id in pred_ids:
        pred_mask = (pred_instance_map == pred_id)
        
        best_iou = 0
        best_gt_id = None
        
        for gt_id in gt_ids:
            if gt_id in matched_gt:  # Skip already matched GT instances
                continue
            
            gt_mask = (gt_instance_map == gt_id)
            
            # Calculate IoU
            intersection = np.sum(gt_mask & pred_mask)
            union = gt_areas[gt_id] + pred_areas[pred_id] - intersection
            iou = intersection / union if union > 0 else 0
            
            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id
            
            # Early exit for perfect match
            if iou == 1.0:
                break
        
        if best_iou >= iou_thresh:
            if best_gt_id not in matched_gt:
                tp_list.append(1)
                matched_gt.add(best_gt_id)
                matched_pred.add(pred_id)
            else:
                fp_list.append(1)
        else:
            fp_list.append(1)

    # All unmatched ground truths are considered false negatives
    fn = len(gt_ids) - len(matched_gt)
    
    return tp_list, fp_list, fn

def get_instance_map_by_class(class_map, instance_map, target_class):
    """
    Filter the instance map to retain only instances that belong to the target class.

    Parameters:
    - class_map (np.ndarray): Array representing the class of each pixel.
    - instance_map (np.ndarray): Array representing the instance of each pixel.
    - target_class (int): The class to filter by.

    Returns:
    - filtered_instance_map (np.ndarray): Instance map containing only instances of the target class.
    """

    class_mask = (class_map == target_class)
    filtered_instance_map = np.where(class_mask, instance_map, 0)
    
    return filtered_instance_map


def evaluate_instance_segmentation(gt_instance_maps, gt_class_maps, pred_instance_maps, pred_class_maps, pred_instance_ids, pred_scores, classes, iou_thresholds=None):
    if isinstance(classes, int):
        classes = [classes[1:]]

    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    dice_scores = []
    iou_scores = {cls: [] for cls in classes}
    weighted_iou_scores = {cls: [] for cls in classes}
    ap_scores = {cls: {thresh: [] for thresh in iou_thresholds} for cls in classes}
    conf_matrix = np.zeros([len(classes), len(classes)])
    
    for gt_instance_map, gt_class_map, pred_instance_map, pred_class_map, pred_inst_id, pred_score in zip(gt_instance_maps, gt_class_maps, pred_instance_maps, pred_class_maps, pred_instance_ids, pred_scores):
        # Dice Score
        dice = compute_dice(gt_instance_map, pred_instance_map)
        dice_scores.append(dice)
        
        conf_matrix_temp = compute_confusion_matrix(gt_class_map, pred_class_map, classes)
        conf_matrix += conf_matrix_temp
        gt_ids = np.unique(gt_instance_map)[1:]
        for cls in classes:
            gt_mask = (gt_class_map == cls)
            pred_mask = (pred_class_map == cls)
            
            # IoU Score
            iou = compute_iou(gt_mask, pred_mask)
            iou_scores[cls].append(iou)
            
            # Weighted IoU
            weighted_iou = iou * np.sum(gt_mask) / np.sum(gt_class_map > 0)
            weighted_iou_scores[cls].append(weighted_iou)
            
            #get the instance map for the particular class           
            gt_class_instance_map = get_instance_map_by_class(gt_class_map, gt_instance_map, cls)
            pred_class_instance_map = get_instance_map_by_class(pred_class_map, pred_instance_map, cls)
            
            # Object-based Average Precision at each IoU threshold

            tp_dict, fp_dict = match_instances_multithresh(gt_class_instance_map, pred_class_instance_map, pred_inst_id, pred_score, iou_thresholds)
            for iou_thresh in iou_thresholds:
                precisions, recalls = compute_precision_recall_curve(tp_dict[iou_thresh], fp_dict[iou_thresh], num_gt_instances=len(gt_ids))
                ap = compute_average_precision(precisions, recalls)
                ap_scores[cls][iou_thresh].append(ap)
    
    # Calculate final metrics
    dice = np.mean(dice_scores)
    iou = {cls: np.mean(iou_scores[cls]) for cls in classes}
    weighted_iou = {cls: np.mean(weighted_iou_scores[cls]) for cls in classes}
    mAP = {cls: {thresh: np.mean(ap_scores[cls][thresh]) for thresh in iou_thresholds} for cls in classes}
    mean_mAP = {cls: np.mean(list(mAP[cls].values())) for cls in classes}
    mean_of_mean_mAP = np.mean(list(mean_mAP.values()))
    
    return dice, conf_matrix, iou, weighted_iou, mAP, mean_mAP, mean_of_mean_mAP