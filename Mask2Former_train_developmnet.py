# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:53:06 2024

@author: WylezolN
Mask2Former train developmnet
"""
# LIBRARIES
from PIL import Image
import numpy as np
from transformers import Mask2FormerImageProcessor
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Mask2FormerForUniversalSegmentation
from tqdm.auto import tqdm
import os
import scipy.io as sio
from sklearn.metrics import confusion_matrix
import csv

#%%
def compute_confusion_matrix(gt_class_map, pred_class_map):
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
    cm = confusion_matrix(gt_flat, pred_flat)
    
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

def match_instances_old(gt_instance_map, pred_instance_map, iou_thresh):
    matched_gt = set()
    matched_pred = set()
    tp_list, fp_list = [], []
    
    for pred_id in np.unique(pred_instance_map):
        if pred_id == 0:  # Skip background
            continue
        pred_mask = (pred_instance_map == pred_id)
        
        best_iou = 0
        best_gt_id = None
        for gt_id in np.unique(gt_instance_map):
            if gt_id == 0:  # Skip background
                continue
            gt_mask = (gt_instance_map == gt_id)
            
            iou = compute_iou(gt_mask, pred_mask)
            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id
        
        if best_iou >= iou_thresh:
            if best_gt_id not in matched_gt:
                tp_list.append(1)
                fp_list.append(0)
                matched_gt.add(best_gt_id)
                matched_pred.add(pred_id)
            else:
                fp_list.append(1)
                tp_list.append(0)
        else:
            fp_list.append(1)
            tp_list.append(0)

    # All unmatched ground truths are considered false negatives
    fn = len(np.unique(gt_instance_map)) - len(matched_gt) - 1  # Exclude background

    return tp_list, fp_list, fn

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

def evaluate_instance_segmentation(gt_instance_maps, gt_class_maps, pred_instance_maps, pred_class_maps, classes, iou_thresholds=None):
    """
    Evaluate instance segmentation performance using various metrics.

    Parameters:
    - gt_instance_maps (list of np.ndarray): List of ground truth instance maps, each of shape (H, W).
    - gt_class_maps (list of np.ndarray): List of ground truth class maps, each of shape (H, W).
    - pred_instance_maps (list of np.ndarray): List of predicted instance maps, each of shape (H, W).
    - pred_class_maps (list of np.ndarray): List of predicted class maps, each of shape (H, W).
    - classes (list of int): List of class labels to evaluate.
    - iou_thresholds (list of float, optional): List of IoU thresholds for AP calculation. Default is [0.5, 0.6, 0.7, 0.8, 0.9].

    Returns:
    - dice (float): Mean Dice score across all instance maps.
    - conf_matrix (np.ndarray): Combined confusion matrix across all class maps.
    - iou (dict): Mean IoU score for each class.
    - weighted_iou (dict): Weighted mean IoU score for each class.
    - mAP (dict): Mean Average Precision (AP) for each class at each IoU threshold.
    - mean_mAP (dict): Mean AP averaged across all IoU thresholds for each class.
    - mean_of_mean_mAP (float): Mean of mean AP values across all classes.
    """
    
    if isinstance(classes, int):
        classes = [classes]

    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    dice_scores = []
    iou_scores = {cls: [] for cls in classes}
    weighted_iou_scores = {cls: [] for cls in classes}
    ap_scores = {cls: {thresh: [] for thresh in iou_thresholds} for cls in classes}
    conf_matrix = np.zeros([len(classes)+1, len(classes)+1])
    
    for gt_instance_map, gt_class_map, pred_instance_map, pred_class_map in zip(gt_instance_maps, gt_class_maps, pred_instance_maps, pred_class_maps):
        # Dice Score
        dice = compute_dice(gt_instance_map, pred_instance_map)
        dice_scores.append(dice)
        
        conf_matrix += compute_confusion_matrix(gt_class_map, pred_class_map)
        
        for cls in classes:
            gt_mask = (gt_class_map == cls)
            pred_mask = (pred_class_map == cls)
            
            # IoU Score
            iou = compute_iou(gt_mask, pred_mask)
            iou_scores[cls].append(iou)
            
            # Weighted IoU
            weighted_iou = iou * np.sum(gt_mask) / np.sum(gt_class_map > 0)
            weighted_iou_scores[cls].append(weighted_iou)
            
            # Object-based Average Precision at each IoU threshold
            for iou_thresh in iou_thresholds:
                tp_list, fp_list, fn = match_instances(gt_instance_map, pred_instance_map, iou_thresh)
                
                precisions, recalls = compute_precision_recall_curve(tp_list, fp_list, num_gt_instances=len(np.unique(gt_instance_map)) - 1)
                
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
    """
    Compute the confusion matrix for multi-class semantic segmentation.

    Args:
        pred_mask (np.ndarray): Predicted mask of shape (H, W), where each pixel contains the predicted class label.
        true_mask (np.ndarray): Ground truth mask of shape (H, W), where each pixel contains the true class label.
        num_classes (int): The number of classes (including the background if applicable).

    Returns:
        np.ndarray: Confusion matrix of shape (num_classes, num_classes).
    """
    # Flatten the masks to 1D arrays
    pred_mask_flat = pred_mask.flatten()
    true_mask_flat = true_mask.flatten()

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(true_mask_flat, pred_mask_flat, labels=np.arange(num_classes))

    return conf_matrix
def match_instances_multithresh2(gt_instance_map, pred_instance_map, pred_inst_ids, pred_scores, iou_thresholds):
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

def evaluate_instance_segmentation2(gt_instance_maps, gt_class_maps, pred_instance_maps, pred_class_maps, pred_instance_ids, pred_scores, classes, iou_thresholds=None):
    if isinstance(classes, int):
        classes = [classes]

    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    dice_scores = []
    iou_scores = {cls: [] for cls in classes}
    weighted_iou_scores = {cls: [] for cls in classes}
    ap_scores = {cls: {thresh: [] for thresh in iou_thresholds} for cls in classes}
    conf_matrix = np.zeros([len(classes)+1, len(classes)+1])
    
    for gt_instance_map, gt_class_map, pred_instance_map, pred_class_map, pred_inst_id, pred_score in zip(gt_instance_maps, gt_class_maps, pred_instance_maps, pred_class_maps, pred_instance_ids, pred_scores):
        # Dice Score
        dice = compute_dice(gt_instance_map, pred_instance_map)
        dice_scores.append(dice)
        
        conf_matrix += compute_confusion_matrix(gt_class_map, pred_class_map)
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

            tp_dict, fp_dict = match_instances_multithresh2(gt_class_instance_map, pred_class_instance_map, pred_inst_id, pred_score, iou_thresholds)
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
# CUSTOM DATASET CLASS

class ImageSegmentationDatasetMultiClass(Dataset):
    """Image segmentation dataset with multiple classes."""

    def __init__(self, root_dir, processor, transform=None):
        self.root_dir = root_dir
        self.image_list = sorted(os.listdir(os.path.join(root_dir, 'images')))
        self.annotation_list = sorted(os.listdir(os.path.join(root_dir, 'annotations')))
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.annotation_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, 'images', self.image_list[idx])
        annotation_path = os.path.join(self.root_dir, 'annotations', self.annotation_list[idx])

        # Load image
        image = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
        
        # Load instance and class maps
        instance_map = sio.loadmat(annotation_path)["inst_map"]
        class_map = sio.loadmat(annotation_path)["class_map"]

        # Create mapping from instance IDs to their respective class labels
        unique_ids = np.unique(instance_map)
        mapping = {obj_id: class_map[instance_map == obj_id][0] for obj_id in unique_ids if obj_id != 0}

        if self.transform is not None:
            transformed = self.transform(image=image, mask=instance_map)
            image, instance_seg = transformed['image'], transformed['mask']
            image = image.transpose(2, 0, 1)
        else:
            instance_seg = instance_map

        # Prepare inputs based on the mapping
        if len(mapping) == 0:
            inputs = self.processor([image], return_tensors="pt")
            inputs = {k: v.squeeze() for k, v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros((0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1]))
        else:
            try:
                inputs = self.processor([image], [instance_seg], instance_id_to_semantic_id=mapping, return_tensors="pt")
                inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}
            except:
                inputs = self.processor([image], return_tensors="pt")
                inputs = {k: v.squeeze() for k, v in inputs.items()}
                inputs["class_labels"] = torch.tensor([0])
                inputs["mask_labels"] = torch.zeros((0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1]))

        return inputs

class ImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, root_dir, processor, transform=None):
        """
        Args:
            dataset
        """
        self.root_dir = root_dir
        self.image_list = sorted(os.listdir(os.path.join(root_dir, 'images')))
        self.annotation_list = sorted(os.listdir(os.path.join(root_dir, 'annotations')))
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.annotation_list)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, 'images', self.image_list[idx])
        annotation_path = os.path.join(self.root_dir, 'annotations', self.annotation_list[idx])
        
        image = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
        
        instance_map = sio.loadmat(annotation_path)["inst_map"]
        unique_ids = np.unique(instance_map)

        # Creating a dictionary where each object has a class value of 1
        mapping = {obj_id: 1 for obj_id in unique_ids if obj_id != 0}
        #mapping = sio.loadmat(annotation_path)["class_map"]

        # apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=instance_map)
            image, instance_seg = transformed['image'], transformed['mask']
            # convert to C, H, W
        else:
            instance_seg = instance_map
            
        image = image.transpose(2,0,1) #WHY?
        #TODO: check the annotations without objects (they should be added to the dataset)
        if np.all(mapping == 0):
            # Some image does not have annotation (all ignored)
            inputs = self.processor([image], return_tensors="pt")
            inputs = {k:v.squeeze() for k,v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros((0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1]))
        else:
          try:
            inputs = self.processor([image], [instance_seg], instance_id_to_semantic_id=mapping, return_tensors="pt")
            inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()}
          except:
            inputs = self.processor([image], return_tensors="pt")
            inputs = {k:v.squeeze() for k,v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros((0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1]))

        return inputs

class RawValidationDataset(Dataset):
    """Validation dataset to load raw instance_map and class_map."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and annotations.
            transform (callable, optional): Optional transform to be applied
                on an image and mask.
        """
        self.root_dir = root_dir
        self.image_list = sorted(os.listdir(os.path.join(root_dir, 'images')))
        self.annotation_list = sorted(os.listdir(os.path.join(root_dir, 'annotations')))
        self.transform = transform

    def __len__(self):
        return len(self.annotation_list)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, 'images', self.image_list[idx])
        annotation_path = os.path.join(self.root_dir, 'annotations', self.annotation_list[idx])
        
        # Load image and annotation
        image = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
        instance_map = sio.loadmat(annotation_path)["inst_map"]
        class_map = sio.loadmat(annotation_path)["class_map"]

        # Apply optional transforms if needed
        if self.transform is not None:
            transformed = self.transform(image=image, mask=instance_map)
            image = transformed['image']
            instance_map = transformed['mask']

        return {
            "image": image,               # Raw image data as numpy array
            "instance_map": instance_map, # Raw instance segmentation map as numpy array
            "class_map": class_map        # Raw class map as numpy array
        }
#FUNCTIONS
def collate_fn(batch):
    # Remove None values
    batch = [item for item in batch if item is not None]

    if not batch:  # If entire batch has None values, return None
        return None

    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels, "mask_labels": mask_labels}

def eval_collate_fn(batch):
    # Remove None values
    batch = [item for item in batch if item is not None]

    if not batch:  # If entire batch has None values, return None
        return None

    images = [example["image"] for example in batch]
    instance_maps = [example["instance_map"] for example in batch]
    class_maps = [example["class_map"] for example in batch]

    return {
        "images": images,                   # List of raw image numpy arrays
        "instance_maps": instance_maps,     # List of raw instance_map numpy arrays
        "class_maps": class_maps            # List of raw class_map numpy arrays
    }

def check_path_existence(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path '{path}' does not exist.")
        
def postprocess_results(results):
    """
    Postprocesses the instance segmentation results to generate instance and class maps.

    Args:
    - results (dict): The mask2former model's list output, where dict contains keys:
      'segmentation' and 'segmentation_info'. 'segmentation' is the instance map, and
      'segmentation_info' contains a list of dictionaries with 'id', 'label_id', 'was_fused', 'score'.

    Returns:
    - instance_map (np.array): Processed instance map where background is 0 and instance IDs start from 1.
    - class_map (np.array): Class map where each pixel value corresponds to the class ID.
    - instance_ids (np.array): Array of instance IDs.
    - class_ids (np.array): Array of class IDs corresponding to each instance.
    - scores (np.array): Array of confidence scores corresponding to each instance.
    """

    instance_ids = []
    class_ids = []
    scores = []

    # Process the segmentation map
    segmentation = results['segmentation']
    segmentation_info = results['segments_info']

    # Convert background from -1 to 0 and shift all IDs by +1
    segmentation = segmentation.cpu().detach().numpy()
    instance_map = segmentation.copy()
    instance_map += 1

    # Initialize the class_map with zeros (background)
    class_map = np.zeros_like(instance_map, dtype=np.uint8)

    for info in segmentation_info:
        instance_id = info['id'] + 1  # Shift instance IDs by +1
        class_id = info['label_id'] + 1
        score = info['score']

        # Update class map with the corresponding class ID for the instance
        class_map[instance_map == instance_id] = class_id

        # Store the instance ID, class ID, and score
        instance_ids.append(instance_id)
        class_ids.append(class_id)
        scores.append(score)


    # Convert lists to numpy arrays
    instance_ids = np.array(instance_ids)
    class_ids = np.array(class_ids)
    scores = np.array(scores)

    return instance_map, class_map, instance_ids, class_ids, scores
#%%
# MAIN CODE

# Define variables
save_checkpoints_folder = "C:/_research_projects/Adipocyte model project/Mask2Former/trained models/model_1"
train_dataset_path = "C:/_research_projects/Adipocyte model project/Mask2Former/data/training 2"
#train_dataset_path = "C:/_research_projects/Adipocyte model project/Mask2Former_v1/training dataset"
val_dataset_path = "C:/_research_projects/Adipocyte model project/Mask2Former/data/validation 2"  # Add validation dataset path
batch_size = 1
num_epochs = 1
initial_lr = 5e-5
validation_frequency = 1 #every x epoch, default: None
checkpoint_frequency = 1 #every x epoch, default: 5

check_path_existence(train_dataset_path)
check_path_existence(val_dataset_path)  # Check validation path

if not os.path.exists(save_checkpoints_folder):
    os.makedirs(save_checkpoints_folder, exist_ok=True)

# Data Augmentation
train_transform = A.Compose([
    A.GaussianBlur(always_apply=False, p=0.5, blur_limit=(3, 19), sigma_limit=(0, 2)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(always_apply=False, p=0.5),
])

#Note: Instance segmentation dataset labels start from 1 while 0 is reserved for the null / background class to be ignored.
processor = Mask2FormerImageProcessor(reduce_labels=True, ignore_index=255, do_resize=False, do_rescale=False, do_normalize=False)
train_dataset = ImageSegmentationDataset(train_dataset_path, processor, None)  # No augmentation
val_dataset = ImageSegmentationDataset(val_dataset_path, processor, None)  # No augmentation for validation

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


if validation_frequency is not None:
    eval_dataset = RawValidationDataset(val_dataset_path, None)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=eval_collate_fn)

# Model
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-instance", ignore_mismatched_sizes=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Decays LR by a factor of 0.1 every 5 epochs
#%%
# Initialize DataFrame to store metrics
metrics_path = os.path.join(save_checkpoints_folder, 'training_metrics.csv')

with open(metrics_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Learning Rate"])

# # Optionally load validation ground truth data
if validation_frequency is not None:
    metrics_path2 = os.path.join(save_checkpoints_folder, 'validation_metrics.csv')
    with open(metrics_path2, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "DICE", "IoU", "wIoU", "mAP", "mean_mAP", "mean_meanAP", "confusionMatrix"])

#     val_gt_masks = []
#     val_gt_class_labels = []
#     for val_batch in tqdm(val_dataloader):
#         if val_batch is not None:
#             val_gt_masks.append([labels.to(device) for labels in val_batch["mask_labels"]])
#             val_gt_class_labels.append([labels.to(device) for labels in val_batch["class_labels"]])

#%%
for epoch in range(num_epochs):
    print(f"Epoch {epoch} | Training")
    train_loss, val_loss = [], []

    # Training Phase
    model.train()
    for idx, batch in enumerate(tqdm(train_dataloader)):
        if batch is None:
            continue

        # Reset the parameter gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        outputs = model(
            pixel_values=batch["pixel_values"].to(device),
            mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
            class_labels=[labels.to(device) for labels in batch["class_labels"]],
        )

        # Backward propagation
        loss = outputs.loss
        train_loss.append(loss.item())
        loss.backward()

        batch_size = batch["pixel_values"].size(0)
       

        if idx % 50 == 0:
            print("  Training loss: ", round(sum(train_loss)/len(train_loss), 6))
        
    train_loss = round(sum(train_loss)/len(train_loss), 6)
    optimizer.step()
    scheduler.step()

    # Validation Phase
    model.eval()

    print(f"Epoch {epoch} | Validation")
    with torch.no_grad():
        for idx, val_batch in enumerate(tqdm(val_dataloader)):
            if val_batch is None:
                continue

            val_outputs = model(
                pixel_values=val_batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in val_batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in val_batch["class_labels"]],
            )

            # Get validation loss
            loss = outputs.loss
            val_loss.append(loss.item())
            if idx % 50 == 0:
                print("  Validation loss: ", round(sum(val_loss)/len(val_loss), 6))
                
    # Average validation epoch loss
    val_loss = round(sum(val_loss)/len(val_loss), 6)
    current_lr = optimizer.param_groups[0]['lr']
    # Print epoch losses
    print(f"Epoch {epoch + 1} | train_loss: {train_loss} | validation_loss: {val_loss} | learning_rate: {current_lr}")
              
    with open(metrics_path, mode='a', newline='') as file:
        writer = csv.writer(file)       
        writer.writerow([epoch + 1, train_loss, val_loss, current_lr])
    
    #TODO: code with validation dataset
    if validation_frequency is not None:
        if (epoch + 1) % validation_frequency == 0 or epoch == (num_epochs - 1):
            all_pred_instance_maps = []
            all_pred_class_maps = []
            all_pred_instance_ids = []
            all_pred_class_ids = []
            all_pred_scores = []
            all_gt_instance_maps = []
            all_gt_class_maps = []
            target_size = (1024, 1024)
            with torch.no_grad():
                for val_batch, eval_batch in tqdm(zip(val_dataloader, eval_dataloader)):
                    if val_batch is None:
                        continue
                    if eval_batch is None:
                        continue
                    val_outputs = model(
                        pixel_values=val_batch["pixel_values"].to(device),
                        mask_labels=[labels.to(device) for labels in val_batch["mask_labels"]],
                        class_labels=[labels.to(device) for labels in val_batch["class_labels"]],
                    )
                    
                    results = processor.post_process_instance_segmentation(val_outputs, target_sizes=[target_size])
                    pred_inst_map = results[0]["segmentation"].cpu().detach().numpy()
                    
                    for result in results:
                        pred_instance_map, pred_class_map, instance_ids, class_ids, scores = postprocess_results(results[0])
                        all_pred_instance_maps.append(pred_instance_map)
                        all_pred_class_maps.append(pred_class_map)
                        all_pred_instance_ids.append(instance_ids)
                        all_pred_class_ids.append(class_ids)
                        all_pred_scores.append(scores)
                        
                    all_gt_instance_maps.extend(eval_batch["instance_maps"])
                    all_gt_class_maps.extend(eval_batch["class_maps"])
            classes = [1]
            dice, conf_matrix, iou, weighted_iou, mAP, mean_mAP, mean_of_mean_mAP = evaluate_instance_segmentation2(
                     all_gt_instance_maps, all_gt_class_maps, all_pred_instance_maps, all_pred_class_maps, all_pred_instance_ids, all_pred_scores, classes)
            with open(metrics_path2, mode='a', newline='') as file:
                writer = csv.writer(file)       
                writer.writerow([epoch + 1, dice, iou, weighted_iou, mAP, mean_mAP, mean_of_mean_mAP, conf_matrix])
    
        
    # Save checkpoints
    if (epoch + 1) % checkpoint_frequency == 0 or epoch == (num_epochs - 1):
        checkpoint_path = os.path.join(save_checkpoints_folder, f'mask2former_instseg_adipocyte_epoch_{epoch + 1}')
        model.save_pretrained(checkpoint_path)
        processor.save_pretrained(checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
    

#%% EVALUATION
model_path = "C:/_research_projects/Adipocyte model project/Mask2Former_v1/trained models/model Ov1 MTC aug 1024/mask2former_adipocyte_test_epoch_80"
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path).to(device)
#processor = Mask2FormerImageProcessor()
processor = Mask2FormerImageProcessor(reduce_labels=True, ignore_index=255, do_resize=False, do_rescale=False, do_normalize=False)
# Validation Phase
model.eval()
val_running_loss = 0.0
val_num_samples = 0
dice_scores = []
all_ap_scores = []
#%%
all_pred_instance_maps = []
all_pred_class_maps = []
all_pred_instance_ids = []
all_pred_class_ids = []
all_pred_scores = []
all_gt_instance_maps = []
all_gt_class_maps = []
target_size = (1024, 1024)
with torch.no_grad():
    for val_batch, eval_batch in tqdm(zip(val_dataloader, eval_dataloader)):
        if val_batch is None:
            continue
        if eval_batch is None:
            continue
        val_outputs = model(
            pixel_values=val_batch["pixel_values"].to(device),
            mask_labels=[labels.to(device) for labels in val_batch["mask_labels"]],
            class_labels=[labels.to(device) for labels in val_batch["class_labels"]],
        )
        
        results = processor.post_process_instance_segmentation(val_outputs, target_sizes=[target_size])
        pred_inst_map = results[0]["segmentation"].cpu().detach().numpy()
        
        for result in results:
            pred_instance_map, pred_class_map, instance_ids, class_ids, scores = postprocess_results(results[0])
            all_pred_instance_maps.append(pred_instance_map)
            all_pred_class_maps.append(pred_class_map)
            all_pred_instance_ids.append(instance_ids)
            all_pred_class_ids.append(class_ids)
            all_pred_scores.append(scores)
            
        all_gt_instance_maps.extend(eval_batch["instance_maps"])
        all_gt_class_maps.extend(eval_batch["class_maps"])
        
        
#%% EVALUATION v1 but long time to compute mAP 



#%% Example usage:
classes = [1]
dice, conf_matrix, iou, weighted_iou, mAP, mean_mAP, mean_of_mean_mAP = evaluate_instance_segmentation(
     all_gt_instance_maps, all_gt_class_maps, all_pred_instance_maps, all_pred_class_maps, all_pred_inst_ids, all_pred_scores, classes)

#%%

# Example usage:


#%%
# Assuming you have ground truth and predicted class maps with 4 classes (0, 1, 2, 3)
gt_class_map = np.array([[0, 1, 1], [2, 2, 0], [1, 0, 3]])
pred_class_map = np.array([[0, 1, 0], [2, 0, 0], [1, 3, 3]])

num_classes = 4
conf_matrix2 = compute_confusion_matrix(gt_class_map, pred_class_map, num_classes)

print("Confusion Matrix:")
print(conf_matrix)
#%%
conf_matrix3 = conf_matrix + conf_matrix2
#%%
conf_matrix = []
conf_matrix += conf_matrix2
#%%
import matplotlib.pyplot as plt
class_map = gt_masks[0]

for i in range(class_map.shape[0]):
    plt.figure(figsize=(8, 8))
    plt.imshow(class_map[i], cmap='gray')  # You can change 'gray' to other colormaps like 'viridis', 'plasma', etc.
    plt.title(f'Layer {i+1}')
    plt.axis('off')  # Turn off axis labels
    plt.show()
    
#%%
import matplotlib.pyplot as plt
def plot_mask(mask, title="Mask"):
    """
    Plot a 2D mask using Matplotlib.
    
    Args:
        mask (numpy.ndarray): 2D numpy array representing the mask.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='tab20b')  # You can choose different color maps like 'tab20b', 'viridis', etc.
    plt.colorbar()  # Add a color bar to indicate the labels
    plt.title(title)
    plt.axis('off')  # Turn off the axis labels and ticks
    plt.show()
    
#mask = results[0]["segmentation"].cpu().detach().numpy()
plot_mask(gt_instance_map)
plot_mask(pred_instance_map)