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

def get_precision(pred_mask, true_mask, iou_threshold=0.5):
    """
    Compute precision for instance segmentation.

    Args:
        pred_mask (np.ndarray): Predicted instance mask of shape (H, W) where each pixel contains an instance ID.
        true_mask (np.ndarray): Ground truth instance mask of shape (H, W) where each pixel contains an instance ID.
        iou_threshold (float): IoU threshold to consider a predicted instance as a true positive.

    Returns:
        precision (float): Precision for the instance segmentation at the given IoU threshold.
    """
    true_instances = np.unique(true_mask)
    pred_instances = np.unique(pred_mask)

    true_positives = 0
    false_positives = 0

    for pred_id in pred_instances:
        if pred_id == 0:  # Background, skip it
            continue

        pred_instance_mask = (pred_mask == pred_id)
        iou_scores = []

        for true_id in true_instances:
            if true_id == 0:  # Background, skip it
                continue

            true_instance_mask = (true_mask == true_id)
            intersection = np.logical_and(pred_instance_mask, true_instance_mask).sum()
            union = np.logical_or(pred_instance_mask, true_instance_mask).sum()

            iou = intersection / union if union != 0 else 0
            iou_scores.append(iou)

        max_iou = max(iou_scores) if iou_scores else 0

        if max_iou >= iou_threshold:
            true_positives += 1
        else:
            false_positives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0

    return precision

def get_average_precision(pred_mask, true_mask, iou_thresholds=None):
    """
    Compute average precision (AP) for instance segmentation.

    Args:
        pred_mask (np.ndarray): Predicted instance mask of shape (H, W) where each pixel contains an instance ID.
        true_mask (np.ndarray): Ground truth instance mask of shape (H, W) where each pixel contains an instance ID.
        iou_thresholds (list of float): List of IoU thresholds to calculate AP over. Default is [0.5].

    Returns:
        average_precision (float): The mean average precision across IoU thresholds.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    precisions = []
    for threshold in iou_thresholds:
        precision = get_precision(pred_mask, true_mask, iou_threshold=threshold)
        precisions.append(precision)

    average_precision = np.mean(precisions) if precisions else 0.0

    return average_precision

def get_mean_average_precision(pred_masks, true_masks, iou_thresholds=None):
    """
    Compute mean average precision (mAP) for instance segmentation across multiple IoU thresholds.

    Args:
        pred_masks (list of np.ndarray): List of predicted instance masks for different images.
        true_masks (list of np.ndarray): List of ground truth instance masks for different images.
        iou_thresholds (list of float): List of IoU thresholds to calculate AP over. Default is [0.5, 0.75].

    Returns:
        mean_ap (float): The mean average precision across IoU thresholds and instances.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]

    ap_list = []
    for pred_mask, true_mask in zip(pred_masks, true_masks):
        ap = get_average_precision(pred_mask, true_mask, iou_thresholds=iou_thresholds)
        ap_list.append(ap)

    mean_ap = np.mean(ap_list) if ap_list else 0.0

    return mean_ap, ap_list

def get_iou_and_weighted_iou(pred_mask, true_mask, num_classes):
    """
    Compute the IoU and Weighted IoU for multi-class semantic segmentation.

    Args:
        pred_mask (np.ndarray): Predicted mask of shape (H, W), where each pixel contains the predicted class label.
        true_mask (np.ndarray): Ground truth mask of shape (H, W), where each pixel contains the true class label.
        num_classes (int): The number of classes (including background if applicable).

    Returns:
        iou_per_class (np.ndarray): Array of IoU values for each class.
        weighted_iou (float): Weighted IoU across all classes.
    """
    iou_per_class = np.zeros(num_classes)
    class_pixel_count = np.zeros(num_classes)

    for cls in range(num_classes):
        pred_class = (pred_mask == cls)
        true_class = (true_mask == cls)
        
        intersection = np.logical_and(pred_class, true_class).sum()
        union = np.logical_or(pred_class, true_class).sum()
        
        if union == 0:
            iou_per_class[cls] = np.nan  # Or consider assigning a zero, depending on your preference
        else:
            iou_per_class[cls] = intersection / union
        
        class_pixel_count[cls] = true_class.sum()

    # Calculate weighted IoU
    valid_classes = ~np.isnan(iou_per_class)
    weighted_iou = np.nansum(iou_per_class * class_pixel_count) / class_pixel_count[valid_classes].sum()

    return iou_per_class, weighted_iou

def get_multiclass_dice(pred_mask, true_mask, num_classes, smooth=1e-6):
    """
    Calculate the Dice coefficient for multi-class segmentation.

    Args:
        pred_mask (np.ndarray): Predicted mask of shape (H, W), where each pixel contains the predicted class label.
        true_mask (np.ndarray): Ground truth mask of shape (H, W), where each pixel contains the true class label.
        num_classes (int): The number of classes (excluding the background if applicable).
        smooth (float): Small value added to the denominator to avoid division by zero.

    Returns:
        dict: Dice coefficient for each class, where keys are the class indices and values are the Dice scores.
        float: Mean Dice coefficient across all classes.
    """

    dice_scores = {}
    for c in range(num_classes):
        # Create binary masks for the current class
        pred_class = (pred_mask == c).astype(np.float32)
        true_class = (true_mask == c).astype(np.float32)

        # Calculate intersection and union
        intersection = np.sum(pred_class * true_class)
        union = np.sum(pred_class) + np.sum(true_class)

        # Calculate Dice coefficient for the current class
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores[c] = dice

    # Calculate the mean Dice coefficient across all classes
    mean_dice = np.mean(list(dice_scores.values()))

    return dice_scores, mean_dice

def get_dice(pred_mask, true_mask, smooth=1e-6):
    """
    Calculate the Dice coefficient for instance segmentation.

    Args:
        pred_mask (np.ndarray): Predicted binary mask of shape (H, W), where 1 indicates the object.
        true_mask (np.ndarray): Ground truth binary mask of shape (H, W), where 1 indicates the object.
        smooth (float): Small value added to the denominator to avoid division by zero.

    Returns:
        float: Dice coefficient between 0 and 1, where 1 indicates perfect overlap.
    """

    # Flatten the masks to 1D arrays
    pred_mask = pred_mask.flatten()
    true_mask = true_mask.flatten()

    # Calculate the intersection and union
    intersection = np.sum(pred_mask * true_mask)
    union = np.sum(pred_mask) + np.sum(true_mask)

    # Calculate the Dice coefficient
    dice = (2. * intersection + smooth) / (union + smooth)

    return dice

def get_confusion_matrix(pred_mask, true_mask, num_classes):
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
validation_frequency = 5 #every x epoch, default: None
checkpoint_frequency = 5 #every x epoch, default: 5

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

#%%
if validation_frequency is not None:
    eval_dataset = RawValidationDataset(val_dataset_path, None)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=eval_collate_fn)
#%%
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

# Optionally load validation ground truth data
if validation_frequency is not None:
    metrics_path2 = os.path.join(save_checkpoints_folder, 'validation_metrics.csv')
    with open(metrics_path2, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "DICE", "IoU", "wIoU", "AP", "mAP", "confusionMatrix"])

    val_gt_masks = []
    val_gt_class_labels = []
    for val_batch in tqdm(val_dataloader):
        if val_batch is not None:
            val_gt_masks.append([labels.to(device) for labels in val_batch["mask_labels"]])
            val_gt_class_labels.append([labels.to(device) for labels in val_batch["class_labels"]])

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
            validation = 1
            
            with open(metrics_path2, mode='a', newline='') as file:
                writer = csv.writer(file)       
                writer.writerow([epoch + 1, dice, IoU, wIoU, AP, mAP, confusionMatrix])
    
        
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
            pred_instance_map, pred_class_map, _, _, _ = postprocess_results(results[0])
            all_pred_instance_maps.append(pred_instance_map)
            all_pred_class_maps.append(pred_class_map)
            
        all_gt_instance_maps.extend(eval_batch["instance_maps"])
        all_gt_class_maps.extend(eval_batch["class_maps"])
        
#%%

def evaluate_instance_segmentation(gt_instance_maps, gt_class_maps, pred_instance_maps, pred_class_maps, classes):
        
    return dice, IoU, wIoU, AP, mAP, confusionMatrix


classes = 1
dice, IoU, wIoU, AP, mAP, confusionMatrix = evaluate_instance_segmentation(all_gt_instance_maps, all_gt_class_maps, all_pred_instance_maps, all_pred_class_maps, classes)

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
plot_mask(full_instance_map)
