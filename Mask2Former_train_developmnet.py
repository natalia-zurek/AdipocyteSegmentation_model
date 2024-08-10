# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:53:06 2024

@author: WylezolN
Mask2Former train developmnet
"""
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<512>"
#%%
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

# CUSTOM DATASET CLASS
class ImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""

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
        
        image = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
        instance_map = sio.loadmat(annotation_path)["inst_map"]
        unique_ids = np.unique(instance_map)

        # Creating a dictionary where each object has a class value of 1
        mapping = {obj_id: 1 for obj_id in unique_ids if obj_id != 0}

        if self.transform is not None:
            transformed = self.transform(image=image, mask=instance_map)
            image, instance_seg = transformed['image'], transformed['mask']
            image = image.transpose(2, 0, 1)

        if np.all(mapping == 0):
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

# FUNCTIONS
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels, "mask_labels": mask_labels}

def check_path_existence(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path '{path}' does not exist.")

# MAIN CODE

# Define variables
save_checkpoints_folder = "C:/_research_projects/Adipocyte model project/Mask2Former/trained models/model_1"
train_dataset_path = "C:/_research_projects/Adipocyte model project/Mask2Former/data/training 2"
val_dataset_path = "C:/_research_projects/Adipocyte model project/Mask2Former/data/validation 2"  # Add validation dataset path
batch_size = 1
num_epochs = 10
initial_lr = 5e-5

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

processor = Mask2FormerImageProcessor(reduce_labels=True, ignore_index=0, do_resize=False, do_rescale=False, rescale_factor=1/2, do_normalize=False)
train_dataset = ImageSegmentationDataset(train_dataset_path, processor, None)  # No augmentation
val_dataset = ImageSegmentationDataset(val_dataset_path, processor, None)  # No augmentation for validation
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Model
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-instance", ignore_mismatched_sizes=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Decays LR by a factor of 0.1 every 5 epochs

for epoch in range(num_epochs):
    print("Epoch:", epoch)
    
    # Training Phase
    model.train()
    running_loss = 0.0
    num_samples = 0

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
        loss.backward()

        batch_size = batch["pixel_values"].size(0)
        running_loss += loss.item()
        num_samples += batch_size

        if idx % 100 == 0:
            print("Training Loss:", running_loss / num_samples)

        # Optimization
        optimizer.step()

    # Step the scheduler
    scheduler.step()

    # Validation Phase
    model.eval()
    val_running_loss = 0.0
    val_num_samples = 0
    dice_scores = []
    all_ap_scores = []
    all_gt_masks = []
    all_pred_masks = []

    with torch.no_grad():
        for val_batch in tqdm(val_dataloader):
            if val_batch is None:
                continue

            val_outputs = model(
                pixel_values=val_batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in val_batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in val_batch["class_labels"]],
            )

            val_loss = val_outputs.loss
            val_batch_size = val_batch["pixel_values"].size(0)
            val_running_loss += val_loss.item()
            val_num_samples += val_batch_size

            # Calculate Dice coefficient
            # pred_masks = val_outputs.logits.argmax(dim=1).cpu().numpy()
            # gt_masks = [mask.cpu().numpy() for mask in val_batch["mask_labels"]]

            # for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            #     dice_score = dice_coefficient(pred_mask, gt_mask)
            #     dice_scores.append(dice_score)

            #     all_pred_masks.append(pred_mask.flatten())
            #     all_gt_masks.append(gt_mask.flatten())

            # Calculate Average Precision (AP)
            # for i in range(pred_masks.shape[0]):
            #     ap = average_precision_score(gt_masks[i].flatten(), pred_masks[i].flatten())
            #     all_ap_scores.append(ap)

        val_loss_avg = val_running_loss / val_num_samples
        # mean_dice_score = np.mean(dice_scores)
        # mean_ap = np.mean(all_ap_scores)
        # mAP = np.mean([precision_recall_curve(all_gt_masks[i], all_pred_masks[i])[1].max() for i in range(len(all_gt_masks))])
        # print(f"Validation Loss: {val_loss_avg}, Mean Dice: {mean_dice_score}, mAP: {mAP}, Mean AP: {mean_ap}")

    # Save checkpoints
    if (epoch + 1) % 5 == 0 or epoch == (num_epochs - 1):
        checkpoint_path = os.path.join(save_checkpoints_folder, f'mask2former_instseg_adipocyte_epoch_{epoch + 1}')
        model.save_pretrained(checkpoint_path)
        processor.save_pretrained(checkpoint_path)
        print(f"Model saved to {checkpoint_path}")


#%%

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

for epoch in range(num_epochs):
    print("Epoch:", epoch)
    model.train()
    running_loss = 0.0
    num_samples = 0

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
        loss.backward()

        batch_size = batch["pixel_values"].size(0)
        running_loss += loss.item()
        num_samples += batch_size

        if idx % 100 == 0:
            print("Training Loss:", running_loss / num_samples)

        # Optimization
        optimizer.step()

    # Validation Phase
    model.eval()
    val_running_loss = 0.0
    val_num_samples = 0

    with torch.no_grad():
        for val_batch in tqdm(val_dataloader):
            if val_batch is None:
                continue

            val_outputs = model(
                pixel_values=val_batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in val_batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in val_batch["class_labels"]],
            )

            val_loss = val_outputs.loss
            val_batch_size = val_batch["pixel_values"].size(0)
            val_running_loss += val_loss.item()
            val_num_samples += val_batch_size

        val_loss_avg = val_running_loss / val_num_samples
        print("Validation Loss:", val_loss_avg)

    # Save checkpoints
    if (epoch + 1) % 5 == 0 or epoch == (num_epochs - 1):
        checkpoint_path = os.path.join(save_checkpoints_folder, f'mask2former_instseg_adipocyte_epoch_{epoch + 1}')
        model.save_pretrained(checkpoint_path)
        processor.save_pretrained(checkpoint_path)
        print(f"Model saved to {checkpoint_path}")