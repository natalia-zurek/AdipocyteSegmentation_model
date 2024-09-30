# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:53:06 2024

@author: WylezolN
Mask2Former train developmnet
"""
# LIBRARIES
# from PIL import Image
# import numpy as np
from transformers import Mask2FormerImageProcessor, Mask2FormerConfig, Mask2FormerForUniversalSegmentation
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import os
import time
# import scipy.io as sio
import csv
from data.dataloader import (collate_fn, 
                             eval_collate_fn,
                             ImageSegmentationDataset,
                             ImageSegmentationDatasetMultiClass,
                             RawValidationDataset
                             )

from models.inference import postprocess_results

from models.evaluation import evaluate_instance_segmentation, write_metrics_to_json

def check_path_existence(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path '{path}' does not exist.")
        

#%%
# MAIN CODE
# Define variables
train_dataset_path = "C:/_research_projects/Immune infiltrate project/immune infiltrate/Hovernet training Datasets/Dual IHC Mask2Former/training 2"
val_dataset_path = "C:/_research_projects/Immune infiltrate project/immune infiltrate/Hovernet training Datasets/Dual IHC Mask2Former/validation 2" 

save_checkpoints_folder = "C:/_research_projects/Adipocyte model project/Mask2Former/trained models/model_neuly 2"
# train_dataset_path = "C:/_research_projects/Adipocyte model project/Mask2Former/data/training 2"
# #train_dataset_path = "C:/_research_projects/Adipocyte model project/Mask2Former_v1/training dataset"
# val_dataset_path = "C:/_research_projects/Adipocyte model project/Mask2Former/data/validation 2"  # Add validation dataset path
batch_size = 1
num_epochs = 2
initial_lr = 5e-5
validation_frequency = 1 #every x epoch, default: None
checkpoint_frequency = 1 #every x epoch, default: 5

# class_names = ['background', 'adipocyte']   
# classes = [0, 1]
# target_size = (1024, 1024)

class_names = ['background', 'neutrophil', 'lymphocyte', 'other']   
classes = [0, 1, 2, 3]
target_size = (540, 540)

id2label = {0: 'neutrophil', 1: 'lymphocyte', 2: 'other'}
label2id = {'neutrophil': 0, 'lymphocyte': 1, 'other': 2}

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
#train_dataset = ImageSegmentationDataset(train_dataset_path, processor, train_transform)  # With augmentation
train_dataset = ImageSegmentationDatasetMultiClass(train_dataset_path, processor, None)  # No augmentation
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

if validation_frequency != 0:
    val_dataset = ImageSegmentationDatasetMultiClass(val_dataset_path, processor, None)  # No augmentation for validation
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    eval_dataset = RawValidationDataset(val_dataset_path, None)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=eval_collate_fn)

# Initialize model
model_name = "facebook/mask2former-swin-large-coco-instance"
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name, ignore_mismatched_sizes=True, num_labels=len(classes)-1)
# config = Mask2FormerConfig.from_pretrained(model_name)
# config.id2label = id2label
# config.label2id = label2id
# model.config = config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Decays LR by a factor of 0.5 every 5 epochs

# Initialize DataFrame to store metrics
metrics_path = os.path.join(save_checkpoints_folder, 'training_metrics.csv')
with open(metrics_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Learning Rate"])

# Initialize DF to store evaluation metrics
if validation_frequency != 0:
    metrics_path2 = os.path.join(save_checkpoints_folder, 'validation_metrics.csv')
    with open(metrics_path2, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "DICE", "IoU", "wIoU", "mAP", "mean_mAP", "mean_meanAP", "confusionMatrix"])

total_training_time = 0
## ========= START TRAINING ==========          
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    print(f"Epoch {epoch + 1} | Training")
    train_loss, val_loss = [], []

    # Training Phase
    model.train()
    train_start_time = time.time()
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
            print(f"  Training loss: {round(sum(train_loss) / len(train_loss), 6)}\n")
        
    train_loss = round(sum(train_loss)/len(train_loss), 6)
    optimizer.step()
    scheduler.step()
    
    train_end_time = time.time()
    train_time = train_end_time - train_start_time

    print(f"Training time for epoch {epoch + 1}: {train_time:.2f} seconds\n")
    current_lr = optimizer.param_groups[0]['lr']
    
    # ========= Validation Phase ========== 
    if validation_frequency != 0:
        
        model.eval()
        val_start_time = time.time()
        print(f"Epoch {epoch + 1} | Validation")
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
                    print(f"  Validation loss: {round(sum(val_loss) / len(val_loss), 6)}\n")
                    
        # Average validation epoch loss
        val_loss = round(sum(val_loss)/len(val_loss), 6)
        val_end_time = time.time()
        val_time = val_end_time - val_start_time
    
        print(f"Validation time for epoch {epoch + 1}: {val_time:.2f} seconds")
        
        
        print(f"Epoch {epoch + 1} | train_loss: {train_loss} | validation_loss: {val_loss} | learning_rate: {current_lr}\n")
                  
        with open(metrics_path, mode='a', newline='') as file:
            writer = csv.writer(file)       
            writer.writerow([epoch + 1, train_loss, val_loss, current_lr])
    
    else:
        with open(metrics_path, mode='a', newline='') as file:
            writer = csv.writer(file)       
            writer.writerow([epoch + 1, train_loss, 0, current_lr])
            
        print(f"Epoch {epoch + 1} | train_loss: {train_loss} | learning_rate: {current_lr}\n")
        
    # ========= SAVE MODEL ========== 
    if (epoch + 1) % checkpoint_frequency == 0 or epoch == (num_epochs - 1):
        checkpoint_path = os.path.join(save_checkpoints_folder, f'mask2former_instseg_adipocyte_epoch_{epoch + 1}')
        model.save_pretrained(checkpoint_path)
        processor.save_pretrained(checkpoint_path)
        print(f"Model saved to {checkpoint_path}\n")
    
    
    # ========= EVALUATION PHASE ========== 
    if validation_frequency != 0:
        if (epoch + 1) % validation_frequency == 0 or epoch == (num_epochs - 1):
            eval_start_time = time.time()
            all_pred_instance_maps = []
            all_pred_class_maps = []
            all_pred_instance_ids = []
            all_pred_class_ids = []
            all_pred_scores = []
            all_gt_instance_maps = []
            all_gt_class_maps = []
            
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
            
      
            dice, conf_matrix, iou, weighted_iou, mAP, mean_mAP, mean_of_mean_mAP = evaluate_instance_segmentation(
                     all_gt_instance_maps, all_gt_class_maps, all_pred_instance_maps, all_pred_class_maps, all_pred_instance_ids, all_pred_scores, classes)
            
            #saving metrics
            metrics_path3 = os.path.join(save_checkpoints_folder, 'validation_metrics.json')
            write_metrics_to_json(metrics_path3, epoch, dice, iou, weighted_iou, mAP, mean_mAP, mean_of_mean_mAP, conf_matrix, class_names) 
            
            with open(metrics_path2, mode='a', newline='') as file:
                writer = csv.writer(file)       
                writer.writerow([epoch + 1, dice, iou, weighted_iou, mAP, mean_mAP, mean_of_mean_mAP, conf_matrix])      
                
            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time
            print(f"Evaluation time for epoch {epoch + 1}: {eval_time:.2f} seconds\n")

    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    total_training_time += epoch_time
    print(f"Total time for epoch {epoch + 1}: {epoch_time:.2f} seconds\n")

# Total training time
print(f"Total training time: {total_training_time:.2f} seconds\n")


#%%
# Define the name of the model
model_name = "facebook/mask2former-swin-large-coco-instance"
config = Mask2FormerConfig.from_pretrained(model_name)
id2label = {0: 'adipocyte'}
label2id = {'adipocyte': 0}

id2label = {0: 'neutrophil', 1: 'lymphocyte', 2: 'other'}
label2id = {'neutrophil': 0, 'lymphocyte': 1, 'other': 2}

config.id2label = id2label
config.label2id = label2id