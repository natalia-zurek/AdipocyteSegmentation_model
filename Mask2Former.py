# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 15:43:55 2023

@author: WylezolN

Mask2former data preparation and model train
"""
#%% load libraries
from PIL import Image
import numpy as np
from transformers import Mask2FormerImageProcessor
import albumentations as A
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import Mask2FormerForUniversalSegmentation
from tqdm.auto import tqdm
import os
import scipy.io as sio
from datetime import datetime

#%% 
today_date = datetime.today().date()
save_checkpoints_folder =  f'C:/Ovarian cancer project/Adipocyte dataset/Mask2Former/trained models {today_date}/'

if not os.path.exists(save_checkpoints_folder):
    os.makedirs(save_checkpoints_folder)


# CUSTOM DATASET CLASS
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

# DATA AUGMENTATION    
ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

# note that you can include more fancy data augmentation methods here
train_transform = A.Compose([
    A.Resize(width=512, height=512),
    A.HorizontalFlip(p=0.5),  # apply horizontal flip with 50% probability
    A.VerticalFlip(p=0.5),  # apply vertical flip with 50% probability
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.1),  # apply random brightness and contrast adjustment
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.1),  # apply random shift, scale, and rotation
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.25),  # apply random RGB shift
    A.RandomGamma(gamma_limit=(80, 120), p=0.1),  # apply random gamma adjustment
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.1),  # apply random hue, saturation, and value shift
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.15),  # apply random Gaussian noise
    A.Blur(blur_limit=3, p=0.1),  # apply random blur
    # A.OpticalDistortion(p=0.1),  # apply random optical distortion
    # A.GridDistortion(p=0.1),  # apply random grid distortion
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    # A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.25),  # apply random coarse dropout
])
    
#%%
#size = (1024, 1024),

train_transform = A.Compose([
    A.GaussianBlur(always_apply=False, p=0.5, blur_limit=(3, 19), sigma_limit=(0, 2)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    #A.RandomScale(always_apply=False, p=1, interpolation=1, scale_limit=(-0.05, 0.05)), #changing pixel size, interpolation 1 - linear, 
    A.RandomRotate90(always_apply=False, p=0.5), #Randomly rotate the input by 90 degrees zero or more times.
    #A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])
processor = Mask2FormerImageProcessor(reduce_labels=True, ignore_index=0, do_resize=False, do_rescale=False, do_normalize=False)
train_dataset = ImageSegmentationDataset('C:/Ovarian cancer project/Adipocyte dataset/Mask2Former/training dataset', processor, train_transform)

# DATALOADER
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

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)



# MODEL
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-instance", ignore_mismatched_sizes=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

running_loss = 0.0
num_samples = 0
num_epochs = 100


for epoch in range(num_epochs):
  print("Epoch:", epoch)
  model.train()
  for idx, batch in enumerate(tqdm(train_dataloader)):
      if batch is None:  # Skip None batches
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
        print("Loss:", running_loss/num_samples)

      # Optimization
      optimizer.step()

  if (epoch + 1) % 5 == 0 or epoch == (num_epochs - 1):
    #checkpoint_path = f'C:/Ovarian cancer project/Adipocyte dataset/Mask2Former/trained models/mask2former_adipocyte_test_epoch_{epoch+1}.pt'
    #torch.save(model.state_dict(), checkpoint_path)
    checkpoint_path = os.path.join(save_checkpoints_folder, f'mask2former_adipocyte_test_epoch_{epoch+1}')
    model.save_pretrained(checkpoint_path)
    processor.save_pretrained(checkpoint_path)  
    print(f"Model saved to {checkpoint_path}")

