# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 14:38:29 2023

@author: WylezolN
https://github.com/NielsRogge/Transformers-Tutorials/blob/master/MaskFormer/Fine-tuning/Fine_tune_MaskFormer_on_an_instance_segmentation_dataset_(ADE20k_full).ipynb
"""

from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from transformers import MaskFormerImageProcessor
from transformers import Mask2FormerImageProcessor
import albumentations as A
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import MaskFormerForInstanceSegmentation
from transformers import Mask2FormerForUniversalSegmentation
from tqdm.auto import tqdm
#%%
from datasets import load_dataset

dataset = load_dataset("scene_parse_150", "instance_segmentation")
#%%
data = pd.read_csv('C:/Ovarian cancer project/AdipocyteSegmentation_model/instanceInfo100_train.txt',
                   sep='\t', header=0)
data.head(5)
#%%
id2label = {id: label.strip() for id, label in enumerate(data["Object Names"])}
print(id2label)
#%%
example = dataset['train'][1]
image = example['image']
image

example['annotation']

#%%
seg = np.array(example['annotation'])
# get green channel
instance_seg = seg[:, :, 1]
instance_seg


np.unique(instance_seg)

#%%
instance_seg = np.array(example["annotation"])[:,:,1] # green channel encodes instances
class_id_map = np.array(example["annotation"])[:,:,0] # red channel encodes semantic category
class_labels = np.unique(class_id_map)

# create mapping between instance IDs and semantic category IDs
inst2class = {}
for label in class_labels:
    instance_ids = np.unique(instance_seg[class_id_map == label])
    inst2class.update({i: label for i in instance_ids})
print(inst2class)
#%%
print("Visualizing instance:", id2label[inst2class[1] - 1])

# let's visualize the first instance (ignoring background)
mask = (instance_seg == 1)
visual_mask = (mask * 255).astype(np.uint8)
Image.fromarray(visual_mask)

#%%
print("Visualizing instance:", id2label[inst2class[2] - 1])

# let's visualize the second instance
mask = (instance_seg == 2)
visual_mask = (mask * 255).astype(np.uint8)
Image.fromarray(visual_mask)
#%%
R = seg[:, :, 0]
G = seg[:, :, 1]
masks = (R / 10).astype(np.int32) * 256 + (G.astype(np.int32))
visual_mask = (masks * 255).astype(np.uint8)
Image.fromarray(visual_mask)

#%%
processor = MaskFormerImageProcessor(do_reduce_labels=True, ignore_index=255, do_resize=False, do_rescale=False, do_normalize=False)
  
#%%
ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

transform = A.Compose([
    A.Resize(width=512, height=512),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])

transformed = transform(image=np.array(image), mask=instance_seg)
pixel_values = np.moveaxis(transformed["image"], -1, 0)
instance_seg_transformed = transformed["mask"]
print(pixel_values.shape)
print(instance_seg_transformed.shape)

#%% note that you can include more fancy data augmentation methods here
transform = A.Compose([
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
inputs = processor([pixel_values], [instance_seg_transformed], instance_id_to_semantic_id=inst2class, return_tensors="pt")

#%%
for k,v in inputs.items():
  if isinstance(v, torch.Tensor):
    print(k,v.shape)
  else:
    print(k,[x.shape for x in v])
    
#%%
assert not torch.allclose(inputs["mask_labels"][0][0], inputs["mask_labels"][0][1])
inputs["class_labels"]

#%%


# visualize first one
print("Label:", id2label[inputs["class_labels"][0][0].item()])

visual_mask = (inputs["mask_labels"][0][0].numpy() * 255).astype(np.uint8)
Image.fromarray(visual_mask)

#%%

# visualize second one
print("Label:", id2label[inputs["class_labels"][0][1].item()])

visual_mask = (inputs["mask_labels"][0][1].numpy() * 255).astype(np.uint8)
Image.fromarray(visual_mask)

# visualize third one
print("Label:", id2label[inputs["class_labels"][0][2].item()])

visual_mask = (inputs["mask_labels"][0][2].numpy() * 255).astype(np.uint8)
Image.fromarray(visual_mask)

#%%
class ImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, dataset, processor, transform=None):
        """
        Args:
            dataset
        """
        self.dataset = dataset
        self.processor = processor
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image = np.array(self.dataset[idx]["image"].convert("RGB"))

        instance_seg = np.array(self.dataset[idx]["annotation"])[:,:,1]
        class_id_map = np.array(self.dataset[idx]["annotation"])[:,:,0]
        class_labels = np.unique(class_id_map)

        inst2class = {}
        for label in class_labels:
            instance_ids = np.unique(instance_seg[class_id_map == label])
            inst2class.update({i: label for i in instance_ids})

        # apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=instance_seg)
            image, instance_seg = transformed['image'], transformed['mask']
            # convert to C, H, W
            image = image.transpose(2,0,1)

        if class_labels.shape[0] == 1 and class_labels[0] == 0:
            # Some image does not have annotation (all ignored)
            inputs = self.processor([image], return_tensors="pt")
            inputs = {k:v.squeeze() for k,v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros((0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1]))
        else:
          inputs = self.processor([image], [instance_seg], instance_id_to_semantic_id=inst2class, return_tensors="pt")
          inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()}

        return inputs
    
    
#%%

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

# note that you can include more fancy data augmentation methods here
train_transform = A.Compose([
    A.Resize(width=512, height=512),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])

train_dataset = ImageSegmentationDataset(dataset["train"], processor=processor, transform=train_transform)

#%%
inputs = train_dataset[0]
for k,v in inputs.items():
  if isinstance(v, torch.Tensor):
    print(k,v.shape)    

#%% PYTORCH DATALOADER 

def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels, "mask_labels": mask_labels}

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
     
batch = next(iter(train_dataloader))
for k,v in batch.items():
  if isinstance(v, torch.Tensor):
    print(k,v.shape)
  else:
    print(k,len(v))
    
#%%
ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

batch_index = 1

unnormalized_image = (batch["pixel_values"][batch_index].numpy() * np.array(ADE_STD)[:, None, None]) + np.array(ADE_MEAN)[:, None, None]
unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
Image.fromarray(unnormalized_image)

batch["class_labels"][batch_index]
id2label[1]
batch["mask_labels"][batch_index].shape

print("Visualizing mask for:", id2label[batch["class_labels"][batch_index][0].item()])

visual_mask = (batch["mask_labels"][batch_index][0].bool().numpy() * 255).astype(np.uint8)
Image.fromarray(visual_mask)

print("Visualizing mask for:", id2label[batch["class_labels"][batch_index][1].item()])

visual_mask = (batch["mask_labels"][batch_index][1].bool().numpy() * 255).astype(np.uint8)
Image.fromarray(visual_mask)

# Replace the head of the pre-trained model
# We specify ignore_mismatched_sizes=True to replace the already fine-tuned classification head by a new one
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade",
                                                          id2label=id2label,
                                                          ignore_mismatched_sizes=True)

#%% CALCULATE INITIAL LOSS

batch = next(iter(train_dataloader))
for k,v in batch.items():
  if isinstance(v, torch.Tensor):
    print(k,v.shape)
  else:
    print(k,len(v))
    
print([label.shape for label in batch["class_labels"]])
print([label.shape for label in batch["mask_labels"]])

outputs = model(
          pixel_values=batch["pixel_values"],
          mask_labels=batch["mask_labels"],
          class_labels=batch["class_labels"],
      )
outputs.loss

#%% TRAIN THE MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

running_loss = 0.0
num_samples = 0
for epoch in range(2):
#for epoch in range(100):
  print("Epoch:", epoch)
  model.train()
  for idx, batch in enumerate(tqdm(train_dataloader)):
      # Reset the parameter gradients
      optimizer.zero_grad()

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
            
#%% INFERENCE

example = dataset['train'][1]
image = example['image']
image



processor = MaskFormerImageProcessor()

# prepare image for the model
inputs = processor(image, return_tensors="pt").to(device)
for k,v in inputs.items():
  print(k,v.shape)



# forward pass
with torch.no_grad():
  outputs = model(**inputs)

# you can pass them to processor for postprocessing
results = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
print(results.keys())

for segment in results['segments_info']:
  print(segment)
  

def get_mask(segmentation, segment_id):
  mask = (segmentation.cpu().numpy() == segment_id)
  visual_mask = (mask * 255).astype(np.uint8)
  visual_mask = Image.fromarray(visual_mask)

  return visual_mask
     

for segment in results['segments_info']:
    print("Visualizing mask for instance:", model.config.id2label[segment['label_id']])
    mask = get_mask(results['segmentation'], segment['id'])
    display(mask)
    print("------")
      
    



















#%% CODE from
#https://pyimagesearch.com/2023/03/13/train-a-maskformer-segmentation-model-with-hugging-face-transformers/

# Import the necessary packages