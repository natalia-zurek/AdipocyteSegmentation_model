# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 14:20:21 2023

@author: WylezolN

Mask2former coco
"""

from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import json
import albumentations as A
from transformers import Mask2FormerImageProcessor
import torch
#%%
json_dir = 'C:/Ovarian cancer project/Adipocyte dataset/coco/annotations/instances_train2014.json'
#with open('C:/Ovarian cancer project/Adipocyte dataset/coco/annotations/instances_train2014.json') as json_file:
#    data = json.load(json_file)


#%%
coco = COCO('C:/Ovarian cancer project/Adipocyte dataset/coco/annotations/instances_train2014.json')
image_dir = 'C:/Ovarian cancer project/Adipocyte dataset/coco/images/train2014/'
#%%
image_id = 9

img = coco.imgs[image_id]
img
#%%
image = np.array(Image.open(os.path.join(image_dir, img['file_name'])))
plt.imshow(image, interpolation='nearest')
plt.show()
#%%
anns_ids = coco.getAnnIds(imgIds=image_id)
anns = coco.loadAnns(anns_ids)
anns
#%%
mask = coco.annToMask(anns[0])
mask

#%%
np.unique(mask)
mask = coco.annToMask(anns[0])
for i in range(len(anns)):
    mask += coco.annToMask(anns[i])

plt.imshow(mask)

#%%
def generate_instance_map(coco, anns, img_height, img_width):
    """
    Generate an instance segmentation map with unique IDs for each instance.
    """

    # Handle empty annotations
    if not anns:
        return np.zeros((img_height, img_width), dtype=np.uint16), {}

    instance_map = np.zeros((coco.imgs[anns[0]['image_id']]['height'], coco.imgs[anns[0]['image_id']]['width']), dtype=np.uint16)
    instance_id = 1
    mapping = {}
    for ann in anns:
        mask = coco.annToMask(ann)
        instance_map[mask > 0] = instance_id
        mapping[instance_id] = ann['category_id']
        instance_id += 1
    return instance_map, mapping

def get_instance_and_mapping(coco, image_id, img_height, img_width):
    """
    Return an instance map and a mapping between instance IDs and semantic labels for a given image ID.
    """
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)

    # If there are no annotations, return a default instance map and an empty mapping.
    if not anns:
        return np.zeros((img_height, img_width), dtype=np.uint16), {}

    instance_map, mapping = generate_instance_map(coco, anns, img_height, img_width)

    # Check if the instance_map actually has any non-zero instance labels
    if not np.any(instance_map):
        return np.zeros((img_height, img_width), dtype=np.uint16), {}

    return instance_map, mapping

#%%
import numpy as np
from torch.utils.data import Dataset
import glob
class ImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, annotation_file, image_dir, processor, transform=None):
        """
        Args:
            dataset
        """
        self.annotation_file = annotation_file
        self.coco = COCO(annotation_file)
        self.image_dir = glob.glob(image_dir)
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx):
        #keys_list = list(self.coco.imgs.keys())
        #img_idx = keys_list[idx]
        imgInfo = self.coco.imgs[idx]
        image = np.array(Image.open(os.path.join(image_dir, imgInfo['file_name'])))

        instance_map, mapping = get_instance_and_mapping(self.coco, idx, image.shape[0], image.shape[1])

        if not mapping:  # If no annotations
          return None

        # apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=instance_map)
            image, instance_seg = transformed['image'], transformed['mask']
            # convert to C, H, W
            image = image.transpose(2,0,1)

        if not mapping:
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
    
#%%
ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

train_transform = A.Compose([
    A.Resize(width=512, height=512),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])

processor = Mask2FormerImageProcessor(reduce_labels=True, ignore_index=255, do_resize=False, do_rescale=False, do_normalize=False)

train_dataset = ImageSegmentationDataset(json_dir, image_dir, processor=processor, transform=train_transform)
#%%
inputs = train_dataset[9]
for k,v in inputs.items():
  if isinstance(v, torch.Tensor):
    print(k,v.shape)

#%% DATA LOADER
from torch.utils.data import DataLoader

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

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

#%% 

from transformers import Mask2FormerForUniversalSegmentation

# Replace the head of the pre-trained model
# We specify ignore_mismatched_sizes=True to replace the already fine-tuned classification head by a new one
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-instance",
                                                          ignore_mismatched_sizes=True)

#%%
batch = next(iter(train_dataloader))
for k,v in batch.items():
  if isinstance(v, torch.Tensor):
    print(k,v.shape)
  else:
    print(k,len(v))


#%%
print([label.shape for label in batch["class_labels"]])

#%%
outputs = model(
          pixel_values=batch["pixel_values"],
          mask_labels=batch["mask_labels"],
          class_labels=batch["class_labels"],
      )
outputs.loss

#%% TRAIN

from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

running_loss = 0.0
num_samples = 0
num_epochs = 2


for epoch in range(num_epochs):
  print("Epoch:", epoch)
  model.train()
  for idx, batch in enumerate(tqdm(train_dataloader)):
      if batch is None:  # Skip None batches
          continue

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

  if (epoch + 1) % 5 == 0 or epoch == (num_epochs - 1):
    checkpoint_path = f'C:/Ovarian cancer project/trained models/cocotest/mask2former_adipocyte_test_epoch_{epoch+1}.pt'
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

#%%
# initializing dictionary
test_dict = {'Gfg': 1, 'is': 2, 'best': 3}
 
# printing original dictionary
print("The original dictionary is : " + str(test_dict))
 
# Using next() + iter()
# Getting first key in dictionary
res = next(iter(test_dict))
 
# printing initial key
print("The first key of dictionary is : " + str(res))

#%%

for idx, batch in enumerate(tqdm(train_dataloader)):
    print(idx)
    print(batch)


#%%
enumerate(train_dataloader)
