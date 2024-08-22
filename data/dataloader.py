# -*- coding: utf-8 -*-
"""
@author: Natalia Zurek

dataloader
"""
# LIBRARIES
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import scipy.io as sio


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