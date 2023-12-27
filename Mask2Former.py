# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 15:43:55 2023

@author: WylezolN

Mask2former data preparation and model train
"""
#%% load libraries
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from transformers import Mask2FormerImageProcessor
import albumentations as A
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import Mask2FormerForUniversalSegmentation
from tqdm.auto import tqdm
import glob
import os
import scipy.io as sio
#%% 
class ImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, root_dir, processor, transform=None):
        """
        Args:
            dataset
        """
        self.root_dir = root_dir
        self.image_list = os.listdir(os.path.join(root_dir, 'images'))
        self.annotation_list = os.listdir(os.path.join(root_dir, 'annotations'))
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.annotation_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, 'images', self.image_list[idx])
        annotation_path = os.path.join(self.root_dir, 'annotations', self.annotations_list[idx])
        
        image = Image.open(image_path).convert('RGB')
        
        instance_map = sio.loadmat(annotation_path)["inst_map"]
        mapping = sio.loadmat(annotation_path)["class_map"]

        if not mapping:  # If no annotations
          return None

        # apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=instance_map)
            image, instance_seg = transformed['image'], transformed['mask']
            # convert to C, H, W
            image = image.transpose(2,0,1) #WHY?

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
    