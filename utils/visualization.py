# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:06:22 2024

@author: WylezolN
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import Tuple

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
    
# class_map = mask

# for i in range(class_map.shape[0]):
#     plt.figure(figsize=(8, 8))
#     plt.imshow(class_map[i], cmap='gray')  # You can change 'gray' to other colormaps like 'viridis', 'plasma', etc.
#     plt.title(f'Layer {i+1}')
#     plt.axis('off')  # Turn off axis labels
#     plt.show()



#special thanks: https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay
def overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5, 
    resize: Tuple[int, int] = (1024, 1024)
) -> np.ndarray:
    """Combines image and its segmentation mask into a single image.
    
    Params:
        image: Training image.
        mask: Segmentation mask.
        color: Color for segmentation mask rendering.
        alpha: Segmentation mask's transparency.
        resize: If provided, both image and its mask are resized before blending them together.
    
    Returns:
        image_combined: The combined image.
        
    """
    color = np.asarray(color).reshape(3, 1, 1)
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()
    
    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)
    
    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    
    return image_combined
#%%
#mask = results[0]["segmentation"].cpu().detach().numpy()
# plot_mask(gt_instance_map)
# plot_mask(pred_instance_map)