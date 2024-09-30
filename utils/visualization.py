# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:06:22 2024

@author: WylezolN
"""

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
    
class_map = mask

for i in range(class_map.shape[0]):
    plt.figure(figsize=(8, 8))
    plt.imshow(class_map[i], cmap='gray')  # You can change 'gray' to other colormaps like 'viridis', 'plasma', etc.
    plt.title(f'Layer {i+1}')
    plt.axis('off')  # Turn off axis labels
    plt.show()
    
#%%
#mask = results[0]["segmentation"].cpu().detach().numpy()
# plot_mask(gt_instance_map)
# plot_mask(pred_instance_map)