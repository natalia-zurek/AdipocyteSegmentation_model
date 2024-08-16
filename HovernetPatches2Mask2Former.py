# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 22:33:47 2024

@author: WylezolN
"""


import os
import numpy as np
from PIL import Image
import scipy.io as sio

# Define directories
npy_dir = 'D:/Hovernet training data/NeuLy/patches/dualihc/valid/540x540_164x164'  # Change this to the directory containing .npy files
images_dir = 'C:/_research_projects/Immune infiltrate project/immune infiltrate/Hovernet training Datasets/Dual IHC Mask2Former/validation/images'
annotations_dir = 'C:/_research_projects/Immune infiltrate project/immune infiltrate/Hovernet training Datasets/Dual IHC Mask2Former/validation/annotations'

# Create output directories if they don't exist
os.makedirs(images_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)

# List all .npy files in the directory
npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]


#%%
for npy_file in npy_files:
    # Load the .npy file
    data = np.load(os.path.join(npy_dir, npy_file))

    # Separate layers
    image = data[:, :, :3]
    inst_map = data[:, :, 3]
    class_map = data[:, :, 4]

    # Save the image as a .tif file
    image_pil = Image.fromarray(image.astype(np.uint8))
    image_pil.save(os.path.join(images_dir, f'{os.path.splitext(npy_file)[0]}.tif'))

    # Save the instance mask and class mask in a .mat file
    annotation_dict = {'class_map': class_map, 'inst_map': inst_map}
    sio.savemat(os.path.join(annotations_dir, f'{os.path.splitext(npy_file)[0]}.mat'), annotation_dict)

    print(f"Processed and saved {npy_file}")


#%%

# Separate the layers
image = data[:, :, :3]       # The first 3 layers represent the image
instance_mask = data[:, :, 3]  # The 4th layer is the instance mask
class_mask = data[:, :, 4]     # The 5th layer is the class mask

# Visualize the image and masks
plt.figure(figsize=(15, 5))

# Plot the image
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Image')
plt.axis('off')

# Plot the instance mask
plt.subplot(1, 3, 2)
plt.imshow(instance_mask, cmap='gray')
plt.title('Instance Mask')
plt.axis('off')

# Plot the class mask
plt.subplot(1, 3, 3)
plt.imshow(class_mask, cmap='gray')
plt.title('Class Mask')
plt.axis('off')

plt.show()

np.unique(class_mask)