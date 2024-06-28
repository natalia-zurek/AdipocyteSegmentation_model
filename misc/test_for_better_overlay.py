# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:10:58 2024

@author: WylezolN
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from transformers import Mask2FormerImageProcessor
import torch
import os
import cv2
from transformers import Mask2FormerForUniversalSegmentation
from docopt import docopt
from tqdm.auto import tqdm
import scipy.io as sio
# Load your image (replace 'your_image.png' with the path to your image)
image_path = 'C:/Ovarian cancer project/Adipocyte dataset/Mask2Former/test dataset/images inference test/0-11900_GTEX-13QJC_Adipose-Subcutaneous.jpg'  # Read as grayscal
model_path = "C:/Ovarian cancer project/Adipocyte dataset/Mask2Former/trained models/trained models 2024-01-02/mask2former_adipocyte_test_epoch_10"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path).to(device)      
processor = Mask2FormerImageProcessor()

def get_mask(segmentation, segment_id):
  mask = (segmentation.cpu().numpy() == segment_id)
  visual_mask = (mask * 255).astype(np.uint8)
  visual_mask = Image.fromarray(visual_mask)

  return visual_mask      
image = Image.open(os.path.join(image_path)).convert('RGB')
        # prepare image for the model
inputs = processor(image, return_tensors="pt").to(device)
        #for k,v in inputs.items():
        #  print(k,v.shape)
with torch.no_grad():
    outputs = model(**inputs)
            
results = processor.post_process_instance_segmentation(outputs)[0]
        #results = processor.post_process_instance_segmentation(outputs, return_binary_maps = True)[0]
original_image = np.array(image)
final_overlay = np.zeros_like(original_image)
#%%       
        # Iterate over the segments and visualize each mask on top of the original image
for segment in results['segments_info']:
    # Get mask for specific instance
    #mask = get_mask(results['segmentation'], segment['id'])
    mask = (results['segmentation'].cpu().numpy() == segment['id'])
    # Resize mask if necessary
    mask_array = np.array(mask)
    if mask_array.shape != original_image.shape[:2]:
        mask_array = np.array(mask.resize((original_image.shape[1], original_image.shape[0])))

    # Find where the mask is
    mask_location = mask_array == 255

    # Set the mask area to a specific color
    red_channel = final_overlay[:,:,0]
    red_channel[mask_location] = 255  # you may want to ensure that this does not overwrite previous masks
    final_overlay[:,:,0] = red_channel
    
#%%
instance_seg_mask = results["segmentation"].cpu().detach().numpy()
#%%
import random
def visualize_instance_seg_mask(mask):
    # Initialize image with zeros with the image resolution
    # of the segmentation mask and 3 channels
    image = np.zeros((mask.shape[0], mask.shape[1], 3))
    # Create labels
    labels = np.unique(instance_seg_mask)
    label2color = {
        label: (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        for label in labels
    }
    for height in range(image.shape[0]):
        for width in range(image.shape[1]):
            image[height, width, :] = label2color[mask[height, width]]
    image = image / 255
    return image


import numpy as np


def visualize_instance_seg_mask2(mask):
    # Initialize image with zeros with the image resolution
    # of the segmentation mask and 3 channels
    image = np.zeros((mask.shape[0], mask.shape[1], 3))

    # Create labels excluding -1
    labels = np.unique(mask[mask != -1])

    # Select a colormap (e.g., 'jet') and normalize label values
    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(vmin=labels.min(), vmax=labels.max())

    # Assign colors to labels based on the colormap
    label2color = {label: cmap(norm(label))[:3] for label in labels}

    for height in range(image.shape[0]):
        for width in range(image.shape[1]):
            if mask[height, width] != -1:
                image[height, width, :] = label2color[mask[height, width]]

    return image

instance_seg_mask_disp = visualize_instance_seg_mask2(instance_seg_mask)
mask_arr2 = np.resize(instance_seg_mask_disp, (image.width, image.height, 3))
mask_arr3 = cv2.resize(instance_seg_mask_disp, dsize=(image.width, image.height), interpolation=cv2.INTER_NEAREST_EXACT)


plt.figure(figsize=(10, 10))
for plot_index in range(2):
    if plot_index == 0:
        plot_image = image
        title = "Original"
    else:
        plot_image = mask_arr3
        title = "Segmentation"
    
    plt.subplot(1, 2, plot_index+1)
    plt.imshow(plot_image)
    plt.title(title)
    plt.axis("off")              
  
#%%
from matplotlib import cm
from matplotlib.colors import ListedColormap

def overlay_mask(image, mask, opacity=0.5):
    # Create a copy of the original image
    overlay = np.copy(image).astype(np.float32) / 255

    # Create a colormap excluding -1 (background)
    cmap = cm.get_cmap('jet', lut=len(np.unique(mask)) - 1)

    # Create a ListedColormap for visualization excluding background (-1)
    #cmap_listed = ListedColormap(cmap.colors, N=len(cmap.colors))

    # Overlay mask on the image
    for label in np.unique(mask):
        if label != -1:
            overlay[mask == label] = np.array(cmap(label)[:3]) * 255

    # Apply opacity to the mask overlay
    overlay = opacity * overlay + (1 - opacity) * image

    return overlay.astype(np.uint8)

# Assuming you have 'image' (numpy array) and 'mask' (numpy array) loaded

# Call the function to get the overlay
overlay_image = overlay_mask(image, instance_seg_mask, opacity=0.5)

# Display the overlay image
plt.imshow(overlay_image.astype(np.uint8))
plt.axis('off')
plt.show()              

#%%
import cv2
resized_mask = cv2.resize(instance_seg_mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST)
fig, ax = plt.subplots()
ax.imshow(image, cmap='jet')
ax.imshow(resized_mask, cmap='jet', alpha=0.5)
fig.show()
#%%
fig.savefig('C:/Users/wylezoln/Box/Ovarian cancer project/Adipocyte segmentation model/overlapped.png')


#%%
from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
#mask_array = np.array(instance_seg_mask.resize((image.width, image.height)))
mask_arr = np.resize(instance_seg_mask, (image.width, image.height))
#%%
img = np.array(image)

io.imshow(color.label2rgb(mask_arr,img,colors=[(255,0,0),(0,0,255)],alpha=0.01, bg_label=0, bg_color=None))
plt.show()