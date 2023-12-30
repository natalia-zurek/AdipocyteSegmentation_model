# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 19:23:09 2023

@author: WylezolN
"""

#%% load libraries
from PIL import Image
import numpy as np
from transformers import Mask2FormerImageProcessor
import torch
import os
from matplotlib import pyplot as plt
import cv2
#%% MODEL INFERENCE

model_path = "C:/Ovarian cancer project/Adipocyte dataset/Mask2Former/trained models/model12292023/mask2former_adipocyte_test_epoch_40.pt"
image_path = "C:/Ovarian cancer project/Adipocyte dataset/Mask2Former/training dataset/images"
save_dir = "C:/Ovarian cancer project/Adipocyte dataset/Mask2Former/training dataset"

#TODO: load model
#model2 = torch.load(model_path)

#model2 = Model()
model2.load_state_dict(torch.load(model_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Mask2FormerImageProcessor()

image_list = os.listdir(image_path)
image = Image.open(os.path.join('C:/Ovarian cancer project/Adipocyte dataset/Mask2Former/training dataset/images', image_list[0])).convert('RGB')

# prepare image for the model
inputs = processor(image, return_tensors="pt").to(device)
for k,v in inputs.items():
  print(k,v.shape)


with torch.no_grad():
  outputs = model2(**inputs)

image = np.array(image)

#results = processor.post_process_instance_segmentation(outputs, return_binary_maps = True)[0]
results = processor.post_process_instance_segmentation(outputs)[0]
print(results.keys())


def get_mask(segmentation, segment_id):
  mask = (segmentation.cpu().numpy() == segment_id)
  visual_mask = (mask * 255).astype(np.uint8)
  visual_mask = Image.fromarray(visual_mask)

  return visual_mask
     
 
original_image = np.array(image)
final_overlay = np.zeros_like(original_image)

# Iterate over the segments and visualize each mask on top of the original image
for segment in results['segments_info']:
    print("Visualizing mask for instance")

    # Get mask for specific instance
    mask = get_mask(results['segmentation'], segment['id'])

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

cv2.imwrite('C:/Ovarian cancer project/Adipocyte dataset/Mask2Former/predictions/overlay2.png', final_overlay)
# After accumulating all masks, blend final overlay with original image
blended = np.where(final_overlay != [0, 0, 0], final_overlay, original_image * 0.5).astype(np.uint8)

# Display the result
plt.imshow(blended)
plt.show()