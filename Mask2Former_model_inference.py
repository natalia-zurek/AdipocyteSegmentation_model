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
from transformers import Mask2FormerForUniversalSegmentation
from transformers import Mask2FormerConfig, Mask2FormerModel

#%% MODEL INFERENCE

model_path = "C:/Ovarian cancer project/Adipocyte dataset/Mask2Former/trained models 2024-01-02/mask2former_adipocyte_test_epoch_10"
#image_path = "C:/Ovarian cancer project/Adipocyte dataset/Mask2Former/training dataset/images new"
image_path = 'C:/Ovarian cancer project/Adipocyte dataset/Mask2Former/training dataset/to change'
save_dir = "C:/Ovarian cancer project/Adipocyte dataset/Mask2Former/training dataset"




# Accessing the model configuration
#configuration = model2.config

#TODO: load model 
#%%NOT SURE IF THIS WORKS (it doesn't)
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-instance",
                                                          ignore_mismatched_sizes=False)
model.load_state_dict(torch.load(model_path))
#%% doesnt work
# Initializing a Mask2Former facebook/mask2former-swin-small-coco-instance configuration
configuration = Mask2FormerConfig()
# Initializing a model (with random weights) from the facebook/mask2former-swin-small-coco-instance style configuration
model = Mask2FormerModel(configuration)
model.load_state_dict(torch.load(model_path))
#%% size mismatch
configuration = Mask2FormerConfig()
# Initializing a model (with random weights) from the facebook/mask2former-swin-small-coco-instance style configuration
model = Mask2FormerForUniversalSegmentation(configuration)
model.load_state_dict(torch.load(model_path))
#model2 = Mask2FormerForUniversalSegmentation()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path, ignore_mismatched_sizes=False)
#%% different way to load the model

model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path).to(device)
#this processor?
processor = Mask2FormerForUniversalSegmentation.from_pretrained(model_path)


#%%
#TODO: is this processor ok? because I'm saving differently configured processor 
processor = Mask2FormerImageProcessor()

image_list = os.listdir(image_path)
image = Image.open(os.path.join(image_path, image_list[5])).convert('RGB')

# prepare image for the model
inputs = processor(image, return_tensors="pt").to(device)
for k,v in inputs.items():
  print(k,v.shape)


with torch.no_grad():
  outputs = model(**inputs)

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

#cv2.imwrite('C:/Ovarian cancer project/Adipocyte dataset/Mask2Former/predictions/overlay2.png', final_overlay)
# After accumulating all masks, blend final overlay with original image
blended = np.where(final_overlay != [0, 0, 0], final_overlay, original_image * 0.5).astype(np.uint8)

# Display the result
plt.imshow(blended)
plt.show()
#%%
cv2.imwrite(f'C:/Ovarian cancer project/Adipocyte dataset/Mask2Former/predictions/{image_list[5]}', blended)