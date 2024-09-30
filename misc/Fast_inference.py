# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 16:38:39 2024

@author: WylezolN
Quick inference
"""

import os
import cv2
import numpy as np

import torch
import scipy.io as sio
from transformers import Mask2FormerImageProcessor
from transformers import Mask2FormerForUniversalSegmentation
import matplotlib.pyplot as plt
from PIL import Image

def get_mask(segmentation, segment_id):
  mask = (segmentation == segment_id)
  visual_mask = (mask * 255).astype(np.uint8)
  visual_mask = Image.fromarray(visual_mask)

  return visual_mask

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

save_path = "C:/_research_projects/Adipocyte model project/Mask2Former/predictions/Adipocyte_TCGA_MTC_GTEX_augx1_x20_v1/omental mets part 1 0.7"
model_path = "C:/_research_projects/Adipocyte model project/Mask2Former/trained models/Adipocyte_TCGA_MTC_GTEX_augx1_x20_v1/mask2former_instseg_epoch_80"
image_path = "C:/_research_projects/Adipocyte model project/Mask2Former/data/test/images/images omental part 1"
image_list = os.listdir(image_path)
os.makedirs(save_path)
#load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path, ignore_mismatched_sizes=True).to(device)      
processor = Mask2FormerImageProcessor()
model.eval()
#%%
target_size = (1024, 1024)
os.makedirs(os.path.join(save_path, 'overlays'), exist_ok = True)
os.makedirs(os.path.join(save_path, 'masks'), exist_ok = True)
os.makedirs(os.path.join(save_path, 'mat'), exist_ok = True)
for image_name in image_list:

    if not image_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', 'tif', 'tiff')):
        continue
    
    #image = Image.open(os.path.join(image_path, image_name)).convert('RGB')
    image = cv2.imread(os.path.join(image_path, image_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    img_height, img_width = image.shape[:2]               
    original_image = np.array(image)
    final_overlay = np.zeros_like(original_image)

    inputs = processor(image, return_tensors="pt").to(device)
    scores = []
    classes = []
    inst_ids = []
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(outputs, target_sizes=[target_size], threshold = 0.7)[0]              
    mask = results["segmentation"].cpu().detach().numpy()
    for segment_info in results['segments_info']:
        scores.append(segment_info['score'])
        inst_ids.append(segment_info['id'])
        classes.append(segment_info['label_id'])
    
    plot_mask(mask, image_name)
    
    # for inst_id in inst_ids:
    #     # Get mask for specific instance
    #     mask_temp = get_mask(mask, inst_id)

    #     # Resize mask if necessary
    #     mask_array = np.array(mask_temp)
    #     if mask_array.shape != original_image.shape[:2]:
    #         mask_array = np.array(mask.resize((original_image.shape[1], original_image.shape[0])))

    #     # Find where the mask is
    #     mask_location = mask_array == 255

    #     # Set the mask area to a specific color
    #     red_channel = final_overlay[:,:,2]
    #     red_channel[mask_location] = 255  # you may want to ensure that this does not overwrite previous masks
    #     final_overlay[:,:,0] = red_channel
        
    # # After accumulating all masks, blend final overlay with original image    
    # blended = np.where(final_overlay != [0, 0, 0], final_overlay, original_image * 0.5).astype(np.uint8)
    #save overlay
    #cv2.imwrite(os.path.join(save_path, 'overlays', image_name),  cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    #blended.save(os.path.join(save_path, 'overlays', image_name))
    #save mask
    #cv2.imwrite(os.path.join(save_path, 'masks', image_name), mask.astype(np.float16))        
    basename = os.path.splitext(os.path.basename(image_name))[0]
    mat_name = f"{basename}.mat"
    mat_dict = {
        "inst_map" : mask, "inst_scores" : scores, "inst_ids" : inst_ids, "inst_types" : classes}
    sio.savemat(os.path.join(save_path, 'mat', mat_name), mat_dict)