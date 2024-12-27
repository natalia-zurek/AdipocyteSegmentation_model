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
from typing import Tuple



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


image_path = "C:/_research_projects/Adipocyte model project/Mask2Former/data/test/images/images student project x10"
save_path = "C:/_research_projects/Adipocyte model project/Mask2Former/predictions/model TCGA_MTC_GTEX_MTC2_augx3_x10/student project x10 fast 512x512"
model_path = "C:/_research_projects/Adipocyte model project/Mask2Former/trained models/Adipocyte_TCGA_MTC_GTEX_MTC2_augx3_albumentation_x10_v1/mask2former_instseg_epoch_80"

# save_path = "C:/_research_projects/Adipocyte model project/Mask2Former_v1/predictions/model Ov1 MTC aug 1024/student project fast inf"
# model_path = "C:/_research_projects/Adipocyte model project/Mask2Former_v1/trained models/model Ov1 MTC aug 1024/mask2former_adipocyte_test_epoch_80"



image_list = os.listdir(image_path)
os.makedirs(save_path, exist_ok=True)
#load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path, ignore_mismatched_sizes=True).to(device)      
#processor = Mask2FormerImageProcessor(reduce_labels=True, ignore_index=255, do_resize=False, do_rescale=False, do_normalize=False)
processor = Mask2FormerImageProcessor()
model.eval()
#
target_size = (512, 512)
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
    results = processor.post_process_instance_segmentation(outputs, target_sizes=[target_size])[0]              
    mask = results["segmentation"].cpu().detach().numpy()
    for segment_info in results['segments_info']:
        scores.append(segment_info['score'])
        inst_ids.append(segment_info['id'])
        classes.append(segment_info['label_id'])
    
    # plot_mask(mask, image_name)
    # Create a color palette for different instances
    colors = np.random.randint(0, 255, size=(len(inst_ids), 3), dtype=np.uint8)

    # Create a copy of the image to overlay the masks on
    result_image = np.transpose(image.copy(), (2, 0, 1))
    for i, inst_id in enumerate(inst_ids):
        mask_temp = np.array(get_mask(mask, inst_id))
        color = colors[i]

        result_image = overlay(result_image, mask_temp, color, 0.7, resize=None)
    
    ov_img = np.transpose(result_image, (1, 2, 0))
    cv2.imwrite(os.path.join(save_path, 'overlays', image_name),  cv2.cvtColor(ov_img, cv2.COLOR_RGB2BGR))   

    #save mask
    cv2.imwrite(os.path.join(save_path, 'masks', image_name), mask.astype(np.float16))        
    basename = os.path.splitext(os.path.basename(image_name))[0]
    mat_name = f"{basename}.mat"
    mat_dict = {
         "inst_map" : mask, "inst_scores" : scores, "inst_ids" : inst_ids, "inst_types" : classes}
    sio.savemat(os.path.join(save_path, 'mat', mat_name), mat_dict)
    