# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 14:55:52 2024

@author: WylezolN

mask2former - prediction by tiles
"""

#%% load libraries
from PIL import Image
import numpy as np
from transformers import Mask2FormerImageProcessor
import albumentations as A
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import Mask2FormerForUniversalSegmentation
from tqdm.auto import tqdm
import os
import scipy.io as sio
from datetime import datetime
import cv2
import copy
import matplotlib.pyplot as plt

#%% FUNCTIONS

def divide_image_into_tiles(image_path, tile_width, tile_height, overlap=0.2):
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get the dimensions of the image
    image_height, image_width, _ = image.shape

    # Calculate the overlap amount in pixels
    overlap_pixels_x = int(tile_width * overlap)
    overlap_pixels_y = int(tile_height * overlap)

    # Initialize lists to store tiles, positions, and overlap for each tile
    tiles = []
    positions = []
    overlaps = []

    # Divide the image into tiles with overlap
    y = 0
    pos_y = 1
    while y < image_height:
        x = 0
        pos_x = 1
        while x < image_width:
            # Calculate the end coordinates of the tile
            
            tile_end_x = min(x + tile_width, image_width)
            tile_end_y = min(y + tile_height, image_height)

            # Adjust the start coordinates to maintain tile size
            start_x = max(tile_end_x - tile_width, 0)
            start_y = max(tile_end_y - tile_height, 0)

            # Extract the tile from the image
            tile = image[start_y:tile_end_y, start_x:tile_end_x]
            tiles.append(tile)

            # Store the position of the tile
            positions.append((pos_y, pos_x))

            # Calculate the overlap for this tile
            if tile_end_x == img_width:
                overlap_x = (x + overlap_pixels_x) - (img_width - tile_width)
            else:
                overlap_x = overlap_pixels_x
            #    
            if tile_end_y == img_height:
                overlap_y = (y + overlap_pixels_y) - (img_height - tile_height)
            else:
                overlap_y = overlap_pixels_y
                            
            overlaps.append((overlap_x, overlap_y))

            # Update the horizontal position for the next tile
            x += tile_width - overlap_pixels_x
            pos_x += 1

        # Update the vertical position for the next row of tiles
        y += tile_height - overlap_pixels_y
        pos_y += 1
        
    num_rows = pos_y - 1
    num_cols = pos_x - 1   
    return tiles, positions, num_rows, num_cols, overlaps


#%%
model_path = "C:/Ovarian cancer project/Adipocyte dataset/Mask2Former/trained models/model Ov1 MTC aug 1024 intratumoral fat/mask2former_instseg_adipocyte_epoch_80"
# Example usage:
image_path = 'C:/Ovarian cancer project/Adipocyte analysis/tiles fat wsi 2048 0.3/896/896_1.tif'
tile_width = 1024  # Set the width of the tile
tile_height = 1024  # Set the height of the tile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path).to(device)      
processor = Mask2FormerImageProcessor()
#%%
img_width = 2048
img_height = 2048
tiles, positions, num_rows, num_cols, overlaps = divide_image_into_tiles(image_path, tile_width, tile_height)

inference_results = []

for tile in tiles:
    inputs = processor(tile, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(outputs)[0]
    inference_results.append(results)
#%%
def assign_new_indices(mask, ind_list,  start_ind):
    #TODO: check if the start ind is bigger than the max ind
    new_ids = np.arange(start_ind, start_ind + len(ind_list))

    # Step 3: Update the mask with the new instance IDs
    for old_id, new_id in zip(ind_list, new_ids):
        mask[mask == old_id] = new_id

    # Step 4: Update the ind_list list with the new instance IDs
    ind_list = [new_id if old_id in ind_list else old_id for old_id, new_id in zip(ind_list, new_ids)]

    return mask, ind_list

def get_overlap_mask(left_overlap, left_scores, left_classes, left_inst_ids, right_overlap, right_scores, right_classes, right_inst_ids, iou_threshold=0.7):
    # Initialize the consensus mask and lists for consensus instance information
    consensus_mask = np.full(left_overlap.shape, -1)
    consensus_inst_ids = []
    consensus_scores = []
    consensus_classes = []
    right_inst_ids_unpaired = []
    right_inst_ids_unpaired.extend(right_inst_ids) #deep copy of the inst ids so they can be deleted without messing with loop
    # Iterate over each instance in the left mask
    consensus_inst_id = 0
    for left_inst_id, left_score, left_class in zip(left_inst_ids, left_scores, left_classes):
        max_iou = 0
        max_score = 0
        corresponding_class = None

        # Iterate over each instance in the right mask
        for right_inst_id, right_score, right_class in zip(right_inst_ids, right_scores, right_classes):
            # Calculate intersection and union between the masks
            left_mask = np.zeros_like(left_overlap)
            left_mask[left_overlap == left_inst_id] = 1
            
            right_mask = np.zeros_like(right_overlap)
            right_mask[right_overlap == right_inst_id] = 1
            
            
            intersection = np.logical_and(left_mask, right_mask)
            union = np.logical_or(left_mask, right_mask)
            iou = np.sum(intersection) / np.sum(union)

            # Compare scores and update max IoU and corresponding instance ID
            if iou > max_iou:
                max_iou = iou
                max_score = right_score
                corresponding_class = right_class

        # If IoU exceeds the threshold, mark the corresponding instance in the consensus mask
        if max_iou > iou_threshold:
            #paired l_mask
            if left_score > max_score:
                consensus_mask[left_overlap == left_inst_id] = consensus_inst_id
                consensus_inst_ids.append(consensus_inst_id)
                consensus_scores.append(left_score)
                consensus_classes.append(left_class)
            #paired r_mask
            else:
                consensus_mask[right_overlap == right_inst_id] = consensus_inst_id
                consensus_inst_ids.append(consensus_inst_id)
                consensus_scores.append(max_score)
                consensus_classes.append(corresponding_class)
                
            #delete right_inst_id from the unapired_right
            delete_idx = right_inst_ids_unpaired.index(right_inst_id)
            right_inst_ids_unpaired.pop(delete_idx)
        #unpaired l_mask        
        else:
            consensus_mask[left_overlap == left_inst_id] = consensus_inst_id
            consensus_inst_ids.append(consensus_inst_id)
            consensus_scores.append(left_score)
            consensus_classes.append(left_class)
            
        consensus_inst_id = consensus_inst_id + 1    
        
    #unpaired r_mask
    for right_inst_id_unpaired in right_inst_ids_unpaired:
        unpaired_idx = right_inst_ids.index(right_inst_id_unpaired)
        right_score_unpaired = right_scores[unpaired_idx]
        right_class_unpaired = right_classes[unpaired_idx]
        
        consensus_mask[right_overlap == right_inst_id_unpaired] = consensus_inst_id
        consensus_inst_ids.append(consensus_inst_id)
        consensus_scores.append(right_score_unpaired)
        consensus_classes.append(right_class_unpaired)
        consensus_inst_id = consensus_inst_id + 1
    
    
    return consensus_mask, consensus_inst_ids, consensus_scores, consensus_classes
    
  
    
def combine_masks_horizontally(left_mask, left_scores, left_classes, left_inst_ids, right_mask, right_scores, right_classes, right_inst_ids, overlap_width, tile_width):
    # Copy the left mask to the combined mask
    #TODO: list of ids, scores, classes not always a int/float type, instead there is a list inside a list inside a list...
    #TODO: last mask is not the size 2048
    print(overlap_width)
    combined_mask = -1 * np.ones((left_mask.shape[0], overlap_width + 2 * (tile_width - overlap_width)))
    combined_scores = []
    combined_classes = []
    combined_inst_ids = []
    
    #1 assign different inst_ids to right mask to not mix with left one
    start_ind = np.max(np.unique(left_mask))+1  
    right_mask, right_inst_ids = assign_new_indices(right_mask, right_inst_ids, start_ind)
    
    #2 delete objects touching the right edge (left_mask)
    border = left_mask[:, -1]
    border_inst_ids = np.unique(border)
    border_inst_ids = border_inst_ids[border_inst_ids != -1]
    for inst_id in border_inst_ids:
        left_mask[left_mask == inst_id] = -1
        
        row_idx = left_inst_ids.index(inst_id)
        left_scores.pop(row_idx)
        left_classes.pop(row_idx)
        left_inst_ids.pop(row_idx)
    
    #3 delete objects touching the left edge (right_mask)
    border = right_mask[:, 0]
    border_inst_id = np.unique(border)
    border_inst_id = border_inst_id[border_inst_id != -1]
    for inst_id in border_inst_id:
        right_mask[right_mask == inst_id] = -1
        
        row_idx = right_inst_ids.index(inst_id)
        right_scores.pop(row_idx)
        right_classes.pop(row_idx)
        right_inst_ids.pop(row_idx)
    
    #4 move all instances that are only in left_mask
    left_part = left_mask[:, :-overlap_width]
       
    small_mask_rows, small_mask_cols = left_part.shape
    large_mask_rows, large_mask_cols = combined_mask.shape

    # Define the region to copy the smaller mask into the larger mask
    copy_region = (slice(0, small_mask_rows), slice(0, small_mask_cols))
    # Copy the smaller mask into the larger mask
    combined_mask[copy_region] = left_part
    # move scores, classes, ids
    left_part_inst_ids = np.unique(left_part)
    left_part_inst_ids = left_part_inst_ids[left_part_inst_ids != -1]
    
    for inst_id in left_part_inst_ids:
        row_idx = left_inst_ids.index(inst_id)
        combined_inst_ids.append(left_inst_ids[row_idx])
        combined_classes.append(left_classes[row_idx])
        combined_scores.append(left_scores[row_idx])
        left_scores.pop(row_idx)
        left_classes.pop(row_idx)
        left_inst_ids.pop(row_idx)
    
    
    #5 move all instances that are only in rigth_mask
    right_width = right_mask.shape[1] - overlap_width
    right_part = right_mask[:, -right_width:]

    small_mask_rows, small_mask_cols = right_part.shape
    copy_region = (
        slice(0, small_mask_rows),
        slice(large_mask_cols - small_mask_cols, large_mask_cols)
        )

    combined_mask[copy_region] = right_part
    
    right_part_inst_ids = np.unique(right_part)
    right_part_inst_ids = right_part_inst_ids[right_part_inst_ids != -1]
    
    for inst_id in right_part_inst_ids:
        row_idx = right_inst_ids.index(inst_id)
        combined_inst_ids.append(right_inst_ids[row_idx])
        combined_classes.append(right_classes[row_idx])
        combined_scores.append(right_scores[row_idx])
        right_scores.pop(row_idx)
        right_classes.pop(row_idx)
        right_inst_ids.pop(row_idx)
        
    
    #6 clean overlap area from the objects touching the threshold
    left_overlap = copy.deepcopy(left_mask[:, -overlap_width:])
    overlap_thr_left_idx = tile_width - overlap_width - 1;
    thr_left = left_mask[:, overlap_thr_left_idx]
    thr_inst_ids_left = np.unique(thr_left)
    thr_inst_ids_left = thr_inst_ids_left[thr_inst_ids_left != -1]
    for inst_id in thr_inst_ids_left:
        left_overlap[left_overlap == inst_id] = -1
    
    
    right_overlap = copy.deepcopy(right_mask[:, :overlap_width])
    overlap_thr_right_idx = overlap_width
    thr_right = right_mask[:, overlap_thr_right_idx]
    thr_inst_ids_right = np.unique(thr_right)
    thr_inst_ids_right = thr_inst_ids_right[thr_inst_ids_right != -1]
    for inst_id in thr_inst_ids_right:
        right_overlap[right_overlap == inst_id] = -1
    
    #7 merge overlap
    overlap_mask, overlap_inst_ids, overlap_scores, overlap_classes = get_overlap_mask(left_overlap, left_scores, left_classes, left_inst_ids, right_overlap, right_scores, right_classes, right_inst_ids)
    #assign new ids to overlap mask

    ind_combined_max = np.max(np.unique(combined_mask))   
    overlap_mask, overlap_inst_ids = assign_new_indices(overlap_mask, overlap_inst_ids, ind_combined_max+1)
    
    #8 move overlap to combined_mask
    overlap_start_px = combined_mask.shape[1] - tile_width
    combined_mask[:, overlap_start_px:overlap_start_px+overlap_width] = overlap_mask[:]
    combined_inst_ids.append(overlap_inst_ids)
    combined_classes.append(overlap_classes)
    combined_scores.append(overlap_scores)
       
    #8 move the instances that cross the threshold to combined mask
    #left
    target_row_offset = 0
    target_col_offset = 0
    for inst_id in thr_inst_ids_left:
        instance_to_move = np.where(left_mask == inst_id)
        # Copy the instances with the inst_id from the smaller mask to the larger mask
        combined_mask[instance_to_move[0] + target_row_offset, instance_to_move[1] + target_col_offset] = inst_id

        #row_idx = left_inst_ids.index(inst_id)
        #combined_inst_ids.append(left_inst_ids[row_idx])
        #combined_classes.append(left_classes[row_idx])
        #combined_scores.append(left_scores[row_idx])
        #left_scores.pop(row_idx)
        #left_classes.pop(row_idx)
        #left_inst_ids.pop(row_idx)
        #left_mask[left_mask == inst_id] = -1        
    #right
    target_row_offset = 0
    target_col_offset = combined_mask.shape[1] - right_mask.shape[1]
    for inst_id in thr_inst_ids_right:
        instance_to_move = np.where(right_mask == inst_id)
        combined_mask[instance_to_move[0] + target_row_offset, instance_to_move[1] + target_col_offset] = inst_id
        
        #row_idx = right_inst_ids.index(inst_id)
        #combined_inst_ids.append(right_inst_ids[row_idx])
        #combined_classes.append(right_classes[row_idx])
        #combined_scores.append(right_scores[row_idx])
        #right_scores.pop(row_idx)
        #right_classes.pop(row_idx)
        #right_inst_ids.pop(row_idx)
        #right_mask[right_mask == inst_id] = -1
    
    return combined_mask, combined_scores, combined_classes, combined_inst_ids


def combine_masks_vertically(mask_top, mask_bottom):
    
    pass
#%% main code
img_width = 2048
img_height = 2048
combined_mask = np.zeros([img_width, img_height])
mask_rows = []
combined_ids = [[] for _ in range(num_rows)]
combined_scores = []
combined_classes = []
combined_info_idx = 0;
for i in range(0, num_rows*num_cols):
    if (i+1) % num_cols == 0:
        mask_rows.append(left_mask)
        #how to save a list inside a list?
        combined_ids[combined_info_idx] = (left_inst_ids)
        combined_classes.append(left_classes)
        combined_scores.append(left_scores)
        combined_info_idx = combined_info_idx + 1
        continue
    if i % num_cols == 0:
        left_mask = inference_results[i]["segmentation"].cpu().detach().numpy()
        left_mask = cv2.resize(left_mask, dsize=(tile_width, tile_height), interpolation=cv2.INTER_NEAREST_EXACT)
        left_scores = []
        left_classes = []
        left_inst_ids = []
        for segment_info in inference_results[i]['segments_info']:
            left_scores.append(segment_info['score'])
            left_inst_ids.append(segment_info['id'])
            left_classes.append(segment_info['label_id'])
            
        #left_scores = np.array(left_scores).reshape(-1, 1)
        #left_inst_ids = np.array(left_inst_ids).reshape(-1,1)
        #left_classes = np.array(left_classes).reshape(-1,1)    
        

    right_mask = inference_results[i+1]["segmentation"].cpu().detach().numpy()
    right_mask = cv2.resize(right_mask, dsize=(tile_width, tile_height), interpolation=cv2.INTER_NEAREST_EXACT)
    right_scores = []
    right_classes = []
    right_inst_ids = []
    for segment_info in inference_results[i+1]['segments_info']:
        right_scores.append(segment_info['score'])
        right_inst_ids.append(segment_info['id'])
        right_classes.append(segment_info['label_id'])
        
    #right_scores = np.array(right_scores).reshape(-1, 1)
    #right_inst_ids = np.array(right_inst_ids).reshape(-1,1)
    #right_classes = np.array(right_classes).reshape(-1,1) 
    
        
    left_mask, left_scores, left_classes, left_inst_ids = combine_masks_horizontally(left_mask, left_scores, left_classes, left_inst_ids, right_mask, right_scores, right_classes, right_inst_ids, overlaps[i][1], tile_width)
    
#%%    
for i in range(0, num_cols-1):
    top_mask = mask_rows[i]
    bottom_mask = mask_rows[i+1]
    top_mask, top_scores, top_classes, top_inst_ids = combine_masks_vertically(tom_mask, top_scores, top_classes, top_inst_ids, mask_bottom)
    
#%%
for idx, (tile, position) in enumerate(zip(tiles, positions), 1):
        print(f"Tile {idx} at position: {position}")
#%%
import matplotlib.pyplot as plt


#%%
for i, tile in enumerate(tiles):
    plt.imshow(tile)
    plt.axis('off')
    plt.show()

 #%%
 
def locate_instances_left_half(mask, midpoint_x):
    # Get the midpoint of the mask along the horizontal axis
    midpoint_x = 2

    # List to store instance IDs on the left half
    instances_left_half = []

    # Iterate through each unique instance ID in the mask
    instance_ids = np.unique(mask)
    for instance_id in instance_ids:
        if instance_id == -1:  # Skip background
            continue

        # Get the coordinates of all pixels with the current instance ID
        instance_pixels = np.where(mask == instance_id)

        # Calculate the bounding box of the instance
        min_x = np.min(instance_pixels[1])
        max_x = np.max(instance_pixels[1])

        # Check if the bounding box lies completely within the left half of the mask
        if max_x < midpoint_x:
            instances_left_half.append(instance_id)

    return instances_left_half

# Example usage:
mask = np.array([[0, 0, 1, 1, 1],
                 [2, 2, 2, 1, 1],
                 [0, 0, 3, 3, 3]])

instances_on_left_half = locate_instances_left_half(mask)
print("Instance IDs on the left half:", instances_on_left_half)