# -*- coding: utf-8 -*-
#"""
#Created on Fri Dec 29 19:23:09 2023

#@author: Natalia Zurek natalia.zurek@cshs.org
#"""

"""Usage:
    Mask2Former_model_inference.py [options] [--help]

    Options:
      -h --help                             Show this string.
      --save_path=<path>                    Path where results will be saved
      --image_path=<path>                   Path to images
      --model_path=<path>                   Path to model
      --is_nested                           Boolean, set True if the image path is nested
      
      --tile_height=<int>                   Tile height. [default: 1024]
      --tile_width=<int>                    Tile weight. [default: 1024]
      --overlap_fraction=<float>            Overlap between tiles, must be between (0,1) range [default: 0.3]
"""
#TODO: move functions to separate files
#TODO: make this main more ascetic
#TODO: https://github.com/huggingface/transformers/issues/21313
#TODO: processor = MaskFormerImageProcessor.from_pretrained(
    #"adirik/maskformer-swin-base-sceneparse-instance"
#)

#LIBRARIES
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

import copy
# import warnings
# #from transformers import Mask2FormerConfig, Mask2FormerModel

#FUNCTIONS

# Function to check if path exists
def check_path_existence(path):
    if not os.path.exists(path):
        #print(f"The path '{path}' does not exists.")
        raise FileNotFoundError(f"The path '{path}' does not exist.")
        
# Function to get the instance mask
def get_mask(segmentation, segment_id):
  mask = (segmentation == segment_id)
  visual_mask = (mask * 255).astype(np.uint8)
  visual_mask = Image.fromarray(visual_mask)

  return visual_mask

# Function to check if instance IDs are present in the mask
def filter_instances(inst_ids, scores, classes, mask):
    filtered_inst_ids = []
    filtered_scores = []
    filtered_classes = []

    # Iterate through each instance ID in the list
    for idx, inst_id in enumerate(inst_ids):
        # Check if the instance ID is present in the mask
        if inst_id in mask:
            # If present, append the ID, score, and class to the filtered lists
            filtered_inst_ids.append(inst_id)
            filtered_scores.append(scores[idx])
            filtered_classes.append(classes[idx])

    return filtered_inst_ids, filtered_scores, filtered_classes

# Function to divide image into tiles
def divide_image_into_tiles(image, tile_width, tile_height, overlap=0.45):
    # Read the image
    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
    is_end = 0
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
            if tile_end_x == image_width:
                overlap_x = tile_end_x_prev - start_x
            else:
                overlap_x = copy.deepcopy(overlap_pixels_x)
            #    
            if tile_end_y == image_height:
                if is_end == 0:
                    overlap_y = tile_end_y_prev - start_y
                    is_end = 1
            else:
                overlap_y = copy.deepcopy(overlap_pixels_y)
                            
            overlaps.append((overlap_x, overlap_y))

            # Update the horizontal position for the next tile
            x += tile_end_x - overlap_pixels_x 
            pos_x += 1
            
            tile_end_x_prev = copy.deepcopy(tile_end_x)
            tile_end_y_prev = copy.deepcopy(tile_end_y)
        # Update the vertical position for the next row of tiles
        y += tile_end_y - overlap_pixels_y 
        pos_y += 1
        
    num_rows = pos_y - 1
    num_cols = pos_x - 1   
    return tiles, positions, num_rows, num_cols, overlaps

# def divide_image_into_tiles(image, tile_width, tile_height, overlap=0.45):#(image_path, tile_width, tile_height, overlap=0.45):
#     # Read the image
#     #image = cv2.imread(image_path)
#     #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Get the dimensions of the image
#     image_height, image_width, _ = image.shape

#     # Calculate the overlap amount in pixels
#     overlap_pixels_x = int(tile_width * overlap)
#     overlap_pixels_y = int(tile_height * overlap)

#     # Initialize lists to store tiles, positions, and overlap for each tile
#     tiles = []
#     positions = []
#     overlaps = []

#     # Divide the image into tiles with overlap
#     y = 0
#     pos_y = 1
#     while y < image_height:
#         x = 0
#         pos_x = 1
#         while x < image_width:
#             # Calculate the end coordinates of the tile
            
#             tile_end_x = min(x + tile_width, image_width)
#             tile_end_y = min(y + tile_height, image_height)

#             # Adjust the start coordinates to maintain tile size
#             start_x = max(tile_end_x - tile_width, 0)
#             start_y = max(tile_end_y - tile_height, 0)

#             # Extract the tile from the image
#             tile = image[start_y:tile_end_y, start_x:tile_end_x]
#             tiles.append(tile)

#             # Store the position of the tile
#             positions.append((pos_y, pos_x))

#             # Calculate the overlap for this tile
#             if tile_end_x == image_width:
#                 overlap_x = (x + overlap_pixels_x) - (image_width - tile_width)
#             else:
#                 overlap_x = overlap_pixels_x
#             #    
#             if tile_end_y == image_height:
#                 overlap_y = (y + overlap_pixels_y) - (image_height - tile_height)
#             else:
#                 overlap_y = overlap_pixels_y
                            
#             overlaps.append((overlap_x, overlap_y))

#             # Update the horizontal position for the next tile
#             x += tile_width - overlap_pixels_x
#             pos_x += 1

#         # Update the vertical position for the next row of tiles
#         y += tile_height - overlap_pixels_y
#         pos_y += 1
        
#     num_rows = pos_y - 1
#     num_cols = pos_x - 1   
#     return tiles, positions, num_rows, num_cols, overlaps

# Function to assign new ids to the mask
def assign_new_ids(mask, ind_list, start_ind):
    # Check if the start ind is bigger than the max ind
    if not ind_list:
        mask = mask
        int_ind_list = ind_list
    else:
            
        max_ind = np.max(ind_list)
        
        if start_ind <= max_ind:
            #warnings.warn("start_ind must be greater than the maximum index in ind_list", UserWarning)
            start_ind = max_ind+1
            
        new_ids = np.arange(start_ind, start_ind + len(ind_list))
    
        # Update the mask with the new instance IDs
        for old_id, new_id in zip(ind_list, new_ids):
            mask[mask == old_id] = new_id
    
        # Update the ind_list list with the new instance IDs
        ind_list = [new_id if old_id in ind_list else old_id for old_id, new_id in zip(ind_list, new_ids)]
        int_ind_list = [int(x) for x in ind_list] #change the float to int list objects
    return mask, int_ind_list

# Function to obtain mask from overlaped (common) region from tiles
def get_overlap_mask(left_overlap, left_scores, left_classes, left_inst_ids, right_overlap, right_scores, right_classes, right_inst_ids, iou_threshold=0.7):
    # Initialize the consensus mask and lists for consensus instance information
    consensus_mask = np.full(left_overlap.shape, -1)
    consensus_inst_ids = []
    consensus_scores = []
    consensus_classes = []
    
    # Iterate over each instance in the left mask
    consensus_inst_id = 0
    for left_inst_id, left_score, left_class in zip(left_inst_ids, left_scores, left_classes):
        max_iou = 0
        max_score = 0
        corresponding_class = None
        left_mask = np.zeros_like(left_overlap)
        left_mask[left_overlap == left_inst_id] = 1
        
        # Iterate over each instance in the right mask
        for right_inst_id, right_score, right_class in zip(right_inst_ids, right_scores, right_classes):
            # Calculate intersection and union between the masks
            
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
                max_inst_id = right_inst_id
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
                consensus_mask[right_overlap == max_inst_id] = consensus_inst_id
                consensus_inst_ids.append(consensus_inst_id)
                consensus_scores.append(max_score)
                consensus_classes.append(corresponding_class)

            #delete right_inst_id from the unapired_right
            delete_idx = right_inst_ids.index(max_inst_id)
            right_inst_ids.pop(delete_idx)
            right_classes.pop(delete_idx)
            right_scores.pop(delete_idx)
            
        #unpaired l_mask        
        else:
            consensus_mask[left_overlap == left_inst_id] = consensus_inst_id
            consensus_inst_ids.append(consensus_inst_id)
            consensus_scores.append(left_score)
            consensus_classes.append(left_class)

        consensus_inst_id = consensus_inst_id + 1    
        
    #unpaired r_mask
    for right_inst_id_unpaired in right_inst_ids:
        unpaired_idx = right_inst_ids.index(right_inst_id_unpaired)
        right_score_unpaired = right_scores[unpaired_idx]
        right_class_unpaired = right_classes[unpaired_idx]
        
        consensus_mask[right_overlap == right_inst_id_unpaired] = consensus_inst_id
        consensus_inst_ids.append(consensus_inst_id)
        consensus_scores.append(right_score_unpaired)
        consensus_classes.append(right_class_unpaired)
        consensus_inst_id = consensus_inst_id + 1

    
    return consensus_mask, consensus_inst_ids, consensus_scores, consensus_classes
       
# Function to combine masks from rows    
def combine_masks_horizontally(left_mask, left_scores, left_classes, left_inst_ids, right_mask, right_scores, right_classes, right_inst_ids, overlap_width, tile_width):
    # Copy the left mask to the combined mask
    combined_mask = -1 * np.ones((left_mask.shape[0], left_mask.shape[1] + right_mask.shape[1] - overlap_width))
    combined_scores = []
    combined_classes = []
    combined_inst_ids = []
    
    #1 assign different inst_ids to right mask to not mix with left one
    start_ind = np.max(np.unique(left_mask))+1  
    right_mask, right_inst_ids = assign_new_ids(right_mask, right_inst_ids, start_ind)
    
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
    overlap_thr_left_idx = left_mask.shape[1] - overlap_width - 1;
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
    overlap_mask, overlap_inst_ids = assign_new_ids(overlap_mask, overlap_inst_ids, ind_combined_max+1)
    
    #8 move overlap to combined_mask
    overlap_start_px = left_mask.shape[1] - overlap_width
    combined_mask[:, overlap_start_px:overlap_start_px+overlap_width] = overlap_mask[:]
    combined_inst_ids = combined_inst_ids + overlap_inst_ids
    combined_classes = combined_classes + overlap_classes
    combined_scores = combined_scores + overlap_scores
       
    #9 move the instances that cross the threshold to combined mask
    #left
    target_row_offset = 0
    target_col_offset = 0
    for inst_id in thr_inst_ids_left:
        instance_to_move = np.where(left_mask == inst_id)
        # Copy the instances with the inst_id from the smaller mask to the larger mask
        combined_mask[instance_to_move[0] + target_row_offset, instance_to_move[1] + target_col_offset] = inst_id
      
    #right
    target_row_offset = 0
    target_col_offset = combined_mask.shape[1] - right_mask.shape[1]
    for inst_id in thr_inst_ids_right:
        instance_to_move = np.where(right_mask == inst_id)
        combined_mask[instance_to_move[0] + target_row_offset, instance_to_move[1] + target_col_offset] = inst_id
        
    
    return combined_mask, combined_scores, combined_classes, combined_inst_ids

# Function to combine masks by columns
def combine_masks_vertically(top_mask, top_scores, top_classes, top_inst_ids, bottom_mask, bottom_scores, bottom_classes, bottom_inst_ids, overlap_height):
    combined_mask = np.full((top_mask.shape[0] + bottom_mask.shape[0] - overlap_height, top_mask.shape[1]), -1)
    combined_scores = []
    combined_classes = []
    combined_inst_ids = []
    
    #1 rearrange the ids in bottom mask
    start_ind = np.max(np.unique(top_mask))+1  
    bottom_mask, bottom_inst_ids = assign_new_ids(bottom_mask, bottom_inst_ids, start_ind)
        
    #2 delete objects touching the bottom edge (top_mask)
    border = top_mask[-1, :]
    border_inst_ids = np.unique(border)
    border_inst_ids = border_inst_ids[border_inst_ids != -1]
    for inst_id in border_inst_ids:
        top_mask[top_mask == inst_id] = -1
        
        row_idx = top_inst_ids.index(inst_id)
        top_scores.pop(row_idx)
        top_classes.pop(row_idx)
        top_inst_ids.pop(row_idx)
        
    #3 delete objects touching the top edge (bottom_mask)
    border = bottom_mask[0, :]
    border_inst_ids = np.unique(border)
    border_inst_ids = border_inst_ids[border_inst_ids != -1]
    for inst_id in border_inst_ids:
        bottom_mask[bottom_mask == inst_id] = -1
        
        row_idx = bottom_inst_ids.index(inst_id)
        bottom_scores.pop(row_idx)
        bottom_classes.pop(row_idx)
        bottom_inst_ids.pop(row_idx)
   
    #4 move all instances that are only in top_mask
    top_part = top_mask[:-overlap_height, : ]
       
    small_mask_rows, small_mask_cols = top_part.shape
    large_mask_rows, large_mask_cols = combined_mask.shape

    # Define the region to copy the smaller mask into the larger mask
    copy_region = (slice(0, small_mask_rows), slice(0, small_mask_cols))
    # Copy the smaller mask into the larger mask
    combined_mask[copy_region] = top_part
    # move scores, classes, ids
    top_part_inst_ids = np.unique(top_part)
    top_part_inst_ids = top_part_inst_ids[top_part_inst_ids != -1]
    
    for inst_id in top_part_inst_ids:
        row_idx = top_inst_ids.index(inst_id)
        combined_inst_ids.append(top_inst_ids[row_idx])
        combined_classes.append(top_classes[row_idx])
        combined_scores.append(top_scores[row_idx])
        top_scores.pop(row_idx)
        top_classes.pop(row_idx)
        top_inst_ids.pop(row_idx)
        
    
    #5 move all instances that are only in bottom_mask
    bottom_height = bottom_mask.shape[0] - overlap_height
    bottom_part = bottom_mask[-bottom_height:, :]

    small_mask_rows, small_mask_cols = bottom_part.shape
    copy_region = (       
        slice(large_mask_rows - small_mask_rows, large_mask_rows),
        slice(0, small_mask_cols)
        )

    combined_mask[copy_region] = bottom_part
    
    bottom_part_inst_ids = np.unique(bottom_part)
    bottom_part_inst_ids = bottom_part_inst_ids[bottom_part_inst_ids != -1]
    
    for inst_id in bottom_part_inst_ids:
        row_idx = bottom_inst_ids.index(inst_id)
        combined_inst_ids.append(bottom_inst_ids[row_idx])
        combined_classes.append(bottom_classes[row_idx])
        combined_scores.append(bottom_scores[row_idx])
        bottom_scores.pop(row_idx)
        bottom_classes.pop(row_idx)
        bottom_inst_ids.pop(row_idx)
        
    #6 clean overlap area from the objects touching the threshold
    top_overlap = copy.deepcopy(top_mask[-overlap_height:, :])
    overlap_thr_top_idx = top_mask.shape[0] - overlap_height - 1;
    thr_top = top_mask[overlap_thr_top_idx, :]
    thr_inst_ids_top = np.unique(thr_top)
    thr_inst_ids_top = thr_inst_ids_top[thr_inst_ids_top != -1]
    for inst_id in thr_inst_ids_top:
        top_overlap[top_overlap == inst_id] = -1
    
    
    bottom_overlap = copy.deepcopy(bottom_mask[:overlap_height, :])
    overlap_thr_bottom_idx = overlap_height
    thr_bottom = bottom_mask[overlap_thr_bottom_idx, :]
    thr_inst_ids_bottom = np.unique(thr_bottom)
    thr_inst_ids_bottom = thr_inst_ids_bottom[thr_inst_ids_bottom != -1]
    for inst_id in thr_inst_ids_bottom:
        bottom_overlap[bottom_overlap == inst_id] = -1
        
    #7 merge overlap
    overlap_mask, overlap_inst_ids, overlap_scores, overlap_classes = get_overlap_mask(top_overlap, top_scores, top_classes, top_inst_ids, bottom_overlap, bottom_scores, bottom_classes, bottom_inst_ids)
    #assign new ids to overlap mask

    ind_combined_max = np.max(np.unique(combined_mask))   
    overlap_mask, overlap_inst_ids = assign_new_ids(overlap_mask, overlap_inst_ids, ind_combined_max+1)
    
    #8 move overlap to combined_mask
    overlap_start_px = top_mask.shape[0] - overlap_height 
    combined_mask[overlap_start_px:overlap_start_px+overlap_height, :] = overlap_mask[:]
    combined_inst_ids = combined_inst_ids + overlap_inst_ids
    combined_classes = combined_classes + overlap_classes
    combined_scores = combined_scores + overlap_scores
    #9 move the instances that cross the threshold to combined mask
    #top
    target_row_offset = 0
    target_col_offset = 0
    for inst_id in thr_inst_ids_top:
        instance_to_move = np.where(top_mask == inst_id)
        # Copy the instances with the inst_id from the smaller mask to the larger mask
        combined_mask[instance_to_move[0] + target_row_offset, instance_to_move[1] + target_col_offset] = inst_id
      
    #bottom
    target_row_offset = combined_mask.shape[0] - bottom_mask.shape[0]
    target_col_offset = 0
    for inst_id in thr_inst_ids_bottom:
        instance_to_move = np.where(bottom_mask == inst_id)
        combined_mask[instance_to_move[0] + target_row_offset, instance_to_move[1] + target_col_offset] = inst_id
        
    
    return combined_mask, combined_scores, combined_classes, combined_inst_ids

def run_inference(image_path, save_path, tile_width, tile_height):
    os.makedirs(os.path.join(save_path, 'overlays'), exist_ok = True)
    os.makedirs(os.path.join(save_path, 'masks'), exist_ok = True)
    os.makedirs(os.path.join(save_path, 'mat'), exist_ok = True)
    
    image_list = os.listdir(image_path)
    if not image_list:
        print(f"No images found in the directory: {image_path}.")
    else:      
        for image_name in tqdm(image_list):
            
            if not image_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', 'tif', 'tiff')):
                continue
            
            #image = Image.open(os.path.join(image_path, image_name)).convert('RGB')
            image = cv2.imread(os.path.join(image_path, image_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            img_height, img_width = image.shape[:2]               
            original_image = np.array(image)
            final_overlay = np.zeros_like(original_image)
                            
            tiles, positions, num_rows, num_cols, overlaps = divide_image_into_tiles(image, tile_width, tile_height)

            inference_results = []

            for tile in tiles:
                inputs = processor(tile, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                results = processor.post_process_instance_segmentation(outputs)[0]
                inference_results.append(results)               
            
            mask_rows = []
            combined_rows_inst_ids = [[] for _ in range(num_rows)]
            combined_rows_scores = []
            combined_rows_classes = []
            combined_info_idx = 0
            for i in range(0, num_rows*num_cols):
                if (i+1) % num_cols == 0:
                    mask_rows.append(left_mask)
                    #how to save a list inside a list?
                    combined_rows_inst_ids[combined_info_idx] = (left_inst_ids)
                    combined_rows_classes.append(left_classes)
                    combined_rows_scores.append(left_scores)
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

                    left_inst_ids, left_scores, left_classes = filter_instances(left_inst_ids, left_scores, left_classes, left_mask)
                    
                right_mask = inference_results[i+1]["segmentation"].cpu().detach().numpy()
                right_mask = cv2.resize(right_mask, dsize=(tile_width, tile_height), interpolation=cv2.INTER_NEAREST_EXACT)
                right_scores = []
                right_classes = []
                right_inst_ids = []
                for segment_info in inference_results[i+1]['segments_info']:
                    right_scores.append(segment_info['score'])
                    right_inst_ids.append(segment_info['id'])
                    right_classes.append(segment_info['label_id'])
                
                right_inst_ids, right_scores, right_classes = filter_instances(right_inst_ids, right_scores, right_classes, right_mask)
                   
                left_mask, left_scores, left_classes, left_inst_ids = combine_masks_horizontally(left_mask, left_scores, left_classes, left_inst_ids, right_mask, right_scores, right_classes, right_inst_ids, overlaps[i+1][0], tile_width)
                 
            
            overlaps_rows = copy.deepcopy(overlaps[0:len(overlaps)-1:num_cols]) #num_rows or num_cols????
            for i in range(0, num_cols-1):
                if i == 0:
                    top_mask = copy.deepcopy(mask_rows[i])
                    top_scores = copy.deepcopy(combined_rows_scores[i])
                    top_classes = copy.deepcopy(combined_rows_classes[i])
                    top_inst_ids = copy.deepcopy(combined_rows_inst_ids[i])
                
                bottom_mask = copy.deepcopy(mask_rows[i+1])
                bottom_scores = copy.deepcopy(combined_rows_scores[i+1])
                bottom_classes = copy.deepcopy(combined_rows_classes[i+1])
                bottom_inst_ids = copy.deepcopy(combined_rows_inst_ids[i+1])
                
                top_mask, top_scores, top_classes, top_inst_ids = combine_masks_vertically(top_mask, top_scores, top_classes, top_inst_ids, bottom_mask, bottom_scores, bottom_classes, bottom_inst_ids, overlaps_rows[i+1][1])
            
            # Iterate over the segments and visualize each mask on top of the original image
            for inst_id in top_inst_ids:
                # Get mask for specific instance
                mask = get_mask(top_mask, inst_id)

                # Resize mask if necessary
                mask_array = np.array(mask)
                if mask_array.shape != original_image.shape[:2]:
                    mask_array = np.array(mask.resize((original_image.shape[1], original_image.shape[0])))

                # Find where the mask is
                mask_location = mask_array == 255

                # Set the mask area to a specific color
                red_channel = final_overlay[:,:,2]
                red_channel[mask_location] = 255  # you may want to ensure that this does not overwrite previous masks
                final_overlay[:,:,0] = red_channel
                
            # After accumulating all masks, blend final overlay with original image    
            blended = np.where(final_overlay != [0, 0, 0], final_overlay, original_image * 0.5).astype(np.uint8)
            #save overlay
            cv2.imwrite(os.path.join(save_path, 'overlays', image_name),  cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
            #blended.save(os.path.join(save_path, 'overlays', image_name))
            #save mask
            cv2.imwrite(os.path.join(save_path, 'masks', image_name), top_mask.astype(np.float16))        
            basename = os.path.splitext(os.path.basename(image_name))[0]
            mat_name = f"{basename}.mat"
            mat_dict = {
                "inst_map" : top_mask, "inst_scores" : top_scores, "inst_ids" : top_inst_ids, "inst_types" : top_classes}
            sio.savemat(os.path.join(save_path, 'mat', mat_name), mat_dict)
            #print(f'{image_name}... done') #this or tqdm, not both


# MODEL INFERENCE
try:
    if __name__ == "__main__":
        arguments = docopt(__doc__)

        model_path = arguments['--model_path']
        image_path = arguments['--image_path']
        save_path = arguments['--save_path']
        is_nested = arguments['--is_nested']
        #tile_width = int(arguments['--tile_width'])
        #tile_height = int(arguments['--tile_height'])
        # overlap_fraction = float(arguments['--overlap_fraction'])
        tile_width = 1024
        tile_height = 1024
        
        # #TODO:
        # if not 0 < overlap_fraction < 1:
        #     pass#raise OverlapFractionError("Overlap fraction must be within the range (0, 1).")
        
        try:        
            check_path_existence(model_path)
            check_path_existence(image_path)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok = True)  
        
        print("Loading model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path).to(device)      
        processor = Mask2FormerImageProcessor()
            
        if is_nested == False:
            
            run_inference(image_path, save_path, tile_width, tile_height)

        else:
            
            print("Finding nested directories...")
            directories = os.listdir(image_path)
            if len(directories) == 0:
                print(f'No directories in {image_path}')
            
            for directory in directories:
                save_path_directory = os.path.join(save_path, directory)
                image_path_directory = os.path.join(image_path, directory)
                
                run_inference(image_path_directory, save_path_directory, tile_width, tile_height)
            

    while True:
        pass  # Placeholder for your code
except KeyboardInterrupt:
    print("Ctrl+C pressed. Exiting...")
    # Any cleanup or termination actions can be added here
    
    
    
# if __name__ == "__main__":
#     arguments = docopt(__doc__)

#     model_path = arguments['--model_path']
#     image_path = arguments['--image_path']
#     save_path = arguments['--save_path']
#     is_nested = arguments['--is_nested']
#     #tile_width = int(arguments['--tile_width'])
#     #tile_height = int(arguments['--tile_height'])
#     # overlap_fraction = float(arguments['--overlap_fraction'])
#     tile_width = 1024
#     tile_height = 1024
    
#     # #TODO:
#     # if not 0 < overlap_fraction < 1:
#     #     pass#raise OverlapFractionError("Overlap fraction must be within the range (0, 1).")
    
#     try:        
#         check_path_existence(model_path)
#         check_path_existence(image_path)
#     except FileNotFoundError as e:
#         print(f"Error: {e}")
        
#     if not os.path.exists(save_path):
#         os.makedirs(save_path, exist_ok = True)  
    
#     print("Loading model...")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path).to(device)      
#     processor = Mask2FormerImageProcessor()
        
#     if is_nested == False:
        
#         run_inference(image_path, save_path, tile_width, tile_height)

#     else:
        
#         print("Finding nested directories...")
#         directories = os.listdir(image_path)
#         if len(directories) == 0:
#             print(f'No directories in {image_path}')
        
#         for directory in directories:
#             save_path_directory = os.path.join(save_path, directory)
#             image_path_directory = os.path.join(image_path, directory)
            
#             run_inference(image_path_directory, save_path_directory, tile_width, tile_height)
        

            

                                                                     
                