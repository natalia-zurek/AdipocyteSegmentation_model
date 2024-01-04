# -*- coding: utf-8 -*-
#"""
#Created on Fri Dec 29 19:23:09 2023

#@author: Natalia Zurek natalia.zurek@cshs.org
#"""

"""Usage:
    Mask2Former_model_inference.py [options] [--help]

    Options:
      -h --help                             Show this string.
      --save_path=<save_path>               Path where results will be saved
      --image_path=<img_path>               Path to images
      --model_path=<model>                  Path to model

"""
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
#from transformers import Mask2FormerConfig, Mask2FormerModel

#FUNCTIONS
def check_path_existence(path):
    if not os.path.exists(path):
        #print(f"The path '{path}' does not exists.")
        raise FileNotFoundError(f"The path '{path}' does not exist.")
        
        
def get_mask(segmentation, segment_id):
  mask = (segmentation.cpu().numpy() == segment_id)
  visual_mask = (mask * 255).astype(np.uint8)
  visual_mask = Image.fromarray(visual_mask)

  return visual_mask


# MODEL INFERENCE
if __name__ == "__main__":
    arguments = docopt(__doc__)

    model_path = arguments['--model_path']
    image_path = arguments['--image_path']
    save_path = arguments['--save_path']
    
    try:        
        check_path_existence(model_path)
        check_path_existence(image_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok = True)
    
    os.makedirs(os.path.join(save_path, 'overlays'), exist_ok = True)
    os.makedirs(os.path.join(save_path, 'masks'), exist_ok = True)
    os.makedirs(os.path.join(save_path, 'mat'), exist_ok = True)
    
    image_list = os.listdir(image_path)
    if not image_list:
        print("No images found in the specified directory.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path).to(device)      
        processor = Mask2FormerImageProcessor()
        
        for image_name in tqdm(image_list):
            image = Image.open(os.path.join(image_path, image_name)).convert('RGB')
            # prepare image for the model
            inputs = processor(image, return_tensors="pt").to(device)
            #for k,v in inputs.items():
            #  print(k,v.shape)
            with torch.no_grad():
                outputs = model(**inputs)
                
            results = processor.post_process_instance_segmentation(outputs)[0]
            #results = processor.post_process_instance_segmentation(outputs, return_binary_maps = True)[0]
            instance_seg_mask = results["segmentation"].cpu().detach().numpy()
            original_image = np.array(image)
            final_overlay = np.zeros_like(original_image)
            
            # Iterate over the segments and visualize each mask on top of the original image
            for segment in results['segments_info']:
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
            
            # After accumulating all masks, blend final overlay with original image    
            blended = np.where(final_overlay != [0, 0, 0], final_overlay, original_image * 0.5).astype(np.uint8)
            #save overlay
            #cv2.imwrite(os.path.join(save_path, 'overlays', image_name), blended)
            blended.save(os.path.join(save_path, 'overlays', image_name))
            #save mask
            cv2.imwrite(os.path.join(save_path, 'masks', image_name), final_overlay)
            #TODO: save mat file, PLACEHOLDER
            basename = os.path.splitext(os.path.basename(image_name))[0]
            mat_name = f"{basename}.mat"
            mat_dict = {
                "inst_map" : instance_seg_mask}
            sio.savemat(os.path.join(save_path, 'mat', mat_name), mat_dict)
            #print(f'{image_name}... done') #this or tqdm, not both
            