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
#TODO: https://github.com/huggingface/transformers/issues/21313 - DONE
#TODO: processor = MaskFormerImageProcessor.from_pretrained(
    #"adirik/maskformer-swin-base-sceneparse-instance"
#)

#LIBRARIES
from transformers import Mask2FormerImageProcessor
import torch
import os
from transformers import Mask2FormerForUniversalSegmentation
from docopt import docopt
from models.inference import run_inference


#FUNCTIONS

# Function to check if path exists
def check_path_existence(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path '{path}' does not exist.")
        
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
        model.eval()
        if is_nested == False:
            
            run_inference(device, model, processor, image_path, save_path, tile_width, tile_height)

        else:
            
            print("Finding nested directories...")
            directories = os.listdir(image_path)
            if len(directories) == 0:
                print(f'No directories in {image_path}')
            
            for directory in directories:
                save_path_directory = os.path.join(save_path, directory)
                image_path_directory = os.path.join(image_path, directory)
                
                run_inference(device, model, image_path_directory, save_path_directory, tile_width, tile_height)
            

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
        

            

                                                                     
                