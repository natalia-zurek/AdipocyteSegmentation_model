# -*- coding: utf-8 -*-
#"""
#Created on Tue Jan  2 16:19:11 2024

#@author: Natalia Zurek natalia.zurek@cshs.org
#"""

"""Usage:
    Mask2Former_run_train.py [options] [--help]

    Options:
      -h --help                             Show this string.
      --save_folder=<save_folder>           Path where checkpoints will be saved.
      --train_dataset=<train_dataset>       Path to training root folder, with images/ and annotations/ folder
      --batch_size=<batch>                  Training batch size [default: 4]
      --num_epochs=<epochs>                 Number of training epochs [default: 10]
"""
#LIBRARIES
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
from docopt import docopt

#CUSTOM DATASET CLASS
class ImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, root_dir, processor, transform=None):
        """
        Args:
            dataset
        """
        self.root_dir = root_dir
        self.image_list = sorted(os.listdir(os.path.join(root_dir, 'images')))
        self.annotation_list = sorted(os.listdir(os.path.join(root_dir, 'annotations')))
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.annotation_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, 'images 512 rescale', self.image_list[idx])
        annotation_path = os.path.join(self.root_dir, 'annotations 512 rescale', self.annotation_list[idx])
        
        image = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
        
        instance_map = sio.loadmat(annotation_path)["inst_map"]
        unique_ids = np.unique(instance_map)

        # Creating a dictionary where each object has a class value of 1
        mapping = {obj_id: 1 for obj_id in unique_ids if obj_id != 0}
        #mapping = sio.loadmat(annotation_path)["class_map"]

        # apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=instance_map)
            image, instance_seg = transformed['image'], transformed['mask']
            # convert to C, H, W
            image = image.transpose(2,0,1) #WHY?
        #TODO: check the annotations without objects (they should be added to the dataset)
        if np.all(mapping == 0):
            # Some image does not have annotation (all ignored)
            inputs = self.processor([image], return_tensors="pt")
            inputs = {k:v.squeeze() for k,v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros((0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1]))
        else:
          try:
            inputs = self.processor([image], [instance_seg], instance_id_to_semantic_id=mapping, return_tensors="pt")
            inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()}
          except:
            inputs = self.processor([image], return_tensors="pt")
            inputs = {k:v.squeeze() for k,v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros((0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1]))

        return inputs


#FUNCTIONS
def collate_fn(batch):
    # Remove None values
    batch = [item for item in batch if item is not None]

    if not batch:  # If entire batch has None values, return None
        return None

    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels, "mask_labels": mask_labels}


def check_path_existence(path):
    if not os.path.exists(path):
        #print(f"The path '{path}' does not exists.")
        raise FileNotFoundError(f"The path '{path}' does not exist.")


#MAIN CODE

if __name__ == "__main__":
    arguments = docopt(__doc__)

    save_checkpoints_folder = arguments['--save_folder']
    train_dataset_path = arguments['--train_dataset']
    batch_size = int(arguments['--batch_size'])
    num_epochs = int(arguments['--num_epochs'])

    try:        
        check_path_existence(train_dataset_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        
    if not os.path.exists(save_checkpoints_folder):
        os.makedirs(save_checkpoints_folder, exist_ok = True)
        
    
    # DATA AUGMENTATION    
    #ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
    #ADE_STD = np.array([58.395, 57.120, 57.375]) / 255


    #https://demo.albumentations.ai/
    train_transform = A.Compose([
        A.GaussianBlur(always_apply=False, p=0.5, blur_limit=(3, 19), sigma_limit=(0, 2)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        #A.RandomScale(always_apply=False, p=1, interpolation=1, scale_limit=(-0.05, 0.05)), #changing pixel size, interpolation 1 - linear, 
        A.RandomRotate90(always_apply=False, p=0.5), #Randomly rotate the input by 90 degrees zero or more times.
        #A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ])
    
    processor = Mask2FormerImageProcessor(reduce_labels=True, ignore_index=0, do_resize=False, do_rescale=False, rescale_factor=1/2, do_normalize=False)
    train_dataset = ImageSegmentationDataset(train_dataset_path, processor, train_transform)    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # MODEL
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-instance", ignore_mismatched_sizes=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    running_loss = 0.0
    num_samples = 0
    #num_epochs = 10


    for epoch in range(num_epochs):
      print("Epoch:", epoch)
      model.train()
      for idx, batch in enumerate(tqdm(train_dataloader)):
          if batch is None:  # Skip None batches
              continue

          # Reset the parameter gradients
          optimizer.zero_grad(set_to_none=True)

          # Forward pass
          outputs = model(
                  pixel_values=batch["pixel_values"].to(device),
                  mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                  class_labels=[labels.to(device) for labels in batch["class_labels"]],
          )

          # Backward propagation
          loss = outputs.loss
          loss.backward()

          batch_size = batch["pixel_values"].size(0)
          running_loss += loss.item()
          num_samples += batch_size

          if idx % 100 == 0:
            print("Loss:", running_loss/num_samples)

          # Optimization
          optimizer.step()

      if (epoch + 1) % 5 == 0 or epoch == (num_epochs - 1):
        checkpoint_path = os.path.join(save_checkpoints_folder, f'mask2former_adipocyte_test_epoch_{epoch+1}')
        model.save_pretrained(checkpoint_path)
        processor.save_pretrained(checkpoint_path)  
        print(f"Model saved to {checkpoint_path}")



    
    
    
    
    
    
    
    
    
    