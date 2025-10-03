# AdipocyteSegmentation_model
This repository is dedicated to the research paper [Morphometric analysis of adipocytes in matastatic ovarian cancer](link) (**under review**).

# Table of content:
* QuPath annotation code - folder conatining .groovy & MATLAB code for ground truth preparation and export
* - folder contating code used for DLV model training, validation and inference
  - folder containing python code used for M2F model training and inference
  - folder containing code for model evaluation

## Models' weights 
**M2F** model weights can be downloaded [here](link). (not yet available)
**DLV** model weights can be downloaded [here](link). (not yet available)

## Datasets
TCGA, GETX, OM1&2 and GUT images and their corresponding masks (binary and instance) can be found under this [link](link). (not yet available)

## Installation & Prerequisities
```python
pip install -r /path/to/requirements.txt
```

## Usage
### Model training
```python
python Mask2Former_run_train.py --save_folder="path/to/your/output/directory" --train_dataset="path/to/your/training/dataset" --num_epochs=80 --batch_size=32
```

### Model inference
```python
python Mask2Former_model_inference.py --model_path="path/to/your/model" --save_path="path/to/your/output/directory" --image_path="path/to/your/images"
```
 
Contact: [natalia.zurek@cshs.org](mailto:natalia.zurek@cshs.org)
