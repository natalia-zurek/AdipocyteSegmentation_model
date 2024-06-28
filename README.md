# AdipocyteSegmentation_model

# AdipoSeg(?)

This repository is dedicated to the research paper [Morphometric analysis of adipocytes in matastatic ovarian cancer](link) (**under review**).

## AdipoSeg(?)
**AdipoSeg(?)** model can be downloaded [here](link).
Model available upon reasonable request. Contact: [natalia.zurek@cshs.org](mailto:natalia.zurek@cshs.org)


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
 
