# Feature interaction network based on Hierarchical Decoupled Convolution for 3D medical image segmentation
This repository is the work of <a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0288658" target="_blank">"Feature interaction network based on Hierarchical Decoupled Convolution for 3D medical image segmentation"</a> based on PyTorch implementation. <br>
# Requirements
* python 3.6<br>
* pytorch 0.4 or 1.0<br>
* nibabel<br>
* pickle<br>
* imageio<br>
* pyyaml<br>
# Implementation
Download the BraTS2019 dataset and change the path:<br>
```experiments/PATH.yaml```<br>
<br>
Data preprocess:(Convert the .nii files as .pkl files. Normalization with zero-mean and unit variance)<br>
```python preprocess.py```<br>
<br>
(Optional) Split the training set into k-fold for the cross-validation experiment.<br>
```python split.py```<br>
<br>
# Training
Sync bacth normalization is used so that a proper batch size is important to obtain a decent performance. Multiply gpus training with batch_size=8 is recommended.<br>
```python train_all.py --gpu=0,1 --cfg=FHDC_Net --batch_size=8```<br>
# Test
You could obtain the resutls as paper reported by running the following code:<br>
```python test.py --mode=1 --is_out=True --verbose=True --use_TTA=True --postprocess=True --snapshot=True --restore=model_last.pth --cfg=FHDC_Net --gpu=1```<br>
# Evaluation
Submit the results to the online [evaluation server](https://ipp.cbica.upenn.edu/).<br>
