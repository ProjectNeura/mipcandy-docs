# Download a Dataset

We have a few datasets prepared in the nnU-Net format publicly available on the 
[Central Data Storage Server](https://cds.projectneura.org). You can download them using a built-in function in MIP
Candy.

```python
from mipcandy import download_dataset

download_dataset("nnunet_datasets/PH2", "where/to/save")
```

This will create a folder "where/to/save" and subfolders like "where/to/save/imagesTr".

## Available Datasets

### AbdomenCT-1K

A 3D abdomen CT segmentation dataset with four classes: liver, kidney, spleen, and pancreas.

[![IEEE](https://img.shields.io/badge/IEEE-14303e?style=for-the-badge&logo=ieee)](https://ieeexplore.ieee.org/document/9497733)
[![Download](https://img.shields.io/badge/Download-gray?style=for-the-badge)](https://cds.projectneura.org/nnunet_datasets/AbdomenCT-1K.zip)

### PH2

A small 2D binary melanocytic lesions segmentation dataset.

[![IEEE](https://img.shields.io/badge/IEEE-14303e?style=for-the-badge&logo=ieee)](https://ieeexplore.ieee.org/document/6610779)
[![Download](https://img.shields.io/badge/Download-gray?style=for-the-badge)](https://cds.projectneura.org/nnunet_datasets/PH2.zip)