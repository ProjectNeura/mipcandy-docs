# Datasets

MIPCandy provides a comprehensive dataset system designed for medical image processing, with built-in support for supervised and unsupervised learning, K-fold cross validation, and patch-based training.

## Overview

The dataset module offers a flexible hierarchy of dataset classes tailored for medical imaging workflows:

### Dataset Hierarchy

```
_AbstractDataset (base)
├── UnsupervisedDataset
│   ├── DatasetFromMemory
│   ├── PathBasedUnsupervisedDataset
│   │   └── SimpleDataset
│   └── (user custom datasets)
│
└── SupervisedDataset
    ├── MergedDataset
    ├── PathBasedSupervisedDataset
    │   └── NNUNetDataset
    ├── BinarizedDataset
    ├── ROIDataset
    └── (user custom datasets)
```

### Key Features

**Dataset Types:**
- **Supervised**: Image-label pairs for segmentation, classification tasks
- **Unsupervised**: Image-only datasets for unsupervised learning
- **Path-based**: Lazy loading from disk for large datasets
- **Memory-based**: Pre-loaded tensors for fast iteration

**Medical Imaging Support:**
- **nnU-Net Format**: Native support for nnU-Net dataset structure
- **Multimodal**: Handle multiple imaging modalities (CT, MRI sequences, etc.)
- **Isotropic Resampling**: Automatic spacing alignment
- **Format Support**: NIfTI (.nii, .nii.gz), MHA, PNG, JPG via SimpleITK

**Advanced Features:**
- **K-Fold Cross Validation**: Built-in fold splitting with ordered or random strategies
- **Dataset Inspection**: Automatic analysis of foreground regions and statistics
- **Patch Extraction**: ROI-based cropping for patch-based training
- **Dataset Export**: Save dataset paths in CSV, JSON, or TXT formats
- **Transform Pipeline**: Configurable preprocessing for images and labels

**Integration:**
- **PyTorch Compatible**: Inherits from `torch.utils.data.Dataset`
- **Device Management**: Automatic GPU/CPU handling with `HasDevice` mixin
- **Type Safety**: Full generic typing support for IDE autocomplete

## Quick Start

### Basic Usage

```python
from mipcandy import NNUNetDataset
from torch.utils.data import DataLoader

# Create dataset from nnU-Net format folder
dataset = NNUNetDataset("path/to/dataset", device="cuda")

# Access individual samples
image, label = dataset[0]
print(f"Image shape: {image.shape}")
print(f"Label shape: {label.shape}")

# Get dataset size
print(f"Total samples: {len(dataset)}")
```

### K-Fold Cross Validation

```python
from mipcandy import NNUNetDataset
from torch.utils.data import DataLoader

# Create full dataset
full_dataset = NNUNetDataset("path/to/dataset", device="cuda")

# Split into training and validation (fold 0)
train_dataset, val_dataset = full_dataset.fold(fold=0)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Use in training
for images, labels in train_loader:
    # Training step
    pass
```

### Complete Training Workflow

```python
from mipcandy import NNUNetDataset
from torch.utils.data import DataLoader

# Setup dataset with preprocessing
dataset = NNUNetDataset(
    "dataset/",
    split="Tr",
    align_spacing=True,  # Resample to isotropic spacing
    device="cuda"
)

# 5-fold cross validation
for fold_id in range(5):
    print(f"Training fold {fold_id}...")

    # Create fold
    train_dataset, val_dataset = dataset.fold(fold=fold_id)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Train model
    # ... your training code here ...
```

### Simple Directory Dataset

```python
from mipcandy import SimpleDataset

# Load all images from a directory
dataset = SimpleDataset("path/to/images", device="cuda")

# Iterate over images
for image in dataset:
    print(f"Image shape: {image.shape}")
```

### Binary Segmentation

```python
from mipcandy import NNUNetDataset, BinarizedDataset

# Original dataset with multiple classes (0: background, 1: liver, 2: tumor)
base_dataset = NNUNetDataset("dataset/", device="cuda")

# Convert to binary (tumor vs non-tumor)
binary_dataset = BinarizedDataset(base_dataset, positive_ids=(2,))

# Now labels are binary: 0 (non-tumor) and 1 (tumor)
image, binary_label = binary_dataset[0]
print(f"Unique labels: {binary_label.unique()}")  # [0, 1]
```
