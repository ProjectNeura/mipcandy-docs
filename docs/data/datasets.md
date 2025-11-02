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

## Dataset Types

### Base Classes

#### UnsupervisedDataset

Abstract base class for datasets containing images only.

**Type signature:**
```python
UnsupervisedDataset[D]  # D is the type of image storage (e.g., list[str], list[torch.Tensor])
```

**Returns:** `torch.Tensor` (single image)

**Key methods:**
- `load(idx: int) -> torch.Tensor`: Load image at index
- `__len__() -> int`: Return number of samples

**Usage:**
```python
from mipcandy import UnsupervisedDataset

class MyUnsupervisedDataset(UnsupervisedDataset[list[str]]):
    def __init__(self, image_paths: list[str], device: str = "cpu"):
        super().__init__(image_paths, device=device)

    def load(self, idx: int) -> torch.Tensor:
        # Custom loading logic
        return load_image(self._images[idx], device=self._device)
```

#### SupervisedDataset

Abstract base class for datasets containing image-label pairs.

**Type signature:**
```python
SupervisedDataset[D]  # D is the type of image/label storage
```

**Returns:** `tuple[torch.Tensor, torch.Tensor]` (image, label)

**Key methods:**
- `load(idx: int) -> tuple[torch.Tensor, torch.Tensor]`: Load image and label at index
- `__len__() -> int`: Return number of samples
- `construct_new(images: D, labels: D) -> Self`: Create new instance with subset (required for folding)
- `fold(fold, picker) -> tuple[Self, Self]`: Built-in K-fold splitting

**Usage:**
```python
from mipcandy import SupervisedDataset

class MyDataset(SupervisedDataset[list[str]]):
    def __init__(self, image_paths: list[str], label_paths: list[str], device: str = "cpu"):
        super().__init__(image_paths, label_paths, device=device)

    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = load_image(self._images[idx], device=self._device)
        label = load_image(self._labels[idx], is_label=True, device=self._device)
        return image, label

    def construct_new(self, images: list[str], labels: list[str]) -> Self:
        return MyDataset(images, labels, device=self._device)
```

### Concrete Implementations

#### NNUNetDataset

Medical imaging dataset following the nnU-Net format convention.

**Directory structure:**
```
dataset/
├── imagesTr/
│   ├── case_0000_0000.nii.gz  # Single modality
│   ├── case_0001_0000.nii.gz
│   └── ...
├── labelsTr/
│   ├── case_0000.nii.gz
│   ├── case_0001.nii.gz
│   └── ...
├── imagesTs/  # Test split (optional)
└── labelsTs/
```

**Multimodal support:**
```
dataset/
├── imagesTr/
│   ├── case_0000_0000.nii.gz  # Modality 0 (e.g., T1)
│   ├── case_0000_0001.nii.gz  # Modality 1 (e.g., T2)
│   ├── case_0000_0002.nii.gz  # Modality 2 (e.g., FLAIR)
│   ├── case_0001_0000.nii.gz
│   └── ...
└── labelsTr/
    ├── case_0000.nii.gz
    └── ...
```

**Parameters:**
- `folder`: Dataset root directory
- `split`: `"Tr"` (training) or `"Ts"` (test), default: `"Tr"`
- `prefix`: Filter cases by prefix, default: `""`
- `align_spacing`: Resample to isotropic spacing, default: `False`
- `image_transform`: Optional image transform function
- `label_transform`: Optional label transform function
- `device`: Device placement, default: `"cpu"`

**Examples:**

Basic usage:
```python
from mipcandy import NNUNetDataset

# Single modality dataset
dataset = NNUNetDataset("dataset/", device="cuda")
image, label = dataset[0]  # (C, H, W) or (C, D, H, W)
```

Multimodal dataset:
```python
# Automatically concatenates all modalities
dataset = NNUNetDataset("dataset/", device="cuda")
image, label = dataset[0]  # (N_modalities, H, W) or (N_modalities, D, H, W)
```

With preprocessing:
```python
from mipcandy import NNUNetDataset, Normalize

# Resample to isotropic spacing
dataset = NNUNetDataset(
    "dataset/",
    align_spacing=True,
    device="cuda"
)

# With custom transforms
normalizer = Normalize(domain=(0, 1))
dataset = NNUNetDataset(
    "dataset/",
    image_transform=normalizer,
    device="cuda"
)
```

Test split:
```python
# Load test set
test_dataset = NNUNetDataset("dataset/", split="Ts", device="cuda")
```

Filter by prefix:
```python
# Only load cases starting with "BRATS"
dataset = NNUNetDataset("dataset/", prefix="BRATS", device="cuda")
```

**Export dataset:**
```python
# Save current dataset split to new location
dataset.save("Tr", target_folder="processed_dataset/")
```

#### BinarizedDataset

Wrapper that converts multi-class segmentation to binary segmentation.

**Parameters:**
- `base`: Underlying supervised dataset
- `positive_ids`: Tuple of class IDs to treat as positive (foreground)

**Conversion logic:**
- Classes in `positive_ids` → 1 (positive)
- All other non-background classes → 0 (negative)
- Background (0) → 0 (negative)

**Examples:**

Tumor detection:
```python
from mipcandy import NNUNetDataset, BinarizedDataset

# Original: 0=background, 1=liver, 2=tumor
base = NNUNetDataset("dataset/", device="cuda")

# Binary: 0=non-tumor, 1=tumor
binary = BinarizedDataset(base, positive_ids=(2,))
```

Multi-organ grouping:
```python
# Original: 0=background, 1=spleen, 2=kidney_right, 3=kidney_left
base = NNUNetDataset("dataset/", device="cuda")

# Binary: kidneys vs others
binary = BinarizedDataset(base, positive_ids=(2, 3))
```

:::{note}
[`BinarizedDataset`](#mipcandy.data.dataset.BinarizedDataset) does not support `construct_new()` and therefore cannot be used with `fold()`. Apply binarization after folding instead.
:::

Correct usage with K-fold:
```python
# Fold first, then binarize
base = NNUNetDataset("dataset/", device="cuda")
train, val = base.fold(fold=0)

train_binary = BinarizedDataset(train, positive_ids=(2,))
val_binary = BinarizedDataset(val, positive_ids=(2,))
```

#### SimpleDataset

Simple unsupervised dataset loading all files from a directory.

**Parameters:**
- `folder`: Directory containing images
- `device`: Device placement, default: `"cpu"`

**Examples:**
```python
from mipcandy import SimpleDataset

# Load all images from directory (sorted alphabetically)
dataset = SimpleDataset("images/", device="cuda")

# Supports various formats
# images/
# ├── scan001.nii.gz
# ├── scan002.nii.gz
# ├── image001.png
# └── ...

for image in dataset:
    print(image.shape)
```

#### DatasetFromMemory

In-memory dataset for pre-loaded tensors.

**Parameters:**
- `images`: Sequence of pre-loaded tensors
- `device`: Device placement, default: `"cpu"`

**Use cases:**
- Small datasets that fit in memory
- Avoiding repeated I/O operations
- Fast iteration during prototyping

**Examples:**
```python
from mipcandy import DatasetFromMemory
import torch

# Pre-load all tensors
tensors = [torch.rand(3, 256, 256) for _ in range(100)]
dataset = DatasetFromMemory(tensors, device="cuda")

# Fast access (no I/O)
image = dataset[0]
```

Combined with path-based dataset:
```python
from mipcandy import NNUNetDataset, DatasetFromMemory

# Load all to memory
path_dataset = NNUNetDataset("dataset/")
images = [path_dataset[i][0] for i in range(len(path_dataset))]

# Create memory dataset
memory_dataset = DatasetFromMemory(images, device="cuda")
```

#### MergedDataset

Supervised dataset created by merging separate image and label datasets.

**Parameters:**
- `images`: Unsupervised dataset for images
- `labels`: Unsupervised dataset for labels
- `device`: Device placement, default: `"cpu"`

**Examples:**
```python
from mipcandy import SimpleDataset, MergedDataset

# Separate directories for images and labels
images = SimpleDataset("images/", device="cuda")
labels = SimpleDataset("labels/", device="cuda")

# Merge into supervised dataset
dataset = MergedDataset(images, labels, device="cuda")

image, label = dataset[0]
```

With memory datasets:
```python
from mipcandy import DatasetFromMemory, MergedDataset

# Pre-loaded tensors
image_tensors = [...]
label_tensors = [...]

images = DatasetFromMemory(image_tensors, device="cuda")
labels = DatasetFromMemory(label_tensors, device="cuda")

dataset = MergedDataset(images, labels, device="cuda")
```

## K-Fold Cross Validation

All [`SupervisedDataset`](#mipcandy.data.dataset.SupervisedDataset) instances have built-in K-fold cross validation support through the `fold()` method.

### Basic Usage

The `fold()` method splits the dataset into training and validation sets:

```python
from mipcandy import NNUNetDataset

dataset = NNUNetDataset("dataset/", device="cuda")

# Split into training and validation
train_dataset, val_dataset = dataset.fold(fold=0)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
```

**Method signature:**
```python
def fold(
    self,
    *,
    fold: Literal[0, 1, 2, 3, 4, "all"] = "all",
    picker: type[KFPicker] = OrderedKFPicker
) -> tuple[Self, Self]:
    ...
```

**Parameters:**
- `fold`: Which fold to use as validation set
  - `0`, `1`, `2`, `3`, `4`: Use specific fold (0-4) as validation
  - `"all"`: Use all samples for validation (training set is empty)
- `picker`: Strategy for selecting fold samples
  - `OrderedKFPicker`: Sequential splitting (default, reproducible)
  - `RandomKFPicker`: Random sampling for validation fold

**Returns:** `(train_dataset, val_dataset)` tuple

### Fold Selection

Standard 5-fold cross validation:

```python
from mipcandy import NNUNetDataset

dataset = NNUNetDataset("dataset/", device="cuda")

# Fold 0: First 20% as validation
train_0, val_0 = dataset.fold(fold=0)

# Fold 1: Second 20% as validation
train_1, val_1 = dataset.fold(fold=1)

# Fold 2: Third 20% as validation
train_2, val_2 = dataset.fold(fold=2)

# ... and so on for folds 3 and 4
```

Use all samples for evaluation:

```python
# Both train and val contain all samples
train, val = dataset.fold(fold="all")
print(len(train))  # Full dataset size
print(len(val))    # Full dataset size
```

### Picker Strategies

#### OrderedKFPicker (Default)

Splits dataset sequentially into 5 equal parts:

```python
from mipcandy import NNUNetDataset, OrderedKFPicker

dataset = NNUNetDataset("dataset/", device="cuda")

# Sequential splitting (reproducible)
train, val = dataset.fold(fold=0, picker=OrderedKFPicker)
```

**Splitting logic:**
- Dataset size: 100 samples (indices 0-99)
- Fold 0 validation: indices 0-19 (20%)
- Fold 1 validation: indices 20-39 (20%)
- Fold 2 validation: indices 40-59 (20%)
- Fold 3 validation: indices 60-79 (20%)
- Fold 4 validation: indices 80-99 (20%)

**Characteristics:**
- Reproducible: same fold always gives same split
- Maintains data order
- Recommended for most use cases

#### RandomKFPicker

Randomly samples validation indices:

```python
from mipcandy import NNUNetDataset, RandomKFPicker

dataset = NNUNetDataset("dataset/", device="cuda")

# Random sampling (non-reproducible)
train, val = dataset.fold(fold=0, picker=RandomKFPicker)
```

**Characteristics:**
- Non-reproducible: different splits each time
- Breaks data order
- Useful for additional randomization

:::{warning}
[`RandomKFPicker`](#mipcandy.data.dataset.RandomKFPicker) uses random sampling without a fixed seed, so results will vary between runs. For reproducible experiments, use [`OrderedKFPicker`](#mipcandy.data.dataset.OrderedKFPicker).
:::

### Complete 5-Fold Example

```python
from mipcandy import NNUNetDataset
from torch.utils.data import DataLoader

# Load dataset
full_dataset = NNUNetDataset("dataset/", device="cuda")

# Track results across folds
fold_results = []

# 5-fold cross validation
for fold_id in range(5):
    print(f"\n=== Fold {fold_id} ===")

    # Create fold split
    train_dataset, val_dataset = full_dataset.fold(fold=fold_id)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Train model for this fold
    model = create_model()
    for epoch in range(num_epochs):
        # Training
        for images, labels in train_loader:
            # ... training step ...
            pass

        # Validation
        val_score = 0.0
        for images, labels in val_loader:
            # ... validation step ...
            pass

    fold_results.append(val_score)

# Compute cross-validation statistics
import statistics
print(f"\nMean score: {statistics.mean(fold_results):.4f}")
print(f"Std score: {statistics.stdev(fold_results):.4f}")
```

### Training on Full Dataset

After cross-validation, train final model on all data:

```python
from mipcandy import NNUNetDataset
from torch.utils.data import DataLoader

dataset = NNUNetDataset("dataset/", device="cuda")

# Use all samples for training
train_dataset, _ = dataset.fold(fold="all")

# Train final model
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
final_model = create_model()

for epoch in range(num_epochs):
    for images, labels in train_loader:
        # ... training step ...
        pass
```

### Dataset Export by Fold

Save specific fold split to disk:

```python
from mipcandy import NNUNetDataset

dataset = NNUNetDataset("dataset/", split="Tr", device="cuda")

# Create fold split
train, val = dataset.fold(fold=0)

# Save validation fold to separate directory
val.save("Ts", target_folder="dataset_fold0/")

# Now you have:
# dataset_fold0/
# ├── imagesTs/  (validation images from fold 0)
# └── labelsTs/  (validation labels from fold 0)
```

:::{note}
Only [`NNUNetDataset`](#mipcandy.data.dataset.NNUNetDataset) supports the `save()` method. Other dataset types need custom export logic.
:::

### Combining with Transforms

Apply transforms to full dataset before folding:

```python
from mipcandy import NNUNetDataset, Normalize

# Create dataset with transforms
normalizer = Normalize(domain=(0, 1))
dataset = NNUNetDataset(
    "dataset/",
    image_transform=normalizer,
    align_spacing=True,
    device="cuda"
)

# Fold with transforms applied
train, val = dataset.fold(fold=0)

# Both train and val will have normalized, resampled images
```

### Implementation Notes

**Dataset Independence:**

Each fold creates independent dataset instances:

```python
train, val = dataset.fold(fold=0)

# train and val are separate instances
# Modifying one does not affect the other
print(id(train) != id(val))  # True
```

**Memory Sharing:**

Path-based datasets share path lists but create separate instances:

```python
# Original dataset is not modified
train, val = dataset.fold(fold=0)
print(len(dataset))  # Original size unchanged

# But internal path lists are lightweight
# No image data is duplicated until loading
```

**Requirements for Custom Datasets:**

To support `fold()`, custom datasets must implement `construct_new()`:

```python
from mipcandy import SupervisedDataset

class MyDataset(SupervisedDataset[list[str]]):
    def construct_new(self, images: list[str], labels: list[str]):
        # Create new instance with subset
        return MyDataset(images, labels, device=self._device)
```

See [Custom Datasets](#custom-datasets) for more details.

## Dataset Inspection and Patches

Medical imaging datasets often have large volumes with small regions of interest (ROI). The inspection module provides tools for analyzing foreground regions and extracting patches for efficient training.

### Overview

**Why patch-based training?**

Medical images often have:
- Large volumes (e.g., 512×512×512) that don't fit in GPU memory
- Sparse foreground regions (e.g., tumors occupy <5% of volume)
- Variable image sizes across cases

**Patch-based approach:**
- Extract small patches (e.g., 128×128×128) centered on ROI
- Train on informative regions, reducing memory usage
- Handle variable image sizes uniformly

**MIPCandy inspection provides:**
- Automatic foreground region detection
- Statistical analysis of ROI sizes and positions
- Intelligent patch extraction centered on foreground
- Ready-to-use patch datasets for training

### Inspecting Datasets

The [`inspect()`](#mipcandy.data.inspection.inspect) function automatically analyzes a dataset:

```python
from mipcandy import NNUNetDataset, inspect

# Load dataset
dataset = NNUNetDataset("dataset/", device="cuda")

# Inspect all cases
annotations = inspect(dataset, background=0)

print(f"Inspected {len(annotations)} cases")
```

**Parameters:**
- `dataset`: Any [`SupervisedDataset`](#mipcandy.data.dataset.SupervisedDataset)
- `background`: Background class ID (default: `0`)

**Returns:** [`InspectionAnnotations`](#mipcandy.data.inspection.InspectionAnnotations) object

**What it computes:**

For each case:
- Image shape
- Foreground bounding box (minimal box containing all non-background voxels)
- Unique class IDs present in the label

### InspectionAnnotation

Individual case annotation containing metadata:

```python
from mipcandy import inspect, NNUNetDataset

dataset = NNUNetDataset("dataset/", device="cuda")
annotations = inspect(dataset)

# Access individual annotation
annotation = annotations[0]

print(f"Image shape: {annotation.shape}")
print(f"Foreground bbox: {annotation.foreground_bbox}")
print(f"Class IDs: {annotation.ids}")
```

**Attributes:**
- `shape`: Image spatial dimensions `(H, W)` or `(D, H, W)`
- `foreground_bbox`: Bounding box `(y0, y1, x0, x1)` or `(z0, z1, y0, y1, x0, x1)`
- `ids`: Tuple of unique class IDs in label

**Methods:**

```python
# Get foreground region size
fg_shape = annotation.foreground_shape()
print(f"Foreground size: {fg_shape}")  # (H, W) or (D, H, W)

# Get foreground center
center = annotation.center_of_foreground()
print(f"Foreground center: {center}")  # (y, x) or (z, y, x)
```

### InspectionAnnotations

Collection of annotations with statistical analysis:

#### Basic Access

```python
from mipcandy import inspect, NNUNetDataset

dataset = NNUNetDataset("dataset/", device="cuda")
annotations = inspect(dataset)

# Length
print(f"Number of cases: {len(annotations)}")

# Iteration
for annotation in annotations:
    print(annotation.foreground_bbox)

# Indexing
first = annotations[0]
```

#### Statistical Analysis

**Image and foreground shapes:**

```python
# Get all image shapes
depths, heights, widths = annotations.shapes()

if depths:  # 3D dataset
    print(f"Depth range: {min(depths)} - {max(depths)}")
print(f"Height range: {min(heights)} - {max(heights)}")
print(f"Width range: {min(widths)} - {max(widths)}")

# Get all foreground shapes
fg_depths, fg_heights, fg_widths = annotations.foreground_shapes()

if fg_depths:
    print(f"Foreground depth range: {min(fg_depths)} - {max(fg_depths)}")
print(f"Foreground height range: {min(fg_heights)} - {max(fg_heights)}")
print(f"Foreground width range: {min(fg_widths)} - {max(fg_widths)}")
```

**Statistical foreground shape:**

Compute representative foreground size using percentile:

```python
# 95th percentile (default)
stat_shape = annotations.statistical_foreground_shape(percentile=0.95)
print(f"95th percentile foreground: {stat_shape}")

# 99th percentile (larger patches)
stat_shape_99 = annotations.statistical_foreground_shape(percentile=0.99)
print(f"99th percentile foreground: {stat_shape_99}")
```

This is useful for determining patch size that covers most foregrounds.

**Foreground heatmap:**

Generate heatmap showing where foreground regions typically occur:

```python
# Compute heatmap (expensive, computed once and cached)
heatmap = annotations.foreground_heatmap()

print(f"Heatmap shape: {heatmap.shape}")

# Visualize heatmap
from mipcandy import visualize2d, visualize3d

if heatmap.ndim == 2:
    visualize2d(heatmap, title="Foreground Heatmap")
else:
    visualize3d(heatmap, title="Foreground Heatmap 3D")
```

**Center of foregrounds:**

Find the typical center position across all cases:

```python
# Compute global foreground center
center = annotations.center_of_foregrounds()
print(f"Typical foreground center: {center}")

# Offsets from center for each case
offsets = annotations.center_of_foregrounds_offsets()
print(f"Center offsets: {offsets}")
```

#### ROI Shape Determination

Automatically determine optimal ROI size:

```python
# Automatic ROI shape based on statistics
roi_shape = annotations.roi_shape(percentile=0.95)
print(f"Recommended ROI shape: {roi_shape}")

# Manual override
annotations.set_roi_shape((128, 128, 128))
roi_shape = annotations.roi_shape()
print(f"Manual ROI shape: {roi_shape}")
```

:::{note}
The automatic ROI shape is computed as the minimum of:
- Statistical foreground shape (95th percentile by default)
- Minimum image size across all cases

This ensures patches fit within all images while covering most foregrounds.
:::

### ROI Extraction

#### crop_foreground()

Extract foreground region with optional expansion:

```python
from mipcandy import inspect, NNUNetDataset

dataset = NNUNetDataset("dataset/", device="cuda")
annotations = inspect(dataset)

# Crop to exact foreground bbox
image_crop, label_crop = annotations.crop_foreground(0)

# Crop with 1.5x expansion (50% padding)
image_expanded, label_expanded = annotations.crop_foreground(0, expand_ratio=1.5)

print(f"Original foreground: {annotations[0].foreground_shape()}")
print(f"Cropped shape: {image_crop.shape}")
print(f"Expanded shape: {image_expanded.shape}")
```

**Parameters:**
- `i`: Case index
- `expand_ratio`: Expansion factor (default: `1.0`)

#### crop_roi()

Extract ROI centered on foreground:

```python
# Crop to computed ROI shape
image_roi, label_roi = annotations.crop_roi(0)

print(f"ROI shape: {image_roi.shape}")

# Use different percentile
image_roi_99, label_roi_99 = annotations.crop_roi(0, percentile=0.99)
```

**Behavior:**
- Centers patch on case-specific foreground center
- Applies global offsets to align with typical foreground position
- Ensures patch stays within image boundaries
- Returns fixed-size patches (determined by `roi_shape()`)

#### roi()

Get ROI bounding box without cropping:

```python
# Get ROI coordinates
bbox = annotations.roi(0, percentile=0.95)

if len(bbox) == 4:  # 2D
    y0, y1, x0, x1 = bbox
    print(f"2D ROI: y[{y0}:{y1}], x[{x0}:{x1}]")
else:  # 3D
    z0, z1, y0, y1, x0, x1 = bbox
    print(f"3D ROI: z[{z0}:{z1}], y[{y0}:{y1}], x[{x0}:{x1}]")
```

### ROIDataset

Dataset wrapper that yields ROI patches instead of full images:

```python
from mipcandy import NNUNetDataset, inspect, ROIDataset
from torch.utils.data import DataLoader

# Load and inspect dataset
dataset = NNUNetDataset("dataset/", device="cuda")
annotations = inspect(dataset)

# Create ROI dataset
roi_dataset = ROIDataset(annotations, percentile=0.95)

print(f"ROI dataset size: {len(roi_dataset)}")

# Access patches
image_patch, label_patch = roi_dataset[0]
print(f"Patch shape: {image_patch.shape}")

# Use with DataLoader
loader = DataLoader(roi_dataset, batch_size=4, shuffle=True)
for images, labels in loader:
    print(f"Batch images: {images.shape}")
    print(f"Batch labels: {labels.shape}")
    break
```

**Parameters:**
- `annotations`: [`InspectionAnnotations`](#mipcandy.data.inspection.InspectionAnnotations) object
- `percentile`: Percentile for ROI size determination (default: `0.95`)

**Characteristics:**
- Returns fixed-size patches centered on foreground
- Compatible with PyTorch DataLoader
- Inherits device from annotations
- Does not support `fold()` (fold before inspection)

### Saving and Loading Annotations

Save inspection results to avoid re-computation:

```python
from mipcandy import inspect, NNUNetDataset

# Inspect and save
dataset = NNUNetDataset("dataset/", device="cuda")
annotations = inspect(dataset)
annotations.save("annotations.csv")

# Load later (note: requires dataset reference)
# from mipcandy import load_inspection_annotations
# annotations = load_inspection_annotations("annotations.csv")
```

### Complete Patch-based Training Example

```python
from mipcandy import NNUNetDataset, inspect, ROIDataset
from torch.utils.data import DataLoader

# Load dataset
full_dataset = NNUNetDataset("dataset/", device="cuda")

# K-fold split FIRST
train_dataset, val_dataset = full_dataset.fold(fold=0)

# Inspect training set
train_annotations = inspect(train_dataset, background=0)
print(f"Inspected {len(train_annotations)} training cases")

# Analyze dataset
roi_shape = train_annotations.roi_shape(percentile=0.95)
print(f"Recommended ROI shape: {roi_shape}")

# Optionally visualize heatmap
heatmap = train_annotations.foreground_heatmap()
from mipcandy import visualize3d
visualize3d(heatmap, title="Training Foreground Heatmap")

# Create ROI dataset for training
train_roi = ROIDataset(train_annotations, percentile=0.95)

# Inspect validation set separately
val_annotations = inspect(val_dataset, background=0)
val_roi = ROIDataset(val_annotations, percentile=0.95)

# Create loaders
train_loader = DataLoader(train_roi, batch_size=4, shuffle=True)
val_loader = DataLoader(val_roi, batch_size=1, shuffle=False)

# Train on patches
for epoch in range(num_epochs):
    # Training
    for images, labels in train_loader:
        # images: (B, C, H, W) or (B, C, D, H, W)
        # All patches have same size
        pass

    # Validation
    for images, labels in val_loader:
        pass
```

:::{tip}
Always inspect training and validation sets separately after folding to ensure statistics are computed only on training data.
:::

### Use Cases

**Small GPU memory:**
```python
# Large volumes don't fit in GPU
# Use patches instead
annotations = inspect(dataset)
annotations.set_roi_shape((128, 128, 128))  # Fits in GPU
roi_dataset = ROIDataset(annotations)
```

**Variable image sizes:**
```python
# Dataset has different image sizes
# ROIDataset yields uniform patches
shapes = annotations.shapes()
print(f"Variable sizes: {min(shapes[1])}x{min(shapes[2])} to {max(shapes[1])}x{max(shapes[2])}")

roi_dataset = ROIDataset(annotations)
# All patches have same size determined by roi_shape()
```

**Focus on foreground:**
```python
# Sparse foreground regions
# Patches centered on ROI avoid empty patches
annotations = inspect(dataset)
for i in range(len(annotations)):
    fg_ratio = (
        annotations[i].foreground_shape()[0] *
        annotations[i].foreground_shape()[1] /
        (annotations[i].shape[0] * annotations[i].shape[1])
    )
    print(f"Case {i}: {fg_ratio*100:.1f}% foreground")

# ROI patches have much higher foreground ratio
```
