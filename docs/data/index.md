# Data

The data module provides comprehensive tools for loading, processing, and visualizing medical images in MIPCandy.

## Overview

**Datasets:**
- PyTorch-compatible dataset classes for medical imaging
- Built-in K-fold cross validation
- nnU-Net format support with multimodal handling
- Dataset inspection and patch-based training

See [Datasets](datasets.md) for detailed documentation.

**Visualization:**
- 2D and 3D rendering with Matplotlib and PyVista
- Overlay segmentation masks on images
- Automatic value normalization

See [Visualization](visualization.md) for detailed documentation.

## Quick Start

```python
from mipcandy import NNUNetDataset, visualize2d, overlay
from torch.utils.data import DataLoader

# Load dataset with K-fold support
dataset = NNUNetDataset("dataset/", device="cuda")
train, val = dataset.fold(fold=0)

# Create data loader
loader = DataLoader(train, batch_size=4, shuffle=True)

# Visualize sample
image, label = train[0]
overlaid = overlay(image, label)
visualize2d(overlaid)
```
