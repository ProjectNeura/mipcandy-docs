# Visualization

MIPCandy provides comprehensive visualization tools for medical images, supporting both 2D and 3D rendering with multiple backends. The visualization module enables quick inspection of images, labels, and model predictions with minimal code.

## Overview

The visualization module supports:

- **2D Visualization**: Fast rendering of 2D slices with Matplotlib
- **3D Visualization**: Interactive 3D volume rendering with PyVista or Matplotlib
- **Overlay Rendering**: Combine images with segmentation masks or predictions
- **Flexible Backends**: Automatic backend selection or manual configuration
- **Screenshot Export**: Save visualizations to disk
- **Automatic Normalization**: Smart conversion of tensor values for display

## Quick Start

### Visualize a 2D Image

```python
from mipcandy import load_image, visualize2d

image = load_image("scan.nii.gz")
visualize2d(image[0], title="CT Scan Slice")
```

### Visualize a 3D Volume

```python
from mipcandy import load_image, visualize3d

volume = load_image("volume.nii.gz")
visualize3d(volume, title="3D Brain MRI")
```

### Overlay Prediction on Image

```python
from mipcandy import load_image, overlay, visualize2d

image = load_image("image.nii.gz")
label = load_image("prediction.nii.gz")
overlaid = overlay(image[0], label[0])
visualize2d(overlaid, title="Prediction Overlay")
```

## 2D Visualization

The [`visualize2d()`](#mipcandy.data.visualization.visualize2d) function renders 2D images using Matplotlib.

### Basic Usage

```python
from mipcandy import visualize2d
import torch

# Grayscale image
image = torch.rand(256, 256)
visualize2d(image, title="Random Image")

# Multi-channel image
rgb_image = torch.rand(3, 256, 256)
visualize2d(rgb_image, title="RGB Image")
```

### Parameters

- `image`: Input tensor (2D or 3D)
  - 2D: `(H, W)` - displayed as grayscale
  - 3D with 1 channel: `(1, H, W)` - squeezed and displayed as grayscale
  - 3D with multiple channels: `(C, H, W)` - permuted to `(H, W, C)` for RGB display
- `title`: Optional title string
- `cmap`: Matplotlib colormap name (default: `"gray"`)
- `blocking`: If `True`, blocks execution until window is closed (default: `False`)
- `screenshot_as`: Path to save the visualization as an image file

:::{important}
Input tensors are automatically normalized to [0, 255] range using [`auto_convert()`](#mipcandy.data.visualization.auto_convert).
:::

### Colormap Options

```python
from mipcandy import visualize2d
import torch

image = torch.rand(256, 256)

# Grayscale (default)
visualize2d(image, cmap="gray")

# Jet colormap for heatmaps
visualize2d(image, cmap="jet")

# Viridis for perceptually uniform colors
visualize2d(image, cmap="viridis")
```

### Save Screenshot

```python
from mipcandy import visualize2d
import torch

image = torch.rand(256, 256)

# Save and close without displaying
visualize2d(image, screenshot_as="output.png", blocking=True)

# Save and display
visualize2d(image, screenshot_as="output.png", blocking=False)
```

### Dimension Handling

```python
from mipcandy import visualize2d
import torch

# 4D tensor: automatically reduced to 3D
batch_image = torch.rand(2, 3, 256, 256)  # (B, C, H, W)
visualize2d(batch_image)  # Uses first batch item

# 5D tensor: automatically reduced to 3D
volume = torch.rand(1, 1, 32, 256, 256)  # (B, C, D, H, W)
visualize2d(volume)  # Uses first depth slice
```

## 3D Visualization

The [`visualize3d()`](#mipcandy.data.visualization.visualize3d) function renders 3D volumes with interactive viewing.

### Basic Usage

```python
from mipcandy import visualize3d
import torch

# 3D volume
volume = torch.rand(64, 128, 128)  # (D, H, W)
visualize3d(volume, title="3D Volume")
```

### Backend Selection

MIPCandy supports two backends for 3D visualization:

```python
from mipcandy import visualize3d
import torch

volume = torch.rand(64, 128, 128)

# Automatic selection (prefers PyVista if available)
visualize3d(volume, backend="auto")

# Force PyVista (recommended for best quality)
visualize3d(volume, backend="pyvista")

# Force Matplotlib (slower, lower quality)
visualize3d(volume, backend="matplotlib")
```

:::{tip}
Install PyVista for high-quality interactive 3D rendering:
```shell
pip install "mipcandy[standard]"
```
:::

:::{warning}
Using Matplotlib backend for 3D visualization is inefficient and inaccurate. Consider using PyVista for better performance and quality.
:::

### Parameters

- `image`: Input 3D tensor
  - 3D: `(D, H, W)` - displayed as volume
  - Higher dimensions: automatically reduced to 3D using [`ensure_num_dimensions()`](#mipcandy.data.geometric.ensure_num_dimensions)
- `title`: Optional title string
- `cmap`: Colormap name (default: `"gray"`)
- `max_volume`: Maximum number of voxels (default: `1e6`)
  - Volumes larger than this are downsampled using 3D average pooling
- `backend`: Backend selection: `"auto"`, `"matplotlib"`, or `"pyvista"` (default: `"auto"`)
- `blocking`: If `True`, blocks execution until window is closed (default: `False`)
- `screenshot_as`: Path to save the visualization

### Volume Downsampling

Large volumes are automatically downsampled to improve rendering performance:

```python
from mipcandy import visualize3d
import torch

# Large volume (512^3 ≈ 134M voxels)
large_volume = torch.rand(512, 512, 512)

# Automatically downsampled to ~1M voxels
visualize3d(large_volume, max_volume=1e6)

# Higher quality (slower)
visualize3d(large_volume, max_volume=1e7)

# Lower quality (faster)
visualize3d(large_volume, max_volume=1e5)
```

:::{note}
Downsampling uses 3D average pooling with `ceil_mode=True` to preserve volume boundaries.
:::

### PyVista Interactive Controls

When using PyVista backend:

- **Rotate**: Left-click and drag
- **Pan**: Middle-click and drag
- **Zoom**: Scroll wheel or right-click and drag
- **Reset**: Press 'r'

### Non-Blocking Display

```python
from mipcandy import visualize3d
import torch

volume = torch.rand(64, 128, 128)

# Non-blocking: spawns separate process
visualize3d(volume, blocking=False)

# Blocking: waits for window to close
visualize3d(volume, blocking=True)
```

:::{important}
PyVista backend spawns a separate process for non-blocking display, allowing you to continue execution while viewing the volume.
:::

## Overlay Visualization

The [`overlay()`](#mipcandy.data.visualization.overlay) function combines images with segmentation masks or predictions.

### Basic Usage

```python
from mipcandy import overlay, visualize2d
import torch

image = torch.rand(256, 256)
label = torch.randint(0, 2, (256, 256))

# Create overlay
overlaid = overlay(image, label)
visualize2d(overlaid)
```

### Parameters

- `image`: Base image tensor (2D or higher)
  - Automatically converted to 3-channel RGB if single-channel
- `label`: Label/mask tensor (2D or higher)
  - Should have same spatial dimensions as image
- `max_label_opacity`: Maximum opacity for labels (default: `0.5`)
  - Range: 0.0 (transparent) to 1.0 (opaque)
- `label_colorizer`: Optional [`ColorizeLabel`](#mipcandy.common.module.preprocess.ColorizeLabel) instance
  - Default: `ColorizeLabel()` with automatic colormap
  - Set to `None` to use grayscale labels

### Opacity Control

```python
from mipcandy import overlay, visualize2d
import torch

image = torch.rand(256, 256)
label = torch.randint(0, 3, (256, 256))

# Low opacity (more image visible)
overlaid_low = overlay(image, label, max_label_opacity=0.3)
visualize2d(overlaid_low, title="Opacity 0.3")

# High opacity (more label visible)
overlaid_high = overlay(image, label, max_label_opacity=0.7)
visualize2d(overlaid_high, title="Opacity 0.7")

# Full opacity (label only)
overlaid_full = overlay(image, label, max_label_opacity=1.0)
visualize2d(overlaid_full, title="Opacity 1.0")
```

### Custom Colorization

```python
from mipcandy import overlay, visualize2d, ColorizeLabel
import torch

image = torch.rand(256, 256)
label = torch.randint(0, 3, (256, 256))

# Custom colormap
colormap = [
    [255, 0, 0],    # Class 0: red
    [0, 255, 0],    # Class 1: green
    [0, 0, 255],    # Class 2: blue
]
colorizer = ColorizeLabel(colormap=colormap)

overlaid = overlay(image, label, label_colorizer=colorizer)
visualize2d(overlaid)
```

### Disable Colorization

```python
from mipcandy import overlay, visualize2d
import torch

image = torch.rand(256, 256)
label = torch.rand(256, 256)  # Continuous values

# No colorization (grayscale labels)
overlaid = overlay(image, label, label_colorizer=None)
visualize2d(overlaid)
```

### Multi-Class Segmentation

```python
from mipcandy import overlay, visualize2d
import torch

image = torch.rand(256, 256)
# Multi-class segmentation (background + 3 classes)
segmentation = torch.randint(0, 4, (256, 256))

# Default colorizer handles multiple classes
overlaid = overlay(image, segmentation)
visualize2d(overlaid, title="Multi-Class Segmentation")
```

### Alpha Channel Support

The [`ColorizeLabel`](#mipcandy.common.module.preprocess.ColorizeLabel) can return 4-channel output (RGB + alpha):

```python
from mipcandy import overlay, visualize2d, ColorizeLabel
import torch

image = torch.rand(256, 256)
label = torch.rand(256, 256)

# If colorizer returns 4 channels, the 4th channel is used as alpha
colorizer = ColorizeLabel()
overlaid = overlay(image, label, label_colorizer=colorizer)
visualize2d(overlaid)
```

:::{note}
When label values are in [0, 1] range, [`ColorizeLabel`](#mipcandy.common.module.preprocess.ColorizeLabel) automatically includes an alpha channel based on the label values.
:::

## Utilities

### auto_convert()

The [`auto_convert()`](#mipcandy.data.visualization.auto_convert) function normalizes tensors to [0, 255] integer range for display.

```python
from mipcandy import auto_convert
import torch

# Values in [0, 1] range
normalized = torch.rand(256, 256)
converted = auto_convert(normalized)
# Result: values scaled to [0, 255] and converted to int

# Values in arbitrary range
arbitrary = torch.randn(256, 256) * 100 + 50
converted = auto_convert(arbitrary)
# Result: normalized to [0, 255] using Normalize(domain=(0, 255))
```

**Behavior:**
- If `0 <= image.min() < image.max() <= 1`: multiply by 255
- Otherwise: apply [`Normalize(domain=(0, 255))`](#mipcandy.common.module.preprocess.Normalize)
- Convert to integer type

### Integration with Other Utilities

The visualization functions work seamlessly with other MIPCandy utilities:

```python
from mipcandy import (
    load_image,
    visualize2d,
    visualize3d,
    ensure_num_dimensions,
    orthographic_views
)

# Load and visualize
volume = load_image("volume.nii.gz")
visualize3d(volume)

# Visualize orthographic projections
depth_proj, height_proj, width_proj = orthographic_views(volume)
visualize2d(depth_proj[0], title="Depth Projection")
visualize2d(height_proj[0], title="Height Projection")
visualize2d(width_proj[0], title="Width Projection")

# Ensure correct dimensions before visualization
image = torch.rand(256, 256)
image_3d = ensure_num_dimensions(image, 3)  # Add channel dimension
visualize2d(image_3d)
```

### Combining with Normalization

```python
from mipcandy import Normalize, visualize2d
import torch

# Raw image with arbitrary range
image = torch.randn(256, 256) * 1000 + 500

# Manual normalization before visualization
normalizer = Normalize(domain=(0, 1))
normalized = normalizer(image)
visualize2d(normalized, title="Manually Normalized")

# Auto-normalization (handled by visualize2d)
visualize2d(image, title="Auto Normalized")
```
