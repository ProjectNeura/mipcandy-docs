# Utilities

MIPCandy provides utility functions for medical image I/O, geometric transformations, and format conversions.

## Overview

The utilities module includes:

- **I/O Operations**: Load and save medical images with automatic preprocessing
- **Geometric Transformations**: Dimension management, projections, and cropping
- **Format Conversion**: Convert between class IDs and logits for segmentation

```python
from mipcandy import (
    load_image, save_image, resample_to_isotropic, fast_save, fast_load, empty_cache,
    ensure_num_dimensions, orthographic_views, aggregate_orthographic_views, crop,
    convert_ids_to_logits, convert_logits_to_ids
)
```

## I/O Operations

### load_image()

```python
def load_image(path: str | PathLike[str], *, is_label: bool = False,
               align_spacing: bool = False, target_iso: float | None = None,
               device: Device = "cpu") -> torch.Tensor:
```

Load medical images from disk to PyTorch tensors.

#### Parameters

- `path`: File path (supports `.nii`, `.nii.gz`, `.mha`, `.png`, `.jpg`)
- `is_label`: Whether the image is a segmentation label (default: `False`)
  - `True`: Uses nearest neighbor interpolation
  - `False`: Uses B-spline interpolation
- `align_spacing`: Resample to isotropic spacing (default: `False`)
- `target_iso`: Target isotropic spacing when `align_spacing=True` (default: `None`, uses minimum original spacing)
- `device`: Target device (default: `"cpu"`)

#### Usage

```python
from mipcandy import load_image

# Load 3D volume
volume = load_image("scan.nii.gz")

# Load to GPU
volume_gpu = load_image("scan.nii.gz", device="cuda")

# Load label with isotropic resampling
label = load_image("label.nii.gz", is_label=True, align_spacing=True)
```

**Dimension handling:**
- NIfTI/MHA files: Returns 3D `(D, H, W)` or 4D `(C, D, H, W)`
- PNG/JPG files: Returns 3D `(C, H, W)`

### save_image()

```python
def save_image(image: torch.Tensor, path: str | PathLike[str]) -> None:
```

Save PyTorch tensors as medical image files.

#### Parameters

- `image`: Input tensor (automatically detached and moved to CPU)
- `path`: Output file path (format determined by extension)

#### Usage

```python
from mipcandy import save_image
import torch

# Save 3D volume
volume = torch.rand(128, 256, 256)
save_image(volume, "output.nii.gz")

# Save prediction
prediction = torch.randint(0, 4, (128, 256, 256))
save_image(prediction, "prediction.nii.gz")
```

### resample_to_isotropic()

```python
def resample_to_isotropic(image: SpITK.Image, *, target_iso: float | None = None,
                          interpolator: int = SpITK.sitkBSpline) -> SpITK.Image:
```

Resample medical images to isotropic voxel spacing.

#### Parameters

- `image`: Input SimpleITK image
- `target_iso`: Target isotropic spacing (default: minimum original spacing)
- `interpolator`: Interpolation method (default: `sitk.sitkBSpline`)

#### Usage

```python
import SimpleITK as sitk
from mipcandy import resample_to_isotropic

# Load with SimpleITK
image = sitk.ReadImage("scan.nii.gz")
print(image.GetSpacing())  # (0.5, 0.5, 2.0)

# Resample to isotropic
image_iso = resample_to_isotropic(image)
print(image_iso.GetSpacing())  # (0.5, 0.5, 0.5)

# Custom target spacing
image_1mm = resample_to_isotropic(image, target_iso=1.0)
print(image_1mm.GetSpacing())  # (1.0, 1.0, 1.0)

# Labels with nearest neighbor
label = sitk.ReadImage("label.nii.gz")
label_iso = resample_to_isotropic(label, interpolator=sitk.sitkNearestNeighbor)
```

### fast_save()

```python
def fast_save(x: torch.Tensor, path: str | PathLike[str]) -> None:
```

Save a tensor to disk in [safetensors](https://huggingface.co/docs/safetensors) format. The tensor is made contiguous before saving if needed.

#### Parameters

- `x`: Tensor to save
- `path`: Output file path (typically `.safetensors`)

#### Usage

```python
from mipcandy import fast_save
import torch

volume = torch.rand(128, 256, 256)
fast_save(volume, "volume.safetensors")
```

### fast_load()

```python
def fast_load(path: str | PathLike[str], *, device: Device = "cpu") -> torch.Tensor:
```

Load a tensor from a safetensors file.

#### Parameters

- `path`: Path to the safetensors file
- `device`: Target device (default: `"cpu"`)

#### Usage

```python
from mipcandy import fast_load

volume = fast_load("volume.safetensors")
volume_gpu = fast_load("volume.safetensors", device="cuda")
```

:::{tip}
`fast_save` / `fast_load` use the safetensors format, which is faster and safer than `torch.save` / `torch.load` (no pickle execution). Use them when you need to cache intermediate tensors to disk.
:::

### empty_cache()

```python
def empty_cache(device: Device) -> None:
```

Clear memory caches for the specified device. Dispatches to the appropriate backend:

- `"cpu"`: Calls `gc.collect()`
- `"cuda"`: Calls `torch.cuda.empty_cache()`
- `"mps"`: Calls `torch.mps.empty_cache()`

#### Parameters

- `device`: Device whose cache to clear (`"cpu"`, `"cuda"`, or `"mps"`)

#### Usage

```python
from mipcandy import empty_cache

# After releasing large tensors
del large_tensor
empty_cache("cuda")
```

## Geometric Transformations

### ensure_num_dimensions()

```python
def ensure_num_dimensions(x: torch.Tensor, num_dimensions: int, *, append_before: bool = True) -> torch.Tensor:
```

Adjust tensor dimensions by adding or removing dimensions.

#### Parameters

- `x`: Input tensor
- `num_dimensions`: Target number of dimensions
- `append_before`: When `True` (default), adds/removes dimensions at the front. When `False`, adds/removes at the end.

#### Behavior

When `append_before=True` (default):
- **Add dimensions**: Adds leading dimensions of size 1
- **Remove dimensions**: Keeps trailing dimensions

When `append_before=False`:
- **Add dimensions**: Adds trailing dimensions of size 1
- **Remove dimensions**: Keeps leading dimensions

#### Usage

```python
from mipcandy import ensure_num_dimensions
import torch

# Add channel dimension
image_2d = torch.rand(256, 256)
image_3d = ensure_num_dimensions(image_2d, 3)
print(image_3d.shape)  # (1, 256, 256)

# Remove batch and channel dimensions
batch = torch.rand(2, 3, 64, 128, 128)
volume = ensure_num_dimensions(batch, 3)
print(volume.shape)  # (64, 128, 128)
```

### orthographic_views()

```python
def orthographic_views(x: torch.Tensor, reduction: Literal["mean", "sum"] = "mean") -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
```

Generate three orthogonal 2D projections from a 3D volume.

#### Parameters

- `x`: Input tensor (typically 3D or higher)
- `reduction`: Reduction method - `"mean"` or `"sum"` (default: `"mean"`)

#### Returns

Three projections: `(depth_proj, height_proj, width_proj)`

#### Usage

```python
from mipcandy import orthographic_views, visualize2d
import torch

# 3D volume
volume = torch.rand(64, 128, 128)

# Generate projections
axial, coronal, sagittal = orthographic_views(volume)
print(axial.shape)    # (128, 128) - projection along depth
print(coronal.shape)  # (64, 128)  - projection along height
print(sagittal.shape) # (64, 128)  - projection along width

# Visualize
visualize2d(axial, title="Axial View")
visualize2d(coronal, title="Coronal View")
visualize2d(sagittal, title="Sagittal View")
```

### aggregate_orthographic_views()

```python
def aggregate_orthographic_views(d: torch.Tensor, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
```

Reconstruct a 3D volume from three orthogonal projections.

#### Parameters

- `d`: Depth projection (axial view)
- `h`: Height projection (coronal view)
- `w`: Width projection (sagittal view)

#### Returns

Reconstructed 3D tensor computed as outer product of the three projections.

#### Usage

```python
from mipcandy import orthographic_views, aggregate_orthographic_views
import torch

# Original volume
volume = torch.rand(64, 128, 128)

# Generate projections
d, h, w = orthographic_views(volume)

# Reconstruct
reconstructed = aggregate_orthographic_views(d, h, w)
print(reconstructed.shape)  # (64, 128, 128)
```

:::{note}
The reconstruction is an approximation based on outer product, not an exact recovery of the original volume.
:::

### crop()

```python
def crop(t: torch.Tensor, bbox: Sequence[int]) -> torch.Tensor:
```

Extract a region of interest using a bounding box.

#### Parameters

- `t`: Input tensor `(B, C, H, W)` or `(B, C, D, H, W)`
- `bbox`: Bounding box coordinates
  - 2D: `[y_start, y_end, x_start, x_end]`
  - 3D: `[z_start, z_end, y_start, y_end, x_start, x_end]`

#### Usage

```python
from mipcandy import crop
import torch

# 2D crop
image = torch.rand(1, 3, 512, 512)
bbox_2d = [100, 300, 150, 350]
cropped = crop(image, bbox_2d)
print(cropped.shape)  # (1, 3, 200, 200)

# 3D crop
volume = torch.rand(1, 1, 128, 256, 256)
bbox_3d = [20, 100, 50, 200, 60, 220]
cropped = crop(volume, bbox_3d)
print(cropped.shape)  # (1, 1, 80, 150, 160)
```

## Format Conversion

### convert_ids_to_logits()

```python
def convert_ids_to_logits(ids: torch.Tensor, d: Literal[1, 2, 3], num_classes: int) -> torch.Tensor:
```

Convert class ID tensors to one-hot encoded logits.

#### Parameters

- `ids`: Class ID tensor (integer type, non-negative values)
- `d`: Spatial dimensionality - `1`, `2`, or `3`
- `num_classes`: Number of classes

#### Returns

One-hot encoded tensor with shape `(B, num_classes, *spatial_dims)`

**Input format:** The `ids` tensor must include a batch dimension. For `d=2`, the expected shape is `(B, H, W)`; for `d=3`, `(B, D, H, W)`. The tensor must have dtype `torch.int32`.

#### Usage

```python
from mipcandy import convert_ids_to_logits
import torch

# 2D segmentation (batch of 2, 256x256)
ids = torch.randint(0, 3, (2, 256, 256), dtype=torch.int32)
logits = convert_ids_to_logits(ids, d=2, num_classes=3)
print(logits.shape)  # (2, 3, 256, 256)

# 3D segmentation (batch of 1, 64x128x128)
ids_3d = torch.randint(0, 4, (1, 64, 128, 128), dtype=torch.int32)
logits_3d = convert_ids_to_logits(ids_3d, d=3, num_classes=4)
print(logits_3d.shape)  # (1, 4, 64, 128, 128)

# Verify one-hot encoding
assert (logits.sum(dim=1) == 1).all()
```

### convert_logits_to_ids()

```python
def convert_logits_to_ids(logits: torch.Tensor, *, channel_dim: int = 1) -> torch.Tensor:
```

Convert model output logits to predicted class IDs.

#### Parameters

- `logits`: Model output tensor `(..., C, *spatial_dims)`
- `channel_dim`: Channel dimension for argmax (default: `1`)

#### Returns

Class ID tensor. For multi-channel inputs (`C >= 2`), computes `argmax` along `channel_dim` with `keepdim=True`. For single-channel inputs (`C < 2`), rounds values and converts to `torch.int32`.

#### Usage

```python
from mipcandy import convert_logits_to_ids
import torch

# Multi-class 2D segmentation (argmax branch)
logits = torch.randn(1, 3, 256, 256)
ids = convert_logits_to_ids(logits)
print(ids.shape)  # (1, 1, 256, 256) -- channel dim kept with size 1

# Multi-class 3D segmentation
logits_3d = torch.randn(2, 4, 64, 128, 128)
ids_3d = convert_logits_to_ids(logits_3d)
print(ids_3d.shape)  # (2, 1, 64, 128, 128)

# Single-channel (binary) segmentation (round branch)
logits_binary = torch.randn(1, 1, 256, 256)
ids_binary = convert_logits_to_ids(logits_binary)
print(ids_binary.shape)  # (1, 1, 256, 256)
print(ids_binary.dtype)  # torch.int32
```
