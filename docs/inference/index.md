# Inference

MIPCandy provides a flexible inference system centered around the [`Predictor`](#mipcandy.inference.Predictor) class, enabling prediction on various input formats including files, directories, tensors, and datasets.

## Overview

The inference module supports:

- **Flexible Input**: Files, directories, tensors, sequences, or datasets
- **Lazy Model Loading**: Models load only when first prediction is requested
- **Memory Efficient**: Sliding window inference for large volumes
- **Device Management**: Automatic device placement and memory handling
- **Batch Processing**: Efficient batch prediction with automatic padding
- **Easy Export**: Save predictions to files with automatic naming

## Quick Start

```python
from mipcandy_bundles.unet import UNetPredictor

# Create predictor from trained model
predictor = UNetPredictor("experiments/UNet/20240901-1234", device="cuda")

# Predict single image
output = predictor.predict("path/to/image.nii.gz")

# Predict directory of images
outputs = predictor.predict("path/to/images/")

# Save predictions
predictor.predict_to_files("path/to/images/", "path/to/outputs/")
```

## Creating a Predictor

### Basic Predictor Implementation

To create a custom predictor, extend [`Predictor`](#mipcandy.inference.Predictor) and implement `build_network`:

```python
from typing import override, Mapping, Any
import torch
from torch import nn
from mipcandy.inference import Predictor

class MyPredictor(Predictor):
    @override
    def build_network(self, checkpoint: Mapping[str, Any]) -> nn.Module:
        # Build network architecture
        model = MyNetwork()

        # Load weights from checkpoint
        model.load_state_dict(checkpoint["model"])

        return model

# Usage
predictor = MyPredictor(
    "experiments/MyModel/20240901-1234",
    checkpoint="checkpoint_best.pth",
    device="cuda"
)
```

**Parameters:**
- `experiment_folder`: Path to trainer output directory
- `checkpoint`: Checkpoint filename (default: `"checkpoint_best.pth"`)
- `device`: Computing device (default: `"cpu"`)

### Lazy Model Loading

Models are loaded only when first needed, saving memory when predictor is created but not immediately used:

```python
# Predictor created but model not loaded yet
predictor = MyPredictor("experiments/model", device="cuda")

# Model loads on first prediction
output = predictor.predict("image.nii.gz")  # Model loaded here

# Subsequent predictions reuse loaded model
output2 = predictor.predict("image2.nii.gz")  # Model already loaded
```

To explicitly load the model:

```python
predictor = MyPredictor("experiments/model", device="cuda")
predictor.lazy_load_model()  # Explicitly load model
```

## Input Formats

### parse_predictant

The [`parse_predictant`](#mipcandy.inference.parse_predictant) function handles various input types:

```python
from mipcandy.inference import parse_predictant
from mipcandy.data import Loader

# Single file
images, filenames = parse_predictant("image.nii.gz", Loader)
# images: list with 1 tensor
# filenames: ["image.nii.gz"]

# Directory
images, filenames = parse_predictant("images/", Loader)
# images: list with N tensors (one per file in directory)
# filenames: list of filenames

# Single tensor
tensor = torch.randn(1, 128, 128)
images, filenames = parse_predictant(tensor, Loader)
# images: [tensor]
# filenames: None

# List of files
images, filenames = parse_predictant(["img1.nii.gz", "img2.nii.gz"], Loader)
# images: list with 2 tensors
# filenames: ["img1.nii.gz", "img2.nii.gz"]

# List of tensors
images, filenames = parse_predictant([tensor1, tensor2], Loader)
# images: [tensor1, tensor2]
# filenames: None
```

:::{important}
All elements in a sequence must have the same type (all strings or all tensors).
:::

## Prediction Methods

### predict()

Predict and return outputs as tensors:

```python
predictor = MyPredictor("experiments/model", device="cuda")

# Single image
output = predictor.predict("image.nii.gz")
# Returns: list[torch.Tensor] with 1 element

# Multiple images
outputs = predictor.predict("images_directory/")
# Returns: list[torch.Tensor] with N elements

# Tensors
tensor = torch.randn(1, 128, 128).cuda()
outputs = predictor.predict(tensor)
# Returns: list[torch.Tensor]
```

### predict_image()

Predict on a single tensor with optional batching:

```python
# Single image (no batch dimension)
image = torch.randn(1, 128, 128).cuda()
output = predictor.predict_image(image, batch=False)
# Input shape: (C, H, W)
# Output shape: (C, H, W)

# Batch of images
images = torch.randn(4, 1, 128, 128).cuda()
outputs = predictor.predict_image(images, batch=True)
# Input shape: (B, C, H, W)
# Output shape: (B, C, H, W)
```

**Parameters:**
- `image`: Input tensor (with or without batch dimension)
- `batch`: Whether input has batch dimension (default: `False`)

### predict_to_files()

Predict and save directly to files:

```python
# Predict directory and save
predictor.predict_to_files(
    "input_images/",
    "output_predictions/"
)
# Saves predictions with original filenames

# Custom filenames
outputs = predictor.predict("images/")
predictor.save_predictions(
    outputs,
    "output/",
    filenames=["pred_001.nii.gz", "pred_002.nii.gz"]
)
```

### Callable Interface

Predictors can be called directly:

```python
predictor = MyPredictor("experiments/model", device="cuda")

# Equivalent to predictor.predict()
outputs = predictor("images/")
```

## Padding and Restoration

### Automatic Padding

Predictors can optionally implement padding for inputs that don't match required dimensions:

```python
from typing import override
from torch import nn
from mipcandy.inference import Predictor
from mipcandy.common import Pad2d, Restore2d

class PaddedPredictor(Predictor):
    @override
    def build_network(self, checkpoint) -> nn.Module:
        return MyNetwork()

    @override
    def build_padding_module(self) -> nn.Module | None:
        # Pad to multiples of 128
        return Pad2d((128, 128))

    @override
    def build_restoring_module(self, padding_module: nn.Module | None) -> nn.Module | None:
        if padding_module:
            # Restore to original size
            return Restore2d(padding_module)
        return None

# Usage
predictor = PaddedPredictor("experiments/model", device="cuda")

# Input: 100x100
# Automatically padded to 128x128
# Processed by model
# Automatically restored to 100x100
output = predictor.predict_image(torch.randn(1, 100, 100).cuda())
```

The padding and restoring modules are lazily loaded and cached for efficiency.

## Sliding Window Inference

For large volumes that exceed GPU memory, use [`SlidingPredictor`](#mipcandy.inference.SlidingPredictor):

```python
from typing import override
from mipcandy.inference import SlidingPredictor

class MySlidingPredictor(SlidingPredictor):
    @override
    def build_network(self, checkpoint) -> nn.Module:
        return MyNetwork()

    @override
    def get_window_shape(self) -> tuple[int, ...]:
        # IMPORTANT: Return value is the stride (step size)
        # Actual window size will be 2x this value (fixed 50% overlap)
        return (128, 128, 128)  # stride=128, actual window=256x256x256

# Usage
predictor = MySlidingPredictor("experiments/model", device="cuda")

# Large volume (512x512x512) processed in 256x256x256 windows with stride 128
large_volume = torch.randn(1, 512, 512, 512).cuda()
output = predictor.predict_image(large_volume)
```

:::{important}
`get_window_shape()` returns the **stride** (step size), not the actual window size. The actual processing window is always **2x the returned value** with fixed 50% overlap:

```python
# If you return (128, 128, 128):
stride = (128, 128, 128)           # Step size between windows
actual_window = (256, 256, 256)    # Actual processing window size
overlap = 128 voxels per dimension # 50% overlap (fixed)
```
:::

### How Sliding Window Works

1. **Windowing**: Input volume split into overlapping windows of size `2 * get_window_shape()`
2. **Prediction**: Each window processed independently
3. **Reconstruction**: Windows merged with Gaussian-weighted averaging in overlap regions
4. **Padding**: Automatic padding to ensure complete coverage

### Window Configuration

```python
class MySlidingPredictor(SlidingPredictor):
    @override
    def get_window_shape(self) -> tuple[int, ...]:
        # For 2D: return stride, actual window will be 2x
        return (128, 128)  # stride=128, actual window=256x256
```

**Common configurations:**

```python
# For 3D volumes with limited GPU memory
def get_window_shape(self):
    return (64, 64, 64)    # Actual window: 128x128x128

# For 3D volumes with ample GPU memory
def get_window_shape(self):
    return (128, 128, 128)  # Actual window: 256x256x256

# For 2D images
def get_window_shape(self):
    return (256, 256)       # Actual window: 512x512
```

### Automatic Padding for Sliding

`SlidingPredictor` automatically provides padding modules:

```python
# Automatically implemented
def build_padding_module(self) -> nn.Module | None:
    window_shape = self.get_window_shape()
    return Pad2d(window_shape) if len(window_shape) == 2 else Pad3d(window_shape)

def build_restoring_module(self, padding_module) -> nn.Module | None:
    return Restore2d(padding_module) if isinstance(padding_module, Pad2d) else Restore3d(padding_module)
```

## Dataset Integration

Predictors work seamlessly with datasets:

```python
from mipcandy import UnsupervisedDataset

# Create dataset
dataset = UnsupervisedDataset("test_images/", device="cuda")

# Predict entire dataset
outputs = predictor.predict(dataset)

# Process dataset case by case
for i, image in enumerate(dataset):
    output = predictor.predict_image(image)
    predictor.save_prediction(output, f"outputs/case_{i:03d}.nii.gz")
```

## Saving Predictions

### save_prediction()

Save a single prediction:

```python
output = predictor.predict_image(image)
predictor.save_prediction(output, "output.nii.gz")
```

### save_predictions()

Save multiple predictions with automatic or custom naming:

```python
outputs = predictor.predict("images/")

# Automatic naming: prediction_000, prediction_001, ...
predictor.save_predictions(outputs, "output_folder/")

# Custom filenames
predictor.save_predictions(
    outputs,
    "output_folder/",
    filenames=["case1.nii.gz", "case2.nii.gz"]
)
```

**Automatic Naming Format:** `prediction_{i:0Nd}` where `N = ceil(log10(num_cases))`

Example:
- 5 cases: `prediction_0` to `prediction_4`
- 100 cases: `prediction_00` to `prediction_99`
- 1000 cases: `prediction_000` to `prediction_999`

## Complete Example

```python
from typing import override, Mapping, Any
from os import PathLike
from torch import nn
from mipcandy.inference import SlidingPredictor

class UNetPredictor(SlidingPredictor):
    def __init__(self, experiment_folder: str | PathLike[str], *, checkpoint: str = "checkpoint_best.pth", 
                 device: str | torch.device | None = "cuda") -> None:
        super().__init__(experiment_folder, checkpoint=checkpoint, device=device)
        self.num_classes: int = 1

    @override
    def build_network(self, checkpoint: Mapping[str, Any]) -> nn.Module:
        from my_models import UNet

        # Extract model configuration from checkpoint
        model = UNet(
            in_channels=checkpoint["in_channels"],
            num_classes=self.num_classes
        )

        # Load trained weights
        model.load_state_dict(checkpoint["model"])

        return model

    @override
    def get_window_shape(self) -> tuple[int, int, int]:
        return (128, 128, 128)

# Inference pipeline
predictor = UNetPredictor(
    "experiments/UNet/20240901-1234",
    checkpoint="checkpoint_best.pth",
    device="cuda"
)

# Process test dataset
predictor.predict_to_files(
    "data/test_images/",
    "results/predictions/"
)

# Get predictions as tensors for further processing
outputs = predictor.predict("data/test_images/")
for i, output in enumerate(outputs):
    # Post-process predictions
    binary_mask = (output > 0.5).float()

    # Save processed result
    predictor.save_prediction(binary_mask, f"results/binary/case_{i:03d}.nii.gz")
```
