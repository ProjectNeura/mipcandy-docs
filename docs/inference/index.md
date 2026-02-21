# Inference

MIPCandy provides a flexible inference system centered around the [`Predictor`](#mipcandy.inference.Predictor) class, enabling prediction on various input formats including files, directories, tensors, and datasets.

## Overview

The inference module supports:

- **Flexible Input**: Files, directories, tensors, sequences, or datasets
- **Lazy Model Loading**: Models load only when first prediction is requested
- **Device Management**: Automatic device placement and memory handling
- **Batch Processing**: Efficient batch prediction with automatic padding
- **Easy Export**: Save predictions to files with automatic naming

## Quick Start

```python
from mipcandy_bundles.unet import UNetPredictor

# Create predictor from trained model
predictor = UNetPredictor(
    "experiments/UNet/20240901-1234",
    example_shape=(1, 128, 128),
    device="cuda"
)

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
from typing import override
from torch import nn
from mipcandy.inference import Predictor
from mipcandy.types import AmbiguousShape

class MyPredictor(Predictor):
    @override
    def build_network(self, example_shape: AmbiguousShape) -> nn.Module:
        # Build network architecture based on example_shape
        model = MyNetwork(in_channels=example_shape[0])
        return model

# Usage
predictor = MyPredictor(
    "experiments/MyModel/20240901-1234",
    example_shape=(1, 128, 128),
    checkpoint="checkpoint_best.pth",
    device="cuda"
)
```

`build_network` is responsible only for constructing the model architecture. Checkpoint loading is handled automatically by the framework via safetensors -- you do not need to call `load_state_dict` yourself.

**Parameters:**
- `experiment_folder`: Path to trainer output directory
- `example_shape`: A tuple describing the shape of a single input (e.g. `(1, 128, 128)` for single-channel 2D or `(1, 128, 128, 128)` for single-channel 3D). Used by `build_network` to configure the architecture.
- `checkpoint`: Checkpoint filename (default: `"checkpoint_best.pth"`)
- `device`: Computing device (default: `"cpu"`)

### Lazy Model Loading

Models are loaded only when first needed, saving memory when predictor is created but not immediately used:

```python
# Predictor created but model not loaded yet
predictor = MyPredictor("experiments/model", (1, 128, 128), device="cuda")

# Model loads on first prediction
output = predictor.predict("image.nii.gz")  # Model loaded here

# Subsequent predictions reuse loaded model
output2 = predictor.predict("image2.nii.gz")  # Model already loaded
```

To explicitly load the model:

```python
predictor = MyPredictor("experiments/model", (1, 128, 128), device="cuda")
predictor.lazy_load_model()  # Explicitly load model
```

### Model Loading Flow

Internally, when a prediction is first requested, `lazy_load_model` triggers the following chain:

1. `build_network(example_shape)` -- constructs the model architecture
2. `load_checkpoint(model, path)` -- loads weights from the safetensors checkpoint file
3. The model is moved to the configured device and set to eval mode

This separation means `build_network` should return a freshly initialized model without loading any weights.

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

**Parameters:**
- `x`: The input to parse (`SupportedPredictant`)
- `loader`: The loader class to use for file loading (typically `Loader`)
- `as_label`: Whether to load as label data (default: `False`)

**Returns:** `tuple[list[torch.Tensor], list[str] | None]` -- a list of tensors and optionally a list of corresponding filenames.

## Prediction Methods

### predict()

Predict and return outputs as tensors:

```python
predictor = MyPredictor("experiments/model", (1, 128, 128), device="cuda")

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

When `batch=False`, the input is automatically unsqueezed before inference and squeezed back after. Padding and restoring modules (if configured) are applied transparently.

### predict_to_files()

Predict and save directly to files:

```python
# Predict directory and save
filenames = predictor.predict_to_files(
    "input_images/",
    "output_predictions/"
)
# Saves predictions with original filenames
# Returns: list of filenames used, or None

# Custom filenames via save_predictions
outputs = predictor.predict("images/")
predictor.save_predictions(
    outputs,
    "output/",
    filenames=["pred_001.nii.gz", "pred_002.nii.gz"]
)
```

**Returns:** `list[str] | None` -- the filenames used for saving, or `None` if inputs were tensors without associated filenames.

### Callable Interface

Predictors can be called directly:

```python
predictor = MyPredictor("experiments/model", (1, 128, 128), device="cuda")

# Equivalent to predictor.predict()
outputs = predictor("images/")
```

## Padding and Restoration

### Automatic Padding

Predictors can optionally implement padding for inputs that don't match required dimensions:

```python
from typing import override
import torch
from torch import nn
from mipcandy.inference import Predictor
from mipcandy.common import Pad2d, Restore2d
from mipcandy.types import AmbiguousShape

class PaddedPredictor(Predictor):
    @override
    def build_network(self, example_shape: AmbiguousShape) -> nn.Module:
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
predictor = PaddedPredictor("experiments/model", (1, 128, 128), device="cuda")

# Input: 100x100
# Automatically padded to 128x128
# Processed by model
# Automatically restored to 100x100
output = predictor.predict_image(torch.randn(1, 100, 100).cuda())
```

The padding and restoring modules are lazily loaded and cached for efficiency. Both modules are automatically moved to the predictor's configured device.

The default implementations of `build_padding_module` and `build_restoring_module` return `None`, meaning no padding is applied unless explicitly overridden.

## Dataset Integration

Predictors work seamlessly with datasets:

```python
from mipcandy.data import SimpleDataset, PathBasedUnsupervisedDataset

# Create dataset
dataset = SimpleDataset("test_images/", is_label=False, device="cuda")

# Predict entire dataset
outputs = predictor.predict(dataset)

# Process dataset case by case
for i, image in enumerate(dataset):
    output = predictor.predict_image(image)
    predictor.save_prediction(output, f"outputs/case_{i:03d}.nii.gz")
```

When a `PathBasedUnsupervisedDataset` is passed to `predict` or `_predict`, the predictor automatically extracts the file paths from the dataset for use as output filenames.

## Saving Predictions

### save_prediction()

Save a single prediction:

```python
output = predictor.predict_image(image)
Predictor.save_prediction(output, "output.nii.gz")
```

This is a static method that delegates to `save_image`. It can be called on the class directly or on an instance.

### save_predictions()

Save multiple predictions with automatic or custom naming:

```python
outputs = predictor.predict("images/")

# Automatic naming: prediction_00, prediction_01, ...
predictor.save_predictions(outputs, "output_folder/")

# Custom filenames
predictor.save_predictions(
    outputs,
    "output_folder/",
    filenames=["case1.nii.gz", "case2.nii.gz"]
)
```

:::{important}
The output folder must already exist. `save_predictions` raises `FileNotFoundError` if the folder does not exist.
:::

**Automatic Naming Format:** `prediction_{i:0Nd}` where `N = ceil(ln(num_cases))` (natural logarithm).

The file extension is chosen automatically based on tensor dimensionality:
- `.png` for 3D tensors with 1 or 3 channels (2D images)
- `.mha` for all other shapes (3D volumes)

Example:
- 5 cases (`N=2`): `prediction_00` to `prediction_04`
- 100 cases (`N=5`): `prediction_00000` to `prediction_00099`
- 1000 cases (`N=7`): `prediction_0000000` to `prediction_0000999`

## Complete Example

```python
from typing import override
from os import PathLike
import torch
from torch import nn
from mipcandy.inference import Predictor
from mipcandy.types import AmbiguousShape, Device

class UNetPredictor(Predictor):
    def __init__(self, experiment_folder: str | PathLike[str], example_shape: AmbiguousShape, *,
                 checkpoint: str = "checkpoint_best.pth",
                 device: Device = "cuda") -> None:
        super().__init__(experiment_folder, example_shape, checkpoint=checkpoint, device=device)
        self.num_classes: int = 1

    @override
    def build_network(self, example_shape: AmbiguousShape) -> nn.Module:
        from my_models import UNet

        model = UNet(
            in_channels=example_shape[0],
            num_classes=self.num_classes
        )
        return model

# Inference pipeline
predictor = UNetPredictor(
    "experiments/UNet/20240901-1234",
    example_shape=(1, 128, 128),
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
