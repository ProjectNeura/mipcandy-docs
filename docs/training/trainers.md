# Trainers

A [`Trainer`](#mipcandy.training.Trainer) is where you define the training loop logic. It handles the entire training pipeline, from model initialization to checkpoint management, metrics tracking, and validation.

## Overview

The trainer system in MIPCandy follows a hierarchical design:

1. **Base Trainer** - Abstract base class defining the training framework
2. **Sliding Trainer** - Extends base trainer with sliding window mechanism for large volumes
3. **Segmentation Trainer** - Pre-configured trainer for segmentation tasks
4. **Sliding Segmentation Trainer** - Combines sliding window with segmentation features
5. **Model-specific Trainers** - Ready-to-use trainers like UNetTrainer and CMUNeXtTrainer

## TrainerToolbox

[`TrainerToolbox`](#mipcandy.training.TrainerToolbox) is a dataclass that bundles all essential training components together:

```python
from dataclasses import dataclass
from torch import nn, optim

@dataclass
class TrainerToolbox:
    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler
    criterion: nn.Module
    ema: optim.swa_utils.AveragedModel | None = None
```

This toolbox is passed to training methods, providing clean access to all components needed during the forward and backward passes.

## Base Trainer

The base [`Trainer`](#mipcandy.training.Trainer) class provides a complete training framework. You need to implement several abstract methods to create a custom trainer:

### Required Abstract Methods

#### `build_network(example_shape: tuple[int, ...]) -> nn.Module`

Constructs the neural network architecture based on the input shape.

```python
from typing import override
import torch
from torch import nn
from mipcandy import Trainer

class MyTrainer(Trainer):
    @override
    def build_network(self, example_shape: tuple[int, ...]) -> nn.Module:
        # example_shape is (C, H, W) for 2D or (C, D, H, W) for 3D
        in_channels = example_shape[0]
        return MyNetwork(in_channels, num_classes=1)
```

#### `build_optimizer(params: Params) -> optim.Optimizer`

Creates the optimizer for training.

```python
from torch import optim
from mipcandy import Params

@override
def build_optimizer(self, params: Params) -> optim.Optimizer:
    return optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
```

#### `build_scheduler(optimizer: optim.Optimizer, num_epochs: int) -> optim.lr_scheduler.LRScheduler`

Creates the learning rate scheduler.

```python
from mipcandy import AbsoluteLinearLR

@override
def build_scheduler(self, optimizer: optim.Optimizer, num_epochs: int) -> optim.lr_scheduler.LRScheduler:
    # Linear decay: lr = kx + b
    return AbsoluteLinearLR(optimizer, k=-8e-6 / len(self._dataloader), b=1e-2)
```

:::{tip}
[`AbsoluteLinearLR`](#mipcandy.common.optim.lr_scheduler.AbsoluteLinearLR) implements `lr = kx + b` with optional minimum learning rate and restart capability.
:::

#### `build_criterion() -> nn.Module`

Creates the loss function.

```python
from mipcandy import DiceBCELossWithLogits

@override
def build_criterion(self) -> nn.Module:
    return DiceBCELossWithLogits(num_classes=1)
```

#### `backward(images, labels, toolbox) -> tuple[float, dict[str, float]]`

Defines the forward pass and loss computation during training.

```python
@override
def backward(self, images: torch.Tensor, labels: torch.Tensor,
             toolbox: TrainerToolbox) -> tuple[float, dict[str, float]]:
    predictions = toolbox.model(images)
    loss, metrics = toolbox.criterion(predictions, labels)
    loss.backward()
    return loss.item(), metrics
```

:::{important}
The backward method should call `loss.backward()` but NOT call `optimizer.step()`. The base trainer handles optimizer stepping automatically.
:::

#### `validate_case(image, label, toolbox) -> tuple[float, dict[str, float], torch.Tensor]`

Validates a single case (without batch dimension).

```python
@override
def validate_case(self, image: torch.Tensor, label: torch.Tensor,
                  toolbox: TrainerToolbox) -> tuple[float, dict[str, float], torch.Tensor]:
    image, label = image.unsqueeze(0), label.unsqueeze(0)
    with torch.no_grad():
        prediction = (toolbox.ema if toolbox.ema else toolbox.model)(image)
        loss, metrics = toolbox.criterion(prediction, label)
    return -loss.item(), metrics, prediction.squeeze(0)
```

:::{note}
The validation score is typically the negative loss (higher is better). The trainer tracks the best score for checkpoint saving.
:::

### Experiment Management

The trainer automatically manages experiments:

```python
from mipcandy_bundles.unet import UNetTrainer
from torch.utils.data import DataLoader

trainer = UNetTrainer("experiments", train_loader, val_loader, device="cuda")
trainer.train(100)
```

This creates a timestamped experiment folder:
```
experiments/
└── UNetTrainer/
    └── 20251101-14-a3f2/
        ├── logs.txt
        ├── metrics.csv
        ├── checkpoint_best.pth
        ├── checkpoint_latest.pth
        ├── progress.png
        └── ...
```

The experiment ID format is `YYYYMMDD-HH-XXXX` where `XXXX` is a 4-character hash ensuring uniqueness.

### Logging System

The trainer provides a logging system that writes to both console and file:

```python
self.log("Custom message")  # Logs to console and logs.txt
self.log("Internal message", on_screen=False)  # Only to logs.txt
```

### Metrics Tracking

Record metrics during training:

```python
self.record("loss", 0.5)  # Record single metric
self.record_all({"dice": 0.85, "iou": 0.78})  # Record multiple metrics
```

Metrics are automatically:
- Averaged per epoch
- Saved to `metrics.csv`
- Visualized as curves in PNG files

### Checkpointing

The trainer automatically saves:
- **Latest checkpoint**: After every epoch
- **Best checkpoint**: When validation score improves
- **Periodic checkpoints**: At intervals (e.g., every 20 epochs if `num_checkpoints=5` and `num_epochs=100`)

```python
trainer.train(100, num_checkpoints=10)  # Saves 10 evenly spaced checkpoints
```

## Custom Trainer Example

Here's a complete example of a custom trainer:

```python
from typing import override
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from mipcandy import Trainer, TrainerToolbox, Params, DiceBCELossWithLogits, AbsoluteLinearLR


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.encoder: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.decoder: nn.Module = nn.Conv2d(64, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.decoder(features)


class MySegmentationTrainer(Trainer):
    num_classes: int = 1

    @override
    def build_network(self, example_shape: tuple[int, ...]) -> nn.Module:
        return SimpleUNet(example_shape[0], self.num_classes)

    @override
    def build_optimizer(self, params: Params) -> optim.Optimizer:
        return optim.AdamW(params, lr=1e-3)

    @override
    def build_scheduler(self, optimizer: optim.Optimizer, num_epochs: int) -> optim.lr_scheduler.LRScheduler:
        return AbsoluteLinearLR(optimizer, -8e-6 / len(self._dataloader), 1e-2)

    @override
    def build_criterion(self) -> nn.Module:
        return DiceBCELossWithLogits(self.num_classes)

    @override
    def backward(self, images: torch.Tensor, labels: torch.Tensor,
                 toolbox: TrainerToolbox) -> tuple[float, dict[str, float]]:
        predictions = toolbox.model(images)
        loss, metrics = toolbox.criterion(predictions, labels)
        loss.backward()
        return loss.item(), metrics

    @override
    def validate_case(self, image: torch.Tensor, label: torch.Tensor,
                      toolbox: TrainerToolbox) -> tuple[float, dict[str, float], torch.Tensor]:
        image, label = image.unsqueeze(0), label.unsqueeze(0)
        with torch.no_grad():
            prediction = (toolbox.ema if toolbox.ema else toolbox.model)(image)
            loss, metrics = toolbox.criterion(prediction, label)
        return -loss.item(), metrics, prediction.squeeze(0)


# Usage
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
trainer = MySegmentationTrainer("experiments", train_loader, val_loader, device="cuda")
trainer.train(100)
```

## Segmentation Trainer

[`SegmentationTrainer`](#mipcandy.presets.segmentation.SegmentationTrainer) provides pre-configured defaults for segmentation tasks. It implements all required methods with sensible defaults:

### Pre-configured Components

- **Loss Function**: [`DiceBCELossWithLogits`](#mipcandy.common.optim.loss.DiceBCELossWithLogits) - Combines Dice loss and Binary Cross Entropy
- **Optimizer**: `AdamW` with default parameters
- **Scheduler**: [`AbsoluteLinearLR`](#mipcandy.common.optim.lr_scheduler.AbsoluteLinearLR) with linear decay
- **Preview Generation**: Automatic 2D/3D visualization with overlays

### Preview Visualization

The segmentation trainer automatically generates preview images comparing expected vs. actual predictions:

```python
@override
def save_preview(self, image: torch.Tensor, label: torch.Tensor, output: torch.Tensor,
                 *, quality: float = .75) -> None:
    # Saves:
    # - input (preview).png
    # - label (preview).png
    # - prediction (preview).png
    # - expected (preview).png (overlay of image + label)
    # - actual (preview).png (overlay of image + prediction)
```

:::{tip}
For 3D volumes, set `preview_quality` to control the maximum number of voxels rendered (default: 0.75 million voxels).
:::

### Customization

You only need to implement `build_network()`:

```python
from typing import override
from torch import nn
from mipcandy import SegmentationTrainer


class MySegTrainer(SegmentationTrainer):
    num_classes: int = 3  # Multi-class segmentation

    @override
    def build_network(self, example_shape: tuple[int, ...]) -> nn.Module:
        return MyNetwork(example_shape[0], self.num_classes)
```

## Sliding Window Trainer

[`SlidingTrainer`](#mipcandy.training.SlidingTrainer) extends the base trainer with sliding window capability for processing large medical volumes that don't fit in GPU memory.

### How It Works

1. **Windowing**: The large volume is split into overlapping windows
2. **Processing**: Each window is processed independently
3. **Reconstruction**: Windows are merged back using Gaussian-weighted averaging

### Window Shape

Define the window size by implementing `get_window_shape()`:

```python
from typing import override
from mipcandy import SlidingTrainer


class MySlidingTrainer(SlidingTrainer):
    @override
    def get_window_shape(self) -> tuple[int, int] | tuple[int, int, int]:
        return (128, 128)  # 2D windows of 128x128
        # or
        # return (64, 128, 128)  # 3D windows
```

### Sliding Window Methods

Instead of `backward()` and `validate_case()`, implement the windowed versions:

#### `backward_windowed(images, labels, toolbox, metadata) -> tuple[float, dict[str, float]]`

Processes a batch of windows during training.

```python
@override
def backward_windowed(self, images: torch.Tensor, labels: torch.Tensor,
                      toolbox: TrainerToolbox, metadata: SWMetadata) -> tuple[float, dict[str, float]]:
    # images: (B*N, C, H, W) where N is number of windows
    predictions = toolbox.model(images)
    loss, metrics = toolbox.criterion(predictions, labels)
    loss.backward()
    return loss.item(), metrics
```

#### `validate_case_windowed(images, labels, toolbox, metadata) -> tuple[float, dict[str, float], torch.Tensor]`

Validates all windows from a single volume.

```python
@override
def validate_case_windowed(self, images: torch.Tensor, labels: torch.Tensor,
                           toolbox: TrainerToolbox, metadata: SWMetadata) -> tuple[float, dict[str, float], torch.Tensor]:
    with torch.no_grad():
        predictions = (toolbox.ema if toolbox.ema else toolbox.model)(images)
        loss, metrics = toolbox.criterion(predictions, labels)
    # Return predictions for all windows; base class handles reconstruction
    return -loss.item(), metrics, predictions
```

### SWMetadata

[`SWMetadata`](#mipcandy.sliding_window.SWMetadata) contains information about the sliding window operation:

```python
@dataclass
class SWMetadata:
    kernel: tuple[int, int] | tuple[int, int, int]  # Window size
    stride: tuple[int, int] | tuple[int, int, int]  # Step size
    ndim: Literal[2, 3]  # Number of dimensions
    batch_size: int  # Original batch size
    out_size: tuple[int, int] | tuple[int, int, int]  # Output volume size
    n: int  # Number of windows
```

### Gaussian Weighting

The trainer uses Gaussian weighting to smoothly blend overlapping windows:

```python
# 1D Gaussian for each dimension
g = exp(-0.5 * (x / sigma)^2)

# 2D weight: outer product of 1D Gaussians
w2d = g_h[:, None] * g_w[None, :]

# Final reconstruction: weighted sum / weight sum
output = numerator / denominator
```

This prevents artifacts at window boundaries.

## Sliding Segmentation Trainer

[`SlidingSegmentationTrainer`](#mipcandy.presets.segmentation.SlidingSegmentationTrainer) combines the sliding window mechanism with segmentation-specific features.

### Configuration

```python
from typing import override
from torch import nn
from mipcandy import SlidingSegmentationTrainer


class MySlidingSegTrainer(SlidingSegmentationTrainer):
    num_classes: int = 1
    sliding_window_shape: tuple[int, int] = (256, 256)  # Override default (128, 128)

    @override
    def build_network(self, example_shape: tuple[int, ...]) -> nn.Module:
        return MyNetwork(example_shape[0], self.num_classes)
```

### Usage Example

```python
from mipcandy import SlidingSegmentationTrainer, NNUNetDataset
from torch.utils.data import DataLoader


class MyTrainer(SlidingSegmentationTrainer):
    sliding_window_shape: tuple[int, int] = (192, 192)  # Custom window size

    @override
    def build_network(self, example_shape: tuple[int, ...]) -> nn.Module:
        return MyLargeNetwork(example_shape[0], num_classes=1)


dataset, val_dataset = NNUNetDataset("path/to/dataset").fold()
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

trainer = MyTrainer("experiments", train_loader, val_loader, device="cuda")
trainer.train(200, note="Large volume segmentation with sliding windows")
```

## Predefined Trainers

MIPCandy Bundles provides ready-to-use trainers for popular architectures.

:::{tip}
`mipcandy_bundles` needs to be installed separately or with `"mipcandy[all]"`.

```shell
pip install "mipcandy[all]"
```
:::

### UNetTrainer

[`UNetTrainer`](#mipcandy_bundles.unet.unet_trainer.UNetTrainer) provides a U-Net implementation supporting both 2D and 3D segmentation.

```python
from mipcandy_bundles.unet import UNetTrainer
from torch.utils.data import DataLoader

trainer = UNetTrainer("experiments", train_loader, val_loader, device="cuda")
trainer.num_classes = 3  # Multi-class segmentation
trainer.num_dims = 2  # 2D segmentation (default)
# trainer.num_dims = 3  # For 3D segmentation
trainer.train(100)
```

**Attributes**:
- `num_dims: int` - Set to `2` for 2D or `3` for 3D segmentation (default: 2)
- `num_classes: int` - Number of segmentation classes (default: 1)

### CMUNeXtTrainer

[`CMUNeXtTrainer`](#mipcandy_bundles.cmunext.cmunext_trainer.CMUNeXtTrainer) implements the CMUNeXt architecture, a ConvNeXt-inspired model optimized for medical imaging.

```python
from mipcandy_bundles.cmunext import CMUNeXtTrainer
from torch.utils.data import DataLoader

trainer = CMUNeXtTrainer("experiments", train_loader, val_loader, device="cuda")
trainer.num_classes = 1
trainer.variant = "s"  # Small variant; use "l" for large
trainer.train(100)
```

**Attributes**:
- `variant: Literal["s", "l"] | None` - Model size: "s" (small) or "l" (large) (default: None for auto-detection)
- `num_classes: int` - Number of segmentation classes (default: 1)

**Special Features**:
- Custom padding to multiples of 16
- SGD optimizer with momentum (instead of AdamW)
- Adaptive normalization (BatchNorm for batch_size > 1, GroupNorm otherwise)

```python
from mipcandy_bundles.cmunext import CMUNeXtTrainer
from mipcandy import NNUNetDataset
from torch.utils.data import DataLoader

dataset, val_dataset = NNUNetDataset("path/to/MSD/Task03_Liver").fold()
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

trainer = CMUNeXtTrainer("liver_seg", train_loader, val_loader, device="cuda")
trainer.num_classes = 2  # Background + liver
trainer.variant = "l"  # Large variant for complex task
trainer.train(200, note="Liver segmentation with CMUNeXt-L")
```

## Training Parameters

The `train()` method accepts many parameters to customize training behavior:

```python
trainer.train(
    num_epochs=100,
    note="Experiment description",
    num_checkpoints=10,
    ema=True,
    seed=42,
    early_stop_tolerance=10,
    val_score_prediction=True,
    val_score_prediction_degree=5,
    save_preview=True,
    preview_quality=0.75
)
```

### Parameter Details

- **`num_epochs: int`** - Total number of training epochs
- **`note: str`** - Description logged in experiment folder (default: "")
- **`num_checkpoints: int`** - Number of evenly-spaced checkpoints to save (default: 5)
- **`ema: bool`** - Enable Exponential Moving Average of model weights (default: True)
- **`seed: int | None`** - Random seed for reproducibility; random if None (default: None)
- **`early_stop_tolerance: int`** - Stop if validation score doesn't improve for N epochs (default: 5)
- **`val_score_prediction: bool`** - Enable validation score prediction using quotient regression (default: True)
- **`val_score_prediction_degree: int`** - Polynomial degree for score prediction (default: 5)
- **`save_preview: bool`** - Generate visualization previews (default: True)
- **`preview_quality: float`** - Quality for 3D previews, controls max voxels in millions (default: 0.75)

### Loading Settings from Configuration

You can load default training parameters from a YAML configuration file:

```python
# settings.yml
note: "Default experiment note"
num_checkpoints: 20
ema: true
seed: 42

# In code
trainer.train_with_settings(100, note="Override default note")
```

The `train_with_settings()` method merges settings from `settings.yml` with provided kwargs.

## Frontend Integration

Trainers support integration with experiment tracking platforms. See [Frontends](frontends.md) for detailed setup.

```python
from mipcandy import NotionFrontend

trainer = MyTrainer("experiments", train_loader, val_loader, device="cuda")
trainer.set_frontend(NotionFrontend)
trainer.train(100)
```

The frontend receives callbacks for:
- `on_experiment_created` - When training starts
- `on_experiment_updated` - After each epoch
- `on_experiment_completed` - When training finishes
- `on_experiment_interrupted` - If an exception occurs

## Advanced Features

### Exponential Moving Average (EMA)

EMA maintains a moving average of model parameters, often improving validation performance:

```python
trainer.train(100, ema=True)
```

During validation, the EMA model is used instead of the raw model if available:

```python
model_to_use = toolbox.ema if toolbox.ema else toolbox.model
```

### Validation Score Prediction

The trainer uses quotient regression to predict the maximum achievable validation score:

```python
# Fits: score(epoch) ≈ P(epoch) / Q(epoch)
# where P and Q are polynomials of specified degree
max_epoch, max_score = self.predict_maximum_validation_score(num_epochs, degree=5)
```

This provides:
- **Maximum score prediction**: Expected best validation score
- **Target epoch prediction**: When the maximum will be reached
- **Estimated time of completion (ETC)**: Time until target epoch

```
Maximum validation score 0.8523 predicted at epoch 87
Estimated time of completion in 1245.3 seconds at 11-01 15:32:10
```

:::{note}
Score prediction starts after `val_score_prediction_degree` epochs to ensure sufficient data points.
:::

### Early Stopping

Training stops automatically if the validation score doesn't improve for `early_stop_tolerance` consecutive epochs:

```python
trainer.train(200, early_stop_tolerance=10)
```

```
Early stopping triggered because the validation score has not improved for 10 epochs
```

This prevents overfitting and saves computation time.

### Seed Setting

Set a seed for reproducible results:

```python
trainer.train(100, seed=42)
```

This sets seeds for:
- NumPy random
- PyTorch CPU random
- PyTorch CUDA random
- Python random
- PYTHONHASHSEED environment variable

### Device Management

All trainers inherit from [`HasDevice`](#mipcandy.layer.HasDevice), providing automatic device management:

```python
trainer = MyTrainer("experiments", train_loader, val_loader, device="cuda")
# All operations automatically use the specified device
```

Data is automatically moved to the device during training and validation.

### Padding Module

Some architectures require input dimensions to be multiples of certain values. Override `build_padding_module()` to handle this:

```python
from typing import override
from torch import nn
from mipcandy import Trainer, Pad2d

class MyTrainer(Trainer):
    @override
    def build_padding_module(self) -> nn.Module | None:
        return Pad2d(32)  # Pad to multiples of 32
```

The padding module is automatically applied to inputs during training and validation.

## Metrics Display

The trainer displays detailed metrics tables after each epoch:

```
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric       ┃ Mean Value ┃ Span             ┃ Diff     ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ combined loss│ 0.2431     │ [0.1821, 0.3156] │ -0.0123  │
│ soft dice    │ 0.8456     │ [0.7892, 0.8901] │ +0.0089  │
│ bce loss     │ 0.1234     │ [0.0923, 0.1567] │ -0.0045  │
└──────────────┴────────────┴──────────────────┴──────────┘
```

- **Mean Value**: Current epoch's average
- **Span**: [min, max] range across all epochs
- **Diff**: Change from previous epoch
