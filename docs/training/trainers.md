# Trainers

A [`Trainer`](#mipcandy.training.Trainer) is where you define the training loop logic. It handles the entire training pipeline, from model initialization to checkpoint management, metrics tracking, and validation.

## Overview

The trainer system in MIPCandy follows a two-level hierarchy:

1. **Base Trainer** - Abstract base class defining the training framework
2. **Segmentation Trainer** - Pre-configured trainer for segmentation tasks with deep supervision support
3. **Model-specific Trainers** - Ready-to-use trainers like UNetTrainer and CMUNeXtTrainer (from bundles)

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
    ema: nn.Module | None = None
```

This toolbox is passed to training methods, providing clean access to all components needed during the forward and backward passes.

## TrainerTracker

[`TrainerTracker`](#mipcandy.training.TrainerTracker) is a dataclass that tracks training state across epochs:

```python
from dataclasses import dataclass

@dataclass
class TrainerTracker:
    epoch: int = 0
    best_score: float = float("-inf")
    worst_case: int | None = None
```

The tracker maintains three pieces of state:

- **`epoch`** - The current epoch number, used for recovery and progress tracking
- **`best_score`** - The best validation score achieved so far, used for checkpoint saving
- **`worst_case`** - The index of the worst-performing validation case in the current epoch, used for preview generation

The trainer automatically updates the tracker during validation. When the worst validation case is identified, its input, label, and output tensors are saved so that preview images always show the hardest case.

## Base Trainer

The base [`Trainer`](#mipcandy.training.Trainer) class provides a complete training framework. It inherits from [`WithPaddingModule`](#mipcandy.layer.WithPaddingModule) and [`WithNetwork`](#mipcandy.layer.WithNetwork).

### Constructor

```python
Trainer(
    trainer_folder: str | PathLike[str],
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    validation_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    recoverable: bool = True,
    profiler: bool = False,
    device: torch.device | str = "cpu",
    console: Console = Console()
)
```

- **`trainer_folder`** - Root directory for experiment outputs
- **`dataloader`** - Training data loader
- **`validation_dataloader`** - Validation data loader (must have `batch_size=1`)
- **`recoverable`** - Enable training recovery support (default: `True`)
- **`profiler`** - Enable performance profiling (default: `False`)
- **`device`** - Computation device (default: `"cpu"`)
- **`console`** - Rich console for output (default: `Console()`)

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
from mipcandy.types import Params

@override
def build_optimizer(self, params: Params) -> optim.Optimizer:
    return optim.SGD(params, lr=1e-2, weight_decay=3e-5, momentum=.99, nesterov=True)
```

#### `build_scheduler(optimizer: optim.Optimizer, num_epochs: int) -> optim.lr_scheduler.LRScheduler`

Creates the learning rate scheduler.

```python
from mipcandy.common import PolyLRScheduler

@override
def build_scheduler(self, optimizer: optim.Optimizer, num_epochs: int) -> optim.lr_scheduler.LRScheduler:
    return PolyLRScheduler(optimizer, 1e-2, num_epochs * len(self._dataloader))
```

:::{tip}
[`PolyLRScheduler`](#mipcandy.common.optim.lr_scheduler.PolyLRScheduler) implements polynomial decay: `lr = initial_lr * (1 - step / max_steps) ^ exponent`. [`AbsoluteLinearLR`](#mipcandy.common.optim.lr_scheduler.AbsoluteLinearLR) implements linear decay: `lr = kx + b` with optional minimum learning rate and restart capability.
:::

#### `build_criterion() -> nn.Module`

Creates the loss function.

```python
from mipcandy.common import DiceBCELossWithLogits

@override
def build_criterion(self) -> nn.Module:
    return DiceBCELossWithLogits()
```

#### `build_ema(model: nn.Module) -> nn.Module`

Creates the Exponential Moving Average model for validation.

```python
from torch import nn, optim

@override
def build_ema(self, model: nn.Module) -> nn.Module:
    return optim.swa_utils.AveragedModel(model)
```

:::{note}
[`SegmentationTrainer`](#mipcandy.presets.segmentation.SegmentationTrainer) provides a default implementation using `AveragedModel`.
:::

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
The backward method should call `loss.backward()` but NOT call `optimizer.step()`. The base trainer handles optimizer stepping automatically in `train_batch()`.
:::

#### `validate_case(idx, image, label, toolbox) -> tuple[float, dict[str, float], torch.Tensor]`

Validates a single case. The `idx` parameter is the case index in the validation set.

```python
@override
def validate_case(self, idx: int, image: torch.Tensor, label: torch.Tensor,
                  toolbox: TrainerToolbox) -> tuple[float, dict[str, float], torch.Tensor]:
    image, label = image.unsqueeze(0), label.unsqueeze(0)
    output = (toolbox.ema if toolbox.ema else toolbox.model)(image)
    loss, metrics = toolbox.criterion(output, label)
    return -loss.item(), metrics, output.squeeze(0)
```

:::{note}
The validation score is typically the negative loss (higher is better). The trainer tracks the best score for checkpoint saving and automatically identifies the worst-performing case for preview generation.
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
  UNetTrainer/
    20251101-14-a3f2/
      logs.txt
      metrics.csv
      checkpoint_best.pth
      checkpoint_latest.pth
      checkpoint_0.pth
      progress.png
      state_dicts.pth
      state_orb.json
      worst_input.pt
      worst_label.pt
      worst_output.pt
      ...
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
self.record_all({"dice": [0.85], "iou": [0.78]})  # Record multiple metrics (averaged per epoch)
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

### Worst Validation Case Tracking

During validation, the trainer automatically tracks the worst-performing case. For each validation case, if its score is lower than the current worst, the trainer saves:

- `worst_input.pt` - The input tensor
- `worst_label.pt` - The ground truth label
- `worst_output.pt` - The model output

When the best validation score improves, the saved worst case is used to generate preview images. This ensures previews always show the model's weakest prediction, giving the most informative visual feedback.

## Custom Trainer Example

Here is a complete example of a custom trainer built directly on the base `Trainer`:

```python
from typing import override
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from mipcandy.training import Trainer, TrainerToolbox
from mipcandy.types import Params
from mipcandy.common import DiceBCELossWithLogits, PolyLRScheduler


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
        return optim.SGD(params, lr=1e-2, weight_decay=3e-5, momentum=.99, nesterov=True)

    @override
    def build_scheduler(self, optimizer: optim.Optimizer, num_epochs: int) -> optim.lr_scheduler.LRScheduler:
        return PolyLRScheduler(optimizer, 1e-2, num_epochs * len(self._dataloader))

    @override
    def build_criterion(self) -> nn.Module:
        return DiceBCELossWithLogits()

    @override
    def build_ema(self, model: nn.Module) -> nn.Module:
        return optim.swa_utils.AveragedModel(model)

    @override
    def backward(self, images: torch.Tensor, labels: torch.Tensor,
                 toolbox: TrainerToolbox) -> tuple[float, dict[str, float]]:
        predictions = toolbox.model(images)
        loss, metrics = toolbox.criterion(predictions, labels)
        loss.backward()
        return loss.item(), metrics

    @override
    def validate_case(self, idx: int, image: torch.Tensor, label: torch.Tensor,
                      toolbox: TrainerToolbox) -> tuple[float, dict[str, float], torch.Tensor]:
        image, label = image.unsqueeze(0), label.unsqueeze(0)
        output = (toolbox.ema if toolbox.ema else toolbox.model)(image)
        loss, metrics = toolbox.criterion(output, label)
        return -loss.item(), metrics, output.squeeze(0)


# Usage
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
trainer = MySegmentationTrainer("experiments", train_loader, val_loader, device="cuda")
trainer.train(100)
```

## Segmentation Trainer

[`SegmentationTrainer`](#mipcandy.presets.segmentation.SegmentationTrainer) provides pre-configured defaults for segmentation tasks. It implements all required abstract methods with sensible defaults, so you only need to implement `build_network()`.

### Class Attributes

```python
class SegmentationTrainer(Trainer, metaclass=ABCMeta):
    num_classes: int = 1
    include_background: bool = True
    deep_supervision: bool = False
    deep_supervision_scales: Sequence[float] | None = None
    deep_supervision_weights: Sequence[float] | None = None
```

- **`num_classes`** - Number of segmentation classes. Set to `1` for binary segmentation, `>= 2` for multiclass (default: `1`)
- **`include_background`** - Whether to include the background class in loss computation. Must be `True` for binary segmentation (default: `True`)
- **`deep_supervision`** - Enable deep supervision training (default: `False`)
- **`deep_supervision_scales`** - List of scale factors for deep supervision outputs. Used to auto-compute weights if `deep_supervision_weights` is not set
- **`deep_supervision_weights`** - Explicit weight factors for each supervision level. If not set but `deep_supervision_scales` is provided, weights are computed as `1 / 2^i` normalized to sum to 1

### Pre-configured Components

- **Loss Function**: [`DiceBCELossWithLogits`](#mipcandy.common.optim.loss.DiceBCELossWithLogits) for binary segmentation (`num_classes < 2`), or [`DiceCELossWithLogits`](#mipcandy.common.optim.loss.DiceCELossWithLogits) for multiclass segmentation (`num_classes >= 2`)
- **Optimizer**: `SGD` with `lr=1e-2`, `weight_decay=3e-5`, `momentum=0.99`, `nesterov=True`
- **Scheduler**: [`PolyLRScheduler`](#mipcandy.common.optim.lr_scheduler.PolyLRScheduler) with polynomial decay over total training steps
- **EMA**: `AveragedModel` from `torch.optim.swa_utils`
- **Gradient Clipping**: `clip_grad_norm_` with max norm of `12`
- **Preview Generation**: Automatic 2D/3D visualization with overlays

### Loss Classes

All segmentation loss functions inherit from [`Loss`](#mipcandy.common.optim.loss.Loss), which provides a `validation_mode` property. When set, the setter automatically propagates the value to all child `Loss` modules. [`SegmentationLoss`](#mipcandy.common.optim.loss.SegmentationLoss) extends `Loss` with `logitfy_no_grad()` for automatic ID-to-logit conversion.

```python
from mipcandy.common import Loss, SegmentationLoss
```

### DiceBCELossWithLogits

Binary segmentation loss combining Dice and BCE:

**Parameters:**
- `lambda_bce`: Weight for BCE loss (default: `1`)
- `lambda_soft_dice`: Weight for Dice loss (default: `1`)
- `smooth`: Smoothing constant for Dice (default: `1e-5`)

In validation mode, additionally computes `binary_dice` on thresholded outputs.

```python
from mipcandy import DiceBCELossWithLogits

criterion = DiceBCELossWithLogits()
# labels: (B, 1, H, W) float tensor
```

### DiceCELossWithLogits

Multiclass segmentation loss combining Dice and Cross Entropy:

**Parameters:**
- `num_classes`: Number of output classes
- `lambda_ce`: Weight for CE loss (default: `1`)
- `lambda_soft_dice`: Weight for Dice loss (default: `1`)
- `smooth`: Smoothing constant for Dice (default: `1e-5`)
- `include_background`: Include background in Dice computation (default: `True`)

In validation mode, additionally computes per-class `binary_dice` and overall `dice_similarity_coefficient`.

```python
from mipcandy import DiceCELossWithLogits

criterion = DiceCELossWithLogits(num_classes=4, include_background=False)

# Supports two label formats:
# 1. One-hot encoded: (B, num_classes, H, W) float tensor
# 2. Class indices: (B, 1, H, W) int tensor - automatically converted to one-hot
```

:::{note}
When `num_classes > 1` and labels have shape `(B, 1, H, W)`, the loss function automatically converts integer class indices to one-hot encoding internally. This allows using the same label format as standard segmentation datasets.
:::

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

### Class Percentages

The `class_percentages()` method computes the distribution of class labels in a segmentation tensor:

```python
def class_percentages(self, ids: torch.Tensor) -> dict[int, float]:
    bin_count = torch.bincount(ids.flatten(), minlength=self.num_classes)
    distribution = (bin_count / bin_count.sum()).cpu().tolist()
    return dict(enumerate(distribution))
```

During validation, this is used to report the percentage of each class in both the ground truth label and the model prediction. The companion `format_class_percentages()` static method formats these into metric dictionaries with keys like `% label class 0`, `% output class 1`, etc.

### Customization

You only need to implement `build_network()`:

```python
from typing import override
from torch import nn
from mipcandy.presets.segmentation import SegmentationTrainer


class MySegTrainer(SegmentationTrainer):
    num_classes: int = 3  # Multi-class segmentation
    include_background: bool = False  # Exclude background from Dice computation

    @override
    def build_network(self, example_shape: tuple[int, ...]) -> nn.Module:
        return MyNetwork(example_shape[0], self.num_classes)
```

:::{tip}
Set `include_background=False` when training multiclass segmentation where background dominates. This focuses the Dice loss on foreground classes and often improves segmentation quality.
:::

### Loss Function Selection

The loss function is automatically selected based on `num_classes`:

```python
@override
def build_criterion(self) -> nn.Module:
    if self.num_classes < 2:
        if not self.include_background:
            raise ValueError("Binary segmentation models must include background class")
        loss = DiceBCELossWithLogits()          # Binary: Dice + BCE
    else:
        loss = DiceCELossWithLogits(             # Multiclass: Dice + CE
            self.num_classes, include_background=self.include_background
        )
    # Wrapped with DeepSupervisionWrapper if deep_supervision is enabled
    ...
```

## Deep Supervision

[`DeepSupervisionWrapper`](#mipcandy.presets.segmentation.DeepSupervisionWrapper) extends [`Loss`](#mipcandy.common.optim.loss.Loss) and wraps a loss function to support multi-scale supervision during training.

### How It Works

Deep supervision computes the loss at multiple resolutions. The model produces outputs at several scales, and the ground truth labels are downsampled to match each scale. The final loss is a weighted sum across all scales.

```python
class DeepSupervisionWrapper(Loss):
    def __init__(self, loss: nn.Module, *, weight_factors: Sequence[float] | None = None) -> None:
        ...

    def forward(self, outputs: Sequence[torch.Tensor],
                targets: Sequence[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        ...
```

- **`loss`** - The base loss function to apply at each scale
- **`weight_factors`** - Per-scale weights. If `None`, all scales are weighted equally (`1.0`)

### Automatic Weight Computation

When `deep_supervision=True` and `deep_supervision_scales` is set but `deep_supervision_weights` is not, the weights are computed automatically:

```python
weights = [1 / (2 ** i) for i in range(len(deep_supervision_scales))]
# Normalized: e.g., for 3 scales -> [0.571, 0.286, 0.143]
```

This gives the highest weight to the full-resolution output and exponentially decreasing weights to lower resolutions.

### Enabling Deep Supervision

```python
from typing import override
from torch import nn
from mipcandy.presets.segmentation import SegmentationTrainer


class MyDeepSupTrainer(SegmentationTrainer):
    num_classes: int = 1
    deep_supervision: bool = True
    deep_supervision_scales: tuple[float, ...] = (1.0, 0.5, 0.25)

    @override
    def build_network(self, example_shape: tuple[int, ...]) -> nn.Module:
        return MyMultiScaleNetwork(example_shape[0], self.num_classes)
```

During training, the `backward()` method handles deep supervision automatically:
1. The model output is split into per-scale outputs (unbinding along dimension 1 if needed)
2. Labels are downsampled to match each output scale via `prepare_deep_supervision_targets()`
3. The `DeepSupervisionWrapper` computes the weighted sum of losses

During validation, only the highest-resolution output (index 0) is used.

### Metrics Under Deep Supervision

The wrapper produces per-scale metrics with `_ds{i}` suffixes (e.g., `soft_dice_ds0`, `bce_loss_ds1`). The main (scale 0) metric is also recorded without a suffix for consistency.

## Training Recovery

Training recovery allows you to resume interrupted training sessions. This is controlled by the `recoverable` parameter in the `Trainer` constructor.

### How Recovery Works

When `recoverable=True` (the default), the trainer saves recovery state after every epoch:

- **`state_dicts.pth`** - Optimizer, scheduler, and criterion state dictionaries
- **`state_orb.json`** - Tracker state (epoch, best_score, worst_case) and training arguments

### Recovering a Training Session

```python
trainer = MyTrainer("experiments", train_loader, val_loader, device="cuda")
trainer.recover_from("20251101-14-a3f2")  # Provide the experiment ID
trainer.continue_training(num_epochs=50)   # Continue for 50 more epochs
```

The `recover_from()` method:
1. Sets the experiment ID to the specified value
2. Verifies the experiment folder exists
3. Loads saved metrics and tracker state
4. Marks the trainer as recovered

The `continue_training()` method:
1. Verifies the trainer is in recovery mode
2. Loads the saved training arguments
3. Calls `train()` with the loaded arguments plus the new `num_epochs`

### Recovery State Methods

```python
trainer.recover_from(experiment_id)           # Enter recovery mode
trainer.continue_training(num_epochs)         # Resume training

# Lower-level methods
trainer.save_everything_for_recovery(toolbox, tracker, **training_arguments)
trainer.load_state_orb()                      # Load full recovery state
trainer.load_tracker()                        # Load TrainerTracker
trainer.load_training_arguments()             # Load saved train() kwargs
trainer.load_metrics()                        # Load metrics from CSV
trainer.load_toolbox(num_epochs, ...)         # Rebuild toolbox with saved state
```

:::{note}
When `recoverable=False`, calling `save_everything_for_recovery()` is a no-op. The `recovery()` method returns `True` only after `recover_from()` has been called.
:::

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
- SGD optimizer with momentum (instead of the default)
- Adaptive normalization (BatchNorm for batch_size > 1, GroupNorm otherwise)

```python
from mipcandy_bundles.cmunext import CMUNeXtTrainer
from mipcandy.data import NNUNetDataset
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
    compile_model=True,
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
- **`note: str`** - Description logged in experiment folder (default: `""`)
- **`num_checkpoints: int`** - Number of evenly-spaced checkpoints to save (default: `5`)
- **`compile_model: bool`** - Enable `torch.compile()` for optimized model execution (default: `True`)
- **`ema: bool`** - Enable Exponential Moving Average of model weights (default: `True`)
- **`seed: int | None`** - Random seed for reproducibility; random if `None` (default: `None`)
- **`early_stop_tolerance: int`** - Stop if validation score doesn't improve for N epochs (default: `5`)
- **`val_score_prediction: bool`** - Enable validation score prediction using quotient regression (default: `True`)
- **`val_score_prediction_degree: int`** - Polynomial degree for score prediction (default: `5`)
- **`save_preview: bool`** - Generate visualization previews (default: `True`)
- **`preview_quality: float`** - Quality for 3D previews, controls max voxels in millions (default: `0.75`)

### Loading Settings from Configuration

You can load default training parameters from a YAML configuration file:

```yaml
# settings.yml
note: "Default experiment note"
num_checkpoints: 20
ema: true
seed: 42
```

```python
# In code
trainer.train_with_settings(100, note="Override default note")
```

The `train_with_settings()` method merges settings from `settings.yml` with provided kwargs.

### Model Compilation

When `compile_model=True`, MIPCandy uses `torch.compile()` to optimize the model for faster execution. This requires a GPU with sufficient compute capability:

| CUDA Version | Minimum Compute Capability |
|--------------|---------------------------|
| cu118, cu121, cu124 | 7.0 |
| cu128+ | 7.5 |

:::{warning}
GPUs like P100 (CC 6.0) or GTX 1080 Ti (CC 6.1) do not meet these requirements. If you encounter an error like `CUDA capability >= 7.0`, disable model compilation:

```python
trainer.train(100, compile_model=False)
```
:::

## Frontend Integration

Trainers support integration with experiment tracking platforms. See [Frontends](frontends.md) for detailed setup.

```python
from mipcandy.frontend import NotionFrontend

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
# Fits: score(epoch) = P(epoch) / Q(epoch)
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

### Sanity Check

The `sanity_check()` method validates the model before training by running a forward pass with a template input:

```python
from mipcandy.sanity_check import SanityCheckResult

result: SanityCheckResult = trainer.sanity_check(template_model, example_shape)
# Returns MACs, parameter count, and example output info
# The template_model is automatically deleted after the check
```

### Profiler Memory Diagnostics

When the profiler is enabled (`profiler=True`), the `record_profiler_allocated_tensors()` method logs a diff of all live tensors between calls, helping diagnose memory leaks:

```python
trainer = MyTrainer("experiments", train_loader, val_loader, device="cuda", profiler=True)
# During training, the trainer automatically calls record_profiler_allocated_tensors()
# after each training epoch and after recovery saves
```

The profiler output includes added/removed tensors with their sizes, shapes, and gradient status. See [`dump_allocated_tensors()`](../data/utilities.md#dump_allocated_tensors) for the underlying utility.

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
from mipcandy.training import Trainer
from mipcandy.common.module import Pad2d

class MyTrainer(Trainer):
    @override
    def build_padding_module(self) -> nn.Module | None:
        return Pad2d(32)  # Pad to multiples of 32
```

The padding module is automatically applied to inputs during training and validation.

### Training Recovery

Resume interrupted training sessions using `recover_from()` and `continue_training()`:

```python
from mipcandy_bundles.unet import UNetTrainer

# Create trainer with same configuration
trainer = UNetTrainer("experiments", train_loader, val_loader, device="cuda")

# Recover from a specific experiment
trainer.recover_from("20251201-14-a3f2")

# Continue training for additional epochs
trainer.continue_training(50)  # Add 50 more epochs
```

**How it works:**
1. `recover_from(experiment_id)` loads:
   - Saved metrics history
   - Training tracker state (best score, epoch count)
   - Training arguments from the original run

2. `continue_training(num_epochs)` resumes with:
   - Model loaded from `checkpoint_latest.pth`
   - Optimizer and scheduler states restored
   - Metrics continuing from where training stopped

**Use cases:**

Recovering from crashes:
```python
# Original training crashed at epoch 75
trainer = UNetTrainer("experiments", train_loader, val_loader, device="cuda")
trainer.recover_from("20251201-14-a3f2")
trainer.continue_training(25)  # Complete remaining 25 epochs
```

Extending successful training:
```python
# Model still improving at epoch 100, extend training
trainer.recover_from("20251201-14-a3f2")
trainer.continue_training(100)  # Add 100 more epochs
```

:::{note}
Recovery requires the experiment folder to exist with `checkpoint_latest.pth`, `optimizer.pth`, `scheduler.pth`, and `criterion.pth` files.
:::

### Loading Toolbox

Load a complete training toolbox from a saved experiment:

```python
trainer = UNetTrainer("experiments", train_loader, val_loader, device="cuda")
trainer.recover_from("20251201-14-a3f2")

# Load toolbox with model, optimizer, scheduler, criterion
toolbox = trainer.load_toolbox(num_epochs=100, example_shape=(1, 256, 256))

# Access individual components
model = toolbox.model
optimizer = toolbox.optimizer
scheduler = toolbox.scheduler
criterion = toolbox.criterion
```

This is useful for:
- Custom inference pipelines
- Fine-tuning on new data
- Analyzing trained models

## Metrics Display

The trainer displays detailed metrics tables after each epoch:

```
+--------------+------------+------------------+----------+
| Metric       | Mean Value | Span             | Diff     |
+--------------+------------+------------------+----------+
| combined loss| 0.2431     | [0.1821, 0.3156] | -0.0123  |
| soft dice    | 0.8456     | [0.7892, 0.8901] | +0.0089  |
| bce loss     | 0.1234     | [0.0923, 0.1567] | -0.0045  |
+--------------+------------+------------------+----------+
```

- **Mean Value**: Current epoch's average
- **Span**: [min, max] range across the epoch
- **Diff**: Change from previous epoch

Per-case metrics are also displayed during validation, showing individual scores for each validation case.
