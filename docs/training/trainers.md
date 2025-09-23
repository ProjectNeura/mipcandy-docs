from torch import optimfrom torch import optimfrom torch import nn

# Trainers

A [`Trainer`](#mipcandy.training.Trainer) is where you define the training loop logic.

```python
from typing import override

import torch
from torch import nn, optim

from mipcandy import Trainer, TrainerToolbox, Params


class MyTrainer(Trainer):
    @override
    def build_scheduler(self, optimizer: optim.Optimizer, num_epochs: int) -> optim.lr_scheduler.LRScheduler:
        pass

    @override
    def build_criterion(self) -> nn.Module:
        pass

    @override
    def build_network(self, example_shape: tuple[int, ...]) -> nn.Module:
        pass

    @override
    def build_optimizer(self, params: Params) -> optim.Optimizer:
        pass

    @override
    def validate_case(self, image: torch.Tensor, label: torch.Tensor, toolbox: TrainerToolbox) -> tuple[float, dict[
        str, float], torch.Tensor]:
        pass

    @override
    def backward(self, images: torch.Tensor, labels: torch.Tensor, toolbox: TrainerToolbox) -> tuple[float, dict[
        str, float]]:
        pass
```

## Sliding Window Trainer

A [`SlidingWindowTrainer`](#mipcandy.training.SlidingTrainer) defines a sliding window mechanism for training.

## Segmentation Trainer

A [`SegmentationTrainer`](#mipcandy.presets.segmentation.SegmentationTrainer) defines some common training logic for
segmentation tasks.

## Predefined Trainers

:::{tip}

`mipcandy_bundles` needs to be installed separately or with `"mipcandy[all]"`.

```shell
pip install "mipcandy[all]"
```

:::

### CMUNeXt

[`CMUNeXtTrainer`](#mipcandy_bundles.cmunext.cmunext_trainer.CMUNeXtTrainer) supports both 2D and 3D segmentation.

```python
from mipcandy_bundles.cmunext import CMUNeXtTrainer
```

### UNetTrainer

[`UNetTrainer`](#mipcandy_bundles.unet.unet_trainer.UNetTrainer) supports both 2D and 3D segmentation.

```python
from mipcandy_bundles.unet import UNetTrainer
```