# MIP Candy: A Candy for Medical Image Processing

![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/ProjectNeura/MIPCandy)
![PyPI](https://img.shields.io/pypi/v/mipcandy)
![GitHub Release](https://img.shields.io/github/v/release/ProjectNeura/MIPCandy)
![GitHub Release Date - Published_At](https://img.shields.io/github/release-date/ProjectNeura/MIPCandy)

## Installation

```shell
pip install "mipcandy[standard]"
```

## Quick Start

:::{tip}

`mipcandy_bundles` needs to be installed separately or with `"mipcandy[all]"`.

```shell
pip install "mipcandy[all]"
```

:::

```python
from typing import override

from monai.networks.nets import BasicUNet
from monai.transforms import Resized
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from mipcandy import SegmentationTrainer, AmbiguousShape, download_dataset, JointTransform, MONAITransform, Normalize,
    NNUNetDataset


class UNetTrainer(SegmentationTrainer):
    @override
    def build_network(self, example_shape: AmbiguousShape) -> nn.Module:
        return BasicUNet(2, example_shape[0], self.num_classes)


download_dataset("nnunet_datasets/PH2", "tutorial/datasets/PH2")
transform = JointTransform(transform=Compose([
    Resized(("image", "label"), (560, 768)), MONAITransform(Normalize(domain=(0, 1), strict=True))
]))
dataset, val_dataset = NNUNetDataset("tutorial/datasets/PH2", transform=transform, device="cuda").fold()
dataloader = DataLoader(dataset, 8, shuffle=True)
val_dataloader = DataLoader(val_dataset, 1, shuffle=False)
trainer = UNetTrainer("tutorial", dataloader, val_dataloader, device="cuda")
trainer.train(1000, note="a nnU-Net style example")
```

```{toctree}
:hidden:
:glob:
:caption: 🛠️ Framework
download-dataset.md
metrics.md
layer.md
```

```{toctree}
:hidden:
:glob:
:caption: 📊 Data
data/index.md
data/datasets.md
data/visualization.md
data/utilities.md
```

```{toctree}
:hidden:
:glob:
:caption: 🐎 Training
training/index.md
training/trainers.md
training/frontends.md
```

```{toctree}
:hidden:
:glob:
:caption: 🐏 Inference
inference/index.md
```

```{toctree}
:hidden:
:glob:
:caption: 🐂 Evaluation
evaluation/index.md
```

```{toctree}
:hidden:
:glob:
:caption: ⚙️ API
apidocs/index
```
