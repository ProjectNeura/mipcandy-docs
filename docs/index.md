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
from torch.utils.data import DataLoader
from mipcandy import NNUNetDataset
from mipcandy_bundles.unet import UNetTrainer

dataset = NNUNetDataset("path/to/your/dataset", device="cuda")
val_dataset = NNUNetDataset("path/to/your/dataset", split="Ts", device="cuda")
dataloader = DataLoader(dataset, 2, shuffle=True)
val_dataloader = DataLoader(val_dataset, 1, shuffle=False)
trainer = UNetTrainer("path/to/your/trainer/folder", dataloader, val_dataloader, device="cuda")
trainer.train(1000, note="a nnU-Net style example")
```

```{toctree}
:hidden:
:glob:
:caption: 🛠️ Framework
metrics.md
layer.md
```

```{toctree}
:hidden:
:glob:
:caption: 🐎 Training
training/index.md
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
