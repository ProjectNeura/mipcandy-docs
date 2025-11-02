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

import torch
from mipcandy_bundles.unet import UNetTrainer
from torch.utils.data import DataLoader

from mipcandy import download_dataset, NNUNetDataset


class PH2(NNUNetDataset):
    @override
    def load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = super().load(idx)
        return image.unsqueeze(1).permute(2, 0, 1), label


download_dataset("nnunet_datasets/PH2", "tutorial/datasets/PH2")
dataset, val_dataset = PH2("tutorial/datasets/PH2", device="cuda").fold()
dataloader = DataLoader(dataset, 2, shuffle=True)
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
