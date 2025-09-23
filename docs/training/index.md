# Training

To train your model, you'll need three things: a training dataset, a validation dataset, and a trainer.

:::{tip}

`mipcandy_bundles` needs to be installed separately or with `"mipcandy[all]"`.

```shell
pip install "mipcandy[all]"
```

:::

```python
from mipcandy_bundles.cmunext import CMUNeXtTrainer
from torch.utils.data import DataLoader

from mipcandy import NNUNetDataset

train_dataset, val_dataset = NNUNetDataset("a dataset with 10 classes").fold(fold="all")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
trainer_folder = "trainers"
trainer = CMUNeXtTrainer(trainer_folder, train_loader, val_loader, device="cuda")
trainer.num_classes = 10
trainer.train(100, note="training CMUNeXt for 100 epochs", num_checkpoints=10, ema=False, seed=42,
              early_stop_tolerance=10, val_score_prediction=False, preview_quality=.5)
```