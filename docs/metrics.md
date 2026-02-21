# Metrics

MIPCandy provides Dice-family evaluation metrics for medical image segmentation. The module supports binary masks, one-hot encoded tensors, and differentiable soft Dice on logits, with configurable reduction across batch dimensions.

## Overview

The metrics module contains three Dice score functions, each operating on a different tensor format:

| Function | Input Format | dtype | Use Case |
|---|---|---|---|
| [`binary_dice`](#mipcandy.metrics.binary_dice) | Boolean masks `(B, 1, ...)` | `torch.bool` | Evaluation of binary segmentation |
| [`dice_similarity_coefficient`](#mipcandy.metrics.dice_similarity_coefficient) | One-hot float `(B, N, ...)` | `torch.float` | Evaluation of multiclass segmentation |
| [`soft_dice`](#mipcandy.metrics.soft_dice) | Logits/probabilities `(B, C, ...)` | `torch.float` | Differentiable loss during training |

All functions share a common validation layer ([`_args_check`](#mipcandy.metrics._args_check)) and support flexible output aggregation via [`do_reduction`](#mipcandy.metrics.do_reduction).

## Dice Metrics

### Binary Dice

[`binary_dice`](#mipcandy.metrics.binary_dice) computes the Dice score on boolean tensors with shape `(B, 1, ...)`.

```python
binary_dice(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    if_empty: float = 1,
    reduction: Reduction = "mean",
) -> torch.Tensor
```

The spatial dimensions (all axes from index 2 onward) are summed to compute per-sample volume overlap. The `reduction` parameter then aggregates across the batch.

**Formula:**

$$\text{Dice} = \frac{2|A \cap B|}{|A| + |B|}$$

**Example:**

```python
import torch
from mipcandy.metrics import binary_dice

# Batch of 2 binary masks, single channel, 4x4 spatial
outputs = torch.zeros(2, 1, 4, 4, dtype=torch.bool)
labels = torch.zeros(2, 1, 4, 4, dtype=torch.bool)

outputs[0, 0, :2, :2] = True  # 4 positive voxels in sample 0
labels[0, 0, :3, :2] = True   # 6 positive voxels in sample 0

score = binary_dice(outputs, labels)
# Intersection = 4, sum = 4 + 6 = 10
# Dice for sample 0 = 2 * 4 / 10 = 0.8
# Sample 1 is empty on both sides -> if_empty = 1.0
# Mean = (0.8 + 1.0) / 2 = 0.9
```

:::{note}
Both `outputs` and `labels` must be `torch.bool`. Passing float or integer tensors raises a `TypeError`.
:::

### Dice Similarity Coefficient

[`dice_similarity_coefficient`](#mipcandy.metrics.dice_similarity_coefficient) computes Dice on one-hot encoded float tensors with shape `(B, N, ...)`, where `N` is the number of classes.

```python
dice_similarity_coefficient(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    if_empty: float = 1,
    reduction: Reduction = "mean",
) -> torch.Tensor
```

The function computes true positives, false positives, and false negatives per class and per sample across spatial dimensions:

$$\text{DSC} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

If any class has a zero denominator (i.e., no predictions and no ground truth for that class), the function returns `if_empty` immediately.

**Example:**

```python
import torch
from mipcandy.metrics import dice_similarity_coefficient

# Batch=1, 3 classes, 8x8 spatial
outputs = torch.zeros(1, 3, 8, 8, dtype=torch.float)
labels = torch.zeros(1, 3, 8, 8, dtype=torch.float)

# Class 0: full overlap
outputs[0, 0, :4, :4] = 1.0
labels[0, 0, :4, :4] = 1.0

# Class 1: partial overlap
outputs[0, 1, 4:8, :4] = 1.0
labels[0, 1, 4:8, 2:6] = 1.0

score = dice_similarity_coefficient(outputs, labels)
```

:::{tip}
This function is intended for hard one-hot predictions during evaluation. For differentiable training objectives, use [`soft_dice`](#mipcandy.metrics.soft_dice) instead.
:::

### Soft Dice

[`soft_dice`](#mipcandy.metrics.soft_dice) computes a differentiable Dice score on float tensors (logits or probabilities) with shape `(B, C, ...)`.

```python
soft_dice(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    smooth: float = 1,
    batch_dice: bool = True,
    reduction: Reduction = "mean",
) -> torch.Tensor
```

**Formula:**

$$\text{Soft Dice} = \frac{2 \sum (p \cdot g) + \epsilon}{\sum p + \sum g + \epsilon}$$

where $p$ is the predicted tensor, $g$ is the ground truth tensor, and $\epsilon$ is the `smooth` parameter.

**Parameters:**

- `smooth` -- Laplace smoothing constant added to both numerator and denominator to prevent division by zero and stabilize gradients. Default: `1`.
- `batch_dice` -- When `True`, spatial and batch dimensions are aggregated together before computing the per-class score. When `False`, each sample in the batch is scored independently. Default: `True`.
- `reduction` -- Aggregation method applied to the resulting per-class scores. Default: `"mean"`.

**Example:**

```python
import torch
from mipcandy.metrics import soft_dice

# Logits: batch=4, 3 classes, 64x64 spatial
outputs = torch.randn(4, 3, 64, 64)
labels = torch.randint(0, 2, (4, 3, 64, 64)).float()

# Batch-level soft Dice (default)
score = soft_dice(outputs.sigmoid(), labels)

# Per-sample soft Dice
score = soft_dice(outputs.sigmoid(), labels, batch_dice=False)
```

:::{warning}
`soft_dice` does not apply sigmoid or softmax internally. You must apply the appropriate activation to `outputs` before calling this function if your model produces raw logits.
:::

## Utilities

### Argument Validation

[`_args_check`](#mipcandy.metrics._args_check) validates that `outputs` and `labels` are compatible in shape, dtype, and device.

```python
_args_check(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: Device | None = None,
) -> tuple[torch.dtype, Device]
```

**Checks performed:**

1. **Shape**: `outputs.shape == labels.shape`, otherwise raises `ValueError`.
2. **Dtype**: Both tensors must share the same dtype. If `dtype` is specified, both must match it exactly. Raises `TypeError` on mismatch.
3. **Device**: Both tensors must reside on the same device. If `device` is specified, both must be on that device. Raises `RuntimeError` on mismatch.

Returns the validated `(dtype, device)` tuple.

```python
import torch
from mipcandy.metrics import _args_check

a = torch.zeros(2, 1, 8, 8, dtype=torch.bool, device="cpu")
b = torch.zeros(2, 1, 8, 8, dtype=torch.bool, device="cpu")

dtype, device = _args_check(a, b, dtype=torch.bool)
# dtype = torch.bool, device = cpu
```

:::{note}
All three Dice functions call `_args_check` internally with an explicit `dtype` constraint (`torch.bool` for `binary_dice`, `torch.float` for the other two). You generally do not need to call `_args_check` yourself unless you are implementing a custom metric.
:::

### Reduction

[`do_reduction`](#mipcandy.metrics.do_reduction) applies an aggregation method to a tensor of per-sample or per-class scores.

```python
do_reduction(x: torch.Tensor, method: Reduction) -> torch.Tensor
```

The `Reduction` type is defined as:

```python
type Reduction = Literal["mean", "median", "sum", "none"]
```

**Supported methods:**

| Method | Behavior |
|---|---|
| `"mean"` | Arithmetic mean of all elements |
| `"median"` | Median value |
| `"sum"` | Sum of all elements |
| `"none"` | No reduction; returns the tensor unchanged |

```python
import torch
from mipcandy.metrics import do_reduction

scores = torch.tensor([0.8, 0.9, 0.7, 0.85])

do_reduction(scores, "mean")    # tensor(0.8125)
do_reduction(scores, "median")  # tensor(0.825)
do_reduction(scores, "sum")     # tensor(3.25)
do_reduction(scores, "none")    # tensor([0.8, 0.9, 0.7, 0.85])
```

## Handling Empty Regions

The `if_empty` parameter in [`binary_dice`](#mipcandy.metrics.binary_dice) and [`dice_similarity_coefficient`](#mipcandy.metrics.dice_similarity_coefficient) controls the return value when both `outputs` and `labels` contain no positive elements.

- **Default: `1`** -- An empty prediction matching an empty ground truth is considered a perfect score.
- Set to `0` if you want empty-vs-empty cases to be penalized.

```python
import torch
from mipcandy.metrics import binary_dice

# Both masks empty
outputs = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
labels = torch.zeros(1, 1, 8, 8, dtype=torch.bool)

binary_dice(outputs, labels, if_empty=1.0)  # tensor(1.)
binary_dice(outputs, labels, if_empty=0.0)  # tensor(0.)

# Only prediction is empty, label is not
labels[0, 0, :4, :4] = True
binary_dice(outputs, labels)  # tensor(0.) -- computed normally
```

:::{important}
In `binary_dice`, the empty check applies per-sample before reduction: if a single sample has zero volume on both sides, that sample receives `if_empty`. In `dice_similarity_coefficient`, the check applies globally across all classes: if **any** class has a zero denominator, the entire result is replaced by `if_empty`.
:::
