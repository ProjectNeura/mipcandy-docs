# Metrics

MIPCandy provides a comprehensive suite of evaluation metrics specifically designed for medical image segmentation tasks. All metrics support both binary and multiclass scenarios.

## Overview

The metrics module offers:

- **Binary Metrics**: Direct comparison between binary masks
- **Multiclass Metrics**: Automatic per-class evaluation with configurable reduction
- **Empty Region Handling**: Configurable behavior for empty predictions/labels via `if_empty` parameter
- **GPU Acceleration**: Optional CuPy backend for faster computation
- **Type Safety**: Automatic dtype and device validation

## Core Metrics

### Dice Similarity Coefficient

The Dice coefficient (also known as F1-score in segmentation contexts) measures the overlap between prediction and ground truth.

#### Binary Dice

[`dice_similarity_coefficient_binary`](#mipcandy.metrics.dice_similarity_coefficient_binary) computes Dice for binary masks:

```python
import torch
from mipcandy.metrics import dice_similarity_coefficient_binary

# Binary masks (bool tensors)
output = torch.tensor([[True, True, False],
                       [True, False, False]], dtype=torch.bool)
label = torch.tensor([[True, False, False],
                      [True, True, False]], dtype=torch.bool)

dice = dice_similarity_coefficient_binary(output, label)
# Result: 2 * 2 / (3 + 3) = 0.6667
```

**Parameters:**
- `output`: Binary prediction tensor (dtype: `torch.bool`)
- `label`: Binary ground truth tensor (dtype: `torch.bool`)
- `if_empty`: Return value when both masks are empty (default: `1.0`)

**Formula:**
$\text{Dice} = \frac{2|A \cap B|}{|A| + |B|}$

#### Multiclass Dice

[`dice_similarity_coefficient_multiclass`](#mipcandy.metrics.dice_similarity_coefficient_multiclass) computes per-class Dice and aggregates:

```python
from mipcandy.metrics import dice_similarity_coefficient_multiclass

# Integer class labels (0 = background, 1-3 = classes)
output = torch.tensor([[0, 1, 2],
                       [1, 2, 3]], dtype=torch.int)
label = torch.tensor([[0, 1, 1],
                      [1, 2, 3]], dtype=torch.int)

# Compute mean Dice across classes 1-3 (excluding background class 0)
dice = dice_similarity_coefficient_multiclass(output, label, num_classes=3)
```

**Parameters:**
- `output`: Integer prediction tensor (dtype: `torch.int`)
- `label`: Integer ground truth tensor (dtype: `torch.int`)
- `num_classes`: Number of classes (excluding background). If `None`, inferred from max value
- `if_empty`: Return value for empty class pairs (default: `1.0`)

:::{note}
Multiclass metrics exclude class 0 (background) and compute scores for classes 1 to `num_classes`.
:::

**Returns:** Mean Dice coefficient across all classes (1 to `num_classes`)

#### Soft Dice Coefficient

[`soft_dice_coefficient`](#mipcandy.metrics.soft_dice_coefficient) computes differentiable Dice for probability maps:

```python
from mipcandy.metrics import soft_dice_coefficient

# Probability maps (float tensors)
output = torch.rand(2, 1, 128, 128)  # Batch of 2, single channel
label = torch.randint(0, 2, (2, 1, 128, 128)).float()

dice = soft_dice_coefficient(output, label)
```

**Parameters:**
- `output`: Float prediction tensor (typically after sigmoid)
- `label`: Float ground truth tensor
- `smooth`: Smoothing constant to avoid division by zero (default: `1e-5`)
- `include_bg`: Whether to include background class (channel 0) in computation (default: `True`)

**Multiclass usage:**
```python
# Multiclass prediction: (batch, num_classes, H, W)
output = torch.rand(2, 4, 128, 128)  # 4 classes including background
label = torch.zeros(2, 4, 128, 128)

# Include all classes (default)
dice_all = soft_dice_coefficient(output, label, include_bg=True)

# Exclude background (class 0)
dice_fg = soft_dice_coefficient(output, label, include_bg=False)
```

:::{tip}
Soft Dice is primarily used as a differentiable loss function during training, not for evaluation. Set `include_bg=False` when background dominates and you want to focus on foreground classes.
:::

### Accuracy

Measures the proportion of correctly classified pixels/voxels.

#### Binary Accuracy

[`accuracy_binary`](#mipcandy.metrics.accuracy_binary) computes pixel-wise accuracy:

```python
from mipcandy.metrics import accuracy_binary

output = torch.tensor([[True, True, False]], dtype=torch.bool)
label = torch.tensor([[True, False, False]], dtype=torch.bool)

acc = accuracy_binary(output, label)
# Result: (TP + TN) / (TP + TN + FP + FN) = 2/3 = 0.6667
```

**Formula:**
$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$

#### Multiclass Accuracy

[`accuracy_multiclass`](#mipcandy.metrics.accuracy_multiclass) computes per-class accuracy:

```python
from mipcandy.metrics import accuracy_multiclass

output = torch.tensor([[0, 1, 2]], dtype=torch.int)
label = torch.tensor([[0, 1, 1]], dtype=torch.int)

acc = accuracy_multiclass(output, label, num_classes=2)
```

### Precision

Measures the proportion of true positives among all positive predictions.

#### Binary Precision

[`precision_binary`](#mipcandy.metrics.precision_binary):

```python
from mipcandy.metrics import precision_binary

output = torch.tensor([[True, True, False]], dtype=torch.bool)
label = torch.tensor([[True, False, False]], dtype=torch.bool)

prec = precision_binary(output, label)
# Result: TP / (TP + FP) = 1/2 = 0.5
```

**Formula:**
$\text{Precision} = \frac{TP}{TP + FP}$

:::{note}
The `if_empty` parameter controls return value when no positive predictions exist (denominator = 0).
:::

#### Multiclass Precision

[`precision_multiclass`](#mipcandy.metrics.precision_multiclass):

```python
from mipcandy.metrics import precision_multiclass

output = torch.tensor([[0, 1, 2]], dtype=torch.int)
label = torch.tensor([[0, 1, 1]], dtype=torch.int)

prec = precision_multiclass(output, label, num_classes=2)
```

### Recall (Sensitivity)

Measures the proportion of true positives among all actual positives.

#### Binary Recall

[`recall_binary`](#mipcandy.metrics.recall_binary):

```python
from mipcandy.metrics import recall_binary

output = torch.tensor([[True, True, False]], dtype=torch.bool)
label = torch.tensor([[True, False, True]], dtype=torch.bool)

rec = recall_binary(output, label)
# Result: TP / (TP + FN) = 1/2 = 0.5
```

**Formula:**
$\text{Recall} = \frac{TP}{TP + FN}$

:::{note}
The `if_empty` parameter controls return value when no positive labels exist (denominator = 0).
:::

#### Multiclass Recall

[`recall_multiclass`](#mipcandy.metrics.recall_multiclass):

```python
from mipcandy.metrics import recall_multiclass

output = torch.tensor([[0, 1, 2]], dtype=torch.int)
label = torch.tensor([[0, 1, 1]], dtype=torch.int)

rec = recall_multiclass(output, label, num_classes=2)
```

### Intersection over Union (IoU)

Also known as Jaccard Index, measures the overlap between prediction and ground truth regions.

#### Binary IoU

[`iou_binary`](#mipcandy.metrics.iou_binary):

```python
from mipcandy.metrics import iou_binary

output = torch.tensor([[True, True, False]], dtype=torch.bool)
label = torch.tensor([[True, False, False]], dtype=torch.bool)

iou = iou_binary(output, label)
# Result: |A ∩ B| / |A ∪ B| = 1/2 = 0.5
```

**Formula:**
$\text{IoU} = \frac{|A \cap B|}{|A \cup B|}$

**Relationship to Dice:**
\begin{align}
\text{Dice} &= \frac{2 \cdot \text{IoU}}{1 + \text{IoU}} \\
\text{IoU} &= \frac{\text{Dice}}{2 - \text{Dice}}
\end{align}

#### Multiclass IoU

[`iou_multiclass`](#mipcandy.metrics.iou_multiclass):

```python
from mipcandy.metrics import iou_multiclass

output = torch.tensor([[0, 1, 2]], dtype=torch.int)
label = torch.tensor([[0, 1, 1]], dtype=torch.int)

iou = iou_multiclass(output, label, num_classes=2)
```

## Advanced Usage

### Handling Empty Regions

The `if_empty` parameter controls behavior when masks are empty:

```python
# Both masks empty - perfect match
output = torch.zeros((10, 10), dtype=torch.bool)
label = torch.zeros((10, 10), dtype=torch.bool)
dice = dice_similarity_coefficient_binary(output, label, if_empty=1.0)
# Returns: 1.0

# Prediction empty but label non-empty - complete miss
output = torch.zeros((10, 10), dtype=torch.bool)
label = torch.ones((10, 10), dtype=torch.bool)
dice = dice_similarity_coefficient_binary(output, label, if_empty=1.0)
# Returns: 0.0 (computed normally since label is not empty)
```

:::{note}
**Default values:**
- Most metrics: `if_empty=1.0` (perfect score for empty pairs)
- Rationale: Empty prediction matching empty ground truth is considered a correct prediction
:::

### Multiclass Metric Computation

All multiclass metrics use the same underlying pattern via [`apply_multiclass_to_binary`](#mipcandy.metrics.apply_multiclass_to_binary):

```python
# Pseudocode for multiclass metrics
for class_id in range(1, num_classes + 1):
    binary_output = (output == class_id)
    binary_label = (label == class_id)
    score[class_id] = binary_metric(binary_output, binary_label, if_empty=if_empty)

return mean(score)  # or sum, depending on reduction parameter
```

### GPU Acceleration

MIPCandy automatically uses CuPy for distance transform operations when available:

```python
# Automatically uses CuPy if installed and data is on GPU
try:
    from cupy import from_dlpack
    from cupyx.scipy.ndimage import distance_transform_edt
    # Use GPU-accelerated implementation
except ImportError:
    from numpy import from_dlpack
    from scipy.ndimage import distance_transform_edt
    # Fallback to CPU implementation
```

### Type and Device Safety

:::{important}
All metrics include automatic validation to prevent common errors:

**Validation checks:**
1. Shape compatibility: `output.shape == label.shape`
2. Dtype compatibility: Both tensors must have the same dtype
3. Device compatibility: Both tensors must be on the same device
4. Expected dtype: Binary metrics require `torch.bool`, multiclass require `torch.int`

```python
from mipcandy.metrics import dice_similarity_coefficient_binary

output = torch.tensor([[True]], dtype=torch.bool, device="cuda")
label = torch.tensor([[True]], dtype=torch.bool, device="cpu")

# Raises RuntimeError: tensors must be on the same device
dice = dice_similarity_coefficient_binary(output, label)
```
:::

## Reduction Methods

The [`do_reduction`](#mipcandy.metrics.do_reduction) utility function provides flexible aggregation:

```python
from mipcandy.metrics import do_reduction

scores = torch.tensor([0.8, 0.9, 0.7, 0.85])

mean_score = do_reduction(scores, method="mean")    # 0.8125
median_score = do_reduction(scores, method="median")  # 0.825
sum_score = do_reduction(scores, method="sum")      # 3.25
all_scores = do_reduction(scores, method="none")    # [0.8, 0.9, 0.7, 0.85]
```

**Supported methods:**
- `"mean"`: Arithmetic mean (default for most metrics)
- `"median"`: Median value
- `"sum"`: Sum of all values
- `"none"`: No reduction, return all values

## Metric Protocol

The [`Metric`](#mipcandy.metrics.Metric) protocol defines the interface for all metric functions:

```python
from typing import Protocol
import torch

class Metric(Protocol):
    def __call__(
        self,
        output: torch.Tensor,
        label: torch.Tensor,
        *,
        if_empty: float = ...
    ) -> torch.Tensor: ...
```

This protocol enables type-safe metric composition and custom metric implementation.
