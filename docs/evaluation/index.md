# Evaluation

MIPCandy provides a standardized evaluation framework through the [`Evaluator`](#mipcandy.evaluation.Evaluator) class and its companion data structures [`EvalCase`](#mipcandy.evaluation.EvalCase) and [`EvalResult`](#mipcandy.evaluation.EvalResult). Together, they enable structured evaluation of model predictions against ground truth labels, with support for custom metric functions, per-case analysis, and integration with the inference pipeline.

## Overview

The evaluation module offers:

- **Structured Results**: Per-case evaluation results stored as [`EvalCase`](#mipcandy.evaluation.EvalCase) dataclass instances
- **Collection Operations**: [`EvalResult`](#mipcandy.evaluation.EvalResult) provides aggregation (mean), selection (min/max), and indexing over evaluation cases
- **Flexible Input**: Evaluate from datasets, raw tensors, file paths, or directories via [`SupportedPredictant`](#mipcandy.types.SupportedPredictant)
- **Predictor Integration**: Run inference and evaluation in a single call with [`predict_and_evaluate`](#mipcandy.evaluation.Evaluator.predict_and_evaluate)
- **Custom Metrics**: Any callable with signature `(Tensor, Tensor) -> Tensor` can serve as a metric function

## Quick Start

```python
import torch
from mipcandy.evaluation import Evaluator
from mipcandy.metrics import binary_dice

# Create evaluator with one or more metric functions
evaluator = Evaluator(binary_dice)

# Evaluate predictions against ground truth
result = evaluator.evaluate(
    "path/to/predictions/",
    "path/to/labels/"
)

# Inspect aggregated metrics
print(result.mean_metrics)
# {'binary_dice': 0.87}

# Find the worst-performing case
worst = result.min("binary_dice")
print(worst.filename, worst.metrics)
```

## EvalCase

[`EvalCase`](#mipcandy.evaluation.EvalCase) is a dataclass representing the evaluation result for a single case.

```python
from dataclasses import dataclass
import torch

@dataclass
class EvalCase:
    metrics: dict[str, float]
    output: torch.Tensor
    label: torch.Tensor
    image: torch.Tensor | None = None
    filename: str | None = None
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `metrics` | `dict[str, float]` | Metric name to score mapping for this case |
| `output` | `torch.Tensor` | The model prediction tensor |
| `label` | `torch.Tensor` | The ground truth tensor |
| `image` | `torch.Tensor \| None` | The input image (populated by `predict_and_evaluate`) |
| `filename` | `str \| None` | Source filename (populated when input is file-based) |

### Accessing Case Data

```python
result = evaluator.evaluate("predictions/", "labels/")

case = result[0]  # First case
print(case.metrics)       # {'binary_dice': 0.92}
print(case.output.shape)  # torch.Size([1, 128, 128])
print(case.filename)      # 'case_001.nii.gz'
```

## EvalResult

[`EvalResult`](#mipcandy.evaluation.EvalResult) is a `Sequence[EvalCase]` that holds evaluation results for all cases and provides aggregation and selection utilities.

### Constructor

```python
class EvalResult(Sequence[EvalCase]):
    def __init__(
        self,
        metrics: dict[str, list[float]],
        outputs: list[torch.Tensor],
        labels: list[torch.Tensor],
        *,
        images: list[torch.Tensor] | None = None,
        filenames: list[str] | None = None
    ) -> None: ...
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `metrics` | `dict[str, list[float]]` | Metric name to per-case score lists |
| `outputs` | `list[torch.Tensor]` | All prediction tensors |
| `labels` | `list[torch.Tensor]` | All ground truth tensors |
| `images` | `list[torch.Tensor] \| None` | Input images (optional) |
| `filenames` | `list[str] \| None` | Source filenames (optional) |

:::{note}
`outputs` and `labels` must have the same length. A `ValueError` is raised otherwise.
:::

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `metrics` | `dict[str, list[float]]` | Per-case scores for each metric |
| `mean_metrics` | `dict[str, float]` | Mean score for each metric across all cases |
| `outputs` | `list[torch.Tensor]` | All prediction tensors |
| `labels` | `list[torch.Tensor]` | All ground truth tensors |
| `images` | `list[torch.Tensor] \| None` | Input images (if available) |
| `filenames` | `list[str] \| None` | Source filenames (if available) |

### Sequence Interface

`EvalResult` implements `Sequence[EvalCase]`, so it supports `len()` and indexing:

```python
result = evaluator.evaluate("predictions/", "labels/")

print(len(result))  # Number of cases

# Iterate over individual cases
for case in result:
    print(case.filename, case.metrics)

# Direct indexing
first_case = result[0]
last_case = result[-1]
```

### Selection Methods

#### min

```python
def min(self, metric: str) -> EvalCase
```

Return the case with the lowest score for the given metric.

```python
worst_case = result.min("binary_dice")
print(f"Worst Dice: {worst_case.metrics['binary_dice']:.4f}")
print(f"File: {worst_case.filename}")
```

#### min_n

```python
def min_n(self, metric: str, n: int) -> tuple[EvalCase, ...]
```

Return the `n` cases with the lowest scores for the given metric, sorted in ascending order.

```python
# Get 5 worst-performing cases
worst_5 = result.min_n("binary_dice", 5)
for case in worst_5:
    print(f"{case.filename}: {case.metrics['binary_dice']:.4f}")
```

#### max

```python
def max(self, metric: str) -> EvalCase
```

Return the case with the highest score for the given metric.

```python
best_case = result.max("binary_dice")
print(f"Best Dice: {best_case.metrics['binary_dice']:.4f}")
```

#### max_n

```python
def max_n(self, metric: str, n: int) -> tuple[EvalCase, ...]
```

Return the `n` cases with the highest scores for the given metric, sorted in descending order.

```python
# Get top 3 cases
top_3 = result.max_n("binary_dice", 3)
for case in top_3:
    print(f"{case.filename}: {case.metrics['binary_dice']:.4f}")
```

### Aggregated Metrics

`mean_metrics` is computed at construction time and contains the arithmetic mean of each metric across all cases:

```python
result = evaluator.evaluate("predictions/", "labels/")

# Access mean metrics
print(result.mean_metrics)
# {'binary_dice': 0.87, 'soft_dice': 0.78}

# Access per-case metric values
print(result.metrics["binary_dice"])
# [0.92, 0.85, 0.79, 0.91, ...]
```

## Evaluator

[`Evaluator`](#mipcandy.evaluation.Evaluator) is the main class for running evaluations. It accepts one or more metric functions and provides three evaluation methods.

### Constructor

```python
class Evaluator:
    def __init__(
        self,
        *metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> None: ...
```

**Parameters:**

- `*metrics`: One or more metric functions. Each function must accept two tensors (output, label) and return a scalar tensor. The function's `__name__` attribute is used as the metric key in results.

```python
from mipcandy.evaluation import Evaluator
from mipcandy.metrics import binary_dice, dice_similarity_coefficient, soft_dice

evaluator = Evaluator(
    binary_dice,
    dice_similarity_coefficient,
    soft_dice
)
```

### evaluate_dataset

```python
def evaluate_dataset(self, x: SupervisedDataset) -> EvalResult
```

Evaluate all cases in a [`SupervisedDataset`](#mipcandy.data.SupervisedDataset). The dataset yields `(output, label)` pairs directly.

```python
from mipcandy import NNUNetDataset
from mipcandy.evaluation import Evaluator
from mipcandy.metrics import binary_dice

# Prepare dataset where images are model outputs and labels are ground truth
dataset = NNUNetDataset("path/to/dataset").fold(fold=0)[1]  # validation fold

evaluator = Evaluator(binary_dice)
result = evaluator.evaluate_dataset(dataset)

print(f"Mean Dice: {result.mean_metrics['binary_dice']:.4f}")
print(f"Cases evaluated: {len(result)}")
```

:::{note}
When using `evaluate_dataset`, the first element of each dataset pair is treated as the output (prediction), and the second as the label. This method is useful when predictions have already been stored into a dataset structure.
:::

### evaluate

```python
def evaluate(
    self,
    outputs: SupportedPredictant,
    labels: SupportedPredictant
) -> EvalResult
```

Evaluate arbitrary outputs against labels. Both `outputs` and `labels` accept any [`SupportedPredictant`](#mipcandy.types.SupportedPredictant) format: file paths, directories, tensors, or sequences thereof.

```python
evaluator = Evaluator(binary_dice)

# From directories
result = evaluator.evaluate("predictions/", "labels/")

# From file lists
result = evaluator.evaluate(
    ["pred_001.nii.gz", "pred_002.nii.gz"],
    ["label_001.nii.gz", "label_002.nii.gz"]
)

# From tensors
outputs = [torch.randint(0, 2, (1, 128, 128), dtype=torch.bool) for _ in range(10)]
labels = [torch.randint(0, 2, (1, 128, 128), dtype=torch.bool) for _ in range(10)]
result = evaluator.evaluate(outputs, labels)
```

Internally, `evaluate` uses [`parse_predictant`](#mipcandy.inference.parse_predictant) to convert inputs to tensors. Filenames are extracted automatically when file-based inputs are provided.

### predict_and_evaluate

```python
def predict_and_evaluate(
    self,
    x: SupportedPredictant,
    labels: SupportedPredictant,
    predictor: Predictor
) -> EvalResult
```

Run a [`Predictor`](#mipcandy.inference.Predictor) on the inputs, then evaluate the predictions against the labels. This combines inference and evaluation in a single call.

```python
from mipcandy_bundles.unet import UNetPredictor
from mipcandy.evaluation import Evaluator
from mipcandy.metrics import binary_dice, soft_dice

predictor = UNetPredictor("experiments/UNet/20240901-1234", (1, 128, 128), device="cuda")
evaluator = Evaluator(binary_dice, soft_dice)

result = evaluator.predict_and_evaluate(
    "data/test_images/",
    "data/test_labels/",
    predictor
)

print(result.mean_metrics)
# {'binary_dice': 0.89, 'soft_dice': 0.81}
```

:::{tip}
`predict_and_evaluate` populates both `result.images` (the input images) and `result.filenames` (the source filenames), making it the most informative evaluation method.
:::

## Custom Metric Functions

Any function matching the signature `(torch.Tensor, torch.Tensor) -> torch.Tensor` can be used as a metric. The function's `__name__` attribute becomes the metric key in the results dictionary.

### Writing a Custom Metric

```python
import torch

def hausdorff_distance(output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Compute the Hausdorff distance between two binary masks."""
    output_points = torch.nonzero(output)
    label_points = torch.nonzero(label)

    if len(output_points) == 0 or len(label_points) == 0:
        return torch.tensor(float("inf"))

    d_ol = torch.cdist(output_points.float(), label_points.float()).min(dim=1).values.max()
    d_lo = torch.cdist(label_points.float(), output_points.float()).min(dim=1).values.max()
    return torch.max(d_ol, d_lo)

# Use with Evaluator
evaluator = Evaluator(binary_dice, hausdorff_distance)
result = evaluator.evaluate("predictions/", "labels/")
print(result.mean_metrics["hausdorff_distance"])
```

### Using Lambda Functions

Lambda functions can be used, but note that their `__name__` will be `"<lambda>"`, which makes results harder to interpret. Prefer named functions.

```python
# Works but metric key will be "<lambda>"
evaluator = Evaluator(lambda o, l: (o == l).float().mean())

# Prefer a named function
def pixel_accuracy(output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return (output == label).float().mean()

evaluator = Evaluator(pixel_accuracy)
```

### Combining Built-in and Custom Metrics

```python
from mipcandy.metrics import binary_dice, dice_similarity_coefficient, soft_dice

def volume_difference(output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Relative volume difference between prediction and ground truth."""
    vol_output = output.sum().float()
    vol_label = label.sum().float()
    if vol_label == 0:
        return torch.tensor(0.0)
    return torch.abs(vol_output - vol_label) / vol_label

evaluator = Evaluator(
    binary_dice,
    dice_similarity_coefficient,
    soft_dice,
    volume_difference
)
```

## Usage Examples

### Evaluate a Validation Fold

```python
from mipcandy import NNUNetDataset
from mipcandy.evaluation import Evaluator
from mipcandy.metrics import binary_dice
from mipcandy_bundles.unet import UNetPredictor

# Load validation data
_, val_dataset = NNUNetDataset("path/to/dataset").fold(fold=0)

# Set up predictor and evaluator
predictor = UNetPredictor("experiments/UNet/20240901-1234", (1, 128, 128), device="cuda")
evaluator = Evaluator(binary_dice)

# Predict on raw images and evaluate against labels
result = evaluator.predict_and_evaluate(
    [img for img, _ in val_dataset],
    [lbl for _, lbl in val_dataset],
    predictor
)

print(f"Mean Dice: {result.mean_metrics['binary_dice']:.4f}")
```

### Analyze Per-Case Performance

```python
result = evaluator.evaluate("predictions/", "labels/")

# Summary statistics
metric = "binary_dice"
scores = result.metrics[metric]
print(f"Mean:  {result.mean_metrics[metric]:.4f}")
print(f"Min:   {min(scores):.4f}")
print(f"Max:   {max(scores):.4f}")

# Identify failure cases (Dice < 0.5)
for case in result:
    if case.metrics[metric] < 0.5:
        print(f"Low score: {case.filename} = {case.metrics[metric]:.4f}")
```

### Compare Two Models

```python
from mipcandy.evaluation import Evaluator
from mipcandy.metrics import binary_dice, soft_dice
from mipcandy_bundles.unet import UNetPredictor
from mipcandy_bundles.cmunext import CMUNeXtPredictor

evaluator = Evaluator(binary_dice, soft_dice)

predictor_a = UNetPredictor("experiments/UNet/20240901-1234", (1, 128, 128), device="cuda")
predictor_b = CMUNeXtPredictor("experiments/CMUNeXt/20240905-5678", (3, 128, 128), device="cuda")

result_a = evaluator.predict_and_evaluate("test_images/", "test_labels/", predictor_a)
result_b = evaluator.predict_and_evaluate("test_images/", "test_labels/", predictor_b)

for metric in result_a.mean_metrics:
    score_a = result_a.mean_metrics[metric]
    score_b = result_b.mean_metrics[metric]
    print(f"{metric}: UNet={score_a:.4f}, CMUNeXt={score_b:.4f}")
```

### Inspect Best and Worst Cases

```python
result = evaluator.predict_and_evaluate("test_images/", "test_labels/", predictor)
metric = "binary_dice"

# Best case
best = result.max(metric)
print(f"Best:  {best.filename} ({best.metrics[metric]:.4f})")

# Worst case
worst = result.min(metric)
print(f"Worst: {worst.filename} ({worst.metrics[metric]:.4f})")

# Top 3 and bottom 3
print("\nTop 3:")
for case in result.max_n(metric, 3):
    print(f"  {case.filename}: {case.metrics[metric]:.4f}")

print("\nBottom 3:")
for case in result.min_n(metric, 3):
    print(f"  {case.filename}: {case.metrics[metric]:.4f}")
```

### Visualize Evaluation Results

```python
from mipcandy.data import overlay, visualize2d

result = evaluator.predict_and_evaluate("test_images/", "test_labels/", predictor)

# Visualize worst case
worst = result.min("binary_dice")
if worst.image is not None:
    # Overlay prediction on input image
    vis = overlay(worst.image, worst.output)
    visualize2d(vis, title=f"{worst.filename} (Dice={worst.metrics['binary_dice']:.4f})")
```
