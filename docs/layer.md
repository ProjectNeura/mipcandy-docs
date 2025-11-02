# Layer Configuration System

MIPCandy provides a dynamic layer configuration system centered around [`LayerT`](#mipcandy.layer.LayerT), which enables flexible module instantiation with deferred configuration.

## LayerT

[`LayerT`](#mipcandy.layer.LayerT) is a configuration container that stores a module type and its keyword arguments for lazy instantiation. This pattern is particularly useful when building neural networks with configurable components.

### Basic Usage

```python
from torch import nn
from mipcandy.layer import LayerT

# Create a LayerT configuration for Conv2d
conv = LayerT(nn.Conv2d, out_channels=64, kernel_size=3, padding=1)

# Assemble the module with additional arguments
conv_layer = conv.assemble(in_channels=3)
# Equivalent to: nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
```

### Key Methods

#### `assemble(*args, **kwargs) -> nn.Module`

Instantiates the module with stored arguments merged with provided arguments.

```python
# Configuration stored in LayerT
conv = LayerT(nn.Conv2d, out_channels=64, kernel_size=3)

# Instantiate with additional arguments
layer = conv.assemble(in_channels=32, padding=1)
# Result: nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
```

#### `update(*, must_exist: bool = True, **kwargs) -> Self`

Updates the stored keyword arguments. By default, only updates existing keys unless `must_exist=False`. Returns `self` for method chaining.

```python
# Define norm with string placeholders
norm = LayerT(nn.GroupNorm, num_groups="in_ch", num_channels="in_ch")

# Update num_groups, then assemble substitutes num_channels
# This is necessary for GroupNorm which has two mandatory parameters
in_ch = 64
layer = norm.update(num_groups=in_ch).assemble(in_ch=in_ch)
# Result: nn.GroupNorm(num_groups=64, num_channels=64)

# Update existing parameter
conv = LayerT(nn.Conv2d, kernel_size=3, padding=1)
conv.update(kernel_size=5)  # Changes kernel_size to 5

# Add new parameter (requires must_exist=False)
conv.update(must_exist=False, bias=False)  # Adds bias parameter
```

#### `__init__(m: type[nn.Module], **kwargs)`

Creates a new LayerT configuration.

```python
# Store module type and default parameters
norm = LayerT(nn.BatchNorm2d, eps=1e-5, momentum=0.1)
```

### String Parameter Substitution

LayerT supports string-based parameter substitution, where string values in stored kwargs are replaced by corresponding values from `assemble()` kwargs:

```python
# Use string as placeholder
norm = LayerT(nn.BatchNorm2d, num_features="in_ch")

# Substitute during assembly
bn = norm.assemble(in_ch=64)
# Result: nn.BatchNorm2d(num_features=64)
```

This pattern is crucial when the same parameter value needs to be passed to multiple components:

```python
# Always provide parameters that any possible entity might use
def build_layer(in_ch: int, norm: LayerT):
    # Even if current norm doesn't use in_ch, always pass it
    return norm.assemble(in_ch=in_ch)
```

### Design Pattern in MIPCandy

LayerT is extensively used throughout MIPCandy for configurable module construction:

```python
import torch
from torch import nn
from mipcandy.layer import LayerT

class ConfigurableBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        conv: LayerT = LayerT(nn.Conv2d, kernel_size=3, padding=1),
        norm: LayerT = LayerT(nn.BatchNorm2d),
        act: LayerT = LayerT(nn.ReLU, inplace=True)
    ) -> None:
        super().__init__()
        self.conv: nn.Module = conv.assemble(in_channels=in_ch, out_channels=out_ch)
        self.norm: nn.Module = norm.assemble(num_features=out_ch)
        self.act: nn.Module = act.assemble()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))

# Usage with different configurations
# Default configuration
block1 = ConfigurableBlock(32, 64)

# Custom normalization
block2 = ConfigurableBlock(32, 64, norm=LayerT(nn.GroupNorm, num_groups=8))

# Custom activation
block3 = ConfigurableBlock(32, 64, activation=LayerT(nn.GELU))
```

## Device Management

### HasDevice

[`HasDevice`](#mipcandy.layer.HasDevice) is a base class that provides device management capabilities:

```python
import torch
from mipcandy.layer import HasDevice

class MyComponent(HasDevice):
    def __init__(self, device: str | torch.device | None = "cuda") -> None:
        super().__init__(device)
        # self._device is now available

    def process(self, data: torch.Tensor) -> torch.Tensor:
        # Get current device
        device = self.device()
        # Move data to device
        return data.to(device)
```

### WithPaddingModule

[`WithPaddingModule`](#mipcandy.layer.WithPaddingModule) extends `HasDevice` with lazy-loaded padding and restoring modules:

```python
import torch
from torch import nn
from mipcandy.layer import WithPaddingModule

class MyPredictor(WithPaddingModule):
    def __init__(self, device: str | torch.device | None = "cuda") -> None:
        super().__init__(device)

    def build_padding_module(self) -> nn.Module | None:
        # Return padding module or None
        from mipcandy.common import Pad2d
        return Pad2d((128, 128))

    def build_restoring_module(self, padding_module: nn.Module | None) -> nn.Module | None:
        # Return restoring module or None
        if padding_module:
            from mipcandy.common import Restore2d
            return Restore2d(padding_module)
        return None

    def process(self, image: torch.Tensor) -> torch.Tensor:
        # Lazy load padding module
        padding = self.get_padding_module()
        if padding:
            image = padding(image)

        # ... processing ...

        # Lazy load restoring module
        restoring = self.get_restoring_module()
        if restoring:
            image = restoring(image)

        return image
```

The padding and restoring modules are built only once on first access, then cached for subsequent calls.

## Utility Functions

### batch_int_multiply

Multiplies multiple integers by a float, ensuring results are integers:

```python
from mipcandy.layer import batch_int_multiply

# Scale multiple dimensions
scaled = list(batch_int_multiply(0.5, 128, 256, 512))
# Result: [64, 128, 256]

# Raises ValueError if result is not an integer
try:
    list(batch_int_multiply(0.3, 100))  # 100 * 0.3 = 30.0
except ValueError:
    print("Inequivalent conversion")
```

### batch_int_divide

Divides multiple integers by a float, ensuring results are integers:

```python
from mipcandy.layer import batch_int_divide

# Downscale dimensions
downscaled = list(batch_int_divide(2, 128, 256, 512))
# Result: [64, 128, 256]
```

## Best Practices

1. **Always pass potential parameters**: Even if the current module doesn't use a parameter, always pass it during assembly to support different module types:
   ```python
   # Good: Always pass in_ch
   norm.assemble(in_ch=64)

   # Avoid: Conditional parameter passing
   if needs_in_ch:
       norm.assemble(in_ch=64)
   else:
       norm.assemble()
   ```

2. **Use keyword-only arguments**: When designing configurable components, use `*` to separate required and optional parameters:
   ```python
   def __init__(self, in_ch: int, out_ch: int, *, conv: LayerT = ...):
       pass
   ```

3. **Avoid redundant parameter copying**: Pass LayerT instances directly without copying:
   ```python
   # Good
   MyModule(conv=conv, norm=norm)

   # Avoid
   MyModule(conv=LayerT(conv.m, **conv.kwargs), norm=norm)
   ```

4. **Use update() for dynamic configuration**: Modify configurations dynamically based on runtime conditions:
   ```python
   if use_bias:
       conv.update(must_exist=False, bias=True)
   ```