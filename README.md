<div align="center">

<img src="docs/source/proxtorch-logo.jpg" alt="drawing" width="200"/>

# ProxTorch

</div>

`ProxTorch` is a modern Python library that provides an assortment of proximal operators built on top of PyTorch. By leveraging the power of PyTorch, `ProxTorch` is efficient, differentiable, and GPU-compatible, making it suitable for a wide range of optimization and machine learning tasks.

## Features

- **GPU-Compatible**: Achieve faster computation speeds with CUDA support.
- **Differentiable**: Seamlessly integrate with PyTorch-based projects, taking advantage of the autograd feature.
- **Extensive Library**: From `L0`, `L1`, `L2` norms to advanced regularizations like Total Variation, Fused Lasso, and more.
- **User-Friendly**: Easily plug-and-play into existing projects with minimal code changes.

## Installation

Install `ProxTorch` using `pip`:

```
pip install proxtorch
```

## Quick Start

Here's a simple example showcasing how to use `ProxTorch`:

```python
import torch
from proxtorch.operators import L1

# Create a tensor
x = torch.tensor([0.5, -1.2, 0.3, -0.4, 0.7])

# Initialize L1Prox proximal operator
l1_prox = L1(sigma=0.1)

# Compute the value of the regularization term
reg_value = l1_prox(x)
print("Regularization Value:", reg_value)

# Apply the proximal operator
result = l1_prox.prox(x)
print("Prox Result:", result)

```

## Supported Proximal Operators

- L0, L0 ball
- L1, L1 ball
- L2 (Ridge)
- Total Variation (2D, 3D)
- Elastic Net
- Fused Lasso
- Group Lasso
- Huber
- Trace Norm
- Non-negative
- Frobenius Norm


## Documentation

Work in progress.

## Credits

This work is inspired by the following projects:

- [pyproximal](https://github.com/PyLops/pyproximal)
- [ProxGradPyTorch](https://github.com/KentonMurray/ProxGradPytorch)

We believe that `ProxTorch` is a valuable addition to the PyTorch ecosystem, and we hope that you find it useful in your projects.

## Contributing

We welcome contributions!

## License

`ProxTorch` is released under the [MIT License](link-to-license-file).