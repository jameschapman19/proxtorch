<div align="center">

<img src="docs/source/proxtorch-logo.jpg" alt="drawing" width="200"/>

# ProxTorch

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5748062.svg)](https://doi.org/10.5281/zenodo.4382739)
[![codecov](https://codecov.io/gh/jameschapman19/ProxTorch/branch/main/graph/badge.svg?token=JHG9VUB0L8)](https://codecov.io/gh/jameschapman19/ProxTorch)
[![version](https://img.shields.io/pypi/v/ProxTorch)](https://pypi.org/project/ProxTorch/)
[![downloads](https://img.shields.io/pypi/dm/ProxTorch)](https://pypi.org/project/ProxTorch/)

</div>

`ProxTorch` is a modern Python library offering a diverse range of proximal operators and constraints built on PyTorch. Leveraging PyTorch's capabilities, `ProxTorch` ensures efficiency and GPU compatibility, making it indispensable for various optimization and machine learning tasks.

## Features

- **GPU-Compatible**: Elevate computation speeds with comprehensive CUDA support.
- **PyTorch Harmony**: Perfectly fits within PyTorch-powered projects.
- **Robust Library**: From basic norms like `L0`, `L1`, `L2`, `L∞` to sophisticated regularizations including Total Variation, Fused Lasso, and beyond.
- **User Centric**: Seamlessly assimilate into your current projects requiring minimal modifications.

## Installation

Fetch `ProxTorch` with `pip`:

```bash
pip install proxtorch
```

## Quick Start

Here's a simple example showcasing how to use `ProxTorch`:

```python
import torch
from proxtorch.operators import L1

# Sample tensor
x = torch.tensor([0.5, -1.2, 0.3, -0.4, 0.7])

# Kickstart L1Prox proximal operator
l1_prox = L1(sigma=0.1)

# Determine the value of the regularization component
reg_value = l1_prox(x)
print("Regularization Value:", reg_value)

# Invoke the proximal operator
result = l1_prox.prox(x)
print("Prox Result:", result)
```

## Supported Proximal Operators

### Regularizers

- **L1**
- **L2 (Ridge)**
- **ElasticNet**
- **GroupLasso**
- **TV**: 
  - TV_2D
  - TV_3D 
  - TVL1_2D 
  - TVL1_3D
- **Frobenius**
- **Norms**:
  - TraceNorm
  - NuclearNorm
- **FusedLasso**
- **Huber**

### Constraints

- **L0Ball**
- **L1Ball**
- **L2Ball**
- **L∞Ball (Infinity Norm)**
- **Frobenius**
- **TraceNorm**
- **Box**


## Documentation

Still shaping up.

## Credits

This work is inspired by the following projects:

- [pyproximal](https://github.com/PyLops/pyproximal)
- [ProxGradPyTorch](https://github.com/KentonMurray/ProxGradPytorch)

We believe that `ProxTorch` is a valuable addition to the PyTorch ecosystem, and we hope that you find it useful in your projects.

## Contributing

We welcome contributions!

## License

`ProxTorch` is released under the [MIT License](link-to-license-file).
