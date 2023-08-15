<div align="center">

<img src="docs/source/proxtorch-logo.jpg" alt="ProxTorch Logo" width="200"/>

# ProxTorch 

**Unleashing Proximal Gradient Descent on PyTorch** 🚀

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5748062.svg)](https://doi.org/10.5281/zenodo.4382739)
[![codecov](https://codecov.io/gh/jameschapman19/ProxTorch/graph/badge.svg?token=909RDXcEZK)](https://codecov.io/gh/jameschapman19/ProxTorch)
[![version](https://img.shields.io/pypi/v/ProxTorch)](https://pypi.org/project/ProxTorch/)
[![downloads](https://img.shields.io/pypi/dm/ProxTorch)](https://pypi.org/project/ProxTorch/)

</div>

🔍 **What is ProxTorch?**  
Dive into a rich realm of proximal operators and constraints with `ProxTorch`, a state-of-the-art Python library crafted on PyTorch. Whether it's optimization challenges or the complexities of machine learning, `ProxTorch` is designed for speed, efficiency, and seamless GPU integration.

## ✨ **Features**

- **🚀 GPU-Boosted**: Experience lightning-fast computations with extensive CUDA support.
- **🔥 PyTorch Synergy**: Naturally integrates with all your PyTorch endeavours.
- **📚 Expansive Library**: From elemental norms (`L0`, `L1`, `L2`, `L∞`) to advanced regularizations like Total Variation and Fused Lasso.
- **🤝 User-Friendly**: Jump right in! Intuitive design means minimal disruptions to your existing projects.

## 🛠 **Installation**

Getting started with `ProxTorch` is a breeze:

```bash
pip install proxtorch
```

## 🚀 **Quick Start**

Dive in with this straightforward example:

```python
import torch
from proxtorch.operators import L1

# Define a sample tensor
x = torch.tensor([0.5, -1.2, 0.3, -0.4, 0.7])

# Initialize the L1Prox proximal operator
l1_prox = L1(sigma=0.1)

# Compute the regularization component value
reg_value = l1_prox(x)
print("Regularization Value:", reg_value)

# Apply the proximal operator
result = l1_prox.prox(x)
print("Prox Result:", result)
```

## 📜 **Diverse Proximal Operators**

### **Regularizers**

- **L1**, **L2 (Ridge)**, **ElasticNet**, **GroupLasso**, **TV** (includes TV_2D, TV_3D, TVL1_2D, TVL1_3D), **Frobenius**  
- **Norms**: TraceNorm, NuclearNorm
- **FusedLasso**, **Huber**

### **Constraints**

- **L0Ball**, **L1Ball**, **L2Ball**, **L∞Ball (Infinity Norm)**, **Frobenius**, **TraceNorm**, **Box**

## 📖 **Documentation**

Explore the comprehensive documentation on [Read the Docs](https://proxtorch.readthedocs.io/en/latest/).

## 🙌 **Credits**

`ProxTorch` stands on the shoulders of giants:

- [pyproximal](https://github.com/PyLops/pyproximal)
- [ProxGradPyTorch](https://github.com/KentonMurray/ProxGradPytorch)

We're thrilled to introduce `ProxTorch` as an exciting addition to the PyTorch ecosystem. We're confident you'll love it!

## 🤝 **Contribute to the ProxTorch Revolution**

Got ideas? Join our vibrant community and make `ProxTorch` even better!

## 📜 **License**

`ProxTorch` is proudly released under the [MIT License](link-to-license-file).
```