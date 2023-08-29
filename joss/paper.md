---
title: 'ProxTorch: Unleashing Proximal Gradient Descent on PyTorch'
tags:
  - Python
  - PyTorch
  - optimization
  - proximal gradient descent
  - machine learning
authors:
  - name: James Chapman
    orcid: 0000-0002-9364-8118
    affiliation: "1"
  # Add more authors and their details as necessary
affiliations:
  - name: Centre for Medical Image Computing, University College London, London, UK
    index: 1
  # Add more affiliations as necessary
date: 16 August 2023
bibliography: paper.bib
---

# Summary

Imagine embarking on a road trip. Some parts of your journey are smooth, freshly-paved highways, while others are rough, potholed streets. Gradient descent techniques are like driving at a constant speed, which works well on the highways but poses risks on the rough terrains. Proximal gradient descent, in contrast, is akin to an adaptive cruise control system in your car. It lets you speed on the highways and carefully maneuver through the challenging parts, ensuring you reach your destination effectively.

In the realm of computational optimization, this translates to navigating a landscape with both smooth and non-smooth terrains. PyTorch [@paszke2019pytorch], a leading deep learning framework, excels in the 'highway' scenarios but is less equipped for the 'potholed streets'. Enter `ProxTorch`: a library crafted to seamlessly blend the power of proximal gradient descent with the prowess of PyTorch, ensuring an optimized journey through the diverse landscapes of optimization problems.

# Statement of Need

The challenges in optimization continue to evolve as the machine learning and data science landscape broadens. PyTorch is versatile, offering state-of-the-art modeling capabilities. Yet, its conventional mechanisms sometimes fall short in landscapes replete with non-smooth complexities. 

`ProxTorch` introduces regularization-style proximal operators, which act like "soft constraints", along with constraint-style operators for stricter requirements. This versatility becomes pivotal for tasks that need specific features in their solutions, such as sparsity or bounded values.

For researchers navigating the terrains of compressed sensing, image reconstruction, and sparse modeling, `ProxTorch` offers a smoother ride. It harmoniously combines proximal gradient descent's nuanced adaptability with PyTorch's GPU-accelerated computational strengths.

Inspiration for `ProxTorch` stems from PyProximal [@pyproximal], a Python library rich in proximal operators and algorithms. However, PyProximal doesn't gel with PyTorch's GPU capabilities. Overcoming this, `ProxTorch` provides linear operators that sync flawlessly with PyTorch tensors and devices, while also leveraging PyTorch’s automatic differentiation to efficiently compute gradients.

# Documentation

Journey deeper into `ProxTorch` with our comprehensive documentation at [https://proxtorch.readthedocs.io/en/latest/](https://arxiv.org/abs/1912.01703).

# Acknowledgements

JC is grateful for the support from the EPSRC-funded UCL Centre for Doctoral Training in Intelligent, Integrated Imaging in Healthcare ( i4health) (EP/S021930/1) and the Department of Health’s NIHR-funded Biomedical Research Centre at University College London Hospitals.

# References
