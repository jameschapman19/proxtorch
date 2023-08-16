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
    orcid: Your ORCID Here
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

Optimization is a cornerstone in many domains, and its intuitive idea can be likened to navigating a landscape of valleys and mountains. Classic gradient descent is reminiscent of a ball smoothly rolling down a hill. But when landscapes have peculiar terrains like sharp cliffs or plateaus, proximal gradient descent comes into play, blending the simplicity of gradient descent with a "magnetic pull" from these unique terrains, the non-smooth component.

PyTorch, with its GPU acceleration, has emerged as the de facto standard for gradient descent tasks amongst researchers. Yet, merging the capabilities of proximal gradient descent into PyTorch is non-trivial. Enter `ProxTorch`: a library integrating the might of proximal gradient descent within the optimized framework of PyTorch.

# Statement of Need

As machine learning and data science evolve, the challenges in optimization diversify. While PyTorch offers advanced modeling capabilities, it's less equipped for problems where the optimization landscape incorporates complex features or constraints. 

`ProxTorch` ushers in regularization-style proximal operators, which can be perceived as "soft constraints", alongside constraint-style operators that enforce stricter requirements. Such dual capabilities are invaluable for tasks demanding specific features in solutions, like sparsity or bounded values.

Researchers venturing into fields like compressed sensing, image reconstruction, and sparse modeling will find `ProxTorch` indispensable. It synergizes with PyTorch, enabling users to harness the combined strengths of proximal gradient descent and GPU-powered computations.

# Non-mathematical Intuition

Imagine an optimization problem as a landscape filled with hills, valleys, cliffs, and plateaus. The goal is to find the lowest point. Gradient descent is like a ball rolling down, always seeking the easiest path downhill. Proximal gradient descent introduces a unique twist. In areas with sharp cliffs or plateaus (our non-smooth components), it's as if the ball has a magnet pulling it towards specific points of interest, ensuring it doesn't just get stuck or go astray.

# Mathematics

Diving into the formalities, the optimization goal is to minimize a function which is a composite of a smooth component \( f \) and a non-smooth component \( g \):

\[ \min_{x} \{ f(x) + g(x) \} \]

The proximal gradient descent refines solutions through:

1. Gradient descent for the smooth part, and
2. A proximal step for the non-smooth part.

Mathematically, the iteration process is captured as:

\[ x^{(k+1)} = \text{prox}_{\alpha g}(x^{(k)} - \alpha \nabla f(x^{(k)})) \]

Where \( \alpha \) represents a step size, and \( \text{prox}_{\alpha g} \) signifies the proximal operator of function \( g \) scaled by \( \alpha \). `ProxTorch` seamlessly evaluates and integrates such proximal operators within PyTorch's ecosystem.

# Documentation

`ProxTorch`'s documentation is available at [https://proxtorch.readthedocs.io/en/latest/](https://proxtorch.readthedocs.io/en/latest/).

# Acknowledgements

JC is supported by the EPSRC-funded UCL Centre for Doctoral Training in Intelligent, Integrated Imaging in Healthcare ( i4health) (EP/S021930/1) and the Department of Health’s NIHR-funded Biomedical Research Centre at University College London Hospitals.

# References