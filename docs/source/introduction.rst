Introduction
============

Welcome to the documentation of ProxTorch!

ProxTorch is a cutting-edge library that provides a suite of proximal operators, fundamental building blocks in the realm of optimization, especially for problems with non-smooth regularization terms.

**What are Proximal Operators?**
---------------------------------

Mathematically, given a convex function \( f: \mathbb{R}^n \to \mathbb{R} \cup \{ +\infty \} \), the proximal operator of \( f \) is defined as:

\[ \text{prox}_f(v) = \arg \min_x \left\{ f(x) + \frac{1}{2} \| x - v \|_2^2 \right\} \]

In simple terms, the proximal operator is a tool that takes an input vector \( v \) and returns a vector that is closer to the minimizer of \( f \). The operator effectively "pushes" \( v \) closer to the region where \( f \) is small.

**How Do Proximal Operators Enhance Gradient Descent?**
-------------------------------------------------------

Gradient Descent is a popular iterative method used to find the local minimum of a differentiable scalar function. The idea is to take repeated steps in the direction of the steepest decrease of the function.

When dealing with non-smooth functions or problems with constraints, the conventional gradient descent can be ill-behaved or even diverge. Proximal operators come to the rescue in these cases. By combining gradient descent with proximal operations, we obtain the Proximal Gradient Descent method. This hybrid method leverages the strength of gradient information and the structure of the problem (via proximal operators) to ensure convergence and find solutions that respect constraints or non-smooth structures.

In other words, while gradient descent provides the direction to move, proximal operators ensure that this movement adheres to the inherent structure or constraints of the problem.

**Why ProxTorch on PyTorch?**
------------------------------

PyTorch is a leading framework designed for gradient-based optimization in deep learning. However, as the boundaries between traditional optimization problems and deep learning blur, there's an increasing need for sophisticated optimization tools within the PyTorch ecosystem.

With ProxTorch, researchers and developers can seamlessly integrate advanced optimization techniques into their PyTorch-based machine learning workflows. Whether you're solving a traditional convex optimization problem, training a neural network with custom regularizers, or anything in between, ProxTorch provides the necessary tools, all with the familiarity and flexibility of PyTorch.