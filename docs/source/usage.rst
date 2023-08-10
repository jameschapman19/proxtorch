Usage
=====

Armed with an understanding of the mathematical foundation behind proximal operators and their importance in optimization, let's dive into how to utilize ProxTorch in your PyTorch-based projects.

**Step-by-Step Guide to Using ProxTorch**
-----------------------------------------

Getting started with ProxTorch is simple and intuitive. Here's a basic guide:

1. **Initialize an operator**

   To utilize a proximal operator in your optimization routine, first, you need to instantiate it. For instance, to leverage the L1 proximal operator:

   .. code-block:: python

      from proxtorch import L1Prox
      l1_prox = L1Prox(sigma=0.1)

   Here, `sigma` is a parameter that controls the strength of the regularization. Adjusting `sigma` will influence the solution's sparsity when using the L1 proximal operator.

2. **Apply the operator**

   Once you've initialized the desired proximal operator, apply it to a tensor. This "proximal step" will push the tensor closer to the minimizer of the associated function:

   .. code-block:: python

      result = l1_prox.prox(some_tensor)

   The result is a tensor that has undergone the proximal transformation based on the L1 regularizer.

3. **Evaluate the regularization term**

   Beyond applying the proximal transformation, you can also assess the value of the regularization term for a given tensor using the `__call__` method:

   .. code-block:: python

      regularization_value = l1_prox(some_tensor)

   This will provide you with the regularization value associated with the tensor for the specific proximal operator.

**Expanding Your Horizons**
---------------------------

ProxTorch offers a rich library of proximal operators beyond the L1 operator, suitable for various optimization challenges. To explore the full capabilities and available operators, refer to the API Reference. As you delve deeper, you'll find tools tailored for specific problem structures, enabling you to harness the power of proximal methods in diverse optimization landscapes.
