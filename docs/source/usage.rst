Usage
=====

With the foundational knowledge of proximal operators and constraints and their roles in optimization, it's time to explore the practical application of ProxTorch in PyTorch-centric endeavors.

Step-by-Step Guide to Using ProxTorch
-----------------------------------------

Venturing into ProxTorch is a straightforward process. Here's your primer:

1. **Initialize an operator or constraint**

   To embed a proximal operator or constraint in your optimization sequence, initiate it. For instance, for the L1 proximal operator:

   .. code-block:: python

      from proxtorch.operators import L1Prox
      l1_prox = L1Prox(sigma=0.1)

   Meanwhile, for constraints like the L1Ball:

   .. code-block:: python

      from proxtorch.constraints import L1Ball
      l1_constraint = L1Ball(radius=0.5)

   In the L1 proximal operator, `sigma` tweaks the regularization strength. A tweak in `sigma` modifies the sparsity of the solution under the L1 proximal operator. For the L1Ball constraint, `radius` defines the feasible set's boundary.

2. **Apply the operator or enforce the constraint**

   Having initialized your chosen proximal operator or constraint, apply it to a tensor. This step either nudges the tensor closer to the function's minimizer (for operators) or ensures it abides by the defined constraint:

   .. code-block:: python

      result_operator = l1_prox.prox(some_tensor)
      result_constraint = l1_constraint.prox(some_tensor)

   The `result_operator` tensor has felt the influence of the L1 regularizer, while the `result_constraint` tensor adheres to the L1Ball constraint.

3. **Assess the regularization or constraint term**

   Beyond the application phase, one can also gauge the value of the regularization term or constraint penalty for any tensor via the `__call__` method:

   .. code-block:: python

      regularization_value = l1_prox(some_tensor)
      constraint_penalty = l1_constraint(some_tensor)

   This yields the regularization value or penalty related to the tensor for the chosen proximal operator or constraint, respectively.

Broadening Your Scope
-------------------------

ProxTorch offers a comprehensive library of proximal operators and constraints suitable for various optimization challenges. Explore the API Reference for detailed insights. Throughout the documentation, you'll find tools tailored for specific problem structures, allowing you to use proximal strategies in different optimization settings.
