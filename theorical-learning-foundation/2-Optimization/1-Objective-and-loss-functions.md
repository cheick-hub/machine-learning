# 2. Optimization — Objective and Loss Functions

## Key Takeaways

- A **loss function** measures the error made on one observation.
- The **empirical risk** averages the loss over the training dataset.
- The **expected risk** measures performance over the true data distribution.
- Machine Learning minimizes empirical risk because the true distribution is unknown.
- The **generalization gap** measures how far empirical risk is from expected risk.
- A regularized objective balances data fitting and model complexity.
- The chosen objective determines what behavior the model learns.

---

## 1. Introduction

Optimization begins by defining what the model is trying to minimize.

A Machine Learning model is represented by:

```math
f_\theta : \mathcal{X} \rightarrow \mathcal{Y}
```

Training consists of finding parameters that minimize an objective:

```math
\theta^*=\arg\min_\theta J(\theta)
```

The objective function translates the learning problem into a mathematical optimization problem.

---

## 2. Loss Function

A **loss function** measures the error made by the model on a single observation.

```math
\ell(f_\theta(x),y)
```

It assigns a small value to accurate predictions and a larger value to inaccurate predictions.

**Illustration:** In regression, a prediction close to the target should produce a smaller loss than a distant prediction.

---

## 3. Objective Function

Given:

```math
D=\{(x_i,y_i)\}_{i=1}^{n}
```

the empirical objective is:

```math
J(\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
\ell(f_\theta(x_i),y_i)
```

Training consists of solving:

```math
\theta^*
=
\arg\min_\theta J(\theta)
```

The loss evaluates one observation, while the objective evaluates the model over the dataset or a mini-batch.

**Illustration:** A model may perform well on one observation but still have a poor objective value over the complete dataset.

---

## 4. Expected Risk

The true objective is to minimize the expected loss over the unknown data distribution.

```math
R(\theta)
=
\mathbb{E}_{(X,Y)\sim P}
\left[
\ell(f_\theta(X),Y)
\right]
```

This quantity is called the **expected risk** or **population risk**.

Because the true distribution is unknown, expected risk cannot be computed directly.

**Illustration:** The real objective is to perform well on future data, not only on the training sample.

---

## 5. Empirical Risk

Expected risk is approximated using the training sample:

```math
\hat{R}_n(\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
\ell(f_\theta(x_i),y_i)
```

Training therefore follows:

```math
\hat{\theta}
=
\arg\min_\theta
\hat{R}_n(\theta)
```

This principle is called **Empirical Risk Minimization**.

**Illustration:** A model can achieve a low empirical risk while still performing poorly on unseen data.

---

## 6. Generalization Gap

The generalization gap is usually measured as the absolute difference between expected and empirical risk:

```math
\left|R(\theta)-\hat{R}_n(\theta)\right|
```

A small gap means training performance is close to population performance, while a large gap indicates poor generalization. In practice, expected risk is unknown, so test or validation risk is used as an estimate.

**Illustration:** Low training error combined with high test error indicates a large generalization gap.

---

## 7. Regularized Objective Functions

A regularization term can be added to control model complexity:

```math
J_{\text{reg}}(\theta)
=
\hat{R}_n(\theta)
+
\lambda\Omega(\theta)
```

where $\lambda$ controls regularization strength.

**Illustration:** Regularization discourages the model from relying on excessively large or unnecessary parameters.

---

## 8. L2 Regularization

L2 regularization uses:

```math
\Omega(\theta)=\|\theta\|_2^2
```

The objective becomes:

```math
J(\theta)
=
\hat{R}_n(\theta)
+
\lambda\sum_j\theta_j^2
```

It discourages large parameter values and typically produces smooth solutions.

**Illustration:** L2 regularization reduces sensitivity by preventing individual parameters from becoming excessively large.

---

## 9. L1 Regularization

L1 regularization uses:

```math
\Omega(\theta)=\|\theta\|_1
```

The objective becomes:

```math
J(\theta)
=
\hat{R}_n(\theta)
+
\lambda\sum_j|\theta_j|
```

L1 encourages sparsity, meaning some parameters can become exactly zero.

**Illustration:** L1 regularization can remove unimportant features by forcing their coefficients to zero.

---

## 10. Differentiability and Optimization

Gradient-based optimization requires a differentiable or subdifferentiable objective.

```math
\nabla_\theta J(\theta)
```

Gradient Descent updates parameters using:

```math
\theta_{t+1}
=
\theta_t
-
\eta\nabla_\theta J(\theta_t)
```

The shape of the objective determines gradient magnitude, stability and convergence behavior.

**Illustration:** Differentiable objectives provide informative gradients that guide parameter updates.

---

## 11. Optimization Hierarchy

```text
Per-sample loss
        ↓
Empirical risk
        ↓
Regularized objective
        ↓
Parameter optimization
```

More formally:

```math
\ell(f_\theta(x_i),y_i)
```

measures one prediction,

```math
\hat{R}_n(\theta)
=
\frac1n\sum_i\ell(f_\theta(x_i),y_i)
```

measures training performance, and:

```math
J(\theta)
=
\hat{R}_n(\theta)
+
\lambda\Omega(\theta)
```

is the complete objective optimized by the learning algorithm.
