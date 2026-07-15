# 2. Optimization — Types of Loss Functions

## Key Takeaways

- Regression and classification require different loss functions.
- MSE strongly penalizes large regression errors.
- MAE is more robust to outliers.
- Huber loss combines smooth optimization with robustness.
- Cross-entropy is the standard loss for probabilistic classification.
- Hinge loss encourages a classification margin.
- Surrogate losses are used when the true task metric is not differentiable.
- Training loss and evaluation metric are not necessarily the same.
- The loss function should reflect statistical assumptions and operational priorities.

---

## 1. Introduction

Different learning tasks require different loss functions.

The loss function determines how errors are measured, which gradients are produced and which behavior the model learns.

---

## 2. Regression Loss Functions

Regression predicts continuous values:

```math
y\in\mathbb{R}
```

The loss measures the difference between prediction $\hat{y}$ and target $y$.

---

## 2.1 Mean Squared Error

```math
\ell(\hat{y},y)
=
(\hat{y}-y)^2
```

The Mean Squared Error is:

```math
J(\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
(f_\theta(x_i)-y_i)^2
```

MSE strongly penalizes large errors and corresponds to maximum likelihood under Gaussian observation noise with fixed variance.

**Illustration:** MSE is suitable when large prediction errors should be penalized much more strongly than small ones.

### Limitation

MSE is sensitive to outliers because large errors are squared.

---

## 2.2 Mean Absolute Error

```math
\ell(\hat{y},y)
=
|\hat{y}-y|
```

The Mean Absolute Error is:

```math
J(\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
|f_\theta(x_i)-y_i|
```

MAE penalizes errors linearly and is more robust to outliers.

**Illustration:** MAE is preferable when extreme observations should not dominate training.

### Limitation

MAE is not differentiable when $\hat{y}=y$, although subgradients can be used.

---

## 2.3 Huber Loss

```math
\ell_\delta(\hat{y},y)
=
\begin{cases}
\frac{1}{2}(\hat{y}-y)^2,
&
|\hat{y}-y|\leq\delta
\\
\delta
\left(
|\hat{y}-y|-\frac{\delta}{2}
\right),
&
|\hat{y}-y|>\delta
\end{cases}
```

Huber loss behaves like MSE for small errors and like MAE for large errors.

**Illustration:** Huber loss is useful when most observations are reliable but a few contain large errors.

---

## 3. Classification Loss Functions

For binary classification:

```math
Y\in\{0,1\}
```

The model often outputs:

```math
p_\theta(x)=P(Y=1 \mid X=x)
```

where $p_\theta(x)$ must lie strictly between $0$ and $1$ when used inside the logarithm.

---

## 3.1 Zero-One Loss

```math
\ell(\hat{y},y)
=
\mathbf{1}[\hat{y}\neq y]
```

The empirical zero-one risk is:

```math
\hat{R}(\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbf{1}[f_\theta(x_i)\neq y_i]
```

It directly measures classification error.

**Illustration:** Zero-one loss evaluates whether the predicted class is correct, regardless of confidence.

### Limitation

It is discontinuous and non-differentiable, making it unsuitable for Gradient Descent.

---

## 3.2 Binary Cross-Entropy

```math
\ell(p,y)
=
-
\left[
y\log p
+
(1-y)\log(1-p)
\right]
```

If $y=1$:

```math
\ell(p,1)=-\log p
```

If $y=0$:

```math
\ell(p,0)=-\log(1-p)
```

Cross-entropy strongly penalizes confident incorrect predictions and is the negative log-likelihood of a Bernoulli model for binary labels.

**Illustration:** A confidently incorrect prediction receives a much larger penalty than an uncertain prediction.

---

## 3.3 Multi-Class Cross-Entropy

For $K$ classes:

```math
p_k=P(Y=k \mid X=x)
```

with:

```math
\sum_{k=1}^{K}p_k=1
```

The loss is:

```math
\ell(p,y)
=
-\sum_{k=1}^{K}
y_k\log p_k
```

For one-hot targets:

```math
\ell(p,y)
=
-\log p_y
```

Softmax produces class probabilities:

```math
p_k
=
\frac{e^{z_k}}
{\sum_{j=1}^{K}e^{z_j}}
```

**Illustration:** The model is rewarded for assigning high probability to the correct class.

---

## 3.4 Hinge Loss

For labels:

```math
y\in\{-1,+1\}
```

hinge loss is:

```math
\ell(f(x),y)
=
\max(0,1-yf(x))
```

where $f(x)$ is a signed score, not a probability.

If:

```math
yf(x)\geq1
```

the loss is zero.

**Illustration:** Hinge loss encourages predictions to remain on the correct side of the decision boundary with a margin.

---

## 4. Loss Functions as Surrogates

The true task objective may be difficult to optimize directly.

Differentiable **surrogate losses** such as cross-entropy and hinge loss are therefore used.

A useful surrogate should be computationally tractable, provide informative gradients and encourage improvement in the real task metric.

**Illustration:** Cross-entropy is optimized during training even when the final model is evaluated using accuracy or F1-score.

---

## 5. Loss Function vs Evaluation Metric

| Task | Training Loss | Evaluation Metric |
|---|---|---|
| Regression | MSE, MAE, Huber | RMSE, MAE, $R^2$ |
| Binary classification | Cross-entropy | Accuracy, precision, recall, F1 |
| Ranking | Pairwise or listwise loss | NDCG, MAP |
| Object detection | Composite loss | mAP |
| Language modeling | Cross-entropy | Perplexity, task accuracy |

A loss is optimized during training, while a metric evaluates the final system.

**Illustration:** A fraud model may be trained with cross-entropy but evaluated primarily using recall.

---

## 6. Choosing a Loss Function

The choice of loss should depend on:

- the task;
- probabilistic assumptions;
- sensitivity to outliers;
- class imbalance;
- operational costs;
- differentiability;
- the evaluation metric.

Typical choices:

- **MSE:** regression with approximately Gaussian noise;
- **MAE:** regression with significant outliers;
- **Huber:** robust regression with smooth optimization;
- **Cross-entropy:** probabilistic classification;
- **Hinge loss:** margin-based classification;
- **Weighted cross-entropy:** imbalanced classification.

**Illustration:** In an imbalanced medical problem, the loss may need to penalize false negatives more strongly than false positives.
