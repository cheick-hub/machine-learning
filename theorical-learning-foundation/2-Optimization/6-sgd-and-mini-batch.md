# 6. Optimization — SGD and Mini-Batch Gradient Descent

## Key Takeaways

- Full-batch Gradient Descent computes the gradient using the entire training dataset before every parameter update.
- Stochastic Gradient Descent (SGD) replaces the full gradient with a gradient computed from a randomly selected training example.
- A stochastic gradient is noisy, but under uniform sampling it is an unbiased estimate of the full empirical-risk gradient.
- Mini-batch SGD averages gradients over a small subset of examples and reduces variance compared with single-example SGD.
- Smaller batches give cheaper but noisier updates; larger batches give more stable but more expensive updates.
- An epoch is one full pass through the training dataset, while an iteration is one parameter update.
- Mini-batch SGD is the practical default for training many modern machine-learning models.

---

## 1. Empirical Risk and Full-Batch Gradient Descent

For a training dataset with (n) examples, empirical risk is:

```math
\hat R_n(\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
\ell_i(\theta),
```

where:

- (	heta) is the model parameter vector;
- (ell_i(	heta)) is the loss on example (i).

The gradient of the empirical risk is:

```math
\nabla \hat R_n(\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
\nabla \ell_i(\theta).
```

Full-batch Gradient Descent updates the parameters using:

```math
\theta_{t+1}
=
\theta_t
-
\eta
\nabla \hat R_n(\theta_t).
```

This uses the exact gradient of the empirical objective at each iteration.

**Illustration:** Full-batch Gradient Descent asks every training example how the parameters should change before making a single update.

---

## 2. The Computational Problem with Full Gradients

If the dataset is large, evaluating:

```math
\nabla \hat R_n(\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
\nabla \ell_i(\theta)
```

requires processing all (n) examples before every update.

For large datasets or expensive models, this can make each iteration costly.

The key question becomes:

```text
Do we really need the exact full gradient before every parameter update?
```

SGD answers: no.

Instead, we can use a cheaper estimate of the full gradient and update the parameters more frequently.

**Illustration:** Rather than consulting the entire dataset before moving, stochastic methods estimate a useful direction from only part of it.

---

## 3. Stochastic Gradient Descent

In Stochastic Gradient Descent, choose one training example (i_t) at iteration (t).

Define the stochastic gradient:

```math
g_t
=
\nabla \ell_{i_t}(\theta_t).
```

Then update:

```math
\theta_{t+1}
=
\theta_t
-
\eta g_t.
```

Equivalently:

```math
\theta_{t+1}
=
\theta_t
-
\eta
\nabla \ell_{i_t}(\theta_t).
```

Unlike full-batch Gradient Descent, this update does not use the exact empirical-risk gradient.

It uses only one example.

Therefore (g_t) can point somewhat differently from:

```math
\nabla \hat R_n(\theta_t).
```

The update is cheaper, but noisier.

**Illustration:** One example gives a quick estimate of the downhill direction, but that estimate may not exactly match the direction suggested by the whole dataset.

---

## 4. Why the Stochastic Gradient Is Unbiased

Suppose (i_t) is sampled uniformly from:

```math
\{1,2,\dots,n\}.
```

Then every example has probability:

```math
P(i_t=i)=\frac{1}{n}.
```

The expected stochastic gradient is:

```math
\mathbb{E}[g_t\mid\theta_t]
=
\mathbb{E}
\left[
\nabla \ell_{i_t}(\theta_t)
\mid\theta_t
\right].
```

Because sampling is uniform:

```math
\mathbb{E}[g_t\mid\theta_t]
=
\frac{1}{n}
\sum_{i=1}^{n}
\nabla \ell_i(\theta_t).
```

Therefore:

```math
\boxed{
\mathbb{E}[g_t\mid\theta_t]
=
\nabla \hat R_n(\theta_t)
}
```

So the stochastic gradient is an **unbiased estimator** of the full gradient.

This does not mean every stochastic gradient equals the full gradient.

It means that if we repeatedly sampled gradients at the same parameter point and averaged them, the average would equal the full gradient.

**Illustration:** Individual stochastic gradients are noisy, but their average points toward the same gradient as the complete dataset.

---

## 5. Gradient Noise and Variance

Because:

```math
g_t
\neq
\nabla \hat R_n(\theta_t)
```

in general, we can write:

```math
g_t
=
\nabla \hat R_n(\theta_t)
+
\varepsilon_t,
```

where (arepsilon_t) represents gradient noise.

Under unbiased sampling:

```math
\mathbb{E}[\varepsilon_t\mid\theta_t]=0.
```

However, the noise may have nonzero variance:

```math
\mathbb{E}
\left[
\|\varepsilon_t\|_2^2
\mid\theta_t
\right]
>0.
```

This means SGD updates fluctuate around the direction of the full gradient.

The noise can make the optimization path less smooth, but it also allows many inexpensive updates.

**Illustration:** SGD does not descend along a perfectly smooth path; it moves downhill while repeatedly wobbling around the full-gradient direction.

---

## 6. Mini-Batch Gradient Descent

Instead of using one example, we can use a subset of examples.

Let:

```math
B_t
\subset
\{1,\dots,n\}
```

be a mini-batch containing (b) examples.

The mini-batch gradient is:

```math
g_t
=
\frac{1}{b}
\sum_{i\in B_t}
\nabla \ell_i(\theta_t).
```

The update becomes:

```math
\boxed{
\theta_{t+1}
=
\theta_t
-
\eta
\frac{1}{b}
\sum_{i\in B_t}
\nabla \ell_i(\theta_t)
}
```

Mini-batch SGD lies between the two extremes:

```text
Single-example SGD  ←  Mini-batch SGD  →  Full-batch GD
```

If:

```math
b=1,
```

we recover single-example SGD.

If:

```math
b=n,
```

we recover full-batch Gradient Descent.

**Illustration:** A mini-batch uses enough examples to stabilize the gradient estimate without paying the cost of processing the entire dataset before each update.

---

## 7. Batch Size and Gradient Variance

The mini-batch gradient averages several individual gradients.

Averaging reduces random variation.

Under approximately independent sampling, gradient variance decreases roughly as batch size increases:

```math
\operatorname{Var}(g_t)
\propto
\frac{1}{b}.
```

This gives the basic trade-off:

```math
\boxed{
\text{smaller batch}
\Rightarrow
\text{cheaper but noisier gradient}
}
```

and:

```math
\boxed{
\text{larger batch}
\Rightarrow
\text{more expensive but more stable gradient}
}
```

A larger batch does not automatically mean faster optimization in wall-clock time because each update requires more computation.

**Illustration:** Increasing batch size makes each gradient estimate more reliable, but each estimate costs more to compute.

---

## 8. Epochs, Batches, and Iterations

These terms describe different parts of the training process.

### Epoch

An **epoch** is one complete pass through the training dataset.

If the dataset contains (n) examples, one epoch processes all (n) examples once.

### Batch

A **batch** is the subset of examples used to compute one gradient estimate.

If batch size is (b), each batch contains (b) examples.

### Iteration

An **iteration** is one parameter update.

With mini-batch training, the approximate number of iterations per epoch is:

```math
\frac{n}{b}.
```

For example, reducing the batch size increases the number of parameter updates per epoch.

**Illustration:** An epoch measures dataset coverage, while an iteration measures how many optimization updates have occurred.

---

## 9. Shuffling the Dataset

In practice, training examples are commonly shuffled before each epoch.

The dataset is then divided into mini-batches.

This reduces systematic ordering effects.

Without shuffling, examples with similar structure may appear repeatedly in the same sequence, producing correlated updates.

A typical training cycle is:

```text
shuffle dataset
      ↓
split into mini-batches
      ↓
compute mini-batch gradient
      ↓
update parameters
      ↓
continue until all batches are processed
      ↓
start next epoch
```

Shuffling does not change the empirical objective itself; it changes the sequence of gradient estimates used during optimization.

**Illustration:** Shuffling prevents the optimizer from repeatedly seeing training examples in the same potentially biased order.

---

## 10. Full-Batch GD vs SGD vs Mini-Batch SGD

| Method | Examples per update | Gradient quality | Cost per update | Update behavior |
| --- | ---: | --- | --- | --- |
| Full-batch GD | (n) | Exact empirical gradient | High | Stable |
| SGD | (1) | Very noisy estimate | Very low | Highly stochastic |
| Mini-batch SGD | (b) | Lower-noise estimate | Moderate | Stochastic but more stable |

The methods optimize the same empirical objective but use different approximations to its gradient.

The main difference is not the objective function.

It is how much data is used to estimate the gradient before each update.

**Illustration:** Full-batch, stochastic, and mini-batch methods differ mainly in how accurately and expensively they estimate the same underlying gradient.

---

## 11. Why Mini-Batch SGD Is Common in Practice

Mini-batch SGD provides a useful compromise between computational efficiency and gradient stability.

It allows:

- more frequent updates than full-batch Gradient Descent;
- lower gradient variance than single-example SGD;
- efficient vectorized computation;
- effective use of parallel hardware such as GPUs.

The core update remains:

```math
\theta_{t+1}
=
\theta_t
-
\eta g_t,
```

but now (g_t) is a mini-batch estimate of the full gradient.

The conceptual progression is:

```text
Full empirical gradient
        ↓
expensive exact update
        ↓
stochastic gradient estimate
        ↓
cheap but noisy update
        ↓
mini-batch averaging
        ↓
balance between cost and stability
```

The central theoretical property is:

```math
\boxed{
\mathbb{E}[g_t\mid\theta_t]
=
\nabla \hat R_n(\theta_t)
}
```

while the central practical trade-off is:

```math
\boxed{
\text{batch size controls the balance between gradient noise and computation.}
}
```
