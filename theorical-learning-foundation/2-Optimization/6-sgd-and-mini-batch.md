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

### Expectation

The **expectation** of a random variable is its probability-weighted average value:

```math
\mathbb{E}[X]
=
\sum_x x\,P(X=x).
```

For the stochastic gradient:

```math
g_t
=
\nabla \ell_{i_t}(\theta_t),
```

its conditional expectation is:

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

But this is exactly the full empirical-risk gradient:

```math
\boxed{
\mathbb{E}[g_t\mid\theta_t]
=
\nabla \hat R_n(\theta_t)
}
```

### Unbiased Estimator

An estimator is **unbiased** when its expected value equals the quantity it is trying to estimate:

```math
\mathbb{E}[\hat q]=q.
```

Therefore, (g_t) is an unbiased estimator of the full gradient.

This does **not** mean:

```math
g_t
=
\nabla \hat R_n(\theta_t)
```

for every individual step.

It means that if we repeatedly sampled stochastic gradients at the same parameter point and averaged them, that average would equal the full gradient.

### Quick Example

Suppose the gradients from three training examples are:

```math
g_1=2,
\qquad
g_2=4,
\qquad
g_3=6.
```

The full gradient is:

```math
\frac{2+4+6}{3}
=
4.
```

If SGD selects one example uniformly, the stochastic gradient may be (2), (4), or (6).

Its expectation is:

```math
\mathbb{E}[g]
=
\frac13(2)
+
\frac13(4)
+
\frac13(6)
=
4.
```

So an individual stochastic gradient can differ from the full gradient, while its average remains correct.

```math
\boxed{
\text{individual stochastic gradient}
\neq
\text{full gradient in general}
}
```

but:

```math
\boxed{
\mathbb{E}[\text{stochastic gradient}]
=
\text{full gradient}
}
```

**Illustration:** Individual stochastic gradients are noisy, but their average points toward the same gradient as the complete dataset.

---

## 5. Gradient Noise and Variance

Because a stochastic gradient is computed from only part of the dataset, it generally differs from the full empirical-risk gradient:

```math
g_t
\neq
\nabla \hat R_n(\theta_t).
```

We can write this difference as:

```math
g_t
=
\nabla \hat R_n(\theta_t)
+
\varepsilon_t,
```

where:

```math
\varepsilon_t
=
g_t
-
\nabla \hat R_n(\theta_t)
```

is the **gradient noise**.

Under unbiased sampling:

```math
\mathbb{E}[\varepsilon_t\mid\theta_t]=0.
```

This means the noise has no systematic direction: if we repeatedly sampled stochastic gradients at the same parameter point and averaged them, their average would approach the full-dataset gradient.

However, an individual stochastic gradient can still differ from the full gradient. This local variability is measured by:

```math
\mathbb{E}
\left[
\|\varepsilon_t\|_2^2
\mid\theta_t
\right]
>0.
```

Here, `\|\varepsilon_t\|_2^2` is the squared distance between the stochastic gradient and the full gradient. A positive expected value means the stochastic gradients still fluctuate around the full gradient even though their average is correct.

So the key distinction is:

```math
\boxed{
\mathbb{E}[\varepsilon_t\mid\theta_t]=0
\quad\Rightarrow\quad
\text{no systematic error}
}
```

while:

```math
\boxed{
\mathbb{E}[\|\varepsilon_t\|_2^2\mid\theta_t]>0
\quad\Rightarrow\quad
\text{local variance remains}
}
```

**Illustration:** SGD is unbiased because its average direction matches the full gradient, but it is noisy because each individual update can still deviate from that direction.
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

## 7. Practical Essentials

Batch size controls the trade-off between computation and noise: smaller batches are cheaper but produce noisier gradients, while larger batches are more stable but cost more per update. An **iteration** is one parameter update, an **epoch** is one complete pass through the dataset, and training data is usually shuffled before each epoch to avoid systematic ordering effects. Full-batch GD uses all (n) examples, SGD uses one example, and mini-batch SGD uses (b) examples; mini-batches are the practical default because they combine reasonably stable gradient estimates with efficient vectorized computation on modern hardware.
