# 9. Optimization — Non-Differentiable Optimization

## Key Takeaways

- Ordinary Gradient Descent requires differentiability, but some important objectives are non-differentiable.
- For convex non-differentiable functions, **subgradients** generalize gradients.
- The set of all subgradients at a point is the **subdifferential**.
- The **subgradient method** extends Gradient Descent to convex non-smooth objectives.
- For objectives of the form (F=f+g), where (f) is smooth and (g) is non-smooth, **proximal gradient methods** are often more effective.
- The proximal operator is obtained by solving a small optimization problem associated with (g).
- For L1 regularization, the proximal operator is soft-thresholding, which can set parameters exactly to zero.

---

## 1. Why Non-Differentiability Matters

So far, Gradient Descent has relied on:

```math
\nabla J(\theta).
```

But some important objectives are not differentiable everywhere.

A classic example is:

```math
f(x)=|x|.
```

At:

```math
x=0,
```

the ordinary derivative does not exist.

This matters in machine learning because non-smooth terms appear naturally, especially with L1 regularization:

```math
\lambda\|\theta\|_1.
```

When ordinary gradients fail, we need other tools.

---

## 2. Subgradients

For a convex function (f), a vector (g) is a **subgradient** at (x) if:

```math
f(y)
\ge
f(x)
+
g^\top(y-x)
```

for every (y).

Compare this with the differentiable convex condition:

```math
f(y)
\ge
f(x)
+
\nabla f(x)^\top(y-x).
```

So the subgradient plays the same supporting-hyperplane role as the gradient.

The difference is that a non-differentiable point may have several valid subgradients.

The set of all subgradients at (x) is called the **subdifferential**:

```math
\partial f(x).
```

---

## 3. Example: Absolute Value

For:

```math
f(x)=|x|,
```

the subdifferential is:

```math
\partial f(x)
=
\begin{cases}
\{-1\}, & x<0,\\
[-1,1], & x=0,\\
\{1\}, & x>0.
\end{cases}
```

At (x=0), the derivative does not exist, but the whole interval:

```math
[-1,1]
```

contains valid subgradients.

To see why, apply the scalar subgradient condition at (x=0):

```math
|y|
\ge
g y.
```

For (y>0), this requires (g\le 1). For (y<0), it requires (g\ge -1). Therefore:

```math
-1
\le
g
\le
1,
```

so:

```math
\partial |0|
=
[-1,1].
```

---

## 4. Subgradient Method

The subgradient method replaces the gradient with any valid subgradient:

```math
\boxed{
\theta_{t+1}
=
\theta_t
-
\eta_t g_t,
\qquad
g_t\in\partial f(\theta_t)
}
```

This looks similar to Gradient Descent, but (g_t) may not be unique.

Subgradient methods work for convex non-smooth objectives, but they generally have weaker and slower convergence guarantees than smooth Gradient Descent.

The learning-rate schedule is therefore especially important.

---

## 5. Composite Objectives

A common structure is:

```math
F(\theta)
=
f(\theta)
+
g(\theta),
```

where:

- (f) is smooth and differentiable;
- (g) is convex but may be non-differentiable.

A common example is:

```math
F(\theta)
=
f(\theta)
+
\lambda\|\theta\|_1.
```

Instead of treating the whole objective with a subgradient method, we can exploit this structure using a **proximal method**.

---

## 6. Proximal Operator

For a function (g), the proximal operator is defined by:

```math
\boxed{
\mathrm{prox}_{\eta g}(v)
=
\arg\min_w
\left[
g(w)
+
\frac{1}{2\eta}
\|w-v\|_2^2
\right]
}
```

The operator is not chosen arbitrarily.

It is obtained by solving this optimization problem with respect to (w).

The two terms have different roles:

```math
g(w)
```

encodes the non-smooth structure we want to enforce, while:

```math
\frac{1}{2\eta}
\|w-v\|_2^2
```

keeps the new point (w) close to the input (v).

So the proximal operator finds a compromise between respecting (g) and remaining close to the current candidate point.

---

## 7. Proximal Gradient Descent

For:

```math
F(\theta)
=
f(\theta)
+
g(\theta),
```

proximal gradient descent first performs a normal gradient step on the smooth part:

```math
v
=
\theta_t
-
\eta\nabla f(\theta_t).
```

Then it applies the proximal operator of the non-smooth part:

```math
\boxed{
\theta_{t+1}
=
\mathrm{prox}_{\eta g}(v)
}
```

Equivalently:

```math
\boxed{
\theta_{t+1}
=
\mathrm{prox}_{\eta g}
\left(
\theta_t
-
\eta\nabla f(\theta_t)
\right)
}
```

So conceptually:

```text
gradient step on smooth part
        ↓
proximal correction for non-smooth part
```

The proximal step does not change the gradient that was already computed.

It changes the resulting parameter vector, and the gradient is recomputed at that new point during the next iteration.

---

## 8. Deriving the L1 Proximal Operator

Take:

```math
g(w)
=
\lambda|w|.
```

Then:

```math
\mathrm{prox}_{\eta\lambda|\cdot|}(v)
=
\arg\min_w
\left[
\lambda|w|
+
\frac{1}{2\eta}(w-v)^2
\right].
```

For:

```math
w>0,
```

we have (|w|=w), so:

```math
\lambda
+
\frac{1}{\eta}(w-v)
=
0,
```

which gives:

```math
w
=
v
-
\eta\lambda.
```

For:

```math
w<0,
```

we have (|w|=-w), so:

```math
-\lambda
+
\frac{1}{\eta}(w-v)
=
0,
```

which gives:

```math
w
=
v
+
\eta\lambda.
```

For:

```math
|v|
\le
\eta\lambda,
```

the minimizer is:

```math
w=0.
```

This last case can be demonstrated using the optimality condition at the non-differentiable point, but that derivation is omitted here.

Combining the three cases gives the **soft-thresholding operator**:

```math
\boxed{
\mathrm{prox}_{\eta\lambda|\cdot|}(v)
=
\mathrm{sign}(v)
\max(
|v|-\eta\lambda,
0
)
}
```

For vectors, this operator is applied coordinate-wise.

---

## 9. Vector Example: Gradient Step + L1 Prox

Take:

```math
\theta_t
=
\begin{bmatrix}
0.5\\
-0.2
\end{bmatrix}
```

and suppose:

```math
\nabla f(\theta_t)
=
\begin{bmatrix}
0.4\\
-0.6
\end{bmatrix}.
```

Let:

```math
\eta=0.1.
```

First perform the ordinary gradient step:

```math
v
=
\theta_t
-
\eta
\nabla f(\theta_t).
```

Therefore:

```math
v
=
\begin{bmatrix}
0.5\\
-0.2
\end{bmatrix}
-
0.1
\begin{bmatrix}
0.4\\
-0.6
\end{bmatrix}
=
\begin{bmatrix}
0.46\\
-0.14
\end{bmatrix}.
```

Now suppose:

```math
g(\theta)
=
\lambda\|\theta\|_1
```

with:

```math
\lambda=1.
```

Then:

```math
\eta\lambda
=
0.1.
```

Apply soft-thresholding to each coordinate.

For the first coordinate:

```math
0.46
\rightarrow
0.46-0.1
=
0.36.
```

For the second coordinate:

```math
-0.14
\rightarrow
-(0.14-0.1)
=
-0.04.
```

So:

```math
\boxed{
\theta_{t+1}
=
\begin{bmatrix}
0.36\\
-0.04
\end{bmatrix}
}
```

The complete update is:

```text
[ 0.5  ]
[-0.2  ]
    ↓ gradient step
[ 0.46 ]
[-0.14 ]
    ↓ L1 proximal step
[ 0.36 ]
[-0.04 ]
```

If one coordinate after the gradient step had magnitude below (0.1), for example:

```math
v_2=-0.07,
```

the proximal operator would set it exactly to:

```math
0.
```

This is the sparsity effect of L1 regularization.

---

## 10. Why Proximal Methods Matter

The subgradient method treats the full non-smooth objective directly.

Proximal gradient methods instead exploit structure:

```math
F=f+g.
```

They use ordinary gradients where gradients are available and handle the non-smooth part through its proximal operator.

For structured regularizers such as L1, this often gives a cleaner and more effective optimization procedure.

The key idea is:

```math
\boxed{
\text{smooth part}
\rightarrow
\text{gradient step}
}
```

and:

```math
\boxed{
\text{non-smooth part}
\rightarrow
\text{proximal step}
}
```
