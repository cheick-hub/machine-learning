# 7. Optimization — Convexity and Critical Points

## Key Takeaways

- A **critical point** satisfies (
abla J(	heta^*)=0), but it is not necessarily a minimum.
- The Hessian helps classify critical points through local curvature.
- For convex functions, every local minimum is also a global minimum.
- For differentiable convex functions, any stationary point is a global minimizer.
- Jensen's inequality is a direct consequence of convexity and compares the function of an average with the average of the function.
- Strong convexity adds a guaranteed amount of positive curvature and gives stronger optimization guarantees.
- For twice-differentiable functions, convexity and strong convexity can be characterized using the Hessian.

---

## 1. Critical and Stationary Points

A point (	heta^*) is called a **critical point** or **stationary point** when:

```math
\nabla J$\theta^*$=0.
```

At such a point, the first-order information vanishes.

However:

```math
\nabla J$\theta^*$=0
```

does **not** automatically imply that (	heta^*) is a minimum.

A stationary point may be:

- a local minimum;
- a local maximum;
- a saddle point.

**Illustration:** A zero slope can occur at the bottom of a valley, the top of a hill, or on a saddle-shaped surface.

---

## 2. Local Minimum, Maximum, and Saddle Point

A point (	heta^*) is a **local minimum** if nearby points have objective values at least as large:

```math
J$\theta^*$
\le
J$\theta$
```

for all (	heta) sufficiently close to (	heta^*).

Similarly, (	heta^*) is a **local maximum** if:

```math
J$\theta^*$
\ge
J$\theta$
```

for nearby (	heta).

A **saddle point** is a stationary point that is neither a local minimum nor a local maximum.

At a saddle point, the objective may increase in some directions and decrease in others.

**Illustration:** A saddle point is locally uphill in one direction and downhill in another.

---

## 3. Hessian-Based Classification

For a twice-differentiable objective, the Hessian is:

```math
H$\theta$
=
\nabla^2 J$\theta$.
```

At a stationary point (	heta^*), the Hessian describes the local curvature.

If:

```math
H$\theta^*$ \succ 0,
```

the Hessian is positive definite, and (	heta^*) is a strict local minimum.

If:

```math
H$\theta^*$ \prec 0,
```

the Hessian is negative definite, and (	heta^*) is a strict local maximum.

If the Hessian has both positive and negative eigenvalues, it is indefinite, and (	heta^*) is a saddle point.

If the Hessian is only semidefinite, the second-order test may be inconclusive.

**Illustration:** The Hessian tells us whether the surface bends upward, downward, or in mixed directions around a stationary point.

---

## 4. Convex Sets

Before defining convex functions, we define convex sets.

A set $C$ is convex if, for any $x,y\in C$ and any $\lambda\in[0,1]$,

```math
\lambda x + (1-\lambda)y
\in C.
```

This means the entire line segment between any two points in the set remains inside the set.

Convexity of the domain is important because convex functions are defined over convex sets.

**Illustration:** A filled disk is convex because every line segment between two points stays inside it; a crescent-shaped region is not.

---

## 5. Convex Functions

A function $J$ defined on a convex set is convex if, for any (	heta_1,	heta_2) and any $\lambda\in[0,1]$,

```math
J(
\lambda\theta_1
+
(1-\lambda)\theta_2
)
\le
\lambda J(\theta_1)
+
(1-\lambda)J(\theta_2).
```

Geometrically, the line segment joining two points on the graph lies above the graph of the function.

The main optimization consequence is:

```math
\boxed{
\text{every local minimum of a convex function is global}
}
```

This removes the distinction between local and global minima.

**Illustration:** A convex objective has no isolated bad local minima.

---

## 6. First-Order Characterization of Convexity

If $J$ is differentiable, convexity is equivalent to:

```math
J$y$
\ge
J$x$
+
\nabla J$x$^\top(y-x)
```

for all $x$ and $y$.

The right-hand side is the first-order Taylor approximation of $J$ around $x$.

So for a convex function, the tangent plane is always a global lower bound.

This gives an important result.

If:

```math
\nabla J$\theta^*$=0,
```

then:

```math
J$\theta$
\ge
J$\theta^*$
```

for every (	heta).

Therefore:

```math
\boxed{
\nabla J$\theta^*$=0
\quad\Rightarrow\quad
\theta^* \text{ is a global minimizer}
}
```

for differentiable convex objectives.

**Illustration:** For a convex function, once the tangent plane becomes horizontal, there is nowhere lower to go.

---

## 7. Second-Order Characterization of Convexity

If $J$ is twice differentiable, then $J$ is convex if:

```math
\nabla^2 J$\theta$
\succeq
0
```

for every (	heta) in its domain.

This means the Hessian is positive semidefinite everywhere.

Equivalently, for every direction $v$,

```math
v^\top
\nabla^2 J$\theta$
v
\ge
0.
```

So the function never has negative curvature.

**Illustration:** Positive-semidefinite curvature means the objective never bends downward along any direction.

---

## 8. Jensen's Inequality

Jensen's inequality is one of the most important consequences of convexity.

For a convex function $f$, weights $\lambda_i\ge0$, and:

```math
\sum_{i=1}^{m}\lambda_i=1,
```

we have:

```math
\boxed{
f\left(
\sum_{i=1}^{m}
\lambda_i x_i
\right)
\le
\sum_{i=1}^{m}
\lambda_i f(x_i)
}
```

In probability notation:

```math
\boxed{
f(\mathbb{E}[X])
\le
\mathbb{E}[f(X)]
}
```

whenever the expectations exist.

The intuition is:

```math
\boxed{
\text{function of the average}
\le
\text{average of the function}
}
```

for convex $f$.

Jensen's inequality appears throughout machine learning, probability, expected loss analysis, information theory, and variational methods.

**Illustration:** Averaging inputs before applying a convex function cannot produce a value larger than averaging the function values afterward.

---

## 9. Strong Convexity

Convexity requires nonnegative curvature.

Strong convexity requires a **strictly positive lower bound** on curvature.

A differentiable function $J$ is $\mu$-strongly convex if there exists:

```math
\mu>0
```

such that for all $x,y$,

```math
J$y$
\ge
J$x$
+
\nabla J$x$^\top(y-x)
+
\frac{\mu}{2}
\|y-x\|_2^2.
```

Compare this with ordinary convexity:

```math
J$y$
\ge
J$x$
+
\nabla J$x$^\top(y-x).
```

Strong convexity adds:

```math
\frac{\mu}{2}
\|y-x\|_2^2.
```

This forces the function to bend upward by at least a certain amount.

If $J$ is twice differentiable, strong convexity is equivalent to:

```math
\boxed{
\nabla^2 J$\theta$
\succeq
\mu I
}
```

for every (	heta).

This means every eigenvalue of the Hessian is at least $\mu>0$.

The parameter $\mu$ measures the minimum guaranteed curvature.

Important consequences include:

- the global minimizer is unique;
- there are no flat directions at the optimum;
- the objective grows at least quadratically away from the minimizer;
- Gradient Descent can obtain stronger convergence guarantees under suitable smoothness assumptions.

### Example

Consider:

```math
f$x$=x^2.
```

Then:

```math
f''$x$=2.
```

So the function is $2$-strongly convex.

Now consider:

```math
f$x$=x^4.
```

It is convex, but:

```math
f''(0)=0.
```

Therefore it is not strongly convex over all of $\mathbb{R}$.

**Illustration:** Convex means bowl-shaped; strongly convex means bowl-shaped with a guaranteed minimum amount of curvature everywhere.

---

## 10. Why Convexity Matters in Optimization

Without convexity:

```math
\nabla J$\theta$=0
```

may correspond to a minimum, maximum, or saddle point.

With convexity:

```math
\nabla J$\theta^*$=0
```

implies that (	heta^*) is a global minimizer.

With strong convexity, that global minimizer is unique.

So the progression is:

```text
critical point
      ↓
zero gradient
      ↓
Hessian describes local curvature
      ↓
convexity removes bad local minima
      ↓
strong convexity adds uniqueness and stronger guarantees
```

The central distinction is:

```math
\boxed{
\nabla J$\theta$=0
\text{ means stationary, not necessarily minimum}
}
```

while for differentiable convex objectives:

```math
\boxed{
\nabla J$\theta^*$=0
\Rightarrow
\theta^* \text{ is globally optimal}
}
```
