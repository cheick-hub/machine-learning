# 8. Optimization — Convergence and Learning Rate

## Key Takeaways

- Gradient Descent requires a suitable learning rate to decrease the objective reliably.
- An (L)-Lipschitz gradient means the gradient cannot change arbitrarily fast.
- For twice-differentiable objectives, (L) can be bounded using the Hessian.
- The descent lemma gives a quadratic upper bound on how much an (L)-smooth function can change.
- A common safe learning-rate scale is (eta le 1/L).
- Smoothness is not mandatory for Gradient Descent to work, but it provides useful convergence guarantees.
- Convexity determines whether stationary points are globally optimal.
- Strong convexity gives faster geometric convergence.
- The condition number (kappa=L/mu) measures how difficult the optimization landscape is.

---

## 1. Learning Rate and Convergence

Gradient Descent updates parameters using:

```math
\theta_{t+1}
=
\theta_t
-
\eta
\nabla J(\theta_t).
```

The direction:

```math
-\nabla J(\theta_t)
```

tells us where to move, while the learning rate:

```math
\eta
```

controls how far we move.

If (eta) is too small, optimization is stable but slow.

If (eta) is too large, the optimizer may overshoot, oscillate, or diverge.

To reason formally about safe step sizes, we introduce smoothness.

**Illustration:** The gradient gives the downhill direction, while the learning rate determines whether we take a careful step or jump too far.

---

## 2. Lipschitz Gradient and Smoothness

A differentiable function (J) has an (L)-Lipschitz gradient if:

```math
\|\nabla J(x)-\nabla J(y)\|_2
\le
L\|x-y\|_2
```

for all (x,y).

A function satisfying this condition is also called **(L)-smooth**.

The constant (L) limits how quickly the gradient can change.

A larger (L) means the objective may have stronger curvature, so smaller steps are usually required.

```math
\boxed{
L\text{ large}
\Rightarrow
\text{gradient changes quickly}
\Rightarrow
\text{smaller learning rate}
}
```

Smoothness is not required for Gradient Descent to exist or even to work in practice. It is mainly an assumption that allows us to derive useful descent and convergence guarantees.

**Illustration:** Smoothness prevents the local slope from changing too violently when the parameters move slightly.

---

## 3. How to Compute or Bound (L)

For a twice-differentiable function, a global Lipschitz constant can be obtained from the Hessian:

```math
L
=
\sup_\theta
\|\nabla^2 J(\theta)\|_2.
```

For a symmetric Hessian, the spectral norm is controlled by the eigenvalues.

For a convex function:

```math
L
=
\sup_\theta
\lambda_{\max}
\left(
\nabla^2J(\theta)
\right).
```

So (L) measures the maximum curvature of the objective.

### Quadratic Example

Consider:

```math
J(\theta)
=
\frac12
\theta^\top A\theta,
```

with symmetric (A).

Then:

```math
\nabla J(\theta)=A\theta
```

and:

```math
\nabla^2J(\theta)=A.
```

Therefore:

```math
L
=
\|A\|_2.
```

If (A\succeq0):

```math
\boxed{
L=\lambda_{\max}(A)
}
```

### Least-Squares Example

For:

```math
J(\theta)
=
\frac{1}{2n}
\|X\theta-y\|_2^2,
```

the Hessian is:

```math
\nabla^2J(\theta)
=
\frac{1}{n}
X^\top X.
```

Hence:

```math
\boxed{
L
=
\lambda_{\max}
\left(
\frac1n X^\top X
\right)
=
\frac{\sigma_{\max}(X)^2}{n}
}
```

**Illustration:** For smooth objectives, (L) can often be understood as the largest curvature present anywhere in the optimization landscape.

---

## 4. Example Without a Globally Lipschitz Gradient

Consider:

```math
f(x)=x^4.
```

Its gradient is:

```math
f'(x)=4x^3.
```

and its second derivative is:

```math
f''(x)=12x^2.
```

Since:

```math
\sup_x |f''(x)|=\infty,
```

there is no finite global Lipschitz constant (L) over all of (mathbb{R}).

Therefore, (f(x)=x^4) does not have a globally Lipschitz gradient.

However, on a bounded interval ([-R,R]):

```math
|f''(x)|
\le
12R^2,
```

so the gradient is locally Lipschitz there with:

```math
L=12R^2.
```

Thus:

```math
\boxed{
\text{not globally Lipschitz}
\neq
\text{not locally Lipschitz}
}
```

**Illustration:** A function can have well-behaved curvature in a bounded region even if its curvature grows without bound globally.

---

## 5. The Descent Lemma

For an (L)-smooth function:

```math
J(y)
\le
J(x)
+
\nabla J(x)^\top(y-x)
+
\frac{L}{2}
\|y-x\|_2^2.
```

The descent lemma says that an (L)-smooth function cannot increase faster than this quadratic upper bound around the current point.

Applied to Gradient Descent, it gives a condition on the step size that guarantees the objective decreases.

Take:

```math
y
=
x
-
\eta\nabla J(x).
```

Then:

```math
y-x
=
-\eta\nabla J(x).
```

Substituting into the descent lemma:

```math
J(x-\eta\nabla J(x))
\le
J(x)
-
\eta
\|\nabla J(x)\|_2^2
+
\frac{L\eta^2}{2}
\|\nabla J(x)\|_2^2.
```

Factorizing:

```math
\boxed{
J(x-\eta\nabla J(x))
\le
J(x)
-
\eta
\left(
1-\frac{L\eta}{2}
\right)
\|\nabla J(x)\|_2^2
}
```

Unlike a first-order Taylor approximation, this is an actual inequality under the smoothness assumption.

**Illustration:** Smoothness converts the local Gradient Descent intuition into a formal bound on how much the objective can change.

---

## 6. Learning-Rate Condition

To guarantee decrease through the previous bound, we need:

```math
1-\frac{L\eta}{2}>0.
```

Therefore:

```math
\boxed{
0<\eta<\frac{2}{L}
}
```

is sufficient for descent.

A common conservative choice is:

```math
\boxed{
\eta\le\frac{1}{L}
}
```

This explains why the learning rate depends on curvature.

If (L) is large, the function can bend sharply and smaller steps are safer.

If (L) is small, larger steps may be possible.

**Illustration:** The maximum curvature determines how aggressively Gradient Descent can move without losing stability.

---

## 7. Descent vs Convergence

These two ideas are related but different.

**Descent** means:

```math
J(\theta_{t+1})
<
J(\theta_t).
```

The objective decreases from one iteration to the next.

**Convergence** means that the optimization sequence approaches some limiting behavior, for example:

```math
\theta_t
\to
\theta^\*
```

or:

```math
J(\theta_t)
\to
J(\theta^\*).
```

For a general smooth non-convex objective, Gradient Descent does not guarantee convergence to a global minimum.

A typical guarantee is instead that the gradient becomes small:

```math
\|\nabla J(\theta_t)\|_2
\to
0.
```

This means the algorithm approaches a stationary point, which may be a minimum, maximum, or saddle point.

**Illustration:** Decreasing the objective is a local property of each step; convergence describes the long-term behavior of the whole sequence.

---

## 8. Convergence for Convex Objectives

If (J) is convex and (L)-smooth, Gradient Descent with a suitable learning rate converges toward a global minimum.

A standard rate is:

```math
J(\theta_t)
-
J(\theta^\*)
=
O\left(\frac{1}{t}\right).
```

This means the optimization error decreases inversely with the number of iterations.

So, approximately, increasing the number of iterations reduces the error progressively, but not geometrically fast.

Convexity is what lets us interpret convergence toward a stationary point as convergence toward a global optimum.

**Illustration:** Smoothness controls the steps, while convexity tells us that the destination is globally optimal.

---

## 9. Strong Convexity and Linear Convergence

If (J) is both (L)-smooth and (mu)-strongly convex, Gradient Descent obtains a stronger convergence rate.

With a suitable learning rate, for example:

```math
\eta
=
\frac{1}{L},
```

a typical bound is:

```math
J(\theta_t)-J(\theta^\*)
\le
C
\left(
1-\frac{\mu}{L}
\right)^t
```

for some constant (C).

This is called **linear convergence** or **geometric convergence**.

The error shrinks by approximately a constant factor at each iteration.

This is substantially faster than the (O(1/t)) rate associated with general smooth convex optimization.

**Illustration:** Strong convexity gives the objective enough curvature to pull the iterates toward a unique minimizer at a geometric rate.

---

## 10. Condition Number

For an (L)-smooth and (mu)-strongly convex objective, define the condition number:

```math
\boxed{
\kappa
=
\frac{L}{\mu}
}
```

Here:

- (L) measures the largest curvature;
- (mu) measures the smallest guaranteed curvature.

If:

```math
\kappa\approx1,
```

curvature is similar in all directions and optimization is relatively well conditioned.

If:

```math
\kappa\gg1,
```

the landscape may contain long narrow valleys.

Gradient Descent can then zig-zag across steep directions while making slow progress along flatter ones.

So a large condition number usually means slower optimization.

**Illustration:** The condition number measures how uneven the curvature is across different directions.

---

## 11. Learning-Rate Schedules

A constant learning rate is not always ideal.

In stochastic optimization, gradient noise can keep the parameters fluctuating around the optimum.

A decreasing learning rate:

```math
\eta_t
\downarrow
0
```

can reduce these fluctuations over time.

Common schedules include:

```math
\eta_t
=
\frac{\eta_0}{1+ct}
```

or:

```math
\eta_t
=
\frac{\eta_0}{\sqrt{t}}.
```

The exact schedule depends on the optimization method and theoretical assumptions.

The key idea is:

```math
\boxed{
\text{large steps early, smaller steps later}
}
```

**Illustration:** Early training benefits from larger movement, while later training often benefits from smaller refinements around a good solution.

---

## 12. Conceptual Summary

The main chain is:

```text
Lipschitz gradient
      ↓
bounded gradient variation
      ↓
descent lemma
      ↓
safe learning-rate range
      ↓
controlled Gradient Descent
```

Then:

```text
smoothness
+
convexity
      ↓
global convergence
```

and:

```text
smoothness
+
strong convexity
      ↓
geometric convergence
```

The main practical interpretation is:

```math
\boxed{
\text{smoothness controls safe step sizes}
}
```

while:

```math
\boxed{
\text{convexity controls what solution we can guarantee}
}
```

and:

```math
\boxed{
\text{strong convexity controls how fast convergence can be}
}
```


---

## 13. Assumptions and Guarantees Summary

| Assumption | What it gives you |
|---|---|
| **Smooth** | Gradient changes gradually; Gradient Descent can make controlled descent with a suitable step size. |
| **Convex** | Every local minimum is global; for differentiable objectives, a stationary point is a global minimum. |
| **Strongly convex** | The global minimizer is unique. |
| **Smooth + convex** | Gradient Descent converges to the global optimal value, typically at a rate of (O(1/t)). |
| **Smooth + strongly convex** | Gradient Descent converges geometrically to the unique global minimizer, typically at a rate proportional to (left(1-mu/Light)^t). |

A useful mental model is:

```text
Smoothness
    ↓
controlled gradient variation
    ↓
descent with a suitable step size
```

```text
Convexity
    ↓
stationarity implies global optimality
```

```text
Strong convexity
    ↓
the global optimum is unique
```

Therefore, combining smoothness and strong convexity gives both:

- favorable optimization dynamics;
- a unique optimization target.
