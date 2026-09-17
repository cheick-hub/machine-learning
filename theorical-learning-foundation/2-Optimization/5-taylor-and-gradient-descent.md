# 3. Optimization — Taylor Approximation and Gradient Descent

## Key Takeaways

- Taylor approximation describes a function locally using its derivatives.
- The **first-order Taylor approximation** gives a local linear model of the objective.
- Choosing a step in the direction of the **negative gradient** decreases the first-order approximation of the objective.
- Gradient Descent repeatedly applies this local idea:

```math
\theta_{t+1}
=
\theta_t
-
\eta \nabla J(\theta_t)
```

- The learning rate \(\eta\) controls the step size.
- A descent direction does not guarantee that an arbitrarily large step decreases the objective.
- The **second-order Taylor approximation** adds curvature information through the Hessian.
- Gradient Descent uses only first-order information, while second-order methods also use curvature.

---

## 1. Local Approximation of a Function

Optimization usually deals with objective functions that are too complicated to minimize directly.

Instead of understanding the entire function at once, we can approximate it near the current parameter point.

Consider:

```math
J:\mathbb{R}^d\rightarrow\mathbb{R}
```

and a current parameter vector:

```math
\theta\in\mathbb{R}^d.
```

Suppose we make a small change:

```math
\Delta\theta.
```

We want to estimate:

```math
J(\theta+\Delta\theta)
```

without evaluating the full geometry of the function.

Taylor approximation provides such a local model.

**Illustration:** Instead of modeling an entire mountain, Taylor approximation describes the terrain immediately around the point where we are standing.

---

## 2. First-Order Taylor Approximation

For a differentiable multivariable function, the first-order Taylor approximation around \(\theta\) is:

```math
J(\theta+\Delta\theta)
\approx
J(\theta)
+
\nabla J(\theta)^\top \Delta\theta.
```

The approximation contains two parts:

```math
J(\theta)
```

which is the current function value, and

```math
\nabla J(\theta)^\top \Delta\theta
```

which estimates how the function changes under the small displacement \(\Delta\theta\).

Therefore:

```math
\Delta J
=
J(\theta+\Delta\theta)-J(\theta)
```

is approximately:

```math
\Delta J
\approx
\nabla J(\theta)^\top\Delta\theta.
```

This is exactly the same local geometry that appears in the directional derivative.

**Illustration:** The first-order Taylor approximation replaces a curved surface locally with its tangent plane.

---

## 3. Connection to the Directional Derivative

Suppose the displacement is written as:

```math
\Delta\theta=\alpha u
```

where:

- \(u\) is a unit direction;
- \(\alpha\) is a small step size.

Then:

```math
J(\theta+\alpha u)
\approx
J(\theta)
+
\alpha\nabla J(\theta)^\top u.
```

Since:

```math
D_uJ(\theta)
=
\nabla J(\theta)^\top u,
```

we obtain:

```math
J(\theta+\alpha u)
\approx
J(\theta)
+
\alpha D_uJ(\theta).
```

So the directional derivative tells us the first-order change in the objective for movement along a chosen direction.

If:

```math
D_uJ(\theta)<0,
```

then sufficiently small movement along \(u\) decreases the objective locally.

**Illustration:** The directional derivative is the slope term inside the first-order Taylor approximation.

---

## 4. Deriving the Gradient Descent Direction

From `4-gradients-and-directional-derivatives.md`, we already established that:

```math
-\nabla J(\theta)
```

is the direction of steepest local decrease.

So this section does **not** re-derive which direction is best. Instead, Taylor approximation explains why taking a sufficiently small step in that already-established direction decreases the objective locally.

We therefore choose:

```math
\Delta\theta
=
\eta\left(-\nabla J(\theta)\right)
=
-\eta\nabla J(\theta),
```

where:

- \(-\nabla J(\theta)\) determines the **descent direction**;
- \(\eta>0\) determines the **step size**.

Substitute this into the first-order Taylor approximation:

```math
J(\theta+\Delta\theta)
\approx
J(\theta)
+
\nabla J(\theta)^\top
\left(
-\eta\nabla J(\theta)
\right).
```

Therefore:

```math
J(\theta+\Delta\theta)
\approx
J(\theta)
-
\eta
\nabla J(\theta)^\top
\nabla J(\theta).
```

Since:

```math
\nabla J(\theta)^\top\nabla J(\theta)
=
\|\nabla J(\theta)\|_2^2,
```

we obtain:

```math
J(\theta-\eta\nabla J(\theta))
\approx
J(\theta)
-
\eta
\|\nabla J(\theta)\|_2^2.
```

Because:

```math
\eta>0
```

and:

```math
\|\nabla J(\theta)\|_2^2\geq0,
```

the first-order model predicts a decrease whenever:

```math
\nabla J(\theta)\neq0.
```

So the two results have different roles:

```text
Directional derivatives tell us where to move.
Taylor approximation explains why a sufficiently small step there reduces J.
```

**Illustration:** The negative gradient selects the downhill direction; Taylor approximation explains the local effect of stepping in that direction.

---

## 5. Gradient Descent Update Rule

Gradient Descent repeatedly updates the parameters according to:

```math
\boxed{
\theta_{t+1}
=
\theta_t
-
\eta\nabla J(\theta_t)
}
```

where:

- \(\theta_t\) is the current parameter vector;
- \(\nabla J(\theta_t)\) is the gradient at the current point;
- \(\eta\) is the learning rate.

The procedure is:

```text
1. Evaluate the gradient at the current parameters.
2. Move in the opposite direction of the gradient.
3. Recompute the gradient at the new point.
4. Repeat.
```

The gradient is recalculated after every step because the local geometry changes as the parameters move.

**Illustration:** Gradient Descent repeatedly builds a new local linear approximation rather than relying on one approximation for the entire optimization path.

---

## 6. Why the Learning Rate Matters

The direction:

```math
-\nabla J(\theta)
```

tells us where to move.

The learning rate:

```math
\eta
```

tells us how far to move.

These are separate decisions.

A descent direction guarantees a local decrease only for a sufficiently small step.

### 6.1 Small Learning Rate

If \(\eta\) is very small:

```math
\theta_{t+1}
\approx
\theta_t.
```

The updates are stable, but optimization may progress slowly.

### 6.2 Large Learning Rate

If \(\eta\) is too large, the algorithm can move far outside the region where the first-order approximation is accurate.

The optimizer may:

- overshoot a minimum;
- oscillate around a valley;
- increase the objective;
- diverge.

**Illustration:** The negative gradient can point downhill while an excessively large step still jumps to a higher point on the other side of the valley.

---

## 7. Why Taylor Approximation Is Only Local

The approximation:

```math
J(\theta+\Delta\theta)
\approx
J(\theta)
+
\nabla J(\theta)^\top\Delta\theta
```

becomes less accurate as \(\|\Delta\theta\|\) increases.

The reason is that the actual function may curve.

The first-order approximation captures slope but ignores curvature.

Therefore, Gradient Descent relies on a repeated process:

```text
approximate locally
        ↓
take a small step
        ↓
recompute the gradient
        ↓
build a new local approximation
```

This explains why the learning rate must be controlled.

**Illustration:** A tangent plane is accurate near the point of contact but becomes less representative farther away.

---

## 8. Second-Order Taylor Approximation

The second-order Taylor approximation adds curvature information.

For a twice-differentiable function:

```math
J(\theta+\Delta\theta)
\approx
J(\theta)
+
\nabla J(\theta)^\top\Delta\theta
+
\frac12
\Delta\theta^\top
H(\theta)
\Delta\theta,
```

where:

```math
H(\theta)
=
\nabla^2J(\theta)
```

is the Hessian matrix.

The three terms represent the current function value, the local slope, and the local curvature.

**Illustration:** First-order approximation describes the tangent plane; second-order approximation also describes how the surface bends around that plane.

---

## 9. Role of the Hessian

The Hessian tells us how the gradient changes locally.

If:

```math
H(\theta)
```

has large curvature in some direction, the objective can change rapidly in that direction even if the first-order slope alone appears manageable.

The Hessian therefore helps describe:

- narrow valleys;
- flat directions;
- strong curvature;
- local minima;
- saddle points.

Gradient Descent does not explicitly use the Hessian.

It only uses:

```math
\nabla J(\theta).
```

Second-order optimization methods use both gradient and curvature information.

**Illustration:** The gradient tells us which way is downhill; the Hessian tells us how sharply the terrain bends in different directions.

---

## 10. Gradient Descent vs Second-Order Optimization

Gradient Descent uses only first-order information:

```math
\theta_{t+1}
=
\theta_t
-
\eta\nabla J(\theta_t).
```

Newton's method uses a second-order approximation and produces the update:

```math
\theta_{t+1}
=
\theta_t
-
H(\theta_t)^{-1}
\nabla J(\theta_t).
```

The Hessian rescales the gradient according to local curvature.

Gradient Descent is usually cheaper because computing and storing the Hessian can be expensive for high-dimensional models.

**Illustration:** First-order methods use slope information only, while second-order methods also account for how sharply the objective curves.

---

## 11. Stopping Criteria

Gradient Descent must eventually decide when to stop.

Common stopping conditions include:

### Small Gradient Norm

```math
\|\nabla J(\theta_t)\|_2<\varepsilon.
```

This indicates that the objective is locally flat.

### Small Parameter Change

```math
\|\theta_{t+1}-\theta_t\|_2<\varepsilon.
```

This means the parameters are barely moving.

### Small Objective Improvement

```math
|J(\theta_{t+1})-J(\theta_t)|<\varepsilon.
```

This means additional iterations produce little improvement.

### Maximum Number of Iterations

Optimization may also stop after a predefined computational budget.

A small gradient does not automatically imply a minimum; it may also indicate a maximum or saddle point.

**Illustration:** Stopping criteria detect when further optimization is unlikely to produce meaningful progress.