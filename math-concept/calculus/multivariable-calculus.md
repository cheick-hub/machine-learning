# Multivariable Calculus for Machine Learning

## Key Takeaways

- A **partial derivative** measures how a multivariable function changes when one variable changes and the others are held fixed.
- The **gradient** collects all partial derivatives into one vector.
- A **directional derivative** measures how fast a function changes when moving in an arbitrary direction.
- The gradient points in the direction of **steepest local increase**.
- The negative gradient points in the direction of **steepest local decrease**.
- Gradient Descent follows this local geometry by repeatedly moving opposite to the gradient.

---

## 1. Partial Derivatives

Consider a scalar-valued function of several variables:

```math
J(\theta_1,\theta_2,\dots,\theta_d)
```

A **partial derivative** measures how the function changes with respect to one variable while all other variables are kept fixed.

For parameter \(\theta_j\):

```math
\frac{\partial J}{\partial \theta_j}
```

This quantity is the local slope of \(J\) along the \(\theta_j\) coordinate axis.

**Illustration:** If \(J\) is a loss function, \(\frac{\partial J}{\partial \theta_j}\) tells us how sensitive the loss is to a small change in parameter \(\theta_j\).

---

## 2. Gradient Vector

The **gradient** collects all partial derivatives into a single vector:

```math
\nabla_\theta J(\theta)
=
\begin{bmatrix}
\frac{\partial J}{\partial \theta_1}\\
\frac{\partial J}{\partial \theta_2}\\
\vdots\\
\frac{\partial J}{\partial \theta_d}
\end{bmatrix}
```

For a function of two parameters:

```math
\nabla J(\theta)
=
\begin{bmatrix}
\frac{\partial J}{\partial \theta_1}\\
\frac{\partial J}{\partial \theta_2}
\end{bmatrix}
```

The gradient contains both:

- a **direction**, indicating how the function changes most rapidly;
- a **magnitude**, indicating how steep that change is locally.

**Illustration:** A large gradient norm means the loss surface is locally steep, while a small gradient norm means it is locally flat.

---

## 3. Directional Derivative

Partial derivatives describe movement along coordinate axes, but we may want to move in an arbitrary direction.

Let \(u\) be a direction vector, usually normalized so that:

```math
\|u\|_2=1
```

The **directional derivative** of \(J\) in direction \(u\) is:

```math
D_uJ(\theta)
=
\nabla J(\theta)^\top u
```

It measures the instantaneous rate of change of \(J\) if we move from \(\theta\) in direction \(u\).

Its sign has a direct interpretation:

```math
D_uJ(\theta)>0
```

means the function increases in direction \(u\),

```math
D_uJ(\theta)<0
```

means the function decreases, and

```math
D_uJ(\theta)=0
```

means the function is locally flat in that direction.

**Illustration:** The directional derivative answers: "If I move in this particular direction, how quickly will the loss change?"

---

## 4. Directional Derivative as a Dot Product

The directional derivative is a dot product:

```math
D_uJ(\theta)
=
\nabla J(\theta)^\top u
```

Using the geometric definition of the dot product:

```math
\nabla J(\theta)^\top u
=
\|\nabla J(\theta)\|_2\|u\|_2\cos\phi
```

where \(\phi\) is the angle between the gradient and the chosen direction.

For a unit vector \(u\):

```math
\|u\|_2=1
```

so:

```math
D_uJ(\theta)
=
\|\nabla J(\theta)\|_2\cos\phi
```

At a fixed point \(\theta\), the gradient magnitude is fixed. Therefore, the directional derivative only depends on the alignment between \(u\) and the gradient.

---

## 5. Why the Gradient Points in the Direction of Steepest Increase

For any angle \(\phi\):

```math
-1\leq\cos\phi\leq1
```

The largest possible value is:

```math
\cos\phi=1
```

which occurs when:

```math
\phi=0
```

meaning that \(u\) points in exactly the same direction as the gradient.

Because \(u\) is required to have unit length, the unit vector aligned with the gradient is:

```math
u
=
\frac{\nabla J(\theta)}{\|\nabla J(\theta)\|_2}
```

Substituting this direction into the directional derivative gives:

```math
D_uJ(\theta)
=
\|\nabla J(\theta)\|_2
```

which is the maximum possible directional derivative among all unit directions.

Therefore:

```math
\boxed{\nabla J(\theta)\text{ points in the direction of steepest local increase}}
```

The magnitude:

```math
\|\nabla J(\theta)\|_2
```

is the maximum local rate of increase.

**Illustration:** The gradient direction is the direction in parameter space where the loss rises fastest from the current point.

---

## 6. Why the Negative Gradient Is a Descent Direction

If the gradient points in the direction of maximum increase, the opposite direction should produce the maximum local decrease.

Choose the unit direction:

```math
u
=
-\frac{\nabla J(\theta)}{\|\nabla J(\theta)\|_2}
```

Then:

```math
D_uJ(\theta)
=
\nabla J(\theta)^\top
\left(
-\frac{\nabla J(\theta)}{\|\nabla J(\theta)\|_2}
\right)
```

Since:

```math
\nabla J(\theta)^\top\nabla J(\theta)
=
\|\nabla J(\theta)\|_2^2
```

we obtain:

```math
D_uJ(\theta)
=
-\|\nabla J(\theta)\|_2
```

As long as:

```math
\nabla J(\theta)\neq0
```

we have:

```math
D_uJ(\theta)<0
```

so moving opposite to the gradient locally decreases the function.

Without normalizing the direction, we can choose:

```math
d=-\nabla J(\theta)
```

and obtain:

```math
D_dJ(\theta)
=
\nabla J(\theta)^\top(-\nabla J(\theta))
=
-\|\nabla J(\theta)\|_2^2
<0
```

Therefore:

```math
\boxed{-\nabla J(\theta)\text{ is a descent direction whenever }\nabla J(\theta)\neq0}
```

**Illustration:** If the gradient points uphill on the loss surface, the negative gradient points directly downhill.

---

## 7. Connection to Gradient Descent

Gradient Descent uses the negative gradient to update the parameters:

```math
\theta_{t+1}
=
\theta_t
-
\eta\nabla J(\theta_t)
```

where \(\eta>0\) is the learning rate.

The gradient determines the local descent direction, while the learning rate determines how far the algorithm moves in that direction.

The descent guarantee is local: a sufficiently small step decreases the objective, but an excessively large step can overshoot and increase the loss.

The geometric chain is:

```text
Partial derivatives
        ↓
Gradient
        ↓
Directional derivative
        ↓
Steepest local increase
        ↓
Negative gradient
        ↓
Steepest local decrease
        ↓
Gradient Descent
```
