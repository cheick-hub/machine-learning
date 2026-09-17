# Multivariable Calculus for Machine Learning

## Key Takeaways

- A **partial derivative** measures how a multivariable function changes when one variable changes and the others are held fixed.
- The **gradient** collects all partial derivatives into a vector and gives the local direction of steepest increase.
- A **directional derivative** measures how fast a function changes along an arbitrary direction.
- The **chain rule** explains how derivatives propagate through composed functions and is the mathematical basis of backpropagation.
- A **Jacobian matrix** generalizes the derivative to vector-valued functions.
- A **Hessian matrix** contains second-order partial derivatives and describes local curvature.
- The negative gradient points in the direction of **steepest local decrease**.

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

If \(J\) is a loss function, then:

```math
\frac{\partial J}{\partial \theta_j}>0
```

means that increasing \(\theta_j\) locally increases the loss, while:

```math
\frac{\partial J}{\partial \theta_j}<0
```

means that increasing \(\theta_j\) locally decreases it.

**Illustration:** A partial derivative tells us how sensitive the loss is to one parameter while the others remain unchanged.

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
- a **magnitude**, indicating how steep that local change is.

The gradient norm is:

```math
\|\nabla J(\theta)\|_2
```

A large norm indicates a locally steep region, while a small norm indicates a flatter region.

**Illustration:** In optimization, the gradient summarizes the local effect of changing all model parameters at once.

### 2.1 Geometric Interpretation

For a function of two variables:

```math
J(\theta_1,\theta_2)
```

we can imagine \(J\) as a surface above the parameter plane \((\theta_1,\theta_2)\).

Curves of constant function value are called **level curves** or **contours**:

```math
J(\theta_1,\theta_2)=c
```

The gradient is perpendicular to the level curve passing through the current point.

This is important because moving along a level curve does not change \(J\), while moving perpendicular to it produces the strongest local change.

Therefore:

```math
\nabla J(\theta)
```

points toward increasing values of the function, while:

```math
-\nabla J(\theta)
```

points toward decreasing values.

**Illustration:** On a topographic map, contour lines represent equal altitude and the gradient points directly uphill, perpendicular to those contours.

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

### 3.1 Directional Derivative as a Dot Product

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

At a fixed point \(\theta\), the gradient magnitude is fixed. The directional derivative therefore depends on the alignment between \(u\) and the gradient.

---

## 4. Chain Rule

The **chain rule** differentiates composed functions.

Suppose:

```math
J = f(z)
```

and:

```math
z=g(\theta)
```

Then:

```math
\frac{dJ}{d\theta}
=
\frac{dJ}{dz}
\frac{dz}{d\theta}
```

For multivariable functions, the same principle applies through partial derivatives.

If:

```math
J=f(z_1,z_2,\dots,z_m)
```

and each \(z_i\) depends on \(\theta\), then:

```math
\frac{\partial J}{\partial \theta}
=
\sum_{i=1}^{m}
\frac{\partial J}{\partial z_i}
\frac{\partial z_i}{\partial \theta}
```

The chain rule is fundamental in Machine Learning because neural networks are compositions of many functions.

Backpropagation repeatedly applies the chain rule from the output layer back toward earlier parameters.

**Illustration:** If a parameter influences the loss through several intermediate computations, the chain rule multiplies the local derivatives along that computational path.

---

## 5. Jacobian Matrix

The **Jacobian** generalizes derivatives to vector-valued functions.

Consider:

```math
f:\mathbb{R}^n\rightarrow\mathbb{R}^m
```

with:

```math
f(x)
=
\begin{bmatrix}
f_1(x)\\
f_2(x)\\
\vdots\\
f_m(x)
\end{bmatrix}
```

The Jacobian matrix is:

```math
J_f(x)
=
\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n}\\
\vdots & \ddots & \vdots\\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
```

Each row describes how one output component changes with respect to all input components.

If the output is scalar, \(m=1\), the Jacobian reduces to the gradient up to row/column convention.

The Jacobian appears naturally in the multivariable chain rule. If:

```math
y=f(z),\qquad z=g(x)
```

then:

```math
J_{f\circ g}(x)
=
J_f(g(x))J_g(x)
```

**Illustration:** In a neural-network layer, the Jacobian describes how changes in the input vector affect every component of the output vector.

---

## 6. Hessian Matrix

The **Hessian** contains all second-order partial derivatives of a scalar-valued function.

For:

```math
J:\mathbb{R}^d\rightarrow\mathbb{R}
```

the Hessian is:

```math
H_J(\theta)
=
\nabla^2J(\theta)
=
\begin{bmatrix}
\frac{\partial^2J}{\partial\theta_1^2} & \cdots & \frac{\partial^2J}{\partial\theta_1\partial\theta_d}\\
\vdots & \ddots & \vdots\\
\frac{\partial^2J}{\partial\theta_d\partial\theta_1} & \cdots & \frac{\partial^2J}{\partial\theta_d^2}
\end{bmatrix}
```

While the gradient describes **slope**, the Hessian describes **curvature**.

Near a point \(\theta\):

- positive curvature means the surface bends upward;
- negative curvature means it bends downward;
- mixed curvature indicates a saddle-like geometry.

At a stationary point where:

```math
\nabla J(\theta)=0
```

the Hessian helps classify the point:

- positive definite Hessian: local minimum;
- negative definite Hessian: local maximum;
- indefinite Hessian: saddle point.

Second-order optimization methods, such as Newton's method, explicitly use curvature information from the Hessian.

**Illustration:** The gradient tells us which way the loss slopes, while the Hessian tells us how that slope itself changes nearby.

---

## 7. Why the Gradient Points in the Direction of Steepest Increase

For a unit direction \(u\):

```math
D_uJ(\theta)
=
\|\nabla J(\theta)\|_2\cos\phi
```

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

## 8. Why the Negative Gradient Is a Descent Direction

If the gradient points in the direction of maximum increase, the opposite direction produces the maximum local decrease.

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

Without normalizing the direction, choose:

```math
d=-\nabla J(\theta)
```

Then:

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

## 9. Connection to Gradient Descent

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

The conceptual chain is:

```text
Partial derivatives
        ↓
Gradient
        ↓
Directional derivatives
        ↓
Steepest local increase/decrease
        ↓
Gradient Descent
```

The chain rule and Jacobians explain how gradients propagate through composed vector-valued computations, while the Hessian adds second-order curvature information.