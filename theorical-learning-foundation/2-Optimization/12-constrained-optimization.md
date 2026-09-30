# 12. Optimization — Constrained Optimization

## Key Takeaways

- Constrained optimization minimizes an objective while restricting the solution to an allowed set.
- The set of all points satisfying the constraints is the **feasible set**.
- Equality constraints are handled naturally with **Lagrange multipliers**.
- Inequality constraints lead to the **KKT conditions**.
- KKT conditions describe what must hold at an optimum; they are not themselves an optimization algorithm.
- An inactive constraint has multiplier \(\lambda=0\); an active constraint may have \(\lambda>0\).
- Complementary slackness expresses the rule that either a constraint is inactive or its multiplier can matter.
- Projected Gradient Descent keeps each iterate inside the feasible set.
- Penalty methods convert constraints into additional terms in the objective.
- Many solvers search directly for primal and dual variables that approximately satisfy the KKT conditions.

---

## 1. Constrained Optimization

So far, most optimization problems had the form:

\`\`\`math
\min_\theta J(\theta).
\`\`\`

The parameters were free to move anywhere in the parameter space.

In constrained optimization, we instead solve:

\`\`\`math
\min_\theta J(\theta)
\`\`\`

subject to conditions such as:

\`\`\`math
h_j(\theta)=0
\`\`\`

or:

\`\`\`math
g_i(\theta)\le0.
\`\`\`

The central idea is:

\`\`\`math
\boxed{
\text{find the best point among the allowed points}
}
\`\`\`

A useful intuition is:

> We want to go as low as possible on a hill, but we must stay inside a fence.

---

## 2. Feasible Set

The **feasible set** is the set of all parameter values satisfying every constraint.

For example:

\`\`\`math
\theta_1+\theta_2=1
\`\`\`

defines a line in \(\mathbb{R}^2\).

Only points on that line are feasible.

So unconstrained optimization asks:

\`\`\`text
What is the lowest point anywhere?
\`\`\`

while constrained optimization asks:

\`\`\`text
What is the lowest point that I am allowed to reach?
\`\`\`

A point that violates even one constraint cannot be a valid solution.

---

## 3. Equality Constraints and Lagrange Multipliers

Consider:

\`\`\`math
\min_\theta J(\theta)
\`\`\`

subject to:

\`\`\`math
h(\theta)=0.
\`\`\`

We introduce a new variable \(\lambda\), called a **Lagrange multiplier**, and define the Lagrangian:

\`\`\`math
\boxed{
\mathcal L(\theta,\lambda)
=
J(\theta)
+
\lambda h(\theta)
}
\`\`\`

For several equality constraints:

\`\`\`math
h_j(\theta)=0,
\`\`\`

we write:

\`\`\`math
\mathcal L(\theta,\lambda)
=
J(\theta)
+
\sum_j
\lambda_j h_j(\theta).
\`\`\`

At a constrained optimum, we solve:

\`\`\`math
\nabla_\theta\mathcal L(\theta,\lambda)=0
\`\`\`

together with:

\`\`\`math
h(\theta)=0.
\`\`\`

For one equality constraint:

\`\`\`math
\nabla J(\theta)
+
\lambda\nabla h(\theta)
=
0.
\`\`\`

Therefore:

\`\`\`math
\boxed{
\nabla J(\theta)
=
-\lambda\nabla h(\theta)
}
\`\`\`

---

## 4. Geometric Meaning of Lagrange Multipliers

The gradient:

\`\`\`math
\nabla h(\theta)
\`\`\`

is perpendicular to the constraint surface:

\`\`\`math
h(\theta)=0.
\`\`\`

At a constrained optimum, there can be no feasible direction along the constraint surface that decreases the objective.

Therefore the objective gradient must also be perpendicular to the feasible surface.

So:

\`\`\`math
\nabla J(\theta)
\parallel
\nabla h(\theta).
\`\`\`

The multiplier \(\lambda\) tells us how strongly the constraint contributes to this balance.

This is the main geometric meaning of the Lagrange multiplier condition.

---

## 5. Equality-Constrained Example

Consider:

\`\`\`math
\min_{x,y}
x^2+y^2
\`\`\`

subject to:

\`\`\`math
x+y=1.
\`\`\`

Without the constraint, the minimum is:

\`\`\`math
(x,y)=(0,0),
\`\`\`

but that point is not feasible.

Define:

\`\`\`math
h(x,y)=x+y-1.
\`\`\`

The Lagrangian is:

\`\`\`math
\mathcal L(x,y,\lambda)
=
x^2+y^2
+
\lambda(x+y-1).
\`\`\`

The stationarity equations are:

\`\`\`math
\frac{\partial\mathcal L}{\partial x}
=
2x+\lambda
=
0,
\`\`\`

\`\`\`math
\frac{\partial\mathcal L}{\partial y}
=
2y+\lambda
=
0,
\`\`\`

and feasibility requires:

\`\`\`math
x+y-1=0.
\`\`\`

The first two equations imply:

\`\`\`math
x=y.
\`\`\`

Therefore:

\`\`\`math
x+y=1
\Rightarrow
2x=1,
\`\`\`

so:

\`\`\`math
\boxed{
x^*=y^*=\frac12
}
\`\`\`

and the constrained optimum is:

\`\`\`math
\boxed{
(x^*,y^*)
=
\left(
\frac12,
\frac12
\right).
}
\`\`\`

---

## 6. Inequality Constraints

Now consider constraints of the form:

\`\`\`math
g_i(\theta)\le0.
\`\`\`

An inequality constraint can be either **inactive** or **active**.

### Inactive Constraint

If:

\`\`\`math
g_i(\theta^*)<0,
\`\`\`

the solution lies strictly inside the allowed region.

The constraint is not currently blocking the optimizer.

### Active Constraint

If:

\`\`\`math
g_i(\theta^*)=0,
\`\`\`

the solution lies exactly on the boundary.

The constraint may be preventing the optimizer from moving toward a better unconstrained point.

This distinction is central to the KKT conditions.

---

## 7. Lagrangian with Equality and Inequality Constraints

For:

\`\`\`math
\min_\theta J(\theta)
\`\`\`

subject to:

\`\`\`math
g_i(\theta)\le0
\`\`\`

and:

\`\`\`math
h_j(\theta)=0,
\`\`\`

define:

\`\`\`math
\boxed{
\mathcal L(\theta,\lambda,\nu)
=
J(\theta)
+
\sum_i
\lambda_i g_i(\theta)
+
\sum_j
\nu_j h_j(\theta)
}
\`\`\`

where:

- \(\lambda_i\) are multipliers for inequality constraints;
- \(\nu_j\) are multipliers for equality constraints.

The multipliers \(\lambda_i\) can be interpreted as the **price** or **force** associated with a constraint.

Intuitively:

\`\`\`text
How much is this constraint stopping us from improving the objective?
\`\`\`

---

## 8. KKT Conditions

The **Karush-Kuhn-Tucker conditions** are optimality conditions for constrained optimization problems.

They are best understood as four checks.

### 8.1 Primal Feasibility

The solution must respect the original constraints:

\`\`\`math
g_i(\theta^*)\le0
\`\`\`

and:

\`\`\`math
h_j(\theta^*)=0.
\`\`\`

This simply means:

\`\`\`text
The candidate solution must stay inside the feasible region.
\`\`\`

---

### 8.2 Dual Feasibility

For inequality constraints:

\`\`\`math
\boxed{
\lambda_i^*\ge0.
}
\`\`\`

The multiplier can be interpreted as the price of the constraint.

If a constraint does not affect the optimum, its price is typically zero.

---

### 8.3 Complementary Slackness

For every inequality constraint:

\`\`\`math
\boxed{
\lambda_i^*
g_i(\theta^*)
=
0.
}
\`\`\`

This means that at least one of the two factors must be zero.

If:

\`\`\`math
g_i(\theta^*)<0,
\`\`\`

the constraint is inactive, so:

\`\`\`math
\boxed{
\lambda_i^*=0.
}
\`\`\`

If:

\`\`\`math
\lambda_i^*>0,
\`\`\`

then necessarily:

\`\`\`math
\boxed{
g_i(\theta^*)=0.
}
\`\`\`

So:

\`\`\`math
\boxed{
\text{inactive constraint}
\Rightarrow
\lambda_i=0
}
\`\`\`

while an active constraint may have a positive multiplier.

A useful intuition is:

> If you are not touching the fence, the fence is not stopping you, so its price is zero.

---

### 8.4 Stationarity

At the optimum:

\`\`\`math
\boxed{
\nabla J(\theta^*)
+
\sum_i
\lambda_i^*
\nabla g_i(\theta^*)
+
\sum_j
\nu_j^*
\nabla h_j(\theta^*)
=
0.
}
\`\`\`

Without constraints, a differentiable optimum often satisfies:

\`\`\`math
\nabla J(\theta^*)=0.
\`\`\`

With constraints, reaching that point may be impossible because the feasible region blocks the optimizer.

Instead, the gradient of the objective and the contributions of the active constraints must balance.

Think of it as:

\`\`\`text
objective force
+
constraint forces
=
0
\`\`\`

This is the constrained version of the zero-gradient condition.

---

## 9. KKT Intuition in Four Questions

The four KKT conditions can be remembered as:

1. **Am I respecting all constraints?**
2. **Are the inequality multipliers nonnegative?**
3. **If a constraint is not blocking me, is its multiplier zero?**
4. **Do the objective and active constraints balance at the solution?**

The central idea is:

\`\`\`math
\boxed{
\text{KKT describes what must happen when constraints prevent us from reaching the unconstrained optimum.}
}
\`\`\`

---

## 10. When KKT Characterizes the Optimum

For convex constrained problems, KKT conditions are especially powerful.

Under suitable constraint qualifications, if:

- \(J\) is convex;
- every inequality function \(g_i\) is convex;
- every equality constraint is affine;

then satisfying the KKT conditions can characterize a global optimum.

So KKT plays a role similar to:

\`\`\`math
\nabla J(\theta^*)=0
\`\`\`

in unconstrained convex optimization.

For non-convex problems, KKT conditions are generally necessary local optimality conditions under suitable assumptions, but satisfying them does not automatically guarantee a global minimum.

---

## 11. KKT Is Not an Algorithm

KKT conditions describe the mathematical target.

They do not specify one unique procedure for finding the solution.

Many constrained optimization algorithms search for:

\`\`\`math
\theta
\`\`\`

and the multipliers:

\`\`\`math
\lambda,\nu
\`\`\`

such that the KKT conditions are approximately satisfied.

Common approaches include:

- **SQP (Sequential Quadratic Programming)** — repeatedly solves local quadratic approximations of the constrained problem.
- **Interior-Point Methods** — remain inside the feasible region and progressively approach the constrained optimum.
- **Active-Set Methods** — estimate which inequality constraints are active and solve the corresponding reduced problem.
- **Primal-Dual Methods** — update primal variables and dual multipliers together.

So:

\`\`\`math
\boxed{
\text{KKT conditions}
=
\text{optimality target, not the optimization algorithm itself}
}
\`\`\`

If a solver reports:

\`\`\`math
\text{KKT residual}<10^{-6},
\`\`\`

it means the KKT equations and inequalities are satisfied to a small numerical tolerance.

---

## 12. Projected Gradient Descent

A direct way to handle a feasible set \(C\) is **Projected Gradient Descent**.

First take an ordinary gradient step:

\`\`\`math
v_t
=
\theta_t
-
\eta\nabla J(\theta_t).
\`\`\`

The point \(v_t\) may be infeasible.

We then project it back onto the feasible set:

\`\`\`math
\boxed{
\theta_{t+1}
=
\Pi_C(v_t)
}
\`\`\`

where:

\`\`\`math
\Pi_C(v)
=
\arg\min_{\theta\in C}
\|\theta-v\|_2^2.
\`\`\`

So the procedure is:

\`\`\`text
gradient step
      ↓
possibly leave feasible set
      ↓
project back to closest feasible point
\`\`\`

Projection is also a special case of a proximal operator.

---

## 13. Penalty Methods

Another approach is to move the constraint into the objective.

For an equality constraint:

\`\`\`math
h(\theta)=0,
\`\`\`

we can optimize:

\`\`\`math
J(\theta)
+
\rho
\|h(\theta)\|_2^2.
\`\`\`

A violation of the constraint increases the objective.

Larger values of \(\rho\) penalize violations more strongly.

The advantage is that the constrained problem becomes an unconstrained one.

The limitation is that very large penalty coefficients can make the optimization problem poorly conditioned.

---

## 14. Constraints and Regularization

Constrained optimization is closely related to regularization.

For example, the regularized problem:

\`\`\`math
\min_\theta
\hat R_n(\theta)
+
\lambda\|\theta\|_2^2
\`\`\`

is closely related, under appropriate conditions, to:

\`\`\`math
\min_\theta
\hat R_n(\theta)
\`\`\`

subject to:

\`\`\`math
\|\theta\|_2^2
\le
c.
\`\`\`

The constrained version imposes a hard limit.

The regularized version introduces a soft penalty.

So:

\`\`\`math
\boxed{
\text{constraint}
\rightarrow
\text{hard restriction}
}
\`\`\`

while:

\`\`\`math
\boxed{
\text{regularization}
\rightarrow
\text{soft cost for violating a preferred scale}
}
\`\`\`

This connects constrained optimization directly to the earlier regularization topic.

---

## 15. Conceptual Summary

Constrained optimization changes the problem from:

\`\`\`math
\min_\theta J(\theta)
\`\`\`

to:

\`\`\`math
\min_\theta J(\theta)
\quad
\text{subject to feasibility conditions}.
\`\`\`

The main conceptual chain is:

\`\`\`text
constraints
    ↓
feasible set
    ↓
Lagrangian
    ↓
multipliers
    ↓
KKT conditions
    ↓
algorithms search for an approximate KKT point
\`\`\`

The most important KKT intuition is:

\`\`\`math
\boxed{
\text{primal feasibility}
=
\text{respect the constraints}
}
\`\`\`

\`\`\`math
\boxed{
\text{dual feasibility}
=
\text{constraint prices are valid}
}
\`\`\`

\`\`\`math
\boxed{
\text{complementary slackness}
=
\text{inactive constraint}\Rightarrow\lambda=0
}
\`\`\`

\`\`\`math
\boxed{
\text{stationarity}
=
\text{objective and active-constraint forces balance}
}
\`\`\`

KKT conditions are therefore not the algorithm itself; they describe the mathematical conditions that many constrained optimization algorithms try to satisfy.
