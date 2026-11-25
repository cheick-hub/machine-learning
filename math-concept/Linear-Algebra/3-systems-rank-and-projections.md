# 3. Linear Algebra — Systems, Rank and Projections

## Key Takeaways

- Solving **$Ax=b$** means expressing $b$ as a linear combination of the columns of $A$.
- The **rank** counts independent column directions; the **null space** contains inputs that map to zero.
- The **rank-nullity theorem** explains when solutions are unique or non-unique.
- When an exact solution does not exist, **orthogonal projection** provides the closest representable output.
- **Least squares** minimizes squared residuals and leads to the normal equations.
- Linear regression is geometrically a projection of the target vector onto the column space of the design matrix.

---

## 1. Systems of Linear Equations

A linear system can be written as:

```math
Ax=b,
\qquad
A\in\mathbb{R}^{m\times n},
\quad x\in\mathbb{R}^n,
\quad b\in\mathbb{R}^m
```

The columns $a_i$ of $A$ generate the possible outputs:

```math
Ax=x_1a_1+\cdots+x_na_n
```

The **column space** is the span of those columns:

```math
\mathrm{Col}(A)=\mathrm{span}(a_1,\ldots,a_n)
```

Consequently, the system is solvable exactly if and only if:

```math
Ax=b \text{ has a solution }
\iff b\in\mathrm{Col}(A)
```

A system can have a unique solution, infinitely many solutions, or no solution.

<p align="center">
  <img src="assets/linear_system_solutions.svg" width="95%" alt="Three possible outcomes of a system of two linear equations">
</p>

**Illustration:** A target vector outside the column space cannot be reproduced exactly by a linear model.

---

## 2. Matrix Rank

The **rank** is the number of linearly independent columns (also equal to the number of independent rows):

```math
\mathrm{rank}(A)=\dim(\mathrm{Col}(A))
\leq\min(m,n)
```

For example:

```math
A=
\begin{bmatrix}
1&2\\
2&4\\
3&6
\end{bmatrix},
\qquad
\mathrm{rank}(A)=1
```

The second column is twice the first, so it adds no independent direction.

**Full column rank** means $\mathrm{rank}(A)=n$; for a square matrix, full rank also means the matrix is invertible.

**Illustration:** If features in a design matrix are exact linear combinations of others, the matrix is rank-deficient, causing perfect multicollinearity.

---

## 3. Column Space and Null Space

The column space describes possible **outputs**:

```math
\mathrm{Col}(A)=\{Ax:x\in\mathbb{R}^{n}\}
```

The **null space** describes input directions that disappear under the transformation:

```math
\mathrm{Null}(A)=\{x\in\mathbb{R}^n:Ax=0\}
```

For example:

```math
A=
\begin{bmatrix}
1&2\\
2&4
\end{bmatrix}
\quad\Longrightarrow\quad
\mathrm{Null}(A)=
\mathrm{span}
\left(
\begin{bmatrix}-2\\1\end{bmatrix}
\right)
```

<p align="center">
  <img src="assets/column_null_spaces.svg" width="85%" alt="Geometric comparison of a matrix column space and null space">
</p>

**Illustration:** If $Xw_1=Xw_2$, then $w_1-w_2\in\mathrm{Null}(X)$: two coefficient vectors produce the same predictions.

---

## 4. Rank-Nullity and Solution Uniqueness

For $A\in\mathbb{R}^{m\times n}$:

```math
\boxed{\mathrm{rank}(A)+\dim(\mathrm{Null}(A))=n}
```

If $x_0$ solves $Ax=b$, all solutions have the form:

```math
x=x_0+z,
\qquad z\in\mathrm{Null}(A)
```

Therefore, a **trivial null space** implies at most one solution, while a nontrivial null space gives infinitely many solutions whenever the system is consistent.

- **Overdetermined:** $m>n$ (more equations than unknowns); an exact solution may not exist.
- **Underdetermined:** $m<n$ (more unknowns than equations); any consistent system has infinitely many solutions.

**Illustration:** In high-dimensional regression with more features than observations, unregularized parameters cannot be uniquely identified from the training predictions alone.

---

## 5. Orthogonal Projections

If $b\notin\mathrm{Col}(A)$, there is no exact solution to $Ax=b$. Instead, we find the closest vector in the column space:

```math
\hat b=A\hat x,
\qquad
\hat x=\arg\min_x\|Ax-b\|_2^2
```

The residual is:

```math
r=b-\hat b
```

At the orthogonal projection, it is perpendicular to the entire column space:

```math
r\perp\mathrm{Col}(A)
\qquad\Longleftrightarrow\qquad
A^\top r=0
```

<p align="center">
  <img src="assets/orthogonal_projection.svg" width="85%" alt="Target vector, orthogonal projection onto the column space, and perpendicular residual">
</p>

**Illustration:** A regression model chooses the closest prediction vector to the targets among all predictions representable by its feature columns.

---

## 6. Least Squares and Normal Equations

The **least-squares objective** is:

```math
J(x)=\|Ax-b\|_2^2
=(Ax-b)^\top(Ax-b)
```

Expanding and differentiating:

```math
J(x)=x^\top A^\top Ax-2b^\top Ax+b^\top b
```

```math
\nabla_xJ(x)=2A^\top Ax-2A^\top b
```

Setting the gradient to zero yields the **normal equations**:

```math
\boxed{A^\top A\hat x=A^\top b}
```

If $A$ has full column rank, the solution is unique:

```math
\boxed{\hat x=(A^\top A)^{-1}A^\top b}
```

For rank-deficient matrices, minimizers still exist but may not be unique. In practice, numerical solvers using **QR decomposition or SVD** are preferred over forming the inverse explicitly.

**Illustration:** The normal equations are the algebraic form of the geometric condition $A^	op r=0$.

---

## 7. Connection to Linear Regression

For a dataset $X\in\mathbb{R}^{N\times d}$ and targets $y\in\mathbb{R}^N$, ordinary least squares solves:

```math
w^*=\arg\min_w\|Xw-y\|_2^2
```

The fitted predictions $\hat y=Xw^*$ are the **orthogonal projection** of $y$ onto $\mathrm{Col}(X)$:

```math
X^\top(y-Xw^*)=0
```

If the columns of $X$ are linearly dependent, the coefficient vector may not be unique even though the fitted prediction vector is unique.

**Illustration:** L2 regularization can make the coefficient estimate unique even when features are redundant, while also discouraging large weights.

---

## 8. Conceptual Hierarchy

```text
Linear system: Ax = b
        ↓
Is b in the column space?
    ├── Yes → exact solution(s)
    └── No  → closest approximation
                  ↓
         Orthogonal projection
                  ↓
        Least-squares objective
                  ↓
           Normal equations
                  ↓
           Linear regression
```

The central connection is:

```math
b\notin\mathrm{Col}(A)
\quad\Longrightarrow\quad
\min_x\|Ax-b\|_2^2
\quad\Longrightarrow\quad
A^\top A\hat x=A^\top b
```
