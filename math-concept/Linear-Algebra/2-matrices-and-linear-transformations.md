# 2. Linear Algebra — Matrices and Linear Transformations

## Key Takeaways

- A **matrix** is a rectangular array of numbers that can represent a linear transformation.
- Matrix-vector multiplication computes a linear combination of the matrix's columns.
- A **linear transformation** preserves vector addition and scalar multiplication.
- Matrix multiplication represents the **composition** of transformations; its order generally matters.
- Identity, diagonal, transpose, symmetric and inverse matrices have useful structural properties.
- A neural network layer $Wx+b$ is an **affine transformation**; nonlinear activations enable nonlinear models.

---

## 1. Matrices and Dimensions

A matrix with $m$ rows and $n$ columns is written:

```math
A=
\begin{bmatrix}
a_{11} & \cdots & a_{1n}\\
\vdots & \ddots & \vdots\\
a_{m1} & \cdots & a_{mn}
\end{bmatrix}
\in\mathbb{R}^{m\times n}
```

In Machine Learning, a dataset is commonly represented as:

```math
X\in\mathbb{R}^{N\times d}
```

where $N$ is the number of observations and $d$ is the number of features. Rows represent observations; columns represent features.

---

## 2. Matrix Operations

Matrices of the same shape can be added element by element, while scalar multiplication scales every entry:

```math
(A+B)_{ij}=A_{ij}+B_{ij},
\qquad
(\alpha A)_{ij}=\alpha A_{ij}
```

### Matrix-Vector Multiplication

For $A\in\mathbb{R}^{m\times n}$ and $x\in\mathbb{R}^{n}$:

```math
y=Ax\in\mathbb{R}^{m}
```

If $a_i$ denotes column $i$ of $A$, then:

```math
Ax=x_1a_1+x_2a_2+\cdots+x_na_n
```

The result is therefore a **linear combination of the columns** of $A$:

```math
Ax\in\mathrm{span}(a_1,\ldots,a_n)
```

**Illustration:** Linear regression computes $\hat y=Xw$, a weighted combination of the feature columns.

### Matrix-Matrix Multiplication

For $A\in\mathbb{R}^{m\times n}$ and $B\in\mathbb{R}^{n\times p}$:

```math
AB\in\mathbb{R}^{m\times p},
\qquad
(AB)_{ij}=\sum_{k=1}^{n}A_{ik}B_{kj}
```

Each output entry is the dot product of a row of $A$ and a column of $B$.

---

## 3. Linear Transformations

A **linear transformation** $T:\mathbb{R}^n\rightarrow\mathbb{R}^m$ satisfies:

```math
T(\alpha x+\beta y)=\alpha T(x)+\beta T(y)
```

Any matrix $A\in\mathbb{R}^{m\times n}$ defines such a transformation:

```math
T(x)=Ax
```

Geometrically, matrices can transform vectors through:

- **Scaling:** stretching or shrinking coordinates.
- **Rotation:** changing direction while preserving lengths.
- **Reflection:** mirroring vectors across an axis.
- **Projection:** retaining components along a chosen subspace.

For example, projection onto the horizontal axis is:

```math
P=
\begin{bmatrix}
1&0\\
0&0
\end{bmatrix},
\qquad
P\begin{bmatrix}x\\y\end{bmatrix}
=
\begin{bmatrix}x\\0\end{bmatrix}
```

**Illustration:** Least-squares regression and PCA use projections onto subspaces.

---

## 4. Composition of Transformations

If $T_1(x)=Ax$ and $T_2(x)=Bx$, applying $T_1$ followed by $T_2$ yields:

```math
T_2(T_1(x))=B(Ax)=(BA)x
```

Therefore, multiplying matrices corresponds to composing transformations.

**Order matters:** in general,

```math
AB\neq BA
```

For instance, rotating and then scaling along one coordinate axis generally differs from scaling first and then rotating.

---

## 5. Important Matrix Structures

| Matrix | Property | Interpretation |
|---|---|---|
| **Identity** $I$ | $Ix=x$ | Leaves a vector unchanged |
| **Diagonal** $D$ | $D_{ij}=0$ for $i\neq j$ | Scales coordinates independently |
| **Transpose** $A^\top$ | $(A^\top)_{ij}=A_{ji}$ | Swaps rows and columns |
| **Symmetric** $A$ | $A=A^\top$ | Important for covariance and eigendecomposition |
| **Inverse** $A^{-1}$ | $A^{-1}A=I$ | Undoes an invertible transformation |

Useful identities include:

```math
(AB)^\top=B^\top A^\top
```

```math
Ax=b\quad\Longrightarrow\quad x=A^{-1}b
```

The second identity requires $A$ to be **square and invertible**. In numerical ML, solving a linear system directly is generally preferable to explicitly computing the inverse.

---

## 6. Linear vs. Affine Transformations

A purely linear transformation has the form:

```math
T(x)=Wx
```

and always satisfies $T(0)=0$.

Adding a bias produces an **affine transformation**:

```math
T(x)=Wx+b
```

which can translate the output away from the origin.

Neural-network layers typically combine an affine transformation with a nonlinear activation:

```math
h=\sigma(Wx+b)
```

Without nonlinear activations, multiple layers collapse into a single affine transformation. For example, without biases:

```math
W_2(W_1x)=(W_2W_1)x
```

**Illustration:** A weight matrix $W\in\mathbb{R}^{64\times100}$ maps an input with 100 features into a 64-dimensional representation.

---

## 7. Conceptual Hierarchy

```text
Matrix
   ↓
Matrix-vector multiplication
   ↓
Linear transformation
   ↓
Composition via matrix multiplication
   ↓
Affine transformations + nonlinear activations
   ↓
Neural network layers
```

The key connection is that **matrices encode transformations**, and their multiplication lets Machine Learning models combine those transformations into more complex computations.
