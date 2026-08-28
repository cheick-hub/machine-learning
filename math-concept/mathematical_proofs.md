# Mathematical Proofs: A Formal View

A mathematical proof can be viewed as a process in which we start from **definitions, hypotheses, axioms, and previously established theorems**, then apply valid logical rules to establish that a target proposition is true.

In compact form:

$$
\boxed{
\text{Definitions + Axioms + Hypotheses + Theorems}
\xrightarrow{\text{rules of inference}}
\text{Proposition to prove}
}
$$

## 1. Objects and domains

A proof operates on objects belonging to some domain or set. Variables should therefore have a well-defined universe of discourse.

Examples:

$$
x \in \mathbb{R}, \qquad n \in \mathbb{N}, \qquad f \in \mathcal{F}.
$$

The domain matters because the truth of a statement can depend on it.

## 2. Predicates and propositions

A **proposition** is a statement that can be true or false.

A **predicate** is a proposition whose truth value depends on one or more variables. For example,

$$
Human(x)
$$

is true when $x$ is a human and false otherwise.

A mathematical example is

$$
Even(n) := \exists k \in \mathbb{Z},\; n = 2k.
$$

Here, $Even(n)$ is a predicate over integers.

## 3. Quantifiers

Quantifiers specify how variables are quantified over a domain.

### Universal quantifier

$$
\forall x \in E,\; P(x)
$$

means that $P(x)$ is true for every $x$ in $E$.

### Existential quantifier

$$
\exists x \in E,\; P(x)
$$

means that at least one $x$ in $E$ satisfies $P(x)$.

### Unique existence

$$
\exists! x \in E,\; P(x)
$$

means that exactly one such $x$ exists.

## 4. Logical operators

Propositions can be combined using logical connectives:

| Operator | Meaning |
|---|---|
| $P \land Q$ | P and Q |
| $P \lor Q$ | P or Q |
| $\neg P$ | not P |
| $P \Rightarrow Q$ | P implies Q |
| $P \Leftrightarrow Q$ | P if and only if Q |

These operators define the logical structure of mathematical statements.

## 5. Definitions

Definitions are fundamental because they specify precisely what mathematical concepts mean.

For example,

$$
n \text{ is even} \iff \exists k \in \mathbb{Z},\; n = 2k.
$$

Many proofs begin by **unfolding a definition**, manipulating the resulting statement, and eventually matching another definition.

## 6. Hypotheses and axioms

**Hypotheses** are propositions assumed to be true within the context of a particular theorem.

**Axioms** are foundational statements accepted without proof within a formal system.

They form part of the starting information from which deductions can be made.

## 7. Previously established theorems

Once a theorem has been proved, it can be reused in later proofs.

For example, suppose we already know

$$
A \Rightarrow B
$$

and establish $A$. We may then infer $B$.

This allows mathematics to build increasingly complex results on top of previously established ones.

## 8. Rules of inference

Rules of inference determine which logical transitions are valid. They are what make a sequence of statements a proof rather than merely a sequence of assertions.

A fundamental example is **modus ponens**:

$$
P, \qquad P \Rightarrow Q
$$

therefore

$$
Q.
$$

In other words, if $P$ is true and $P$ implies $Q$, then $Q$ must also be true.

## 9. Proof strategies

Proofs can be organized using several standard reasoning structures, including:

- direct proof;
- proof by contraposition;
- proof by contradiction;
- mathematical induction;
- proof by cases;
- constructive existence proofs;
- uniqueness proofs.

These are not additional logical primitives. They are higher-level strategies for arranging valid logical deductions.

## 10. Example

Consider the statement

$$
\forall n \in \mathbb{Z},\quad Even(n) \Rightarrow Even(n^2).
$$

We want to prove that the square of every even integer is even.

Assume

$$
Even(n).
$$

By definition of an even integer,

$$
\exists k \in \mathbb{Z} : n = 2k.
$$

Therefore,

$$
n^2 = (2k)^2 = 4k^2 = 2(2k^2).
$$

Since $k \in \mathbb{Z}$, we also have

$$
2k^2 \in \mathbb{Z}.
$$

Let

$$
m = 2k^2.
$$

Then

$$
n^2 = 2m
$$

for some $m \in \mathbb{Z}$. By the definition of evenness,

$$
Even(n^2).
$$

Thus,

$$
Even(n) \Rightarrow Even(n^2).
$$

Since $n$ was arbitrary,

$$
\forall n \in \mathbb{Z},\quad Even(n) \Rightarrow Even(n^2).
$$

## 11. Proofs as formal transformations

At a sufficiently formal level, a mathematical proof can therefore be understood as a sequence of transformations:

$$
S_0 \rightarrow S_1 \rightarrow S_2 \rightarrow \cdots \rightarrow S_n,
$$

where each transition is justified by a definition, an axiom, a previously proved theorem, or a valid rule of inference, and where $S_n$ is the proposition we wanted to establish.

The language used to construct these statements consists primarily of:

- objects and domains;
- functions and relations;
- predicates and propositions;
- quantifiers;
- logical connectives;
- definitions;
- axioms and hypotheses;
- theorems;
- rules of inference.

This perspective is closely related to how **proof assistants** such as Lean, Coq, and Isabelle work: a proof is represented as a precise sequence of logically valid steps that can be checked mechanically.
