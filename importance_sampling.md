# Importance Sampling

## 1. What Is Importance Sampling?

**Importance sampling** is a Monte Carlo technique used to estimate an expectation under one distribution while sampling from another distribution.

Instead of sampling directly from the original distribution $p(x)$, we sample from another distribution $q(x)$. This second distribution is usually chosen because it is easier to sample from or because it focuses more on important regions.

Then we correct the bias introduced by sampling from $q(x)$ using a weight.

---

## 2. The Problem

Suppose we want to compute an expectation:

$$
\mathbb{E}_{p}[f(x)]
$$

This means:

$$
\mathbb{E}_{p}[f(x)] = \int f(x)p(x)dx
$$

where:

- $p(x)$ is the target/original distribution.
- $f(x)$ is the function we want to average.
- $x$ is sampled according to $p(x)$.

In standard Monte Carlo, we would sample:

$$
x_1, x_2, ..., x_N \sim p(x)
$$

and estimate:

$$
\mathbb{E}_{p}[f(x)] \approx \frac{1}{N}\sum_{i=1}^{N} f(x_i)
$$

But sometimes sampling from $p(x)$ is difficult, inefficient, or produces too many samples that contribute very little.

---

## 3. Core Idea

Instead of sampling from $p(x)$, we sample from another distribution:

$$
x_1, x_2, ..., x_N \sim q(x)
$$

where $q(x)$ is called the **proposal distribution** or **sampling distribution**.

However, if we sample from $q(x)$, we are no longer directly estimating the expectation under $p(x)$. We are sampling under $q(x)$, so we need to correct the mismatch between $p(x)$ and $q(x)$.

The correction weight is:

$$
w(x) = \frac{p(x)}{q(x)}
$$

This gives the importance sampling identity:

$$
\mathbb{E}_{p}[f(x)]
=
\int f(x)p(x)dx
=
\int f(x)\frac{p(x)}{q(x)}q(x)dx
=
\mathbb{E}_{q}\left[f(x)\frac{p(x)}{q(x)}\right]
$$

So the estimator becomes:

$$
\mathbb{E}_{p}[f(x)]
\approx
\frac{1}{N}\sum_{i=1}^{N} f(x_i)w(x_i)
$$

where:

$$
x_i \sim q(x)
$$

and:

$$
w(x_i)=\frac{p(x_i)}{q(x_i)}
$$

---

## 4. Intuition

Importance sampling intentionally introduces a sampling bias by drawing samples from $q(x)$ instead of $p(x)$.

Then it removes that bias using the weight:

$$
\frac{p(x)}{q(x)}
$$

The weight compares:

- how likely the sample is under the original distribution $p(x)$
- how likely the sample is under the sampling distribution $q(x)$

For a sampled point $x$:

$$
w(x)=\frac{\text{density under }p}{\text{density under }q}
$$

### Interpretation

If:

$$
q(x) > p(x)
$$

then $x$ is sampled too often by $q$, so its weight is small.

If:

$$
q(x) < p(x)
$$

then $x$ is sampled too rarely by $q$, so its weight is large.

So the weight compensates for over-sampling or under-sampling.

---

## 5. Important Detail: The Weight Uses the Same Sample

The weight is computed by evaluating both distributions at the same sampled value $x$.

You do **not** take one sample from $p$ and divide it by one sample from $q$.

You do this:

$$
w(x)=\frac{p(x)}{q(x)}
$$

For example, if the sampled value is:

$$
x=2
$$

then the weight is:

$$
w(2)=\frac{p(2)}{q(2)}
$$

So we evaluate:

- the density of $x=2$ under $p$
- the density of $x=2$ under $q$

and divide them.

---

## 6. Concrete Example

Suppose we want to estimate:

$$
\mathbb{E}_{p}[x^2]
$$

where:

$$
p(x)=\mathcal{N}(0,1)
$$

This means $p(x)$ is a standard normal distribution.

The true value is:

$$
\mathbb{E}[x^2]=1
$$

because the variance of a standard normal distribution is 1.

Now suppose we choose a wider proposal distribution:

$$
q(x)=\mathcal{N}(0,2^2)
$$

This distribution has more probability mass in the tails, so it samples larger values more often.

For each sample $x_i \sim q(x)$, we compute:

$$
x_i^2 \frac{p(x_i)}{q(x_i)}
$$

Then we average the weighted values.

---

## 7. Numerical Example

Assume we sample from $q(x)$:

$$
x = [-2,\ 1.5,\ 0.3,\ 2.5]
$$

For each sampled value, we compute:

$$
w(x)=\frac{p(x)}{q(x)}
$$

| Sample $x$ | $x^2$ | Weight $p(x)/q(x)$ | Weighted contribution |
|---:|---:|---:|---:|
| -2.0 | 4.00 | 0.27 | 1.08 |
| 1.5 | 2.25 | 0.46 | 1.04 |
| 0.3 | 0.09 | 1.88 | 0.17 |
| 2.5 | 6.25 | 0.15 | 0.94 |

The importance sampling estimate is:

$$
\frac{1.08 + 1.04 + 0.17 + 0.94}{4}
=
0.81
$$

The true value is 1.

With more samples, the estimate should become more accurate, assuming $q(x)$ is well chosen.

---

## 8. Why This Works

If we sampled directly from $p(x)$, we could estimate:

$$
\mathbb{E}_{p}[f(x)]
$$

directly.

But since we sampled from $q(x)$, we need to correct each sample.

The correction:

$$
\frac{p(x)}{q(x)}
$$

answers the question:

> How much should this sample count if it had come from $p(x)$ instead of $q(x)$?

So importance sampling can be understood as:

> Sample from a convenient distribution, then reweight the samples so they behave like they came from the original distribution.

---

## 9. Requirements

Importance sampling requires that we can evaluate both distributions:

$$
p(x)
$$

and:

$$
q(x)
$$

or at least their ratio:

$$
\frac{p(x)}{q(x)}
$$

We do not necessarily need to sample from $p(x)$, but we need to evaluate how likely a sample is under $p(x)$.

This is the key reason importance sampling is useful:

- sampling from $p(x)$ may be hard
- sampling from $q(x)$ may be easy
- evaluating $p(x)$ may still be possible

---

## 10. Support Condition

Importance sampling only works correctly if:

$$
q(x) > 0 \quad \text{whenever} \quad p(x) > 0
$$

This means that $q(x)$ must be able to sample all regions where $p(x)$ has probability mass.

If $q(x)=0$ in a region where $p(x)>0$, then that region will never be sampled, and importance sampling cannot correct for it.

---

## 11. Choosing a Good Proposal Distribution

The choice of $q(x)$ is very important.

A good proposal distribution should:

- be easy to sample from
- cover the important regions of $p(x)$
- avoid assigning very small probability where $p(x)$ is large
- produce stable weights

A bad $q(x)$ can cause very large weights.

This leads to high variance and unstable estimates.

In practice, the main failure mode of importance sampling is poor overlap between $p(x)$ and $q(x)$.

---

## 12. Importance Sampling in Reinforcement Learning

In reinforcement learning, importance sampling is often used for **off-policy evaluation**.

Suppose data was collected using one policy:

$$
\mu(a|s)
$$

but we want to evaluate another policy:

$$
\pi(a|s)
$$

The importance weight becomes:

$$
w = \frac{\pi(a|s)}{\mu(a|s)}
$$

where:

- $\mu(a|s)$ is the behavior policy that generated the data.
- $\pi(a|s)$ is the target policy we want to evaluate.

If an action was common under the old policy but rare under the new policy, its weight is small.

If an action was rare under the old policy but common under the new policy, its weight is large.

This corrects the distribution shift between the behavior policy and the target policy.

---

## 13. Summary

Importance sampling is a way to estimate an expectation under a target distribution $p(x)$ while sampling from another distribution $q(x)$.

The key formula is:

$$
\mathbb{E}_{p}[f(x)]
=
\mathbb{E}_{q}\left[f(x)\frac{p(x)}{q(x)}\right]
$$

The weight:

$$
\frac{p(x)}{q(x)}
$$

corrects the bias introduced by sampling from $q(x)$.

In simple terms:

> Importance sampling means sampling from a convenient distribution and correcting each sample so the final estimate matches the original distribution.

The method is powerful, but it depends heavily on choosing a good proposal distribution $q(x)$.
