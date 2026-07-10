# 2. Statistics Fundamentals for Machine Learning

## 1. Introduction

Statistics studies how to learn about an unknown population from finite observations. Machine Learning is deeply statistical because a model only sees a limited training dataset but must perform well on unseen data.

Let the dataset be:

$$
D = \{x_1, x_2, \ldots, x_n\}
$$

where each observation is assumed to be sampled from an unknown distribution $P_X$:

$$
x_i \sim P_X
$$

The true distribution $P_X$ is not directly available. Statistics gives us tools to estimate useful quantities from samples and to reason about how uncertain those estimates are.

**ML intuition:** probability describes the data-generating process; statistics helps us infer that process from data.

---

## 2. Population vs Sample

A **population** is the complete set of possible observations we care about. A **sample** is the finite subset we actually observe.

Machine Learning almost never observes the full population. It observes a sample:

$$
D = \{x_1, \ldots, x_n\}, \qquad x_i \sim P_X
$$

where $D$ is the training or evaluation dataset.

The central problem is that the model is trained and evaluated on samples, but it is expected to generalize to the broader population.

**Simple example:** a fraud detection model is trained on historical transactions, but it must work on future transactions that were not in the training data.

---

## 3. Estimation

An **estimator** is a rule that uses data to estimate an unknown population quantity.

For example, the population mean is:

$$
\mu = \mathbb{E}[X]
$$

Since $\mu$ is unknown, we can estimate it with the sample mean:

$$
\hat{\mu} = \frac{1}{n}\sum_{i=1}^{n}x_i
$$

The hat notation, as in $\hat{\mu}$, means "estimated from data."

More generally, if $\theta$ is an unknown parameter, an estimator $\hat{\theta}$ is computed from the sample:

$$
\hat{\theta} = g(D)
$$

where $g$ is some rule or algorithm.

**Simple example:** measured test accuracy is an estimate of the model's true accuracy on the deployment distribution.

**Engineer note:** model training itself is estimation: the learned weights are estimated from the training data.

---

## 4. Bias and Variance of Estimators

An estimator can be evaluated by its **bias** and **variance**.

Bias measures systematic error:

$$
\operatorname{Bias}(\hat{\theta}) =
\mathbb{E}[\hat{\theta}] - \theta
$$

If the bias is zero, the estimator is unbiased:

$$
\mathbb{E}[\hat{\theta}] = \theta
$$

Variance measures how much the estimator changes from one sample to another:

$$
\operatorname{Var}(\hat{\theta}) =
\mathbb{E}\left[
  \left(\hat{\theta} - \mathbb{E}[\hat{\theta}]\right)^2
\right]
$$

A good estimator should usually have both low bias and low variance, but there is often a tradeoff.

**ML intuition:** a simple model can be systematically wrong, which is high bias. A very flexible model can change a lot depending on the training set, which is high variance.

**Engineer note:** regularization often increases bias slightly but reduces variance enough to improve generalization.

---

## 5. Maximum Likelihood Estimation

Maximum Likelihood Estimation, or MLE, estimates parameters by choosing the values that make the observed data most probable.

For parameters $\theta$ and dataset $D$:

$$
\hat{\theta} =
\arg\max_{\theta} P(D \mid \theta)
$$

If samples are independent and identically distributed, often written i.i.d., then:

$$
P(D \mid \theta) =
\prod_{i=1}^{n} P(x_i \mid \theta)
$$

In practice, we usually maximize the log-likelihood:

$$
\log P(D \mid \theta) =
\sum_{i=1}^{n}\log P(x_i \mid \theta)
$$

The log form is easier to optimize because it turns products into sums. It also avoids numerical underflow when multiplying many small probabilities.

**Simple example:** logistic regression can be derived by maximizing the likelihood of Bernoulli-distributed labels.

**Engineer note:** many common losses are negative log-likelihoods. Minimizing the loss is equivalent to maximizing the likelihood.

---

## 6. Empirical Risk and Expected Risk

In Machine Learning, the true objective is usually the **expected risk**:

$$
R(f) =
\mathbb{E}_{(X,Y)\sim P_{X,Y}}
\left[\ell(f(X), Y)\right]
$$

where:

- $f$ is the model.
- $\ell$ is the loss function.
- $P_{X,Y}$ is the unknown joint distribution of inputs and labels.

Expected risk is the average loss over the real data-generating distribution. Since $P_{X,Y}$ is unknown, we approximate it with **empirical risk**:

$$
\hat{R}(f) =
\frac{1}{n}\sum_{i=1}^{n}\ell(f(x_i), y_i)
$$

This is the average loss on the observed dataset.

**ML intuition:** training minimizes empirical risk, but the real target is expected risk on unseen data.

**Engineer note:** overfitting happens when empirical risk becomes low while expected risk remains high.

---

## 7. Confidence Intervals

A **confidence interval** quantifies uncertainty around an estimate.

A common approximate confidence interval for an estimate $\hat{\theta}$ is:

$$
\hat{\theta}
\pm
z_{1-\alpha/2}\frac{\sigma}{\sqrt{n}}
$$

where:

- $\hat{\theta}$ is the estimate.
- $\sigma$ is the standard deviation of the quantity being estimated.
- $n$ is the sample size.
- $\alpha$ is the allowed error probability outside the interval.
- $z_{1-\alpha/2}$ is a quantile of the standard normal distribution.

For a 95% confidence interval, $\alpha = 0.05$, so:

$$
z_{1-\alpha/2} = z_{0.975} \approx 1.96
$$

The interval usually shrinks as $n$ increases because the standard error term $\sigma / \sqrt{n}$ becomes smaller.

**Simple example:** if a model has 91% accuracy on a finite test set, the confidence interval expresses how uncertain that 91% estimate is.

**Engineer note:** confidence intervals are useful when comparing models. A tiny accuracy difference may not be meaningful if the uncertainty intervals overlap strongly.

---

## 8. Hypothesis Testing

Hypothesis testing evaluates whether an observed difference is large enough to be considered statistically meaningful under a chosen assumption.

A typical setup defines:

- $H_0$: the null hypothesis, usually "no effect" or "no difference."
- $H_1$: the alternative hypothesis, usually "there is an effect" or "there is a difference."

For example, to compare the mean performance of two systems:

$$
H_0: \mu_A = \mu_B
$$

$$
H_1: \mu_A \neq \mu_B
$$

The test asks whether the observed difference would be surprising if $H_0$ were true.

A **p-value** is the probability, under $H_0$, of observing a result at least as extreme as the one measured.

**Simple example:** in A/B testing, hypothesis testing helps decide whether a new model is genuinely better than the previous one or whether the observed improvement could be sampling noise.

**Engineer note:** statistical significance is not the same as practical importance. A very small effect can become statistically significant with a huge dataset.

---

## 9. Concentration Inequalities

Concentration inequalities bound how far an empirical estimate can be from its true expectation.

They answer the question:

> How likely is it that the sample average differs significantly from the true mean?

A central example is Hoeffding's inequality. For independent bounded variables, for example values in $[0, 1]$:

$$
P\left(|\hat{\mu} - \mu| \geq \epsilon\right)
\leq
2e^{-2n\epsilon^2}
$$

where:

- $\hat{\mu}$ is the empirical mean.
- $\mu$ is the true mean.
- $n$ is the sample size.
- $\epsilon$ is the tolerated deviation.

This inequality shows that the probability of a large deviation decreases exponentially as the number of samples increases.

**ML intuition:** more data usually makes empirical estimates more reliable.

**Engineer note:** PAC learning and generalization theory use concentration inequalities to prove that empirical performance converges toward true performance under assumptions such as independence and bounded losses.

---

## 10. How These Pieces Fit Together in ML

The main statistical workflow in ML is:

$$
\text{unknown population}
\rightarrow
\text{finite sample}
\rightarrow
\text{estimator}
\rightarrow
\text{uncertainty about the estimate}
\rightarrow
\text{generalization decision}
$$

Statistics explains this workflow:

- Samples approximate populations.
- Estimators approximate unknown parameters or metrics.
- Bias describes systematic error.
- Variance describes instability across samples.
- Likelihood connects model parameters to observed data.
- Empirical risk is the training objective we can compute.
- Expected risk is the real objective we care about.
- Confidence intervals and hypothesis tests quantify uncertainty in measured results.
- Concentration inequalities justify why empirical averages can generalize.

---

## 11. Key Takeaways

- Statistics estimates unknown population quantities from finite data.
- A sample is only an approximation of the full population.
- Estimators can have bias and variance.
- MLE is one of the most important parameter estimation principles in ML.
- Empirical risk is the measurable training objective, but expected risk is the real objective.
- Confidence intervals quantify uncertainty around estimates.
- Hypothesis tests help evaluate whether observed improvements are likely to be real.
- Concentration inequalities provide mathematical support for generalization guarantees.
