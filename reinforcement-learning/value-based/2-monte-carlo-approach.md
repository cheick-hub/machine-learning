# Monte Carlo Approach in Value-Based Reinforcement Learning

## 0. Context: From Dynamic Programming to Monte Carlo

In Dynamic Programming, the agent updates value functions using the known MDP model:

```math
P(s'|s,a)
```

and

```math
R(s,a,s')
```

This allows the agent to compute expected Bellman backups by averaging over all possible next states. However, in most real Reinforcement Learning problems, the transition function and reward function are unknown.

Monte Carlo methods remove this assumption.

> **Core idea:** Instead of using the environment model, the agent learns value functions from complete sampled episodes.

An episode is a sequence:

```math
S_0, A_0, R_1, S_1, A_1, R_2, \dots, S_T
```

Monte Carlo methods wait until the episode ends, compute the observed return, and use it as a target for updating value estimates.

---

## 1. Objective of Monte Carlo Methods

The goal is still to estimate the value of states or state-action pairs.

The state-value function is:

```math
V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t=s]
```

The action-value function is:

```math
Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t=s, A_t=a]
```

where the return is:

```math
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots + \gamma^{T-t-1}R_T
```

The difference from Dynamic Programming is the target used for learning.

Dynamic Programming uses an expected one-step backup:

```math
R(s,a,s') + \gamma V(s')
```

Monte Carlo uses the full observed return:

```math
G_t
```

> **Interpretation:** Monte Carlo estimates value by averaging what actually happened after visiting a state or taking an action.

---

## 2. Monte Carlo Prediction

Monte Carlo prediction means evaluating a fixed policy $\pi$.

The policy is not improved yet. The goal is only to estimate:

```math
V^\pi(s)
```

or

```math
Q^\pi(s,a)
```

from sampled episodes.

For state values, the update is:

```math
V(S_t) \leftarrow V(S_t) + \alpha \left[G_t - V(S_t)\right]
```

For action values, the update is:

```math
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha \left[G_t - Q(S_t,A_t)\right]
```

This has the same structure as a moving-average update:

```math
\text{new estimate} = \text{old estimate} + \alpha(\text{target} - \text{old estimate})
```

where:

```math
\text{target} = G_t
```

and:

```math
\text{prediction error} = G_t - V(S_t)
```

or:

```math
\text{prediction error} = G_t - Q(S_t,A_t)
```

> **Justification:** Since $G_t$ is the realized return from the episode, it is a sampled estimate of the true expected return.

---

## 3. First-Visit and Every-Visit Monte Carlo

First-visit and every-visit Monte Carlo are not completely separate families of algorithms. They are two ways of deciding which occurrences of a state or state-action pair should be updated inside an episode.

### 3.1 First-Visit Monte Carlo

In first-visit Monte Carlo, the value is updated only the first time a state, or state-action pair, appears in an episode.

Example episode:

```text
A -> B -> A -> C
```

For state $A$, first-visit Monte Carlo updates only from the first occurrence of $A$.

For action values, if $(S_t,A_t)$ appears multiple times in the same episode, only the first occurrence is used.

The update remains:

```math
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha \left[G_t - Q(S_t,A_t)\right]
```

but it is applied only on the first visit.

### 3.2 Every-Visit Monte Carlo

In every-visit Monte Carlo, the value is updated every time the state or state-action pair appears in the episode.

Using the same example:

```text
A -> B -> A -> C
```

Every-visit Monte Carlo updates $A$ twice, once for each occurrence.

The update is still:

```math
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha \left[G_t - Q(S_t,A_t)\right]
```

but it is applied at every visit.

### 3.3 Practical Difference

| Variant | Update rule | Main idea |
|---|---|---|
| First-visit MC | Update only the first occurrence in an episode | Cleaner statistical interpretation |
| Every-visit MC | Update every occurrence in an episode | Uses more data from each episode |

Both methods estimate values from full episode returns. The main difference is only how many times they update from a single episode.

---

## 4. Monte Carlo Control

Monte Carlo control extends Monte Carlo prediction by adding policy improvement.

Prediction answers:

> How good is the current policy?

Control answers:

> How can the agent improve its policy using the estimated values?

For control, it is more practical to estimate the action-value function:

```math
Q(s,a)
```

rather than only the state-value function:

```math
V(s)
```

This is because $Q(s,a)$ allows the agent to choose actions directly without knowing the environment model:

```math
\pi(s) = \arg\max_a Q(s,a)
```

If the agent only knows $V(s)$, it still needs the model $P(s'|s,a)$ to know which action leads to the best next state. With $Q(s,a)$, the action quality is already estimated.

### 4.1 General Monte Carlo Control Loop

Monte Carlo control follows the same structure as Generalized Policy Iteration:

```text
Initialize Q(s,a) arbitrarily
Initialize an exploratory policy π

Repeat:
    1. Generate a complete episode using π
    2. Compute returns G_t from the episode
    3. Update Q(S_t,A_t) using Monte Carlo returns
    4. Improve π to be more greedy with respect to Q
```

The value update is:

```math
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha \left[G_t - Q(S_t,A_t)\right]
```

The policy improvement step is:

```math
\pi(s) = \arg\max_a Q(s,a)
```

However, a purely greedy policy may stop exploring too early. Therefore, Monte Carlo control usually uses an $\epsilon$-greedy policy.

---

## 5. Exploration with $\epsilon$-Greedy Policies

An $\epsilon$-greedy policy chooses a random action with probability $\epsilon$, and the best-known action with probability $1 - \epsilon$:

```math
A_t =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(S_t,a), & \text{with probability } 1 - \epsilon
\end{cases}
```

This prevents the agent from becoming greedy too early when the value estimates are still inaccurate.

For example, suppose the current estimates are:

| Action | Estimated value |
|---|---:|
| $a_1$ | 5.0 |
| $a_2$ | 2.0 |
| $a_3$ | 1.0 |

A greedy policy always selects $a_1$. An $\epsilon$-greedy policy usually selects $a_1$, but sometimes tries $a_2$ or $a_3$ to collect more information.

> **Justification:** Monte Carlo control requires sufficient exploration. If an action is never tried, its value cannot be estimated correctly.

---

## 6. Monte Carlo Control Algorithm

A simplified on-policy Monte Carlo control algorithm is:

```text
Initialize Q(s,a) arbitrarily
Initialize π as an ε-greedy policy with respect to Q

Repeat for each episode:
    Generate an episode using π:
        S_0, A_0, R_1, ..., S_T

    For each timestep t in the episode:
        Compute the return:
            G_t = R_{t+1} + γR_{t+2} + ... + γ^{T-t-1}R_T

        Update Q:
            Q(S_t,A_t) ← Q(S_t,A_t) + α[G_t - Q(S_t,A_t)]

    Improve π:
        Make π ε-greedy with respect to Q
```

The algorithm can use either first-visit or every-visit updates.

Therefore:

```text
Monte Carlo control = Monte Carlo value estimation + policy improvement
```

---

## 7. Essential Trade-offs and Limitations

Monte Carlo methods use the full observed return as the learning target:

```math
G_t
```

They do **not** bootstrap from their own value estimate, unlike Temporal Difference methods:

```math
R_{t+1} + \gamma V(S_{t+1})
```

This creates the main Monte Carlo trade-off:

| Method | Target | Requires full episode? | Bootstraps? | Main trade-off |
|---|---|---:|---:|---|
| Monte Carlo | $G_t$ | Yes | No | Low bias, high variance |
| Temporal Difference | $R_{t+1} + \gamma V(S_{t+1})$ | No | Yes | Higher bias, lower variance |

The essential limitations are:

* Monte Carlo must wait until the episode ends before updating.
* The return $G_t$ can be noisy because it depends on all future rewards.
* Learning can be slower than TD methods in long or continuous tasks.
* Monte Carlo control still requires enough exploration, usually through an $\epsilon$-greedy policy.

> **Key point:** Monte Carlo is simple and model-free, but it is less suitable for online learning than TD methods such as SARSA and Q-learning.

---

## 8. Summary

Monte Carlo methods are model-free value-based methods that estimate value functions from complete sampled episodes.

The key update is:

```math
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha \left[G_t - Q(S_t,A_t)\right]
```

First-visit and every-visit Monte Carlo are two variants of value estimation:

* **First-visit MC** updates only the first occurrence of a state or state-action pair in an episode.
* **Every-visit MC** updates every occurrence.

Monte Carlo control adds policy improvement:

```math
\pi(s) = \arg\max_a Q(s,a)
```

or, more commonly, an $\epsilon$-greedy version to maintain exploration.

The essential idea is:

> Play full episodes, compute the observed returns, update $Q(s,a)$, and improve the policy using the learned action values.
