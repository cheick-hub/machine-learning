# Value-Based Approaches and Dynamic Programming

## 0. Markov Decision Process Definition

Before introducing value-based methods, we first define the environment as a **Markov Decision Process** (MDP). An MDP is written as:

```math
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, \rho_0)
```

Where:
* $\mathcal{S}$ is the **state space**. A state $s \in \mathcal{S}$ describes the current situation of the environment.
* $\mathcal{A}$ is the **action space**. An action $a \in \mathcal{A}$ is a decision available to the agent.
* $P$ is the **transition function**:

```math
P(s'|s,a) = \mathbb{P}(s_{t+1}=s' \mid s_t=s, a_t=a)
```

It gives the probability of moving to state $s'$ after taking action $a$ in state $s$.

* $R$ is the **reward function**:

```math
R(s,a,s') = \mathbb{E}[r_{t+1} \mid s_t=s, a_t=a, s_{t+1}=s']
```

It gives the immediate reward obtained after a transition.

* $\gamma \in [0,1]$ is the **discount factor**. It controls the importance of future rewards.
* $\rho_0$ is the **initial state distribution**. It defines where episodes start.

The MDP relies on the **Markov property**:

```math
\mathbb{P}(s_{t+1}|s_t,a_t,s_{t-1},a_{t-1},\dots)
=
\mathbb{P}(s_{t+1}|s_t,a_t)
```

> **Justification:** The next state depends only on the current state and current action, not on the full past history. Therefore, if the state is well defined, it contains all information needed for decision-making.

The agent follows a policy $\pi$:

```math
\pi(a|s) = \mathbb{P}(a_t=a \mid s_t=s)
```

The objective is to maximize the expected discounted return:

```math
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
```

---

## 1. Introduction and Core Objective

In value-based Reinforcement Learning, the agent does not directly learn a policy. Instead, it learns a **value function** that estimates how good a state or an action is.

There are two main value functions.

### State-Value Function

The state-value function is:

```math
V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t=s]
```

It answers:

> If the agent starts in state $s$ and follows policy $\pi$, how much future reward should it expect?

### Action-Value Function

The action-value function is:

```math
Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t=s, a_t=a]
```

It answers:

> If the agent takes action $a$ in state $s$, then follows policy $\pi$, how much future reward should it expect?

The optimal action-value function is:

```math
Q^*(s,a) = \max_\pi Q^\pi(s,a)
```

Once $Q^*(s,a)$ is known, the optimal policy is obtained by choosing the action with the highest value:

```math
\pi^*(s) = \arg\max_a Q^*(s,a)
```

> **Core idea:** Value-based methods transform the policy learning problem into a value estimation problem. The agent first estimates the value of actions, then acts greedily with respect to those estimates.

---

## 2. Bellman Equations

The Bellman equations are the mathematical foundation of value-based methods. They rely on the recursive structure of the return:

```math
G_t = r_{t+1} + \gamma G_{t+1}
```

This means that the value of the current state can be expressed using the immediate reward and the value of the next state.

### Bellman Equation for $V^\pi$

```math
V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)
\left[ R(s,a,s') + \gamma V^\pi(s') \right]
```

> **Interpretation:** The value of a state is the expected immediate reward plus the discounted value of the next state, averaged over possible actions and transitions.

### Bellman Equation for $Q^\pi$

```math
Q^\pi(s,a) = \sum_{s'} P(s'|s,a)
\left[ R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a') \right]
```

> **Interpretation:** $Q^\pi(s,a)$ evaluates one fixed first action, then evaluates the continuation under policy $\pi$.

### Bellman Optimality Equation for $V^*$

```math
V^*(s) = \max_a \sum_{s'} P(s'|s,a)
\left[ R(s,a,s') + \gamma V^*(s') \right]
```

### Bellman Optimality Equation for $Q^*$

```math
Q^*(s,a) = \sum_{s'} P(s'|s,a)
\left[ R(s,a,s') + \gamma \max_{a'} Q^*(s',a') \right]
```

> **Justification:** In the optimal case, the agent does not average over actions from a policy. It selects the action that maximizes future expected return.

---

## 3. Dynamic Programming Algorithms

Dynamic Programming methods solve MDPs by repeatedly applying Bellman updates. They assume that the environment model is known, meaning the agent has access to:

```math
P(s'|s,a)
```

and

```math
R(s,a,s')
```

Because of this assumption, Dynamic Programming is mainly a theoretical foundation for Reinforcement Learning. It is not always directly usable in real environments, but it explains the logic behind many later algorithms such as Monte Carlo methods, TD learning, SARSA, and Q-learning.

### 3.1 Policy Evaluation

Policy Evaluation estimates the value function of a fixed policy $\pi$.

The update is:

```math
V(s) \leftarrow \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)
\left[ R(s,a,s') + \gamma V(s') \right]
```

The algorithm repeatedly applies this update for every state until the values stabilize.

> **Intuition:** If the policy is fixed, we are not trying to improve the agent yet. We only want to measure how good the current policy is.

A simplified procedure is:

```text
Initialize V(s) arbitrarily for all states
Repeat:
    For each state s:
        Update V(s) using the Bellman expectation equation
Until V converges
```

The result is an approximation of $V^\pi(s)$.

---

### 3.2 Policy Improvement

Once we know the value of the current policy, we can improve the policy by choosing better actions.

The improved policy is:

```math
\pi_{new}(s) = \arg\max_a \sum_{s'} P(s'|s,a)
\left[ R(s,a,s') + \gamma V^\pi(s') \right]
```

> **Intuition:** For each state, we look one step ahead and choose the action that leads to the best expected return.

This works because if one action gives a higher expected return than the action currently chosen by the policy, then selecting that action produces a policy that is at least as good as the previous one.

---

### 3.3 Policy Iteration

Policy Iteration alternates between Policy Evaluation and Policy Improvement.

```text
Initialize a policy π randomly
Repeat:
    1. Policy Evaluation:
       Estimate V^π(s)

    2. Policy Improvement:
       Update π to be greedy with respect to V^π
Until the policy no longer changes
```

The two steps are:

1. **Evaluate the current policy**: compute how good it is.
2. **Improve the policy**: make it greedy with respect to the current value function.

> **Intuition:** Policy Iteration behaves like a loop of measurement and correction. First, the agent evaluates its current behavior. Then it changes its behavior toward better actions.

When the policy stops changing, the algorithm has reached an optimal policy for the MDP.

---

### 3.4 Value Iteration

Value Iteration simplifies Policy Iteration by combining policy evaluation and policy improvement into a single update:

```math
V(s) \leftarrow \max_a \sum_{s'} P(s'|s,a)
\left[ R(s,a,s') + \gamma V(s') \right]
```

Instead of fully evaluating one policy before improving it, Value Iteration directly applies the Bellman optimality equation.

A simplified procedure is:

```text
Initialize V(s) arbitrarily for all states
Repeat:
    For each state s:
        Update V(s) using the Bellman optimality equation
Until V converges

Extract the policy:
    π(s) = argmax_a expected return from action a
```

> **Intuition:** Value Iteration directly asks: what is the best value achievable from this state if the agent acts optimally?

After convergence, the optimal policy is extracted as:

```math
\pi^*(s) = \arg\max_a \sum_{s'} P(s'|s,a)
\left[ R(s,a,s') + \gamma V^*(s') \right]
```

---

### 3.5 Policy Iteration vs Value Iteration

| Method | Main idea | Evaluation depth | Output |
|---|---|---:|---|
| Policy Iteration | Evaluate a policy, then improve it | Full or near-full evaluation | Optimal policy |
| Value Iteration | Apply optimality updates directly | One-step backups | Optimal value function, then policy |

Policy Iteration separates evaluation and improvement clearly. Value Iteration is usually simpler and often faster in practice because it does not require exact policy evaluation at each iteration.

---

## 4. Summary

Value-based methods estimate the value of states or actions, then derive a policy by choosing the action with the highest estimated value.

The key objects are:

```math
V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t=s]
```

```math
Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t=s, a_t=a]
```

Dynamic Programming applies Bellman equations directly when the environment model is known. Its main algorithms are:

* **Policy Evaluation**: estimate the value of a fixed policy.
* **Policy Improvement**: make the policy greedy with respect to the value function.
* **Policy Iteration**: alternate evaluation and improvement.
* **Value Iteration**: directly apply Bellman optimality updates.

These ideas form the basis of later model-free value-based algorithms such as Monte Carlo control, SARSA, Q-learning, and Deep Q-Networks.
