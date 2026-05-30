# Summary Table — Value-Based Reinforcement Learning Approaches

This table summarizes the main value-based reinforcement learning approaches, from classical Dynamic Programming methods to Deep Q-Learning variants.

| Algorithm | Model-free? | On-policy? | Bootstraps? | Main idea |
|---|---:|---:|---:|---|
| Policy Evaluation | No | Depends | Yes | Estimate the value function of a fixed policy using known environment dynamics. |
| Policy Iteration | No | N/A | Yes | Alternate between policy evaluation and policy improvement until convergence. |
| Value Iteration | No | N/A | Yes | Directly apply the Bellman optimality update to compute the optimal value function. |
| Monte Carlo Control | Yes | Can be either | No | Learn action values from complete episode returns, then improve the policy. |
| TD(0) | Yes | Usually on-policy | Yes | Learn state values from one-step temporal-difference targets. |
| SARSA | Yes | Yes | Yes | Learn the value of the action actually taken by the current behavior policy. |
| Q-learning | Yes | No | Yes | Learn the optimal action-value function using the greedy next-action target. |
| Expected SARSA | Yes | Usually on-policy | Yes | Use the expected value over next actions instead of one sampled next action. |
| DQN | Yes | Off-policy | Yes | Approximate Q-learning with a neural network, replay buffer, and target network. |
| Double DQN | Yes | Off-policy | Yes | Reduce overestimation bias by separating action selection and action evaluation. |
| Dueling DQN | Yes | Off-policy | Yes | Decompose Q-values into state value and action advantage components. |
| Prioritized Experience Replay | Yes | Off-policy | Yes | Sample more informative transitions more often based on TD error. |

---

## Key Reading Notes

### Model-free vs model-based

A method is **model-free** if it does not require the transition probability function:

```math
P(s' \mid s,a)
```

Dynamic Programming methods are not model-free because they assume access to the environment model. Monte Carlo, TD, SARSA, Q-learning, and DQN-style methods learn directly from sampled experience.

### On-policy vs off-policy

An **on-policy** method learns the value of the same policy used to generate data.

An **off-policy** method can learn a target policy that differs from the behavior policy used for exploration.

For example, Q-learning is off-policy because it can explore using an $\epsilon$-greedy behavior policy while learning the greedy optimal policy.

### Bootstrapping

A method bootstraps if its update target depends on its own current value estimate.

For example, TD learning uses:

```math
R_{t+1} + \gamma V(S_{t+1})
```

Monte Carlo does not bootstrap because it uses the complete observed return:

```math
G_t
```

---

## Compact Mental Model

| Family | Mental model |
|---|---|
| Dynamic Programming | Use the known MDP model to compute values. |
| Monte Carlo | Estimate values by averaging complete episode returns. |
| Temporal Difference | Estimate values from one-step rewards plus current value estimates. |
| SARSA | Learn from the actions the agent actually takes. |
| Q-learning | Learn what the optimal greedy action would be. |
| DQN variants | Scale Q-learning to large state spaces using neural networks. |
