---
title: Terms and formulas quick reference
sidebar_position: 1.5
displayed_sidebar: foundationsRlForRoboticsSidebar
description: Understand common reinforcement learning symbols, terms, and formulas through robot examples.
---

# Terms and formulas quick reference

When reading a formula, first ask what it calculates, then identify each symbol, and finally try small numbers. This page explains common concepts in RL foundations. The [MDP](/docs/foundations/rl-for-robotics/mdp) and [temporal-difference](/docs/foundations/rl-for-robotics/temporal-difference) chapters also have expandable explanations beside key formulas. Follow a term link to its entry, then use your browser's Back button to resume reading.

## Reading the symbols \{#notation}

| Symbol | Read it as | Small example |
| --- | --- | --- |
| $s_t,a_t,R_{t+1}$ | Current state, current action, reward received afterward | A robot reaches a junction, turns right, then receives reward $1$ |
| $\sum$ | Add the terms in the specified range | $\sum_{k=0}^{2}x_k=x_0+x_1+x_2$ |
| $\gamma^k$ | The discount weight after $k$ steps | $0.9^2=0.81$ |
| $\mathbb{E}[X]$ | Average the outcomes using their probabilities | Outcomes $2,8$ with equal probability have expectation $5$ |
| $P(a\mid s)$ | Probability of action $a$, given state $s$ | The bar means “given,” not division |
| $\leftarrow$ | Replace the stored value on the left with the result on the right | $V\leftarrow V+0.1$ increases $V$ by $0.1$ |
| $\max$ / $\arg\max$ | Highest score / option with the highest score | Left scores $2$, right scores $5$: maximum $5$, maximizing option right |
| $\hat{A},\theta$ | An estimate, model parameters | The hat marks an estimate; $\theta$ can be policy-network weights |

The same letter can mean different things in different chapters. In an MDP, $R_{t+1}$ is a reward; in PPO, $r_t(\theta)$ is a new-to-old policy probability ratio. Read the definition next to the current formula first.

## States, observations, and MDPs \{#mdp}

A **state** contains the information needed for prediction; an **observation** contains what sensors actually measure. A photo can show a ball's position without revealing where it is rolling. Velocity may also belong in the state.

An **MDP (Markov decision process)** organizes a task into states $S$, actions $A$, transition rules $P$, reward rules $R$, and discount factor $\gamma$. It is a problem model, not a particular neural network: what information is available, what can be done, what happens next, and what counts as good?

The **Markov property** requires that, given the current state and action, past history provides no additional information needed to predict the next state. A **POMDP** describes partial observability, such as occluded information. Stacking past frames may improve observations but does not guarantee recovery of the full state.

Return to the chapter: [MDP definition](/docs/foundations/rl-for-robotics/mdp#21-智能体与环境交互), [Markov property](/docs/foundations/rl-for-robotics/mdp#22-马尔可夫性质).

## Rewards, returns, and discounting \{#return}

A **reward** is one step's feedback; a **return** is the total of future rewards from the current time, possibly discounted. A robot might earn $1$ per forward step and $10$ at its destination. One step's reward and the whole journey's return are different quantities.

The **discount factor** $\gamma$ determines how much future rewards count now. Suppose the next two rewards are $2,3$, followed by termination, and $\gamma=0.9$:

$$
G_t=2+0.9\times3=4.7.
$$

The next-time return is $G_{t+1}=3$, so we can also write $G_t=2+0.9G_{t+1}$. No rewards follow termination, and we set $G_T=0$. Infinite-horizon tasks commonly use $0\leq\gamma<1$ with bounded rewards to ensure a finite return; finite episodes can also use $\gamma=1$.

Return to the chapter: [returns and their recursive form](/docs/foundations/rl-for-robotics/mdp#24-目标与回报).

## Policies, expectations, and values \{#value}

A **policy** $\pi$ is an action-selection rule. For discrete actions, $\pi(a\mid s)=0.7$ means a $70\%$ chance of selecting $a$ in state $s$. Continuous actions are usually represented by probability densities: the density at a point is not the probability of selecting that exact action.

An **expectation** is a probability-weighted average, not a hoped-for outcome or a result guaranteed on one run.

- **State value** $V_\pi(s)$: starting here and following $\pi$, what return do we get on average?
- **Action value** $Q_\pi(s,a)$: starting here, taking $a$ first, and then following $\pi$, what return do we get on average?

If left and right have $Q$ values $2,6$ and the policy selects each with probability one half, then $V_\pi(s)=0.5\times2+0.5\times6=4$. This averages according to the policy; it does not take the maximum $6$.

Return to the chapter: [policies and values](/docs/foundations/rl-for-robotics/mdp#25-策略与价值).

## Bellman equation \{#bellman}

The **Bellman equation** splits the average future total into “one immediate step + the average remaining future”:

$$
V_\pi(s)=\mathbb{E}_\pi[R_{t+1}+\gamma V_\pi(S_{t+1})\mid S_t=s].
$$

For a fixed policy, average over both possible actions and possible next states using their probabilities. The equation relates expectations; a particular trajectory that falls into a pit may still return much less than average.

Return to the chapter: [Bellman equations](/docs/foundations/rl-for-robotics/dynamic-programming#32-贝尔曼方程), [from Bellman to the TD target](/docs/foundations/rl-for-robotics/temporal-difference#53-时序差分目标的推导).

## Bootstrapping, TD targets, and learning rate \{#td}

In this context, **bootstrapping** means updating one estimate using another. Before finishing a journey, use your estimate for the next stop to improve your estimate for the current stop. This is different from bootstrap resampling in statistics.

The **TD target** is the observed reward plus the discounted next-state estimate. The **TD error** is that target minus the current estimate. The **learning rate** $\alpha$ determines how far to move toward it:

$$
\text{new estimate}=\text{old estimate}+\alpha(\text{target}-\text{old estimate}).
$$

With an old estimate of $4$, a target of $6.4$, and $\alpha=0.1$, the new estimate is $4.24$. Learning rate and discount factor differ: $\alpha$ controls updating, while $\gamma$ weights future rewards. Do not bootstrap after true termination; whether a time-limit truncation permits bootstrapping depends on the task's termination definition.

Return to the chapter: [calculating a TD update](/docs/foundations/rl-for-robotics/temporal-difference#51-时序差分估计).

## On-policy and off-policy \{#on-off-policy}

The **behavior policy** collects data. The **target policy** is evaluated or improved. When they agree, learning is **on-policy**; when they differ, it is **off-policy**.

Sarsa uses the value of the action actually selected next. Q-learning uses the highest estimated next-action value, even if exploration selected a different action. The distinction concerns whose behavior is learned, not whether a neural network is used.

Return to the chapter: [Q-learning and Sarsa](/docs/foundations/rl-for-robotics/temporal-difference#57-q-learning-算法), [on-policy and off-policy](/docs/foundations/rl-for-robotics/temporal-difference#58-同策略与异策略).

## Actor, Critic, and advantage \{#advantage}

The **Actor** selects actions; the **Critic** estimates value. Think of a decision-making part and a scoring part, but remember that the Critic's scores are learned estimates, not ground truth.

The **advantage function** asks how much better an action is than the current policy's average at that state: $A_\pi(s,a)=Q_\pi(s,a)-V_\pi(s)$. An action value of $6$ and state value of $4$ give advantage $2$; a negative value means below the baseline. PPO's $\hat{A}_t$ is an advantage estimate, not an immediate reward.

Continue reading: [Actor-Critic](/docs/foundations/rl-for-robotics/actor-critic), [PPO](/docs/foundations/rl-for-robotics/ppo).

## PPO probability ratios and clipping \{#ppo-clip}

The **probability ratio** $r_t(\theta)=\pi_\theta(a_t\mid s_t)/\pi_{\text{old}}(a_t\mid s_t)$ compares the new and old policies for the same action. If a discrete action's probability rises from $0.2$ to $0.3$, the ratio is $1.5$; for continuous actions, it compares densities.

**Clipping** limits incentives in the surrogate objective. With advantage $2$, $\epsilon=0.2$, and ratio $1.5$, this sample's PPO-Clip objective is $\min(1.5\times2,1.2\times2)=2.4$. Increasing this ratio further does not increase that sample's contribution to the objective.

For negative advantage, the incentive being limited is the gain from excessively reducing the action's probability. Do not omit the $\min$ or interpret clipping as locking all probabilities into a fixed interval. Clipping does not guarantee that the new and old policies remain close.

Continue reading: [PPO's Clip objective](/docs/foundations/rl-for-robotics/ppo#1021-clip-约束).

## Further references \{#references}

- [Sutton & Barto: Reinforcement Learning, An Introduction](http://incompleteideas.net/book/the-book-2nd.html): Chapters 3, 4, and 6 cover MDPs, Bellman equations, and TD learning.
- [OpenAI Spinning Up: Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html): formal definitions of states, policies, returns, values, and advantages.
- [OpenAI Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html): probability ratios, the clipped objective, and its limitations.
