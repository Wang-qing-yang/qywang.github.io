---
title: Writing technical content in Markdown-test
date: 2019-07-11
math: true
image:
  placement: 2
  caption: 'Image credit: [**John Moeses Bauan**](https://unsplash.com/photos/OGZtQF8iC0g)'
---

## gymnasium

### Termination 与 Truncation

在gym包中只有done用于episode是否结束，但是实际上区分termination和truncation是格外重要的。termination是指到达终止状态，例如完成任务/任务失败，同样也可以来自有限时间的环境，特别的是为了保持MDP的特性，在有限时间的环境，状态必须包括剩余的时间，即将时间作为状态之一。truncation是指MDP外部定义的终止条件，例如针对无限时间的问题，需要对episode进行强制中断，从而进行策略评估，但是此时最后一个状态并不是终止状态。所以对于truncation需要采取bootstrapping进行策略评估，例如$Q_{target} = r_t + \lambda*max_a(Q(o_{t+1},a_{t+1}))$ ，对于termination，则不需要bootstrapping，即$Q_{target} = r_t$ ，所以总的来说应该采取：$Q_{targte} = r_t + \lambda*(1 - terminated)max_a(Q(o_{t+1},a_{t+1}))$。

### Custom Wrapper

Wrapper用于模块化地修改环境，从而避免重做轮子，可以通过继承以下四类：

- gymnasium.ObservationWrapper : 用于自定义环境反馈的状态量
- gymnasium.ActionWrapper : 用于自定义环境中智能体可以采取的动作
- gymnasium.RewardWrapper : 用于自定义环境反馈的奖励
- gymnasium.Wrapper : 以上三类的父类，高度定制的环境

#### gymnasium.ObservationWrapper



## 强化学习的数学基础

### 第三章



## introductuion

### 第五章 Monte-Carlo 方法

在环境未知的情况下，通过采样（与实际/仿真环境交互）得到的经验数据（状态，动作，奖励）来估计价值函数与最优策略。本章的Monte-Carlo方法特指采用平均采样方式，并且只有在每个episode结束后更新价值函数估计与最优策略，因此Monte-Carlo方法是一种episode-by-episode的方法，而不是step-by-step（online）的方法.

#### 5.1 Monte-Carlo 预测

针对已知的策略，通过与环境交互得到n个episodes，针对每个episodes中的每个状态得到V(s)，只要有足够多的episodes即可得到价值函数的无偏估计。first-visit和every-visit的区别在于每个episode中前者只对第一次遇到的s记录V(s)，而不考虑后面遇到的相同的s，后者对每个s记录V(s)，即first-visit中的每个episode对于相同的s只得到一个V(s)，而every-visit有不止一个。

Monte-Carlo不同于DP，DP使用bootstrap即利用估计来更新估计，而Monte-Carlo对于每个状态的价值估计都是相互独立的，甚至可以只考虑某一个状态，从这个状态作为初始状态创建多个episodes从而更新特定的V(s)。

#### 5.2 Monte-Carlo 估计 Action Value

可以通过V(s)与model得到策略，即比较不同action下的$r + V(s_1)$ ，其中需要通过model确定状态的转移。而通过直接估计state-action，即$q(s,a)$ ，即可得到策略为$a = argmax_a(q(s,a))$ ，但是需要考虑探索性，如果采用greedy的策略，那么有些state-action对将永远不会访问到，从而永远不被更新，所以需要通过策略保障每个state-action对都被探索到。例如固定episode的起始包括所有的state-action对，但是这在实践中往往需要实际的与环境交互，而不是指定a；可以通过$\varepsilon-greedy$ 的策略实现。

#### 5.3 Monte-Carlo 控制

用于估计最优策略。包括policy evaluation与policy improvement，其中evaluation即使用多个episodes估计价值函数（假设探索性与无限个数的episodes），improvement即基于值函数进行greedy policy。考虑实践中如何在有限的episodes进行evaluation，第一个是在每一步估计$q_\pi$ ？第二个是不需要完整地估计policy就进行Improvement。

#### 5.4 Monte-Carlo control without Exploring Starts



