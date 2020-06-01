主要符号对照表
==============

.. list-table::
    :header-rows: 1

    * - 符号
      - 说明
    * - RL
      - 强化学习 (Reinforcement Learning)
    * - MFRL
      - 免模型强化学习 (Model-free Reinforcement Learning)
    * - MBRL
      - 基于模型的强化学习 (Model-based Reinforcement Learning)
    * - MARL
      - 多智能体强化学习 (Multi-agent Reinforcement Learning)
    * - MetaRL
      - 元强化学习 （Meta Reinforcement Learning）
    * - IL
      - 模仿学习 (Imitation Learning)
    * - On-policy
      - 同策略
    * - Off-policy
      - 异策略
    * - MDP
      - 马尔科夫决策过程 (Markov Decision Process)
    * - POMDP
      - 部分可观测马尔科夫决策过程 (Partially Observable Markov Decision Process)
    * - Agent
      - 智能体
    * - :math:`\pi`，Policy
      - 策略
    * - Actor
      - 动作（网络），又称作策略（网络）
    * - Critic
      - 评价（网络）
    * - :math:`s\in \mathcal{S}`，State
      - 状态
    * - :math:`o\in \mathcal{O}`，Observation
      - 观测值，为状态的一部分，:math:`o\subseteq s`
    * - :math:`a\in \mathcal{A}`，Action
      - 动作
    * - :math:`r\in \mathcal{R}`，Reward
      - 奖励
    * - :math:`d\in \{0, 1\}`，Done
      - 结束符，0表示未结束，1表示结束
    * - :math:`s_t, o_t, a_t, r_t, d_t`
      - 在一个轨迹中时刻 :math:`t` 的状态、观测值、动作、奖励和结束符
    * - :math:`P_{ss^\prime}^a\in \mathcal{P}`
      - 在当前状态 :math:`s` 采取动作 :math:`a` 之后，转移到状态 :math:`s'` 的概率；:math:`P_{ss^\prime}^a=\mathbb{P}\{s_{t+1}=s^\prime|s_t=s, a_t=a\}`
    * - :math:`R_s^a`
      - 在当前状态 :math:`s` 采取动作 :math:`a` 之后所能获得的期望奖励；:math:`R_s^a=\mathbb{E}[r_t|s_t=s, a_t=a]`
    * - :math:`\gamma`
      - 折扣因子，作为对未来回报不确定性的一个约束项，:math:`\gamma\in [0, 1]`
    * - :math:`G_t`，Return
      - 累计折扣回报，:math:`G_t=\sum_{i=t}^\infty \gamma^{i-t} r_{i}`
    * - :math:`\pi(a|s)`
      - 随机性策略，表示获取状态 :math:`s` 之后采取的动作 :math:`a` 的概率
    * - :math:`\pi(s)`
      - 确定性策略，表示获取状态 :math:`s` 之后采取的动作
    * - :math:`V(s)`
      - 状态值函数（State-Value Function），表示状态 :math:`s` 对应的期望累计折扣回报
    * - :math:`V^\pi(s)`
      - 使用策略 :math:`\pi` 所对应的状态值函数，:math:`V^\pi(s)=\mathbb{E}_{\pi} [G_t|s_t=s]`
    * - :math:`Q(s, a)`
      - 动作值函数（Action-Value Function），表示状态 :math:`s` 下采取动作 :math:`a` 所对应的期望累计折扣回报
    * - :math:`Q^\pi(s, a)`
      - 使用策略 :math:`\pi` 所对应的动作值函数，:math:`Q^\pi(s, a) = \mathbb{E}_{a\sim \pi} [G_t|s_t=s, a_t=a]`
    * - :math:`A(s, a)`
      - 优势函数，:math:`A(s, a) = Q(s, a) - V(s)`
    * - Batch
      - 数据组
    * - Buffer
      - 数据缓冲区
    * - Replay Buffer
      - 重放缓冲区
    * - RNN
      - 循环神经网络（Recurrent Neural Network）

