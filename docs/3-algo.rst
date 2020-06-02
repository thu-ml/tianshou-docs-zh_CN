.. _algo:

平台支持的深度强化学习算法
==========================

本章节将依次介绍天授平台所实现的强化学习算法的原理，以及这些算法在平台内部的具体实现细节。

强化学习的主要目标是让智能体学会一个能够最大化累计奖励的策略。以下仍然使用符号
:math:`\pi_\theta(\cdot)` 表示一个使用参数 :math:`\theta` 参数化的策略
:math:`\pi`\ ，优化目标为最大化 :math:`J(\theta)`\ ，定义为：

.. math:: J(\theta) = \mathbb{E}_{\pi_\theta}[G_t] = \sum_{s\in\mathcal{S}} d^\pi (s)V^\pi(s)=\sum_{s\in\mathcal{S}} d^\pi(s)\sum_{a\in\mathcal{A}}\pi_\theta(a|s)Q^\pi(s,a)
   :label: equ-main

其中 :math:`d^\pi(s)` 是使用策略 :math:`\pi` 在马尔科夫链中达到状态
:math:`s`
的概率。为力求简洁直观地表述各个算法，下文尽量不进行复杂的数学公式推导。此外，下文在描述算法时通常使用状态值
:math:`s_t`，在描述具体实现时通常使用观测值 :math:`o_t`。

基于策略梯度的深度强化学习算法
------------------------------

策略梯度（PG）
~~~~~~~~~~~~~~

策略梯度算法（Policy Gradient :cite:`pg` 
，又称REINFORCE算法、蒙特卡洛策略梯度算法，以下简称PG）是一个较为直观简洁的强化学习算法。它于上世纪九十年代被提出，依靠蒙特卡洛采样直接进行对累计折扣回报的估计，并直接以公式 :eq:`equ-main` 对 :math:`\theta` 进行求导，可推出梯度为：

.. math:: \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} [G_t\nabla_\theta\log\pi_\theta(a_t|s_t)]
   :label: equ-pg0

其中 :math:`a_t`、:math:`s_t` 均为具体采样值。将公式 :eq:`equ-pg0`
反推为目标函数

.. math:: J(\theta) = \mathbb{E}_{\pi_\theta} [G_t\log\pi_\theta(a_t|s_t)]
   :label: equ-pg

可以发现策略梯度算法本质上是最大化好动作的概率 [1]_ 。因此实现PG算法只需求得累计回报 :math:`G_t`、每次采样的数据点在策略函数中的对数概率
:math:`\log\pi_\theta(a_t|s_t)` 之后即可对参数 :math:`\theta`
进行求导，从而使用梯度上升方法更新模型参数。

一个被广泛使用的变种版本是在算法中将 :math:`G_t`
减去一个基准值，在保证不改变偏差的情况下尽可能减小梯度估计的方差。比如减去一个平均值，或者是如果使用状态值函数
:math:`V(s)` 作为一个基准，那么实际所使用的即为优势函数
:math:`A(s, a) = Q(s, a) - V(s)`，为动作值函数与状态值函数的差值。这将在后续描述的算法中进行使用。

策略梯度算法在天授中的实现十分简单：

-  ``process_fn``：计算 :math:`G_t`，具体实现位于 :ref:`sec_gae`；

-  ``forward``：给定 :math:`o_t`
   计算动作的概率分布，并从其中进行采样返回；

-  ``learn``：按照公式 :eq:`equ-pg` 计算 :math:`G_t` 与动作的对数概率
   :math:`\log\pi_\theta(a_t|o_t)`
   的乘积，求导之后进行反向传播与梯度上升，优化参数 :math:`\theta`；

-  采样策略：使用同策略的方法进行采样。

优势动作评价（A2C）
~~~~~~~~~~~~~~~~~~~

优势动作评价算法（Advantage
Actor-Critic :cite:`a2c` ，又被译作优势演员-评论家算法，以下简称A2C）是对策略梯度算法的一个改进。简单来说，策略梯度算法相当于A2C算法中评价网络输出恒为0的版本。该算法从公式 :eq:`equ-pg`
改进如下：

.. math:: \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\Big[\hat{A}(s_t,a_t)\nabla_\theta\log\pi_\theta(a_t|s_t)\Big]

其中 :math:`\hat{A}(s_t,a_t)`
为估计的优势函数，具体定义以及实现见 :ref:`sec_gae`。 :math:`\hat{V}(s_t)`
为评价网络输出的状态值函数。此外为了让评价网络的输出尽可能接近真实的状态值函数，在优化过程中还加上了对应的均方误差项；此外标准的实现中还有关于策略分布的熵正则化项。因此汇总的A2C目标函数为：

.. math:: J(\theta) = \mathbb{E}_{\pi_\theta}\Big[\hat{A}(s_t,a_t)\log\pi_\theta(a_t|s_t)-c_1(\hat{V}(s_t) - G_t)^2 + c_2H(\pi_\theta(\cdot)|s_t)\Big]
   :label: equ-a2c

其中 :math:`c_1, c_2` 是前述两项的对应超参数。

A2C最大的特点就是支持同步的并行采样训练，但由于天授平台支持所有算法的并行环境采样，此处不再赘述。此外A2C相比于异步策略执行版本A3C而言，避免了算法中策略执行不一致的问题，具有更快的收敛速度。

A2C算法在天授中的实现如下：

-  ``process_fn``：计算
   :math:`\hat{A}(s_t, a_t)`，具体实现位于 :ref:`sec_gae`；

-  ``forward``：和策略梯度算法一致，给定观测值
   :math:`o_t`，计算输出的输出策略的概率分布，并从中采样；

-  ``learn``：按照公式 :eq:`equ-a2c` 计算目标函数并求导更新参数；

-  采样策略：使用同策略的方法进行采样。

近端策略优化（PPO）
~~~~~~~~~~~~~~~~~~~

近端策略优化算法（Proximal Policy Optimization :cite:`ppo`，以下简称PPO）是信任区域策略优化算法（Trust Region Policy Optimization :cite:`trpo`，TRPO）的简化版本。由于策略梯度算法对超参数较为敏感，二者对策略的更新进行了一定程度上的限制，避免策略性能在参数更新前后产生剧烈变化，从而导致采样效率低下等问题。

PPO算法通过计算更新参数前后两次策略的比值来确保这个限制。具体目标函数为

.. math:: J^{\mathrm{CLIP}}(\theta) = \mathbb{E}_{\pi_\theta}\Big[ \min\Big( r(\theta)\hat{A}_{\theta_\mathrm{old}}(s_t, a_t), \mathrm{clip}(r(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_{\theta_\mathrm{old}}(s_t, a_t) \Big) \Big]

其中 :math:`\hat{A}(\cdot)`
表示估计的优势函数，因为真实的优势函数无法从训练过程所得数据中进行精确计算；:math:`r(\theta)`
是重要性采样权重，定义为新策略与旧策略的概率比值

.. math:: r(\theta) = \frac{\pi_\theta (a_t|s_t)}{\pi_{\theta_\mathrm{old}}(a_t|s_t)}

函数 :math:`\mathrm{clip}(r(\theta), 1-\epsilon, 1+\epsilon)`
将策略的比值 :math:`r(\theta)` 限制在 :math:`[1-\epsilon, 1+\epsilon]`
之间，从而避免了策略性能上的剧烈变化。在将PPO算法运用在动作评价（Actor-Critic）架构上时，与A2C算法类似，目标函数通常会加入状态值函数项与熵正则化项

.. math:: J(\theta) = \mathbb{E}_{\pi_\theta}[J^{\mathrm{CLIP}}(\theta) - c_1(\hat{V}(s_t) - G_t)^2 + c_2 H(\pi_\theta(\cdot)|s_t)]
   :label: equ-ppo

其中 :math:`c_1, c_2` 为两个超参数，分别对应状态值函数估计与熵正则化两项。

天授中的PPO算法实现大致逻辑与A2C十分类似：

-  ``process_fn``：计算 :math:`\hat{A}(s_t,a_t)` 与
   :math:`G_t`，具体实现位于 :ref:`sec_gae`；

-  ``forward``：按照给定的观测值 :math:`o_t`
   计算概率分布，并从中采样出动作 :math:`a_t`；

-  ``learn``：重新计算每个数据组所对应的对数概率，并按照公式 :eq:`equ-ppo`
   进行目标函数的计算；

-  采样策略：使用同策略的方法进行采样。

.. _sec_gae:

广义优势函数估计器（GAE）
~~~~~~~~~~~~~~~~~~~~~~~~~

广义优势函数估计器（Generalized Advantage Estimator :cite:`gae`，以下简称GAE）是将以上若干种策略梯度算法的优势函数的估计
:math:`\hat{A}(s_t, a_t)`
进行形式上的统一。一般而言，策略梯度算法的梯度估计都遵循如下形式：

.. math:: \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \Big[\Psi_t\nabla_\theta\log\pi_\theta(a_t|s_t)\Big]

其中 :math:`\Psi_t` 具有多种形式，比如PG中为
:math:`\Psi_t=\sum_{i=t}^\infty r_i`，即累计回报函数；A2C中为
:math:`\Psi_t=\hat{A}_t=-V(s_t)+r_t+\gamma r_{t+1}+\cdots+\gamma^{T-t+1}r_{T-1}+\gamma^{T-t}V(s_T)`；PPO中是
:math:`\Psi_t=\hat{A}_t=\delta_t+(\gamma\lambda)\delta_{t+1}+\cdots+(\gamma\lambda)^{T-t+1}\delta_{T-1}`，其中
:math:`\delta_t` 是时序差分误差项（Temporal Difference error，TD
error），:math:`\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)`。GAE将上述若干种估计形式进行统一如下：

.. math:: \hat{A}_t^{\mathrm{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma\lambda)^l\delta_{t+l}=\sum_{l=0}^\infty (\gamma\lambda)^l(r_t+\gamma V(s_{t+l+1}) - V(s_{t+l}))
   :label: equ-gae

其中 :math:`\mathrm{GAE}(\gamma, 0)` 的情况为
:math:`\hat{A}_t=\delta_t=r_t+\gamma V(s_{t+1}) - V(s_t)`，为1步时序差分误差，:math:`\mathrm{GAE}(\gamma, 1)`
的情况为
:math:`\hat{A}_t = \sum_{l=0}^\infty \gamma^l\delta_{t+l} = \sum_{l=0}^\infty \gamma^lr_{t+l}-V(s_t)`，即为A2C中的估计项。
PG中的估计项即为A2C中 :math:`V(s_t)` 恒为0的特殊情况。

天授中GAE实现与其他平台有一些不同之处。比如在OpenAI
Baselines :cite:`baselines` 的实现中，对每个完整轨迹的最后一帧进行特殊判断处理。与此不同，天授使用轨迹中每项的下一时刻观测值
:math:`o_{t+1}`
批量计算状态值函数，避免了特殊判断。天授的GAE实现将大部分操作进行向量化，并且支持同时计算多个完整轨迹的GAE函数，还比Baselines正常使用Python写的循环语句要快不少。

基于Q价值函数的深度强化学习算法
-------------------------------

深度Q网络（DQN）
~~~~~~~~~~~~~~~~

深度Q网络算法（Deep Q Network :cite:`dqn` ，以下简称DQN）是强化学习算法中最经典的算法之一，它在Atari游戏中表现一鸣惊人，由此开启了深度强化学习的新一轮浪潮。DQN算法核心是维护Q函数并使用它进行决策。具体而言，:math:`Q^\pi(s,a)` 为在该策略 :math:`\pi` 下的动作值函数；每次到达一个状态 :math:`s_t` 之后，遍历整个动作空间，将动作值函数最大的动作作为策略：

.. math:: a_t = \arg\max_{a} Q^\pi(s_t, a)

其动作值函数的更新采用贝尔曼方程进行迭代

.. math:: Q^\pi(s_t,a_t) \leftarrow Q^\pi(s_t,a_t)+\alpha_t (r_t+\gamma \max_a Q^\pi(s_{t+1}, a) - Q^\pi(s_t,a_t))
   :label: equ-dqn

其中 :math:`\alpha` 为学习率。通常在简单任务上，使用全连接神经网络来拟合
:math:`Q^\pi`，但是在稍微复杂一点的任务上如Atari游戏，会使用卷积神经网络进行由图像到值函数的映射拟合，这也是深度Q网络中“深度”一词的由来。由于这种表达形式只能处理有限个动作值，因此DQN通常被用在离散动作空间任务中。

为了避免陷入局部最优解，DQN算法通常采用
:math:`\epsilon`-贪心方法进行策略探索，即每次有
:math:`\epsilon\in [0, 1]` 的概率输出随机策略，:math:`1-\epsilon`
的概率输出使用动作值函数估计的最优策略；此外通常把公式 :eq:`equ-dqn` 中
:math:`r_t+\gamma\max_a Q^\pi(s_{t+1},a)` 一项称作目标动作值函数
:math:`Q_\mathrm{target}`，它还可以拓展成不同的形式，比如 :math:`n`
步估计：

.. math:: Q_\mathrm{target}^n(s_t, a_t) = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \max_a\gamma^{n} Q^\pi(s_{t+n}, a)
   :label: equ-target_q

天授中的DQN算法实现如下：

-  ``process_fn``：使用公式 :eq:`equ-target_q`
   计算目标动作函数，与重放缓冲区交互进行计算；

-  ``forward``：给定观测值 :math:`o_t`，输出每个动作对应的动作值函数
   :math:`Q(o_t, \cdot)`，并使用
   :math:`\epsilon`-贪心算法添加噪声，输出动作 :math:`a_t`；

-  ``learn``：使用公式 :eq:`equ-dqn` 进行迭代，在特定时刻可调整
   :math:`\epsilon`-贪心算法中的 :math:`\epsilon` 值；

-  采样策略：使用异策略的方法进行采样。

双网络深度Q学习（DDQN）
~~~~~~~~~~~~~~~~~~~~~~~

双网络深度Q学习算法（Double DQN :cite:`double-dqn`，以下简称DDQN）是DQN算法的重要改进之一。由于在公式 :eq:`equ-dqn`
中使用同一个动作值函数进行对目标动作值函数的估计，会导致策略网络产生过于乐观的估计，从而降低了算法的采样效率。DDQN算法将动作评估与动作选择进行解耦，从而减少高估所带来的负面影响。它将公式 :eq:`equ-dqn`
中的目标动作值函数加以改造如下

.. math:: Q_\mathrm{target}(s_t, a_t) = r_t + \gamma Q^{\pi_\mathrm{old}}\Big(s_{t+1}, \arg\max_a Q^\pi(s_{t+1}, a)\Big)
   :label: equ-ddqn

其中 :math:`Q^{\pi_\mathrm{old}}` 是目标网络（Target Network），为策略网络 :math:`Q^\pi` 的历史版本，专门用来进行动作评估。公式 :eq:`equ-ddqn`
同样可以和公式 :eq:`equ-target_q` 进行结合，推广到 :math:`n`
步估计的情况，此处不再赘述。

由于DDQN与DQN仅有细微区别，因此在天授的实现中将二者封装在同一个类中，改动如下：

-  ``process_fn``：按照公式 :eq:`equ-ddqn` 计算目标动作函数；

-  ``learn``：在需要的时候更新目标网络的参数。

优先级经验重放（PER）
~~~~~~~~~~~~~~~~~~~~~

优先级经验重放（Prioritized Experience Replay :cite:`per`，以下简称PER）是DQN算法的另一个重要改进。该算法也可应用在之后的DDPG算法族中。其核心思想是，根据策略网络输出的动作值函数
:math:`Q^\pi(s_t, a_t)` 与实际采样估计的动作值函数
:math:`Q_\mathrm{target}(s_t,a_t)`
的时序差分误差来给每个样本不同的采样权重，将误差更大的数据能够以更大的概率被采样到，从而提高算法的采样与学习效率。

PER的实现不太依赖于算法层的改动，比较和底层的重放缓冲区相关。相关改动如下：

-  算法层：加入一个接口，传出时序差分误差，作为优先经验重放缓冲区的更新权重；

-  数据层：新建优先经验重放缓冲区类，继承自重放缓冲区类，修改采样函数，并添加更新优先值权重的函数。

综合Q价值函数与策略梯度的深度强化学习算法
-----------------------------------------

深度确定性策略梯度（DDPG）
~~~~~~~~~~~~~~~~~~~~~~~~~~

深度确定性策略梯度算法（Deep Deterministic Policy
Gradient :cite:`ddpg`，以下简称DDPG）是一种同时学习确定性策略函数
:math:`\pi_\theta(s)` 和动作值函数 :math:`Q^\pi(s, a)`
的算法。它主要解决的是连续动作空间内的策略训练问题。在DQN中，由于常规的Q函数只能接受可数个动作，因此无法拓展到连续动作空间中。

DDPG算法假设动作值函数 :math:`Q(s, a)`
在连续动作空间中是可微的，将动作值 :math:`a` 用一个函数
:math:`\pi_\theta(s)` 拟合表示，并将 :math:`\pi_\theta(s)`
称作动作网络，:math:`Q^\pi(s, a)`
称作评价网络。DDPG算法评价网络的更新部分与DQN算法类似，动作网络的更新根据确定性策略梯度定理 :cite:`dpg`，直接对目标函数
:math:`Q^\pi(s, \pi_\theta(s))` 进行梯度上升优化即可。

为了更好地进行探索，原始DDPG算法添加了由Ornstein-Uhlenbeck随机过程 [2]_ 产生的时间相关的噪声项，但在实际测试中，高斯噪声可以达到与其同样的效果 :cite:`td3`；DDPG还采用了目标网络以稳定训练过程，对目标动作网络和目标评价网络进行参数软更新，即
:math:`\theta^\prime \leftarrow \tau \theta + (1 - \tau) \theta^\prime`，以
:math:`\tau` 的比例将新网络的权重 :math:`\theta` 更新至目标网络
:math:`\theta^\prime` 中。

天授对DDPG算法的实现如下：

-  ``process_fn``：和DQN算法类似，其中动作 :math:`a`
   不进行全空间遍历，而是以动作网络的输出作为参考标准；

-  ``forward``：给定观测值 :math:`o_t`，输出动作
   :math:`a_t=\pi_\theta(o_t)`，并添加噪声项；

-  ``learn``：分别计算贝尔曼误差项和 :math:`Q^\pi(s, \pi_\theta(s))`
   并分别优化，之后软更新目标网络的参数；

-  采样策略：使用异策略的方法进行采样。

双延迟深度确定性策略梯度（TD3）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

双延迟深度确定性策略梯度算法（Twin Delayed DDPG :cite:`td3`，以下简称TD3）是DDPG算法的改进版本。学习动作值函数Q的一系列方法一直以来都有过度估计的问题，DDPG也不例外。TD3算法做了如下几点对DDPG进行改进：

-  截断双网络Q学习：截断双网络Q学习使用两个动作值网络，取二者中的最小值作为动作值函数
   Q 的估计，从而有利于减少过度估计：

   .. math:: Q_{\mathrm{target}_i} = r + \min_{j=1, 2} Q^\pi_{\phi_j}(s^\prime, \pi_{\theta}(s^\prime))

-  动作网络延迟更新：相关实验结果表明，同步训练动作网络和评价网络，却不使用目标网络，会导致训练过程不稳定；但是仅固定动作网络时，评价网络往往能够收敛到正确的结果。因此TD3算法以较低的频率更新动作网络，较高频率更新评价网络，通常每两次更新评价网络时，进行一次策略更新。

-  平滑目标策略：TD3算法在动作中加入截断高斯分布产生的随机噪声，避免策略函数
   :math:`\pi_\theta(s)` 陷入Q函数的极值点，从而更有利于收敛：

   .. math::

      \begin{aligned}
          Q_\mathrm{target}&=r+\gamma Q^\pi(s^\prime, \pi_\theta(s^\prime)+\epsilon)\\
          \epsilon&\sim\mathrm{clip}(\mathcal{N}(0, \sigma), -c, c)
      \end{aligned}

与DDPG算法类似，天授在TD3的实现中继承了DDPG算法，只修改了 ``learn``
部分，按照上述三点一一实现代码。

软动作评价（SAC）
~~~~~~~~~~~~~~~~~

软动作评价算法（Soft Actor-Critic :cite:`sac`，以下简称SAC）是基于最大熵强化学习理论提出的一个算法。SAC算法同时具备稳定性好和采样效率高的优点，容易实现，同时融合了动作评价框架、异策略学习框架和最大熵强化学习框架，因此成为强化学习算法中继PPO之后的标杆算法。

SAC的算法结构和TD3也十分类似，同样拥有一个动作网络和两个评价网络。单从最终推导得到的式子来看，和TD3的最大差别是在求目标动作值函数的时候，最后一项加上了较为复杂的熵正则化项，其余的实现十分类似。具体的推导可以在原论文中找到。

由于SAC的实现和TD3十分类似，故此处不再对其进行详细阐述。

部分可观测马尔科夫决策过程的训练
--------------------------------

在实际场景中，智能体往往难以观测到环境中所有的信息，只能观测到状态
:math:`s` 的一个子集 :math:`o`
进行决策，这种场景被称作部分可观测马尔科夫决策过程（Partially Observable Markov Decision Process，简称POMDP）。

POMDP在深度强化学习领域通常有两种解决方案：（1）将过去一段时间内的信息（如过去的观测值、过去的动作和奖励）添加到当前状态中，按照常规方式进行处理；（2）将过去的信息利用循环神经网络（RNN）存储到中间状态中，可以传给后续状态进行使用。

第一种方法只需在重放缓冲区中添加时序采样功能，比如待采样下标是
:math:`t`，需要采样连续 :math:`n`
帧，那么在重放缓冲区中进行一定设置，返回观测值
:math:`\{o_{t-n+1}, \dots, o_{t-1}, o_t\}`，剩下的过程和正常的强化学习训练过程无异。
第二种方法需要在第一种方法的基础上，在所有和神经网络相关的接口中添加对中间状态的支持。天授已经支持上述两种方法的实现。

模仿学习
--------

模仿学习（Imitation
Learning）更偏向于监督学习与半监督学习的范畴。它的核心思想是学习已有的数据，尽可能地还原产生这些数据的原始策略。比如给定一些
:math:`t` 时刻的状态与动作数据对
:math:`(s_t, a_t)`，那么可以使用神经网络来回归映射
:math:`\mathcal{F}: \mathcal{S} \rightarrow \mathcal{A}`，
从而进行模仿学习。更进一步地，还有逆强化学习（Inverse Reinforcement
Learning :cite:`irl`，IRL）和生成式对抗模仿学习（Generative
Adversarial Imitation Learning :cite:`gail`，GAIL）等算法。

目前天授平台实现了最基本的模仿学习算法，具体实现如下：

-  连续动作空间：将其看作回归任务，直接对给定的动作进行回归；

-  离散动作空间：将其看作分类任务，最大化采取给定动作的概率；

-  采样策略：使用参考策略和异策略方法进行不断地采样补充数据。

小结
----

本章节介绍了深度强化学习算法的原理以及在天授平台上的具体实现，包括了9种免模型强化学习算法、循环神经网络模型训练和模仿学习。

.. [1]
   这个视频详细地讲解了策略梯度算法的推导过程： https://youtu.be/XGmd3wcyDg8

.. [2]
   https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
