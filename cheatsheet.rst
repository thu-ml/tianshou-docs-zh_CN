速查手册
========

本页面列举出了一些天授平台的常用使用方法。

.. _network_api:

搭建神经网络
------------

参见 :ref:`build_the_network`。

.. _new_policy:

构建策略
--------

参见 `BasePolicy </en/master/api/tianshou.policy.html#tianshou.policy.BasePolicy>`_。

.. _customize_training:

定制化训练策略
--------------

参见 :ref:`customized_trainer`。

.. _parallel_sampling:

环境并行采样
------------

天授提供了四种类：

- :class:`~tianshou.env.DummyVectorEnv` 使用原始的for循环实现，可用于debug，小规模的环境用这个的开销会比其他三种小

- :class:`~tianshou.env.SubprocVectorEnv` 用多进程来实现的，最常用

- :class:`~tianshou.env.ShmemVectorEnv` 是上面这个多进程实现的一个改进：把环境的obs用一个shared buffer来存储，降低比较大的obs的开销（比如图片等）

- :class:`~tianshou.env.RayVectorEnv` 基于Ray的实现，可以用于多机

这些类虽说是用于不同的场景中，但是他们的API都是一致的，只需要提供一个可调用的env列表即可，像这样：

::

    env_fns = [lambda x=i: MyTestEnv(size=x) for i in [2, 3, 4, 5]]
    venv = SubprocVectorEnv(env_fns)  # DummyVectorEnv/ShmemVectorEnv/RayVectorEnv 都行
    venv.reset()  # 这个会返回每个环境最初的obs
    venv.step(actions)  # actions长度是env的数量，返回也是这些env原始返回值concat起来之后的结果

.. sidebar:: 一个venv同步或异步执行的例子，相同颜色代表这些episode会组合起来由venv返回出去

     .. Figure:: /_static/images/async.png

默认情况下是使用同步模式（sync mode），就像图中最上面那样。如果每个env step耗时差不多的话，这种模式开销最小。但如果每个env step耗时差别很大的话（比如通常1s，偶尔会10s），这个时候async就派上用场了（`Issue 103 <https://github.com/thu-ml/tianshou/issues/103>`_）：只需要多提供两个参数（或者其中之一也行），一个是 ``wait_num``，表示一旦达到这么多env结束就返回（比如4个env，设置 ``wait_num = 3`` 的话，每一步venv.step只会返回4个env中的3个结果）；另一个是 ``timeout``，表示一旦超过这个时间并且有env已经结束了的话就返回结果。

::

    env_fns = [lambda x=i: MyTestEnv(size=x, sleep=x) for i in [2, 3, 4, 5]]
    venv = SubprocVectorEnv(env_fns, wait_num=3, timeout=0.2)
    venv.reset()
    # returns "wait_num" steps or finished steps after "timeout" seconds,
    # whichever occurs first.
    venv.step(actions, ready_id)


.. warning::

    如果自定义环境的话，记得设置 ``seed`` 比如这样：

    ::

        def seed(self, seed):
            np.random.seed(seed)

    如果seed没有被重写，每个环境的seed都是全局的seed，都会产生一样的结果，相当于一个env的数据复制了好多次，没啥用。


.. _preprocess_fn:

批处理数据组
------------

本条目与 `Issue 42 <https://github.com/thu-ml/tianshou/issues/42>`_ 相关。

如果想收集训练log、预处理图像数据（比如Atari要resize到84x84x3 -- 不过这个推荐直接wrapper做）、根据环境信息修改奖励函数的值，可以在Collector中使用 ``preprocess_fn`` 接口，它会在数据存入Buffer之前被调用。

``preprocess_fn`` 有两种输入接口：如果是env.reset()的话，它只会接收obs；如果是正常的env.step()，那么他会接收5个关键字 "obs_next"/"rew"/"done"/"info"/"policy"。返回一个字典或者Batch，里面包含着你想修改的东西。

::

    import numpy as np
    from collections import deque


    class MyProcessor:
        def __init__(self, size=100):
            self.episode_log = None
            self.main_log = deque(maxlen=size)
            self.main_log.append(0)
            self.baseline = 0

        def preprocess_fn(**kwargs):
            """把reward给归一化"""
            if 'rew' not in kwargs:
                # 意味着 preprocess_fn 是在 env.reset() 之后被调用的，此时kwargs里面只有obs
                return Batch()  # 没有变量需要更新，返回空
            else:
                n = len(kwargs['rew'])  # Collector 中的环境数量
                if self.episode_log is None:
                    self.episode_log = [[] for i in range(n)]
                for i in range(n):
                    self.episode_log[i].append(kwargs['rew'][i])
                    kwargs['rew'][i] -= self.baseline
                for i in range(n):
                    if kwargs['done']:
                        self.main_log.append(np.mean(self.episode_log[i]))
                        self.episode_log[i] = []
                        self.baseline = np.mean(self.main_log)
                return Batch(rew=kwargs['rew'])

最终只需要在Collector声明的时候加入一下这个hooker：
::

    test_processor = MyProcessor(size=100)
    collector = Collector(policy, env, buffer, preprocess_fn=test_processor.preprocess_fn)

还有一些示例在 `test/base/test_collector.py <https://github.com/thu-ml/tianshou/blob/master/test/base/test_collector.py>`_ 中可以查看。

.. _rnn_training:

RNN训练
-------

本条目与 `Issue 19 <https://github.com/thu-ml/tianshou/issues/19>`_ 相关

首先在 ReplayBuffer 的声明中加入 ``stack_num`` （堆叠采样）参数，表示在训练RNN的时候要扔给网络多少个timestep进行训练：
::

    buf = ReplayBuffer(size=size, stack_num=stack_num)

然后把神经网络模型中 ``state`` 参数用起来，可以参考 :class:`~tianshou.utils.net.common.Recurrent`、:class:`~tianshou.utils.net.continuous.RecurrentActorProb` 和 :class:`~tianshou.utils.net.continuous.RecurrentCritic`。

以上代码片段展示了如何修改ReplayBuffer和神经网络模型，从而使用堆叠采样的观测值（stacked-obs）来训练RNN。如果想要堆叠别的值（比如stacked-action来训练Q(stacked-obs, stacked-action)），可以使用一个 ``gym.Wrapper`` 来修改状态表示，比如wrapper把状态改为 [s, a] 的元组：

- 之前的数据存储：(s, a, s', r, d)，可以获得堆叠的s
- 采用wrapper之后的存储：([s, a], a, [s', a'], r, d)，可以获得堆叠的[s, a]，拆开来就是堆叠的s和a

.. _self_defined_env:

自定义环境与状态表示
--------------------

本条目与 `Issue 38 <https://github.com/thu-ml/tianshou/issues/38>`_ 和 `Issue 69 <https://github.com/thu-ml/tianshou/issues/69>`_ 相关。

首先，自定义的环境必须遵守OpenAI Gym定义的API规范，下面列出了一些：

- reset() -> state

- step(action) -> state, reward, done, info

- seed(s) -> List[int]

- render(mode) -> Any

- close() -> None

- observation_space: gym.Space

- action_space: gym.Space

环境状态（state）可以是一个 ``numpy.ndarray`` 或者一个Python字典。比如以 ``FetchReach-v1`` 环境为例：
::

    >>> e = gym.make('FetchReach-v1')
    >>> e.reset()
    {'observation': array([ 1.34183265e+00,  7.49100387e-01,  5.34722720e-01,  1.97805133e-04,
             7.15193042e-05,  7.73933014e-06,  5.51992816e-08, -2.42927453e-06,
             4.73325650e-06, -2.28455228e-06]),
     'achieved_goal': array([1.34183265, 0.74910039, 0.53472272]),
     'desired_goal': array([1.24073906, 0.77753463, 0.63457791])}

这个环境（GoalEnv）是个三个key的字典，天授会将其按照如下格式存储：
::

    >>> from tianshou.data import ReplayBuffer
    >>> b = ReplayBuffer(size=3)
    >>> b.add(obs=e.reset(), act=0, rew=0, done=0)
    >>> print(b)
    ReplayBuffer(
        act: array([0, 0, 0]),
        done: array([0, 0, 0]),
        info: Batch(),
        obs: Batch(
                 achieved_goal: array([[1.34183265, 0.74910039, 0.53472272],
                                       [0.        , 0.        , 0.        ],
                                       [0.        , 0.        , 0.        ]]),
                 desired_goal: array([[1.42154265, 0.62505137, 0.62929863],
                                      [0.        , 0.        , 0.        ],
                                      [0.        , 0.        , 0.        ]]),
                 observation: array([[ 1.34183265e+00,  7.49100387e-01,  5.34722720e-01,
                                       1.97805133e-04,  7.15193042e-05,  7.73933014e-06,
                                       5.51992816e-08, -2.42927453e-06,  4.73325650e-06,
                                      -2.28455228e-06],
                                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                       0.00000000e+00],
                                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                       0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                       0.00000000e+00]]),
             ),
        policy: Batch(),
        rew: array([0, 0, 0]),
    )
    >>> print(b.obs.achieved_goal)
    [[1.34183265 0.74910039 0.53472272]
     [0.         0.         0.        ]
     [0.         0.         0.        ]]

也可以很方便地从Buffer中采样出数据：
::

    >>> batch, indice = b.sample(2)
    >>> batch.keys()
    ['act', 'done', 'info', 'obs', 'obs_next', 'policy', 'rew']
    >>> batch.obs[-1]
    Batch(
        achieved_goal: array([1.34183265, 0.74910039, 0.53472272]),
        desired_goal: array([1.42154265, 0.62505137, 0.62929863]),
        observation: array([ 1.34183265e+00,  7.49100387e-01,  5.34722720e-01,  1.97805133e-04,
                             7.15193042e-05,  7.73933014e-06,  5.51992816e-08, -2.42927453e-06,
                             4.73325650e-06, -2.28455228e-06]),
    )
    >>> batch.obs.desired_goal[-1]  # 推荐，没有深拷贝
    array([1.42154265, 0.62505137, 0.62929863])
    >>> batch.obs[-1].desired_goal  # 不推荐
    array([1.42154265, 0.62505137, 0.62929863])
    >>> batch[-1].obs.desired_goal  # 不推荐
    array([1.42154265, 0.62505137, 0.62929863])

因此只需在自定义的网络中，换一下 ``forward`` 函数的 state 写法：
::

    def forward(self, s, ...):
        # s is a Batch
        observation = s.observation
        achieved_goal = s.achieved_goal
        desired_goal = s.desired_goal
        ...

当然如果自定义的环境中，状态是一个自定义的类，也是可以的。不过天授只会把它的地址进行存储，就像下面这样（状态是nx.Graph）：
::

    >>> # 这个例子可能现在不太能work，因为numpy升级了，以及nx.Graph重写了__getitem__，导致np.array([nx.Graph()])会出来空的数组……
    >>> # 不过正常的自定义class应该没啥问题
    >>> import networkx as nx
    >>> b = ReplayBuffer(size=3)
    >>> b.add(obs=nx.Graph(), act=0, rew=0, done=0)
    >>> print(b)
    ReplayBuffer(
        act: array([0, 0, 0]),
        done: array([0, 0, 0]),
        info: Batch(),
        obs: array([<networkx.classes.graph.Graph object at 0x7f5c607826a0>, None,
                    None], dtype=object),
        policy: Batch(),
        rew: array([0, 0, 0]),
    )

由于只存储了引用，因此如果状态修改的话，有可能之前存储的状态也会跟着修改。为了确保不出bug，建议在返回这个状态的时候加上深拷贝（deepcopy）：
::

    def reset():
        return copy.deepcopy(self.graph)
    def step(a):
        ...
        return copy.deepcopy(self.graph), reward, done, {}


.. _marl_example:

多智能体强化学习
----------------------------------

本条目与 `Issue 121 <https://github.com/thu-ml/tianshou/issues/121>`_ 相关。

多智能体强化学习大概可以分为如下三类：

1. Simultaneous move：所有玩家在每个timestep都同时行动，比如moba游戏；

2. Cyclic move：每个玩家轮流行动，比如飞行棋；

3. Conditional move：每个玩家在当前timestep下面所能采取的行动受限于环境，比如 `Pig Game <https://en.wikipedia.org/wiki/Pig_(dice_game)>`_。

这些基本上都能被转换为正常RL的形式。比如第一个 simultaneous move 只需要加一个 ``num_agent`` 标记一下，剩下代码都不用变；2和3的话，可以统一起来：环境在每个timestep选择id为 ``agent_id`` 的玩家进行游戏，更近一步把“所有的玩家”看做一个抽象的玩家的话（可以称之为MultiAgentPolicyManager，多智能体策略代理），就相当于单个玩家的情况，只不过每次多了个信息叫做 ``agent_id``，由这个代理转发给下属的各个玩家即可。至于3的condition，只需要多加一个信息叫做mask就行了。大概像下面这张图一样：

.. image:: /_static/images/marl.png
    :align: center
    :height: 300

可以把上述文字描述形式化为下面的伪代码：
::

    action = policy(state, agent_id, mask)
    (next_state, next_agent_id, next_mask), reward = env.step(action)

于是只要创建一个新的state：``state_ = (state, agent_id, mask)``，就可以使用之前正常的代码：
::

    action = policy(state_)
    next_state_, reward = env.step(action)

基于这种思路，我们写了个用DQN玩 `四子棋 <https://en.wikipedia.org/wiki/Tic-tac-toe>`_ 的demo，可以在 `这里 </en/master/tutorials/tictactoe.html>`_ 查看。
