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

参见 `BasePolicy </en/latest/api/tianshou.policy.html#tianshou.policy.BasePolicy>`_。

.. _customize_training:

定制化训练策略
--------------

参见 :ref:`customized_trainer`。

.. _parallel_sampling:

环境并行采样
------------

使用 `tianshou.env.VectorEnv </en/latest/api/tianshou.env.html#tianshou.env.VectorEnv>`_ 或者 `tianshou.env.SubprocVectorEnv </en/latest/api/tianshou.env.html#tianshou.env.SubprocVectorEnv>`_：
::

    env_fns = [
        lambda: MyTestEnv(size=2),
        lambda: MyTestEnv(size=3),
        lambda: MyTestEnv(size=4),
        lambda: MyTestEnv(size=5),
    ]
    venv = SubprocVectorEnv(env_fns)

其中 ``env_fns`` 是个产生环境的可调用函数的列表。上面的代码也可以写成下面这样带for循环的形式：
::

    env_fns = [lambda x=i: MyTestEnv(size=x) for i in [2, 3, 4, 5]]
    venv = SubprocVectorEnv(env_fns)

.. _preprocess_fn:

批处理数据组
------------

本条目与 `Issue 42 <https://github.com/thu-ml/tianshou/issues/42>`_ 相关。

如果想收集训练log、预处理图像数据（比如Atari要resize到84x84x3）、根据环境信息修改奖励函数的值，可以在Collector中使用 ``preprocess_fn`` 接口，它会在数据存入Buffer之前被调用。

``preprocess_fn`` 接收7个保留关键字（obs/act/rew/done/obs_next/info/policy），返回需要修改的部分，以字典（dict）或者数据组（Batch）的形式返回均可，比如可以像下面这个例子一样：
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
                return {}  # 没有变量需要更新，返回空
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
                # 也可以返回 {'rew': kwargs['rew']}

最终只需要在Collector声明的时候加入一下这个hooker：
::

    test_processor = MyProcessor(size=100)
    collector = Collector(policy, env, buffer, test_processor.preprocess_fn)

还有一些示例在 `test/base/test_collector.py <https://github.com/thu-ml/tianshou/blob/master/test/base/test_collector.py>`_ 中可以查看。

.. _rnn_training:

RNN训练
-------

本条目与 `Issue 19 <https://github.com/thu-ml/tianshou/issues/19>`_ 相关

首先在 ReplayBuffer 的声明中加入 ``stack_num`` （堆叠采样）参数，表示在训练RNN的时候要扔给网络多少个timestep进行训练：
::

    buf = ReplayBuffer(size=size, stack_num=stack_num)

然后把神经网络模型中 ``state`` 参数用起来，可以参考 `代码片段 1 <https://github.com/thu-ml/tianshou/blob/master/test/discrete/net.py>`_ 中的 ``Recurrent``，或者 `代码片段 2 <https://github.com/thu-ml/tianshou/blob/master/test/continuous/net.py>`_ 中的 ``RecurrentActor`` 和 ``RecurrentCritic``。

以上代码片段展示了如何修改ReplayBuffer和神经网络模型，从而使用堆叠采样的观测值（stacked-obs）来训练RNN。如果想要堆叠别的值（比如stacked-action来训练Q(stacked-obs, stacked-action)），可以使用一个 ``gym.wrapper`` 来修改状态表示，比如wrapper把状态改为 [s, a] 的元组：

- 之前的数据存储：(s, a, s', r, d)，可以获得堆叠的s
- 采用wrapper之后的存储：([s, a], a, [s', a'], r, d)，可以获得堆叠的[s, a]，拆开来就是堆叠的s和a

.. _self_defined_env:

自定义环境与状态表示
--------------------

本条目与 `Issue 38 <https://github.com/thu-ml/tianshou/issues/38>`_ 和 `Issue 69 <https://github.com/thu-ml/tianshou/issues/69>`_ 相关。

首先，自定义的环境必须遵守OpenAI Gym定义的API规范，下面列出了一些：

- reset() -> state

- step(action) -> state, reward, done, info

- seed(s) -> None

- render(mode) -> None

- close() -> None

- observation_space

- action_space

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
