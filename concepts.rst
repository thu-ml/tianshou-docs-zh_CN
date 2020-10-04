基本概念
========

天授把一个RL训练流程划分成了几个子模块：trainer（负责训练逻辑）、collector（负责数据采集）、policy（负责训练策略）和 buffer（负责数据存储），此外还有两个外围的模块，一个是env，一个是model（policy负责RL算法实现比如loss function的计算，model就只是个正常的神经网络）。下图描述了这些模块的依赖：

.. image:: /_static/images/concepts_arch.png
    :align: center
    :height: 300


Batch
-----

天授提供了 :class:`~tianshou.data.Batch` 作为内部模块传递数据所使用的数据结构，它既像字典又像数组，可以以这两种方式组织数据和访问数据，像下面这样：
::

    >>> import torch, numpy as np
    >>> from tianshou.data import Batch
    >>> data = Batch(a=4, b=[5, 5], c='2312312', d=('a', -2, -3))
    >>> # the list will automatically be converted to numpy array
    >>> data.b
    array([5, 5])
    >>> data.b = np.array([3, 4, 5])
    >>> print(data)
    Batch(
        a: 4,
        b: array([3, 4, 5]),
        c: '2312312',
        d: array(['a', '-2', '-3'], dtype=object),
    )
    >>> data = Batch(obs={'index': np.zeros((2, 3))}, act=torch.zeros((2, 2)))
    >>> data[:, 1] += 6
    >>> print(data[-1])
    Batch(
        obs: Batch(
                 index: array([0., 6., 0.]),
             ),
        act: tensor([0., 6.]),
    )

总之就是可以定义任何key-value放在Batch里面，然后可以做一些常规的操作比如+-\*、cat/stack之类的。`Understand Batch </en/master/tutorials/batch.html>`_ 里面详细描述了Batch的各种用法，非常值得一看。


Buffer
------

:class:`~tianshou.data.ReplayBuffer` 负责存储数据和采样出来数据用于policy的训练。目前天授保留了7个关键字在Buffer里面：

* ``obs`` :math:`t` 时刻的观测值；
* ``act`` :math:`t` 时刻采取的动作值；
* ``rew`` :math:`t` 时刻环境返回的奖励函数值；
* ``done`` :math:`t` 时刻是否结束这个episode；
* ``obs_next`` :math:`t+1` 时刻的观测值；
* ``info`` :math:`t` 时刻环境给出的额外信息（gym.Env会返回4个东西，最后一个就是它）；
* ``policy`` :math:`t` 时刻由policy计算出的需要额外存储的数据；

下面的代码片段展示了Buffer的一些典型用法：
::

    >>> import pickle, numpy as np
    >>> from tianshou.data import ReplayBuffer
    >>> buf = ReplayBuffer(size=20)
    >>> for i in range(3):
    ...     buf.add(obs=i, act=i, rew=i, done=i, obs_next=i + 1, info={})
    >>> buf.obs
    # 因为设置了 size = 20，所以 len(buf.obs) == 20
    array([0., 1., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0.])
    >>> # 但是里面只有3个合法的数据，因此 len(buf) == 3
    >>> len(buf)
    3
    >>> pickle.dump(buf, open('buf.pkl', 'wb'))  # 把buffer所有数据保存到 "buf.pkl"
    >>> buf2 = ReplayBuffer(size=10)
    >>> for i in range(15):
    ...     buf2.add(obs=i, act=i, rew=i, done=i, obs_next=i + 1, info={})
    >>> len(buf2)
    10
    >>> buf2.obs
    # 因为 buf2 的 size = 10，所以它只会存储最后10步的结果
    array([10., 11., 12., 13., 14.,  5.,  6.,  7.,  8.,  9.])

    >>> # 把 buf2 的数据挪到buf里面，同时保持相对时间顺序
    >>> buf.update(buf2)
    array([ 0.,  1.,  2.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.])

    >>> # 从buffer里面拿一个随机的数据，batch_data就是buf[indice]
    >>> batch_data, indice = buf.sample(batch_size=4)
    >>> batch_data.obs == buf[indice].obs
    array([ True,  True,  True,  True])
    >>> len(buf)
    13
    >>> buf = pickle.load(open('buf.pkl', 'rb'))  # load from "buf.pkl"
    >>> len(buf)
    3

:class:`~tianshou.data.ReplayBuffer` 还支持堆叠采样（为了RNN，详情查看 `Issue 19 <https://github.com/thu-ml/tianshou/issues/19>`_）、不存储obs_next（为了省些内存），以及任意类型的数据结构存储（这个是Batch支持的）：
::

    >>> buf = ReplayBuffer(size=9, stack_num=4, ignore_obs_next=True)
    >>> for i in range(16):
    ...     done = i % 5 == 0
    ...     buf.add(obs={'id': i}, act=i, rew=i, done=done,
    ...             obs_next={'id': i + 1})
    >>> print(buf)  # 可以发现obs_next并不在里面存着
    ReplayBuffer(
        act: array([ 9., 10., 11., 12., 13., 14., 15.,  7.,  8.]),
        done: array([0., 1., 0., 0., 0., 0., 1., 0., 0.]),
        info: Batch(),
        obs: Batch(
                 id: array([ 9., 10., 11., 12., 13., 14., 15.,  7.,  8.]),
             ),
        policy: Batch(),
        rew: array([ 9., 10., 11., 12., 13., 14., 15.,  7.,  8.]),
    )
    >>> index = np.arange(len(buf))
    >>> print(buf.get(index, 'obs').id)
    [[ 7.  7.  8.  9.]
     [ 7.  8.  9. 10.]
     [11. 11. 11. 11.]
     [11. 11. 11. 12.]
     [11. 11. 12. 13.]
     [11. 12. 13. 14.]
     [12. 13. 14. 15.]
     [ 7.  7.  7.  7.]
     [ 7.  7.  7.  8.]]
    >>> # 也可以这样取出stacked过的obs
    >>> np.allclose(buf.get(index, 'obs')['id'], buf[index].obs.id)
    True
    >>> # 可以通过 __getitem__ 来弄出obs_next（虽然并没存）
    >>> print(buf[:].obs_next.id)
    [[ 7.  8.  9. 10.]
     [ 7.  8.  9. 10.]
     [11. 11. 11. 12.]
     [11. 11. 12. 13.]
     [11. 12. 13. 14.]
     [12. 13. 14. 15.]
     [12. 13. 14. 15.]
     [ 7.  7.  7.  8.]
     [ 7.  7.  8.  9.]]

天授还提供了其他类型的buffer比如 :class:`~tianshou.data.ListReplayBuffer` （基于list），:class:`~tianshou.data.PrioritizedReplayBuffer` （基于线段树）。可以访问对应的文档来查看。


Policy
------

天授把一个RL算法用一个继承自 :class:`~tianshou.policy.BasePolicy` 的类来实现，主要的部分有如下几个：

* :meth:`~tianshou.policy.BasePolicy.__init__`：策略初始化，比如初始化自定义的模型等；
* :meth:`~tianshou.policy.BasePolicy.forward`：根据给定的观测值obs，计算出动作值action；
* :meth:`~tianshou.policy.BasePolicy.process_fn`：在获取训练数据之前和buffer进行交互，比如使用GAE或者nstep算法来估计优势函数；
* :meth:`~tianshou.policy.BasePolicy.learn`：使用一个Batch的数据进行策略的更新；
* :meth:`~tianshou.policy.BasePolicy.post_process_fn`：使用一个Batch的数据进行Buffer的更新（比如更新PER）；
* :meth:`~tianshou.policy.BasePolicy.update`：最主要的接口。这个update函数先是从buffer采样出一个batch，然后调用process_fn预处理，然后learn更新策略，然后 post_process_fn完成一次迭代：``process_fn -> learn -> post_process_fn``。


.. _policy_state:

各种状态和阶段
^^^^^^^^^^^^^^

强化学习训练流程可以分为两个部分：训练部分（Training state）和测试部分（Testing State），而训练部分可以细分为采集数据阶段（Collecting state）和更新策略阶段（Updating state），两个阶段在训练过程中交替进行。
顾名思义，采集数据阶段是由collector负责的，而策略更新阶段是由policy.update负责的。

为了区分上述这些状态，可以通过检查 ``policy.training`` 和 ``policy.updating`` 来确定处于哪个状态，这边列了一张表方便查看：

+-----------------------------------+-----------------+-----------------+
|          State for policy         | policy.training | policy.updating |
+================+==================+=================+=================+
|                | Collecting state |       True      |      False      |
| Training state +------------------+-----------------+-----------------+
|                |  Updating state  |       True      |      True       |
+----------------+------------------+-----------------+-----------------+
|           Testing state           |       False     |      False      |
+-----------------------------------+-----------------+-----------------+

``policy.updating`` 实际情况下主要用于exploration，比如在各种Q-Learning算法中，在不同的state切换探索策略。


policy.forward
^^^^^^^^^^^^^^


``forward`` 函数接收obs计算action，输入和输出由于算法的不同而不同，但大部分情况下是这样的：``(batch, state, ...) -> batch``。

输入的Batch是环境中给出的数据（observation、reward、done 和 info)，要么来自 ``tianshou.data.Collector.collect`` （Collecting state），要么来自``tianshou.data.ReplayBuffer.sample``（Updating state）。Batch里面的所有数据第一维都是batch-size。

输出也是一个Batch，必须包含 ``act`` 关键字，可能包含 ``state`` 关键字（用于存放hiddle state，RNN使用）、``policy`` 关键字（policy计算过程中需要存储到buffer里面的中间结果，比如logprob之类的，后续更新网络需要用到），以及其他key（只不过不会被存储到buffer里面）。

比如您想要使用policy单独来evaluate一个episode，不用collect给出的函数，可以像下面这样做：
::

    # env 是 gym.Env
    obs, done = env.reset(), False
    while not done:
        batch = Batch(obs=[obs])  # 第一维是 batch size
        act = policy(batch).act[0]  # policy.forward 返回一个 batch，使用 ".act" 来取出里面action的数据
        obs, rew, done, info = env.step(act)

这边 ``Batch(obs=[obs])`` 会自动为obs下面的所有数据创建第0维，让它为batch size=1，否则nn没法确定batch size。


.. _process_fn:

policy.process_fn
^^^^^^^^^^^^^^^^^

``process_fn`` 用于计算时间相关的序列信息，比如计算n-step returns或者GAE returns。这边拿2-step DQN举例，公式是

.. math::

    G_t = r_t + \gamma r_{t + 1} + \gamma^2 \max_a Q(s_{t + 2}, a)

:math:`\gamma` 是 discount factor，:math:`\gamma \in [0, 1]`。下面给出了未使用天授的训练过程伪代码：
::

    s = env.reset()
    buffer = Buffer(size=10000)
    agent = DQN()
    for i in range(int(1e6)):
        a = agent.compute_action(s)
        s_, r, d, _ = env.step(a)
        buffer.store(s, a, s_, r, d)
        s = s_
        if i % 1000 == 0:
            b_s, b_a, b_s_, b_r, b_d = buffer.get(size=64)
            # 计算 2-step returns，咋算呢？
            b_ret = compute_2_step_return(buffer, b_r, b_d, ...)
            # 更新 DQN policy
            agent.update(b_s, b_a, b_s_, b_r, b_d, b_ret)

从上面伪代码可以看出我们需要一个依赖于时间相关的接口来计算2-step returns。:meth:`~tianshou.policy.BasePolicy.process_fn` 就是用来做这件事的，给它一个replay buffer、采样用的index（相当于时间t）和采样出来的batch就能计算。因为在buffer里面我们按照时间顺序存储各种数据，因此2-step returns的计算可以像下面这样简单：
::

    class DQN_2step(BasePolicy):
        """其他的代码"""

        def process_fn(self, batch, buffer, indice):
            buffer_len = len(buffer)
            batch_2 = buffer[(indice + 2) % buffer_len]
            # 上面这个代码访问batch_2.obs就是s_{t+2}，也可以像下面这样访问：
            #   batch_2_obs = buffer.obs[(indice + 2) % buffer_len]
            # 总之就是 buffer.obs[i] 和 buffer[i].obs是一个意思，但是前面的这种写法效率更高
            Q = self(batch_2, eps=0)  # shape: (batch_size, action_shape)
            maxQ = Q.max(dim=-1)
            batch.returns = batch.rew \
                + self._gamma * buffer.rew[(indice + 1) % buffer_len] \
                + self._gamma ** 2 * maxQ
            return batch

上面这个代码并没考虑 ``done = True`` 的情况，因此正确性不能保证，但是它展示了两种能够访问到 :math:`s_{t + 2}` 的方法。

至于policy的其他功能，可以参考 `tianshou.policy </en/master/api/tianshou.policy.html>`_，在最下面给出了一个宏观解释：:ref:`pseudocode`。


Collector
---------

:class:`~tianshou.data.Collector` 负责policy与env的交互和数据存储。:meth:`~tianshou.data.Collector.collect` 是collector的主要方法，它能够指定让policy和环境交互至少 ``n_step`` 个step或者 ``n_episode`` 个episode，并把该过程中产生的数据存储到buffer中。

为啥这边强调 **至少**？因为天授使用了一个buffer来处理这些事情（当然还有一种方法是每个env对应单独的一个buffer）。如果用一个buffer做的话，需要维护若干个cache buffer，然后必须等到episode结束才能将cache buffer里面的数据转移到main buffer中，否则不能保证其中的时间顺序。

这么做有优点也有缺点，缺点是老是有人提issue，得手动加一个 ``gym.wrappers.TimeLimit`` 在env上面（如果env的done一直是False的话）；优点是delayed update能够带来一定的性能提升，以及会大幅简化其他部分代码（比如PER、nstep、GAE这种就很简单，还有buffer.sample也还算简单，如果n个buffer的话就得多写很多代码）。

:ref:`pseudocode` 给出了一个宏观层面的解释，其他collector的功能可参考对应文档。


Trainer
-------

有了之前声明的collector和policy之后，就可以用trainer把它们包起来。Trainer负责最上层训练逻辑的控制，例如训练多少次之后进行策略和环境的交互。现有的训练器包括同策略学习训练器（On-policy Trainer）和异策略学习训练器（Off-policy Trainer）。

天授未显式地将训练器抽象成一个类，因为在其他现有平台中都将类似训练器的实现抽象封装成一个类，导致用户难以二次开发。因此以函数的方式实现训练器，并提供了示例代码便于研究者进行定制化训练策略的开发。可以参考 :ref:`customized_trainer`。


.. _pseudocode:

宏观解释
--------

接下来将通过一段伪代码的讲解来阐释上述所有抽象模块的应用。
::

    # pseudocode, cannot work                                       # 对应天授实现
    s = env.reset()                                                 # 环境初始化，在env中实现
    buffer = Buffer(size=10000)                                     # buffer = tianshou.data.ReplayBuffer(size=10000)
    agent = DQN()                                                   # policy.__init__(...)
    for i in range(int(1e6)):                                       # 在Trainer中实现
        a = agent.compute_action(s)                                 # act = policy(batch, ...).act
        s_, r, d, _ = env.step(a)                                   # collector.collect(...)
        buffer.store(s, a, s_, r, d)                                # collector.collect(...)
        s = s_                                                      # collector.collect(...)
        if i % 1000 == 0:                                           # 在Trainer中实现
                                                                    # the following is done in policy.update(batch_size, buffer)
            b_s, b_a, b_s_, b_r, b_d = buffer.get(size=64)          # batch, indice = buffer.sample(batch_size)
            # 计算 2-step returns，咋算呢？
            b_ret = compute_2_step_return(buffer, b_r, b_d, ...)    # policy.process_fn(batch, buffer, indice)
            # 更新 DQN policy
            agent.update(b_s, b_a, b_s_, b_r, b_d, b_ret)           # policy.learn(batch, ...)
