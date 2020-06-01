.. _example:

平台使用实例
============

实例一：在CartPole-v0环境中运行DQN算法
--------------------------------------

本实例将使用天授平台从头搭建整个训练流程，并获得一个能在平均10秒之内解决CartPole-v0环境的的DQN :cite:`dqn` 智能体。

首先导入相关的包，并且定义相关参数：

::

    import gym, torch, numpy as np, torch.nn as nn
    from torch.utils.tensorboard import SummaryWriter
    import tianshou as ts

    task = 'CartPole-v0'
    lr = 1e-3
    gamma = 0.9
    n_step = 4
    eps_train, eps_test = 0.1, 0.05
    epoch = 10
    step_per_epoch = 1000
    collect_per_step = 10
    target_freq = 320
    batch_size = 64
    train_num, test_num = 8, 100
    buffer_size = 20000
    writer = SummaryWriter('log/dqn')

创建向量化环境从而能够并行采样：

::

    # you can also use SubprocVectorEnv
    train_envs = ts.env.VectorEnv([
        lambda: gym.make(task) for _ in range(train_num)])
    test_envs = ts.env.VectorEnv([
        lambda: gym.make(task) for _ in range(test_num)])

使用PyTorch原生定义的网络结构，并定义优化器：

::

    class Net(nn.Module):
        def __init__(self, state_shape, action_shape):
            super().__init__()
            self.model = nn.Sequential(*[
                nn.Linear(np.prod(state_shape), 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 128), nn.ReLU(inplace=True),
                nn.Linear(128, np.prod(action_shape))
            ])
        def forward(self, s, state=None, info={}):
            if not isinstance(s, torch.Tensor):
                s = torch.tensor(s, dtype=torch.float)
            batch = s.shape[0]
            logits = self.model(s.view(batch, -1))
            return logits, state

    env = gym.make(task)
    state_shape = env.observation_space.shape \
        or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = Net(state_shape, action_shape)
    optim = torch.optim.Adam(net.parameters(), lr=lr)

初始化策略（Policy）和采集器（Collector）：

::

    policy = ts.policy.DQNPolicy(
        net, optim, gamma, n_step,
        target_update_freq=target_freq)
    train_collector = ts.data.Collector(
        policy, train_envs, ts.data.ReplayBuffer(buffer_size))
    test_collector = ts.data.Collector(policy, test_envs)

开始训练：

::

    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector, epoch, 
        step_per_epoch, collect_per_step, test_num, batch_size, 
        train_fn=lambda e: policy.set_eps(eps_train),
        test_fn=lambda e: policy.set_eps(eps_test),
        stop_fn=lambda x: x >= env.spec.reward_threshold, 
        writer=writer, task=task)
    print(f'Finished training! Use {result["duration"]}')

会有进度条显示，并且在大约10秒内训练完毕，结果如下：

.. code:: bash

    Epoch #1:  88%|#8| 880/1000 [00:05<00:00, ..., v/st=17329.91]
    Finished training! Use 5.45s

可以将训练完毕的策略模型存储至文件中或者从已有文件中导入模型权重：

::

    torch.save(policy.state_dict(), 'dqn.pth')
    policy.load_state_dict(torch.load('dqn.pth'))

可以以每秒35帧的速率查看智能体与环境交互的结果：

::

    collector = ts.data.Collector(policy, env)
    collector.collect(n_episode=1, render=1 / 35)
    collector.close()

查看TensorBoard中存储的结果：

.. code:: bash

    tensorboard --logdir log/dqn

结果如 `图 5.1`_ 所示。

.. figure:: /_static/images/tfb.png
   :scale: 100%
   :name: fig-tfb
   :align: center

   图 5.1：TensorBoard可视化训练过过程

.. _图 5.1: #fig-tfb

当然，如果想要定制化训练策略而不使用训练器提供的现有逻辑，也是可以的。下面的代码展示了如何定制化训练策略：

::

    # pre-collect 5000 frames with random action before training
    policy.set_eps(1)
    train_collector.collect(n_step=5000)

    policy.set_eps(0.1)
    for i in range(int(1e6)):  # total step
        collect_result = train_collector.collect(n_step=10)

        # once if the collected episodes' mean returns reach
        # the threshold, or every 1000 steps, we test it on 
        # test_collector
        if collect_result['rew'] >= env.spec.reward_threshold \ 
                or i % 1000 == 0:
            policy.set_eps(0.05)
            result = test_collector.collect(n_episode=100)
            if result['rew'] >= env.spec.reward_threshold:
                # end of training loop
                break
            else:
                # back to training eps
                policy.set_eps(0.1)

        # train policy with a sampled batch data
        losses = policy.learn(
            train_collector.sample(batch_size=64))

    print('Finished training! Test mean returns:', 
          str(result["rew"]))

实例二：循环神经网络的训练
--------------------------

在POMDP场景中往往需要循环神经网络的训练支持。此处为简单起见，仍然以实例一中的场景和代码为基础进行展示。需要的改动如下：

首先修改模型为LSTM：

::

    class Recurrent(nn.Module):
        def __init__(self, state_shape, action_shape):
            super().__init__()
            self.fc1 = nn.Linear(np.prod(state_shape), 128)
            self.nn = nn.LSTM(input_size=128, hidden_size=128,
                              num_layers=3, batch_first=True)
            self.fc2 = nn.Linear(128, np.prod(action_shape))

        def forward(self, s, state=None, info={}):
            if not isinstance(s, torch.Tensor):
                s = torch.tensor(s, dtype=torch.float)
            # s [bsz, len, dim] (training)
            # or [bsz, dim] (evaluation)
            if len(s.shape) == 2:
                bsz, dim = s.shape
                length = 1 
            else:
                bsz, length, dim = s.shape
            s = self.fc1(s.view([bsz * length, dim]))
            s = s.view(bsz, length, -1) 
            self.nn.flatten_parameters()
            if state is None:
                s, (h, c) = self.nn(s)
            else:
                # we store the stack data with [bsz, len, ...]
                # but pytorch rnn needs [len, bsz, ...]
                s, (h, c) = self.nn(s, (
                    state['h'].transpose(0, 1).contiguous(),
                    state['c'].transpose(0, 1).contiguous()))
            s = self.fc2(s[:, -1])
            # make sure the 0-dim is batch size: [bsz, len, ...]
            return s, {'h': h.transpose(0, 1).detach(),
                       'c': c.transpose(0, 1).detach()}

其次重新定义策略，并将 ``train_collector``
中的重放缓冲区设置成堆叠采样模式，堆叠帧数 :math:`n` 为4：

::

    env = gym.make(task)
    state_shape = env.observation_space.shape \
        or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = Recurrent(state_shape, action_shape)
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    policy = ts.policy.DQNPolicy(
        net, optim, gamma, n_step,
        target_update_freq=target_freq)
    train_collector = ts.data.Collector(
        policy, train_envs,
        ts.data.ReplayBuffer(buffer_size, stack_num=4))
    test_collector = ts.data.Collector(policy, test_envs)

即可使用实例一中的代码进行正常训练，结果如下：

::

    Epoch #1:  83%|#4| 831/1000 [00:19<00:03, ..., v/st=13832.13]
    Finished training! Use 19.63s

实例三：多模态任务训练
----------------------

在像机器人抓取之类的任务中，智能体会获取多模态的观测值。天授完整保留了多模态观测值的数据结构，以数据组的形式给出，并且能方便地支持分片操作。以Gym环境中的“FetchReach-v1”为例，每次返回的观测值是一个字典，包含三个元素“observation”、“achieved\_goal”和“desired\_goal”。

在实例一代码的基础上进行修改：

::

    task = 'FetchReach-v1'
    train_envs = ts.env.VectorEnv([
        lambda: gym.make(task) for _ in range(train_num)]) 
    test_envs = ts.env.VectorEnv([
        lambda: gym.make(task) for _ in range(test_num)])

    class Net(nn.Module):
        def __init__(self, state_shape, action_shape):
            super().__init__()
            self.model = nn.Sequential(*[
                nn.Linear(np.prod(state_shape), 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128), nn.ReLU(inplace=True),
                nn.Linear(128, 128), nn.ReLU(inplace=True),
                nn.Linear(128, np.prod(action_shape))
            ])
        def forward(self, s, state=None, info={}):
            o = s.observation
            # s.achieved_goal, s.desired_goal are also available
            if not isinstance(o, torch.Tensor):
                o = torch.tensor(o, dtype=torch.float)
            batch = o.shape[0]
            logits = self.model(o.view(batch, -1))
            return logits, state

    env = gym.make(task)
    env.spec.reward_threshold = 1e10
    state_shape = env.observation_space.spaces['observation']
    state_shape = state_shape.shape
    action_shape = env.action_space.shape
    net = Net(state_shape, action_shape)
    optim = torch.optim.Adam(net.parameters(), lr=lr)

剩下的代码与实例一一致，可以直接运行。通过对比可以看出，只需改动神经网络中
``forward`` 函数的 :math:`s` 参数的处理即可。
