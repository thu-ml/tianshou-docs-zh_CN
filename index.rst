.. Tianshou documentation master file, created by
   sphinx-quickstart on Sat Mar 28 15:58:19 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

欢迎查看天授平台中文文档
========================

`天授 <https://baike.baidu.com/item/%E5%A4%A9%E6%8E%88>`_ 是一个基于PyTorch的深度强化学习平台，目前实现的算法有：

* DQN :class:`~tianshou.policy.DQNPolicy` `Deep Q-Network <https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf>`_
* 双网络DQN :class:`~tianshou.policy.DQNPolicy` `Double DQN <https://arxiv.org/pdf/1509.06461.pdf>`_
* C51 :class:`~tianshou.policy.C51Policy` `Categorical DQN <https://arxiv.org/pdf/1707.06887.pdf>`_
* QR-DQN :class:`~tianshou.policy.QRDQNPolicy` `Quantile Regression DQN <https://arxiv.org/pdf/1710.10044.pdf>`_
* Rainbow :class:`~tianshou.policy.RainbowPolicy` `Rainbow DQN <https://arxiv.org/pdf/1707.02298.pdf>`_
* IQN :class:`~tianshou.policy.IQNPolicy` `Implicit Quantile Network <https://arxiv.org/pdf/1806.06923.pdf>`_
* FQF :class:`~tianshou.policy.FQFPolicy` `Fully-parameterized Quantile Function <https://arxiv.org/pdf/1911.02140.pdf>`_
* 策略梯度 :class:`~tianshou.policy.PGPolicy` `Policy Gradient <https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf>`_
* 自然策略梯度 :class:`~tianshou.policy.NPGPolicy` `Natural Policy Gradient <https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf>`_
* 优势动作评价 (A2C) :class:`~tianshou.policy.A2CPolicy` `Advantage Actor-Critic <https://openai.com/blog/baselines-acktr-a2c/>`_
* 信任区域策略优化 (TRPO) :class:`~tianshou.policy.TRPOPolicy` `Trust Region Policy Optimization <https://arxiv.org/pdf/1502.05477.pdf>`_
* 近端策略优化 (PPO) :class:`~tianshou.policy.PPOPolicy` `Proximal Policy Optimization <https://arxiv.org/pdf/1707.06347.pdf>`_
* 深度确定性策略梯度 (DDPG) :class:`~tianshou.policy.DDPGPolicy` `Deep Deterministic Policy Gradient <https://arxiv.org/pdf/1509.02971.pdf>`_
* 双延迟深度确定性策略梯度 (TD3) :class:`~tianshou.policy.TD3Policy` `Twin Delayed DDPG <https://arxiv.org/pdf/1802.09477.pdf>`_
* 软动作评价 (SAC) :class:`~tianshou.policy.SACPolicy` `Soft Actor-Critic <https://arxiv.org/pdf/1812.05905.pdf>`_
* 离散软动作评价 :class:`~tianshou.policy.DiscreteSACPolicy` `Discrete Soft Actor-Critic <https://arxiv.org/pdf/1910.07207.pdf>`_
* 模仿学习 :class:`~tianshou.policy.ImitationPolicy` Imitation Learning
* BCQ :class:`~tianshou.policy.DiscreteBCQPolicy` `Discrete Batch-Constrained deep Q-Learning <https://arxiv.org/pdf/1910.01708.pdf>`_
* CQL :class:`~tianshou.policy.DiscreteCQLPolicy` `Discrete Conservative Q-Learning <https://arxiv.org/pdf/2006.04779.pdf>`_
* CRR :class:`~tianshou.policy.DiscreteCRRPolicy` `Critic Regularized Regression <https://arxiv.org/pdf/2006.15134.pdf>`_
* 后验采样强化学习 (PSRL) :class:`~tianshou.policy.PSRLPolicy` `Posterior Sampling Reinforcement Learning <https://www.ece.uvic.ca/~bctill/papers/learning/Strens_2000.pdf>`_
* 优先级经验重放 (PER) :class:`~tianshou.data.PrioritizedReplayBuffer` `Prioritized Experience Replay <https://arxiv.org/pdf/1511.05952.pdf>`_
* 广义优势函数估计器 (GAE) :meth:`~tianshou.policy.BasePolicy.compute_episodic_return` `Generalized Advantage Estimator <https://arxiv.org/pdf/1506.02438.pdf>`_

天授还有如下特点：

* 实现优雅，使用4000多行代码即完全实现上述功能
* 目前为止实现效果最好的 `MuJoCo benchmark <https://github.com/thu-ml/tianshou/tree/master/examples/mujoco>`_
* 支持任意算法的多个环境（同步异步均可的）并行采样，详见 :ref:`parallel_sampling`
* 支持动作网络和价值网络使用循环神经网络（RNN）来实现，详见 :ref:`rnn_training`
* 支持自定义环境，包括任意类型的观测值和动作值（比如一个字典、一个自定义的类），详见 :ref:`self_defined_env`
* 支持自定义训练策略，详见 :ref:`customize_training`
* 支持 N-step bootstrap 采样方式 :meth:`~tianshou.policy.BasePolicy.compute_nstep_return` 和优先级经验重放 :class:`~tianshou.data.PrioritizedReplayBuffer` 在任意基于Q学习的算法上的应用；感谢numba jit的优化让GAE、nstep和PER运行速度变得巨快无比
* 支持多智能体学习，详见 :ref:`marl_example`
* 拥有全面的 `单元测试 <https://github.com/thu-ml/tianshou/actions>`_，包括功能测试、完整训练流程测试、文档测试、代码风格测试和类型测试

与英文文档不同，中文文档提供了一个宏观层面的对天授平台的概览。（其实都是 `毕业论文 <_static/thesis.pdf>`_ 里面弄出来的）


安装
----

天授目前发布在 `PyPI <https://pypi.org/project/tianshou/>`_ 和 `conda-forge <https://github.com/conda-forge/tianshou-feedstock>`_ 中，需要Python版本3.6以上。

通过PyPI进行安装：

.. code-block:: bash

    $ pip install tianshou

通过conda进行安装：

.. code-block:: bash

    $ conda -c conda-forge install tianshou

还可以直接从GitHub源代码最新版本进行安装：

.. code-block:: bash

    $ pip install git+https://github.com/thu-ml/tianshou.git@master --upgrade


在安装完毕后，打开您的Python并输入

::

    import tianshou
    print(tianshou.__version__)

如果没有异常出现，那么说明已经成功安装了。

.. toctree::
   :maxdepth: 1
   :caption: 教程

   slide
   tutorials
   concepts
   benchmark
   cheatsheet

.. toctree::
   :maxdepth: 2
   :caption: 文档

   docs/toc

.. toctree::
   :maxdepth: 1
   :caption: 贡献

   contributing


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
