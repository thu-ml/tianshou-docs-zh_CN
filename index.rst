.. Tianshou documentation master file, created by
   sphinx-quickstart on Sat Mar 28 15:58:19 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

欢迎查看天授平台中文文档
========================

`天授 <https://baike.baidu.com/item/%E5%A4%A9%E6%8E%88>`_ 是一个基于PyTorch的深度强化学习平台，目前实现的算法有：

* 策略梯度 :class:`~tianshou.policy.PGPolicy` `Policy Gradient <https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf>`_
* 深度Q网络 :class:`~tianshou.policy.DQNPolicy` `Deep Q-Network <https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf>`_
* 双网络深度Q学习 :class:`~tianshou.policy.DQNPolicy` `Double DQN <https://arxiv.org/pdf/1509.06461.pdf>`_ with n-step returns
* 优势动作评价 :class:`~tianshou.policy.A2CPolicy` `Advantage Actor-Critic <https://openai.com/blog/baselines-acktr-a2c/>`_
* 深度确定性策略梯度 :class:`~tianshou.policy.DDPGPolicy` `Deep Deterministic Policy Gradient <https://arxiv.org/pdf/1509.02971.pdf>`_
* 近端策略优化 :class:`~tianshou.policy.PPOPolicy` `Proximal Policy Optimization <https://arxiv.org/pdf/1707.06347.pdf>`_
* 双延迟深度确定性策略梯度 :class:`~tianshou.policy.TD3Policy` `Twin Delayed DDPG <https://arxiv.org/pdf/1802.09477.pdf>`_
* 软动作评价 :class:`~tianshou.policy.SACPolicy` `Soft Actor-Critic <https://arxiv.org/pdf/1812.05905.pdf>`_
* 模仿学习 :class:`~tianshou.policy.ImitationPolicy` Imitation Learning
* 优先级经验重放 :class:`~tianshou.data.PrioritizedReplayBuffer` `Prioritized Experience Replay <https://arxiv.org/pdf/1511.05952.pdf>`_
* 广义优势函数估计器 :meth:`~tianshou.policy.BasePolicy.compute_episodic_return` `Generalized Advantage Estimator <https://arxiv.org/pdf/1506.02438.pdf>`_

天授支持所有算法的并行环境采样，所有算法均被重新形式化为基于重放缓冲区的算法。所有算法的Actor均支持循环状态表示（RNN Network）。

与英文文档不同，中文文档提供了一个宏观层面的对天授平台的概览。（其实都是 `毕业论文 </_static/thesis.pdf>`_ 里面弄出来的）

安装
----

天授目前发布在 `PyPI <https://pypi.org/project/tianshou/>`_ 中，可以通过
::

    pip3 install tianshou

来在您的Python环境中直接安装（注意Python版本需要是3.6以上）。当然也可以选择从GitHub源代码直接安装最新开发版本：
::

    pip3 install git+https://github.com/thu-ml/tianshou.git@master

如果使用的Python是托管在Anaconda或者Miniconda中，那么可以用如下命令进行安装：
::

    # 搞个新环境并让它自带pip
    conda create -n myenv pip
    # 激活这个新环境
    conda activate myenv
    # 安装天授
    pip install tianshou

在安装完毕后，打开您的Python并输入
::

    import tianshou as ts
    print(ts.__version__)

如果没有异常出现，那么说明已经成功安装了。

.. toctree::
   :maxdepth: 1
   :caption: 教程

   tutorials

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
