.. _intro:

引言
====

深度强化学习研究背景
--------------------

在2012年AlexNet :cite:`alexnet` 夺得ImageNet图像分类比赛冠军之后，深度神经网络被应用在许多领域，如目标检测和跟踪；
并且在一系列的任务中，深度学习模型的准确率达到甚至超过了人类水准。大规模的商业化应用随之而来，如人脸识别、医疗图像处理等领域使用深度神经网络提高识别的速度和精度。

强化学习概念的提出最早可追溯到20世纪，其在简单场景上的应用于上世纪90年代至本世纪初即被深入研究，比如1992年使用强化学习算法打败了人类西洋双陆棋玩家 :cite:`td`。早期的强化学习算法大多使用线性回归来拟合策略函数，并且需要预先提取人为定义好的特征，实际效果不甚理想。2013年之后，结合了深度学习的优势，深度强化学习使用深度神经网络进行函数拟合，展现出了其强大的威力，如使用DQN :cite:`dqn` 玩Atari游戏达到人类水准、AlphaGo :cite:`alphago` 与人类顶尖围棋选手的划时代人机对战，OpenAI Five :cite:`dota2` 在Dota2 5v5对战比赛中击败人类冠军团队，无论是学术界还是工业界都对这一领域表现出了极大兴趣。深度强化学习如今不但被应用在游戏AI中，还被使用在如机械臂抓取、自动驾驶、高频交易、智能交通等等实际场景中，其前景十分广阔。

深度强化学习平台框架现状
------------------------

现有深度强化学习平台简介
~~~~~~~~~~~~~~~~~~~~~~~~

深度强化学习算法由于其计算模式不规则和高并发的特点，导致其无法像计算机视觉、自然语言处理领域的计算框架一样，从训练数据流的角度进行设计与实现；而强化学习算法形式与实现细节难以统一，又进一步加大了平台的编写难度。一些项目尝试解决通用强化学习算法框架问题，但是通常情况下，深度强化学习领域的研究者不得不从头编写一个特定强化学习算法的程序来满足自己的需求。

以GitHub星标数量大于约一千为标准，现有使用较为广泛的深度强化学习平台有OpenAI的Baselines :cite:`baselines`、SpinningUp :cite:`spinningup`，加州伯克利大学的开源分布式强化学习框架RLlib :cite:`rllib`、rlpyt :cite:`rlpyt`、rlkit :cite:`rlkit`、Garage :cite:`garage`，谷歌公司的Dopamine :cite:`dopamine`、B-suite :cite:`bsuite`，以及其他独立开发的平台Stable-Baselines :cite:`stable-baselines`、keras-rl :cite:`keras-rl`、PyTorch-DRL :cite:`pytorch-drl`、TensorForce :cite:`tensorforce`。`图 1.1`_ 展示了若干主流强化学习算法平台的标志，`表 1.1`_ 列举了各个框架的基本信息。

.. figure:: /_static/images/exist_framework.png
   :width: 400px
   :name: fig-exist
   :align: center

   图 1.1：目前较为主流的深度强化学习算法平台

.. _图 1.1: #fig-exist

几乎所有的强化学习平台都以OpenAI Gym :cite:`gym` 所定义的API作为智能体与环境进行交互的标准接口，以TensorFlow :cite:`tensorflow` 作为后端深度学习框架为主的平台居多，支持了至少4种免模型强化学习算法。大部分平台支持对训练环境进行自定义配置。

PyTorch :cite:`pytorch` 是Facebook公司推出的一款开源深度学习框架，由于其易用性、接口稳定性和社区活跃性，受到越来越多学术界和工业界研究者的青睐，大有超过TensorFlow框架的趋势。然而使用PyTorch编写的深度强化学习框架中，星标最多为PyTorch-DRL :cite:`pytorch-drl` （2400+星标），远远不如TensorFlow强化学习社区中的开源框架活跃。本文将在下一小节分析讨论其详细原因。

现有深度强化学习平台不足
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: 表 1.1：深度强化学习平台总览，按照GitHub星标数从大到小排序，截止2020/05/12
   :name: tab-allframe
   :align: center
   :header-rows: 1

   * - 平台名称
     - 星标数
     - 后端框架
     - 模块化
     - 文档
     - 代码质量
     - 单元测试
     - 上次更新
   * - `Ray/RLlib <https://github.com/ray-project/ray/tree/master/rllib>`_ :cite:`rllib`
     - 11460
     - TF/PyTorch
     - :math:`\surd`
     - 较全
     - 10 / 24065
     - :math:`\surd`
     - 2020.5
   * - `Baselines <https://github.com/openai/baselines>`_ :cite:`baselines`
     - 9764
     - TF
     - :math:`\times`
     - 无
     - 2673 / 10411
     - :math:`\surd`
     - 2020.1
   * - `Dopamine <https://github.com/google/dopamine>`_ :cite:`dopamine`
     - 8845
     - TF1
     - :math:`\surd`
     - 较全
     - 180 / 2519
     - :math:`\surd`
     - 2019.12
   * -  `SpinningUp <https://github.com/openai/spinningup>`_ :cite:`spinningup`
     -  4630 
     -  TF1/PyTorch 
     -  :math:`\times` 
     -  全面 
     -  1656 / 3724
     -  :math:`\times` 
     -  2019.11
   * -  `keras-rl <https://github.com/keras-rl/keras-rl>`_ :cite:`keras-rl`
     -  4612 
     -  Keras 
     -  :math:`\surd`
     -  不全 
     -  522 / 2346 
     -  :math:`\surd`
     -  2019.11
   * -  `Tensorforce <https://github.com/tensorforce/tensorforce>`_ :cite:`tensorforce`
     -  2669 
     -  TF
     -  :math:`\surd`
     -  全面 
     -  3834 / 13609 
     -  :math:`\surd`
     -  2020.5 
   * -  `PyTorch-DRL <https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch>`_ :cite:`pytorch-drl` 
     -  2424 
     -  PyTorch 
     -  :math:`\surd`
     -  无 
     -  2144 / 4307
     -  :math:`\surd`
     -  2020.2 
   * -  `Stable-Baselines <https://github.com/hill-a/stable-baselines>`_ :cite:`stable-baselines`
     -  2054 
     -  TF1 
     -  :math:`\times` 
     -  全面 
     -  2891 / 10989 
     -  :math:`\surd`
     -  2020.5 
   * -  `天授 <https://github.com/thu-ml/tianshou/>`_
     -  1529 
     -  PyTorch 
     -  :math:`\surd`
     -  全面 
     -  0 / 2141 
     -  :math:`\surd`
     -  2020.5 
   * -  `rlpyt <https://github.com/astooke/rlpyt>`_ :cite:`rlpyt`
     -  1448 
     -  PyTorch 
     -  :math:`\surd`
     -  较全 
     -  1191 / 14493 
     -  :math:`\times` 
     -  2020.4 
   * -  `rlkit <https://github.com/vitchyr/rlkit>`_ :cite:`rlkit`
     -  1172 
     -  PyTorch 
     -  :math:`\surd`
     -  不全 
     -  275 / 7824 
     -  :math:`\times` 
     -  2020.3 
   * -  `B-suite <https://github.com/deepmind/bsuite>`_ :cite:`bsuite` 
     -  975
     -  TF2 
     -  :math:`\times` 
     -  无 
     -  220 / 5353 
     -  :math:`\times` 
     -  2020.5 
   * -  `Garage <https://github.com/rlworkgroup/garage>`_ :cite:`garage` 
     -  709
     -  TF1/PyTorch 
     -  :math:`\surd`
     -  不全 
     -  5 / 17820
     -  :math:`\surd`
     -  2020.5 


| 注：TF为TensorFlow缩写，包含版本v1和v2；TF1为TensorFlow v1版本缩写，不包含版本v2；TF2为TensorFlow v2版本缩写，不包含版本v1；代码质量一栏数据格式为“PEP8 **不符合** 规范数 / 项目Python文件行数”。

.. _表 1.1: #tab-allframe

`表 1.1`_ 按照GitHub星标数目降序排列，从后端框架、是否模块化、文档完善程度、代码质量、单元测试和最后维护时间这些维度，列举了比较流行的深度强化学习开源平台框架。这些平台框架在不同评价维度上或多或少有些缺陷，从而降低了用户体验。此处列出一些典型问题，如下所示：

- **算法模块化不足：** 以OpenAI Baselines为代表，将每个强化学习算法单独独立成一份代码，因此无法做到代码之间的复用。用户在使用相关代码时，必须逐一修改每份代码，带来了极大困难。
- **实现算法种类有限：** 以Dopamine和SpinningUp为代表，Dopamine框架只支持DQN算法族，并不支持策略梯度；SpinningUp只支持策略梯度算法族，未实现Q学习的一系列算法。两个著名的平台所支持的强化学习算法均不全面。
- **代码实现复杂度过高：** 以RLlib为代表，代码层层封装嵌套，用户难以进行二次开发。
- **文档不完整：** 文档应包含教程和代码注释，部分平台只实现了其一，甚至完全没有文档，十分影响平台框架的使用。
- **平台性能不佳：** 强化学习算法本身难以调试，如果能够提升平台性能则将会大幅度降低调试难度。仍然以OpenAI Baselines为代表，无法全面支持并行环境采样，十分影响训练效率。
- **缺少完整单元测试：** 单元测试保证了代码的正确性和结果可复现性，但几乎所有平台只做了功能性验证，而没有进行完整的训练过程验证。
- **环境定制支持不足：** 许多非强化学习领域的研究者想使用强化学习算法来解决自己领域内问题，因此所交互的环境并不是Gym已经定制好的，这需要平台框架支持更多种类的环境，比如机械臂抓取所需的多模态环境。以rlpyt为例，该平台将环境进行封装，研究者如果想使用非Atari的环境必须大费周折改动框架代码。

此外值得讨论的是PyTorch深度强化学习框架活跃程度不如TensorFlow社区这个问题。不少使用PyTorch的研究者是编写独立的强化学习算法来满足自己需求，虽然实现较TensorFlow简单很多，但却没有针对数据流、数据存储进行优化；从 `表 1.1`_ 中也可以看出已有基于PyTorch的深度强化学习平台以PyTorch-DRL为代表，文档不全面、代码质量不如独立手写的算法高亦或是封装程度过高、缺乏可靠的单元测试，一定程度上阻碍了这些平台的进一步发展。

主要贡献与论文结构
------------------

主要贡献
~~~~~~~~

.. figure:: /_static/images/intro.png
   :name: fig-framework
   :align: center

   图 1.2：天授平台总体架构

.. _图 1.2: #fig-framework

本文描述了“天授”，一个基于PyTorch的深度强化学习算法平台。`图 1.2`_ 描述了该平台的总体架构。天授平台以PyTorch作为深度学习后端框架，将各个强化学习算法加以模块化，在数据层面抽象出了数据组（Batch）、数据缓冲区（Buffer）、采集器（Collector）三个基本模块，实现了针对任意环境的并行交互与采样功能，算法层面支持丰富多样的强化学习算法，如免模型强化学习（MFRL）中的一系列算法、模仿学习算法（IL）等，从而能够让研究者方便地使用不同算法来测试不同场景。

天授拥有创新的模块化设计，简洁地实现了各种强化学习算法，支持了用户各种各样的需求。在相关的性能实验评测中，天授在众多强化学习平台夺得头筹。种种亮点使其获得了强化学习社区不小的关注度，在GitHub上开源不到短短一个月，星标就超过了基于PyTorch的另一个著名的强化学习平台rlpyt :cite:`rlpyt`。

论文结构
~~~~~~~~

接下来的论文结构安排如下所示：

:ref:`impl`：描述了天授平台的设计与实现，将强化学习算法加以抽象凝练，分析提取出共有部分，介绍模块化的实现；以及介绍平台的其他特点。

:ref:`algo`：描述了天授平台目前所支持的各类深度强化学习算法，介绍各个算法的基本原理以及在天授平台中的实现细节。

:ref:`exp`：对比了天授平台与若干已有的著名深度强化学习平台的优劣之处，包括功能层面和性能层面的测试。

:ref:`example`：列举出了若干天授平台的典型使用样例，使读者能够进一步了解平台的接口和使用方法。

:ref:`conclusion`：对天授平台特点进行总结，并指出后续的工作方向。
