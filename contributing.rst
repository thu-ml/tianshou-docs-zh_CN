参与贡献
=========

安装开发版本
------------

假设现在在仓库的根目录下，使用命令

.. code-block:: bash

    pip3 install -e .[dev]

可以将天授仓库以可编辑模式安装（就是可以直接在源代码中进行改动，而不用重新pip install就可以用）。使用命令

.. code-block:: bash

    python3 setup.py develop --uninstall

PEP8代码风格检测
----------------

本项目遵循原始的PEP8代码风格规定，在项目根目录运行

.. code-block:: bash

    flake8 . --count --show-source --statistics

即可获得检测报告。

本地测试
--------

运行如下命令，即可在本地自动对项目进行单元测试并给出报告结果：

.. code-block:: bash

    pytest test --cov tianshou -s --durations 0 -v

使用GitHub Actions进行测试
--------------------------

1. 点击您fork出来的仓库的 ``Actions`` 图标：

.. image:: _static/images/action1.jpg
    :align: center

2. 点击绿色按钮：

.. image:: _static/images/action2.jpg
    :align: center

3. 会看到 ``Actions Enabled.`` 显示

4. 在此之后，一旦有新的commit被push上来，GitHub Actions会自动帮您运行单元测试：

.. image:: _static/images/action3.png
    :align: center

英文文档
--------

文档在 ``docs/`` 文件夹下，以 ``.rst`` 格式纂写。关于 ReStructuredText 的教程可参考 `这里 <https://pythonhosted.org/an_example_pypi_project/sphinx.html>`_.

API文档由 `Sphinx <http://www.sphinx-doc.org/en/stable/>`_ 自动生成，``docs/api/`` 目录下列出了需要生成的API文档。

如果需要编译出整个文档并以网页版形式预览，需要在 ``docs/`` 文件夹下运行

.. code-block:: bash

    make html

它会将文档生成在 ``docs/_build`` 目录中，可以使用浏览器直接预览。

中文文档
--------

中文文档在 `另外一个仓库中 <https://github.com/thu-ml/tianshou-docs-zh_CN/>`_。

与英文文档不同，此处不提供具体API文档的对应（因为变动可能不同步），而只是提供了一个从宏观层面来了解天授平台的一个教程。文档的编译与发布和英文文档无异。

