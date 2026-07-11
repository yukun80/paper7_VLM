#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可选的 PyTorch/CUDA 最小环境探针。

用途：只打印当前 Python 环境的 torch 版本和 CUDA 可见性。
推荐运行命令：python env_test.py
主要输入：当前激活的 Python/conda 环境。
主要输出：终端中的 ``<torch-version> <cuda-available>``。
写入行为：不写文件。
所属流程：临时环境诊断，不是 benchmark、训练或评估的必需步骤。
"""

import torch


print(torch.__version__, torch.cuda.is_available())
