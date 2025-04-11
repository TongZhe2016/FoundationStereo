# FoundationStereo：零样本立体匹配

## 概述

FoundationStereo是一个为实现强大零样本泛化而设计的立体深度估计基础模型。本文档提供了该项目的详细说明，包括其架构、工作流程和组件。

## 目录

1. [简介](#简介)
2. [项目结构](#项目结构)
3. [核心架构](#核心架构)
4. [特征提取](#特征提取)
5. [几何编码](#几何编码)
6. [视差更新机制](#视差更新机制)
7. [工作流程](#工作流程)
8. [使用方法](#使用方法)
9. [性能](#性能)

## 简介

FoundationStereo是一个零样本立体匹配模型，它接收一对立体图像作为输入，输出密集视差图，可转换为公制尺度的深度图或3D点云。该模型由NVIDIA研究人员开发，并被CVPR 2025接受为口头报告。

FoundationStereo的关键创新包括：
- 大规模（100万对立体图像）高真实感合成训练数据集
- 适应视觉基础模型单目先验的侧调特征骨干网络
- 用于有效成本体积过滤的长程上下文推理
- 无需微调即可在不同领域实现强大的鲁棒性和准确性

该模型在Middlebury和ETH3D排行榜上取得了最先进的性能，为零样本立体深度估计建立了新标准。

## 项目结构

该项目组织为几个关键目录：

- `core/`：包含FoundationStereo模型的主要实现
  - `foundation_stereo.py`：主模型实现
  - `extractor.py`：特征提取模块
  - `geometry.py`：立体匹配的几何编码
  - `update.py`：视差更新机制
  - `submodule.py`：基本构建块和实用函数
  - `utils/`：实用函数
- `depth_anything/`：与Depth Anything模型集成，用于单目深度先验
- `dinov2/`：与DINOv2视觉基础模型集成
- `scripts/`：用于运行演示和可视化的脚本
  - `run_demo.py`：对立体图像对进行推理的主脚本
  - `make_onnx.py`：用于ONNX/TensorRT转换的脚本
  - `vis_dataset.py`：用于可视化FSD数据集的脚本
- `assets/`：样本立体图像和相机参数
- `pretrained_models/`：存储预训练模型权重的目录
- `teaser/`：样本输出和可视化

## 核心架构

FoundationStereo的核心在`core/foundation_stereo.py`中实现。主类`FoundationStereo`继承自`nn.Module`并实现了立体匹配流程。

### 关键组件

1. **特征提取**：使用CNN特征和视觉基础模型（DINOv2和Depth Anything）的组合
2. **成本体积构建**：使用分组相关性和连接构建成本体积
3. **成本体积过滤**：使用带有注意力机制的3D沙漏网络
4. **视差回归**：从过滤后的成本体积进行初始视差估计
5. **迭代细化**：基于GRU的迭代更新以细化视差图
6. **上采样**：上下文感知上采样以生成最终高分辨率视差图

### 主模型类

```python
class FoundationStereo(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, args):
        # 初始化模型组件
        
    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False, low_memory=False, init_disp=None):
        # 前向传播计算视差
        
    def run_hierachical(self, image1, image2, iters=12, test_mode=False, low_memory=False, small_ratio=0.5):
        # 高分辨率图像的分层推理
```

## 特征提取

特征提取是FoundationStereo的关键组件，在`core/extractor.py`中实现。该模型使用混合方法，结合：

1. **基于CNN的特征**：使用EdgeNext作为骨干网络
2. **视觉基础模型**：利用预训练的DINOv2和Depth Anything模型

### Feature类

```python
class Feature(nn.Module):
    def __init__(self, args):
        # 初始化特征提取组件
        
    def forward(self, x):
        # 从输入图像提取特征
```

特征提取过程：
1. 通过EdgeNext处理输入图像以获取多尺度特征
2. 通过冻结的Depth Anything模型传递图像以获取单目深度先验
3. 通过一系列反卷积和连接操作组合这些特征
4. 返回多尺度特征列表和视觉transformer特征

### DepthAnythingFeature

此类集成了Depth Anything模型以获取单目深度先验：

```python
class DepthAnythingFeature(nn.Module):
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }
    
    def __init__(self, encoder='vits'):
        # 初始化Depth Anything模型
        
    def forward(self, x):
        # 提取特征和深度信息
```

## 几何编码

几何编码在`core/geometry.py`中实现。它创建一个组合几何编码体积，帮助建立左右图像之间的对应关系。

### Combined_Geo_Encoding_Volume

```python
class Combined_Geo_Encoding_Volume:
    def __init__(self, init_fmap1, init_fmap2, geo_volume, num_levels=2, dx=None):
        # 初始化几何编码体积
        
    def __call__(self, disp, coords, low_memory=False):
        # 基于当前视差估计生成几何特征
        
    @staticmethod
    def corr(fmap1, fmap2):
        # 计算特征图之间的相关性
```

此类：
1. 在不同尺度构建相关体积金字塔
2. 基于当前视差估计从这些体积中采样
3. 为视差更新提供丰富的几何特征

## 视差更新机制

视差更新机制在`core/update.py`中实现。它使用基于GRU的方法迭代细化视差估计。

### BasicSelectiveMultiUpdateBlock

```python
class BasicSelectiveMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, volume_dim=8):
        # 初始化更新块组件
        
    def forward(self, net, inp, corr, disp, att):
        # 更新视差估计
```

此类：
1. 从当前视差和相关体积编码运动特征
2. 使用选择性GRU单元在多个尺度更新隐藏状态
3. 预测视差更新和用于上下文感知上采样的掩码

### SelectiveConvGRU

```python
class SelectiveConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, small_kernel_size=1, large_kernel_size=3, patch_size=None):
        # 初始化选择性GRU
        
    def forward(self, att, h, *x):
        # 基于注意力更新隐藏状态
```

此类实现了一个注意力引导的GRU，根据注意力图选择性地应用不同的核大小。

## 工作流程

FoundationStereo的整体工作流程可概括如下：

1. **输入处理**：
   - 归一化输入立体图像
   - 将图像填充为32的倍数

2. **特征提取**：
   - 使用Feature类从左右图像提取特征
   - 获取多尺度特征和视觉transformer特征

3. **成本体积构建**：
   - 构建分组相关性体积
   - 构建连接体积
   - 组合这些体积并应用初始过滤

4. **初始视差估计**：
   - 对过滤后的成本体积应用分类器
   - 执行视差回归以获得初始视差图

5. **上下文网络**：
   - 通过上下文网络处理左图像
   - 生成用于视差更新的隐藏状态和上下文信息

6. **迭代细化**：
   - 对于指定的迭代次数：
     - 基于当前视差生成几何特征
     - 使用SelectiveConvGRU更新隐藏状态
     - 预测视差更新并应用
     - 必要时上采样视差

7. **最终输出**：
   - 返回最终视差图
   - 可选择转换为深度图或点云

## 使用方法

### 基本使用

要在一对立体图像上运行FoundationStereo：

```bash
python scripts/run_demo.py --left_file ./assets/left.png --right_file ./assets/right.png --ckpt_dir ./pretrained_models/model_best_bp2.pth --out_dir ./test_outputs/
```

### 关键参数

- `--left_file`, `--right_file`：左右立体图像的路径
- `--intrinsic_file`：相机内参文件的路径
- `--ckpt_dir`：预训练模型的路径
- `--out_dir`：保存结果的目录
- `--scale`：调整输入图像大小的比例因子（必须≤1）
- `--hiera`：为高分辨率图像启用分层推理
- `--valid_iters`：细化迭代次数
- `--get_pc`：生成点云输出
- `--denoise_cloud`：对点云应用去噪

### ONNX/TensorRT推理

为了更快的推理，可以将模型转换为ONNX和TensorRT：

```bash
# 转换为ONNX
export XFORMERS_DISABLED=1
python scripts/make_onnx.py --save_path ./output/foundation_stereo.onnx --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --height 480 --width 640 --valid_iters 22

# 将ONNX转换为TensorRT
trtexec --onnx=./output/foundation_stereo.onnx --saveEngine=./output/foundation_stereo.engine --fp16 --verbose
```

## 性能

FoundationStereo在多个基准测试上取得了最先进的性能：

1. **Middlebury排行榜**：第1名
2. **ETH3D排行榜**：第1名

该模型在不同场景的零样本立体匹配任务中优于现有方法，包括：
- 室内环境
- 室外场景
- 复杂纹理和光照条件

这种性能的关键在于模型无需微调即可泛化到未见数据的能力，利用视觉基础模型的丰富先验和大规模合成训练数据集。

## 要求和依赖

主要依赖项包括：
- PyTorch
- torchvision
- timm
- flash-attention
- open3d（用于点云可视化）
- 支持CUDA的GPU

完整的依赖列表可在`environment.yml`和`requirements.txt`文件中找到。

## 获得最佳结果的提示

- 输入图像应该经过校正和去畸变
- 不要交换左右图像
- 使用无损压缩的PNG文件
- 对于高分辨率图像（>1000px），使用分层推理
- 为了更快的推理，减少输入分辨率和细化迭代次数