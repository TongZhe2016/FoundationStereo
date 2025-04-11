# FoundationStereo：实用指南

本文档为FoundationStereo项目提供实用指南，包括安装、运行演示和解决常见问题。

## 目录

1. [安装](#安装)
2. [数据集和模型权重](#数据集和模型权重)
3. [运行演示](#运行演示)
4. [高级用法](#高级用法)
5. [故障排除](#故障排除)
6. [性能优化](#性能优化)
7. [使用自定义数据](#使用自定义数据)
8. [与其他系统集成](#与其他系统集成)

## 安装

### 前提条件

- 支持CUDA的GPU（推荐：NVIDIA RTX 3090或更好）
- CUDA工具包11.3或更高版本
- Python 3.8或更高版本

### 环境设置

设置环境最简单的方法是使用Conda：

```bash
# 克隆仓库
git clone https://github.com/NVlabs/FoundationStereo.git
cd FoundationStereo

# 创建并激活conda环境
conda env create -f environment.yml
conda activate foundation_stereo
```

如果您在Conda安装中遇到问题，可以尝试使用pip手动安装依赖项：

```bash
pip install -r requirements.txt
```

### 常见安装问题

#### CUDA兼容性

如果遇到CUDA兼容性问题，请确保您的CUDA工具包版本与需求中指定的PyTorch版本兼容。您可能需要安装特定版本的PyTorch：

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Flash注意力

FoundationStereo使用Flash注意力进行高效的transformer操作。如果您在Flash注意力方面遇到问题，可以通过进行以下更改来禁用它：

```python
# 在core/submodule.py中，将flash_attn_func调用替换为标准注意力实现
# 原始：
attn_output = flash_attn_func(Q, K, V, window_size=window_size)

# 修改后：
Q = Q.transpose(1, 2)  # (B, nh, L, d)
K = K.transpose(1, 2)  # (B, nh, L, d)
V = V.transpose(1, 2)  # (B, nh, L, d)
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, L, L)
attn_weights = F.softmax(scores, dim=-1)
attn_output = torch.matmul(attn_weights, V)  # (B, nh, L, d)
attn_output = attn_output.transpose(1, 2).reshape(B, L, -1)  # (B, L, C)
```

## 数据集和模型权重

### 模型权重

从[这里](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing)下载用于零样本推理的基础模型。将整个文件夹（例如`23-51-11`）放在`./pretrained_models/`下。

可用模型：

| 模型 | 描述 |
| ----- | ----------- |
| 23-51-11 | 基于Vit-large的通用最佳性能模型 |
| 11-33-40 | 基于Vit-small的精度略低但推理更快的模型 |

### FSD数据集

Foundation Stereo Dataset (FSD)是一个包含100万对立体图像的大规模合成数据集。您可以从[这里](https://drive.google.com/drive/folders/1YdC2a0_KTZ9xix_HyqNMPCrClpm0-XFU?usp=sharing)下载整个数据集（>1TB）。

要快速查看，您可以下载一个小型[样本数据](https://drive.google.com/file/d/1dJwK5x8xsaCazz5xPGJ2OKFIWrd9rQT5/view?usp=drive_link)（3GB）。

要可视化数据集：

```bash
python scripts/vis_dataset.py --dataset_path ./DATA/sample/manipulation_v5_realistic_kitchen_2500_1/dataset/data/
```

## 运行演示

### 基本用法

在提供的样本立体图像上运行FoundationStereo：

```bash
python scripts/run_demo.py --left_file ./assets/left.png --right_file ./assets/right.png --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --out_dir ./test_outputs/
```

这将：
1. 加载立体图像
2. 运行模型计算视差图
3. 将视差转换为深度图和点云
4. 将结果保存到指定的输出目录
5. 打开点云的可视化

### 命令行参数

`run_demo.py`脚本接受几个命令行参数：

| 参数 | 描述 | 默认值 |
| -------- | ----------- | ------- |
| `--left_file` | 左立体图像的路径 | `./assets/left.png` |
| `--right_file` | 右立体图像的路径 | `./assets/right.png` |
| `--intrinsic_file` | 相机内参文件的路径 | `./assets/K.txt` |
| `--ckpt_dir` | 预训练模型的路径 | `./pretrained_models/23-51-11/model_best_bp2.pth` |
| `--out_dir` | 保存结果的目录 | `./output/` |
| `--scale` | 调整输入图像大小的比例因子（必须≤1） | `1` |
| `--hiera` | 为高分辨率图像启用分层推理 | `0` |
| `--valid_iters` | 细化迭代次数 | `32` |
| `--get_pc` | 生成点云输出 | `1` |
| `--denoise_cloud` | 对点云应用去噪 | `1` |
| `--denoise_nb_points` | 用于半径离群值去除的点数 | `30` |
| `--denoise_radius` | 用于离群值去除的半径 | `0.03` |

### 输出文件

演示脚本生成几个输出文件：

- `vis.png`：输入图像和视差图的可视化
- `depth_meter.npy`：以米为单位的深度图
- `cloud.ply`：3D点云
- `cloud_denoise.ply`：去噪后的3D点云（如果启用了`--denoise_cloud`）

## 高级用法

### 分层推理

对于高分辨率图像（>1000px），您可以启用分层推理以获得更好的性能：

```bash
python scripts/run_demo.py --left_file ./assets/left.png --right_file ./assets/right.png --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --out_dir ./test_outputs/ --hiera 1
```

这将：
1. 下采样输入图像
2. 在下采样图像上运行推理
3. 上采样结果视差图
4. 使用上采样视差作为全分辨率推理的初始化

### 更快的推理

为了更快的推理，您可以：
1. 使用`--scale`参数减少输入图像分辨率
2. 使用`--valid_iters`参数减少细化迭代次数

```bash
python scripts/run_demo.py --left_file ./assets/left.png --right_file ./assets/right.png --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --out_dir ./test_outputs/ --scale 0.5 --valid_iters 16
```

### ONNX/TensorRT推理

为了更快的推理，您可以将模型转换为ONNX和TensorRT：

1. 进行必要的更改以替换flash-attention（参见[此问题](https://github.com/NVlabs/FoundationStereo/issues/13#issuecomment-2708791825)）

2. 转换为ONNX：
```bash
export XFORMERS_DISABLED=1
python scripts/make_onnx.py --save_path ./output/foundation_stereo.onnx --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --height 480 --width 640 --valid_iters 22
```

3. 将ONNX转换为TensorRT：
```bash
trtexec --onnx=./output/foundation_stereo.onnx --saveEngine=./output/foundation_stereo.engine --fp16 --verbose
```

这可以在相同的GPU上提供高达6倍的加速。

## 故障排除

### 常见问题和解决方案

#### Conda安装问题

如果您在Conda安装中遇到问题，请尝试：

```bash
# 更新conda
conda update -n base -c defaults conda

# 使用特定Python版本创建环境
conda create -n foundation_stereo python=3.8
conda activate foundation_stereo

# 手动安装依赖项
pip install -r requirements.txt
```

#### CUDA内存不足

如果遇到CUDA内存不足错误：

1. 使用`--scale`参数减少输入图像分辨率
2. 在模型中启用`low_memory`选项
3. 使用较小的模型（例如，使用`11-33-40`而不是`23-51-11`）

#### Flash注意力问题

如果您的GPU不支持Flash注意力：

1. 进行[Flash注意力](#flash注意力)部分中描述的更改
2. 设置环境变量：`export XFORMERS_DISABLED=1`

#### cuDNN错误：CUDNN_STATUS_NOT_SUPPORTED

此错误可能表示内存不足问题。尝试：

1. 减少图像分辨率
2. 使用内存更大的GPU

#### RealSense集成

有关与RealSense相机一起运行的信息，请参见[此问题](https://github.com/NVlabs/FoundationStereo/issues/26)。

## 性能优化

### 内存使用

要减少内存使用：

1. 使用较小的模型（例如，使用`11-33-40`而不是`23-51-11`）
2. 使用`--scale`参数减少输入图像分辨率
3. 在模型中启用`low_memory`选项

### 推理速度

要提高推理速度：

1. 使用ONNX/TensorRT推理
2. 使用`--valid_iters`参数减少细化迭代次数
3. 使用较小的模型（例如，使用`11-33-40`而不是`23-51-11`）
4. 使用`--scale`参数减少输入图像分辨率

### 质量与速度的权衡

| 设置 | 质量 | 速度 | 内存使用 |
| ------- | ------- | ----- | ------------ |
| 默认（23-51-11，scale=1，valid_iters=32） | 高 | 慢 | 高 |
| 23-51-11，scale=0.5，valid_iters=16 | 中高 | 中 | 中 |
| 11-33-40，scale=0.5，valid_iters=16 | 中 | 快 | 低 |
| TensorRT，11-33-40，scale=0.5，valid_iters=16 | 中 | 非常快 | 低 |

## 使用自定义数据

### 输入要求

为获得最佳结果，您的立体图像应：

- 经过**校正和去畸变**（无鱼眼畸变，水平极线）
- 左图像来自左侧相机（物体在图像中更靠右）
- 为无损压缩的PNG格式
- 具有RGB或灰度色彩空间

### 相机标定

要获取准确的深度和点云信息，您需要提供相机内参：

1. 创建具有以下格式的文本文件：
   - 第1行：展平的1x9内参矩阵（fx, 0, cx, 0, fy, cy, 0, 0, 1）
   - 第2行：基线（左右相机之间的距离），单位为米

示例（`K.txt`）：
```
700 0 320 0 700 240 0 0 1
0.12
```

### 使用多个RGB相机

如果您有两个或更多不在立体设置中的RGB相机：

1. 使用[OpenCV的stereoRectify函数](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6)校正一对图像
2. 将校正后的图像输入FoundationStereo

### 使用RealSense相机

对于RealSense D4XX系列相机：

1. 从左右相机捕获立体IR图像
2. 使用RealSense SDK提供的内参和基线
3. 将图像输入FoundationStereo

## 与其他系统集成

### Python API

您可以将FoundationStereo集成到您的Python项目中：

```python
import torch
from core.foundation_stereo import FoundationStereo
from omegaconf import OmegaConf
from core.utils.utils import InputPadder

# 加载配置
cfg = OmegaConf.load('./pretrained_models/23-51-11/cfg.yaml')
cfg['vit_size'] = 'vitl'
cfg['valid_iters'] = 32

# 初始化模型
model = FoundationStereo(cfg)
ckpt = torch.load('./pretrained_models/23-51-11/model_best_bp2.pth')
model.load_state_dict(ckpt['model'])
model.cuda()
model.eval()

# 处理图像
def process_stereo_pair(left_img, right_img):
    """
    处理立体对并返回视差图。
    
    参数：
        left_img: 作为numpy数组的左图像 (H, W, 3)
        right_img: 作为numpy数组的右图像 (H, W, 3)
        
    返回：
        作为numpy数组的视差图 (H, W)
    """
    # 转换为torch张量
    left_tensor = torch.from_numpy(left_img).cuda().float()[None].permute(0, 3, 1, 2)
    right_tensor = torch.from_numpy(right_img).cuda().float()[None].permute(0, 3, 1, 2)
    
    # 填充图像
    padder = InputPadder(left_tensor.shape, divis_by=32, force_square=False)
    left_padded, right_padded = padder.pad(left_tensor, right_tensor)
    
    # 运行推理
    with torch.cuda.amp.autocast(True):
        disp = model.forward(left_padded, right_padded, iters=cfg['valid_iters'], test_mode=True)
    
    # 去除填充并转换为numpy
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(left_img.shape[0], left_img.shape[1])
    
    return disp
```

### ROS集成

您可以将FoundationStereo与ROS（机器人操作系统）集成：

1. 创建订阅立体图像主题的ROS节点
2. 使用FoundationStereo处理图像
3. 发布视差图和点云

示例ROS节点（简化）：

```python
#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import torch
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from core.foundation_stereo import FoundationStereo
from omegaconf import OmegaConf

class FoundationStereoNode:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('foundation_stereo_node')
        
        # 初始化模型
        cfg = OmegaConf.load('./pretrained_models/23-51-11/cfg.yaml')
        cfg['vit_size'] = 'vitl'
        cfg['valid_iters'] = 32
        
        self.model = FoundationStereo(cfg)
        ckpt = torch.load('./pretrained_models/23-51-11/model_best_bp2.pth')
        self.model.load_state_dict(ckpt['model'])
        self.model.cuda()
        self.model.eval()
        
        # 初始化桥接
        self.bridge = CvBridge()
        
        # 订阅立体图像
        self.left_sub = rospy.Subscriber('/stereo/left/image_rect', Image, self.left_callback)
        self.right_sub = rospy.Subscriber('/stereo/right/image_rect', Image, self.right_callback)
        
        # 发布者
        self.disp_pub = rospy.Publisher('/stereo/disparity', Image, queue_size=1)
        self.pc_pub = rospy.Publisher('/stereo/points', PointCloud2, queue_size=1)
        
        # 图像缓冲区
        self.left_img = None
        self.right_img = None
        
    def left_callback(self, msg):
        self.left_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.process_stereo()
        
    def right_callback(self, msg):
        self.right_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.process_stereo()
        
    def process_stereo(self):
        if self.left_img is None or self.right_img is None:
            return
        
        # 处理立体对
        # ... (类似于Python API示例)
        
        # 发布结果
        # ... (将视差转换为ROS消息并发布)

if __name__ == '__main__':
    node = FoundationStereoNode()
    rospy.spin()
```

### Docker集成

为了更容易部署，您可以使用Docker：

```dockerfile
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# 安装依赖项
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libgl1-mesa-glx \
    libglib2.0-0

# 克隆仓库
WORKDIR /app
RUN git clone https://github.com/NVlabs/FoundationStereo.git
WORKDIR /app/FoundationStereo

# 安装Python依赖项
RUN pip3 install -r requirements.txt

# 下载预训练模型
# ... (添加下载模型的命令)

# 设置入口点
ENTRYPOINT ["python3", "scripts/run_demo.py"]
```

构建并运行Docker容器：

```bash
# 构建容器
docker build -t foundation_stereo .

# 使用GPU支持运行容器
docker run --gpus all -v /path/to/your/data:/data foundation_stereo \
    --left_file /data/left.png \
    --right_file /data/right.png \
    --out_dir /data/output