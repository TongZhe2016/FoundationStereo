# 双目图像相机参数估计与ERP全景图像转换

本项目提供了一套工具，用于从双目图像对中估计相机内参和baseline，然后将pinhole相机数据转换为ERP（Equirectangular Projection，等距矩形投影）全景图像。

## 功能特点

- 使用VGGT（Visual Geometry Grounded Transformer）从双目图像对中估计相机内参和baseline
- 将估计的参数保存为标准格式（与assets/K.txt相同）
- 使用估计的参数将pinhole相机数据转换为ERP全景图像
- 支持批量处理多对图像（高级版本）
- 可视化相机参数的分布（高级版本）

## 环境要求

- Python 3.7+
- PyTorch 1.7+
- CUDA（推荐，用于加速）
- 其他依赖项：numpy, opencv-python, PIL, matplotlib, tqdm

## 安装依赖

```bash
pip install torch numpy opencv-python pillow matplotlib tqdm
```

## 数据集结构

数据集应按以下结构组织：

```
data/
├── left/
│   ├── rgb/         # 左相机RGB图像（jpg格式）
│   └── disparity/   # 视差图（png格式）
└── right/
    └── rgb/         # 右相机RGB图像（jpg格式）
```

## 使用方法

### 基本版本

```bash
python estimate_camera_params.py --data_dir data --output_dir erp_output --params_output estimated_K.txt
```

可选参数：
- `--data_dir`：包含pinhole相机数据的目录（默认：data）
- `--output_dir`：保存ERP数据的目录（默认：erp_output）
- `--params_output`：保存估计的相机参数的文件路径（默认：estimated_K.txt）
- `--left_image`：用于估计相机参数的左图像路径（如果不指定，将使用data/left/rgb中的第一个图像）
- `--right_image`：用于估计相机参数的右图像路径（如果不指定，将使用data/right/rgb中的第一个图像）

### 高级版本

```bash
python estimate_camera_params_advanced.py --data_dir data --output_dir erp_output --params_output estimated_K.txt --num_pairs 5 --visualize
```

可选参数：
- `--data_dir`：包含pinhole相机数据的目录（默认：data）
- `--output_dir`：保存ERP数据的目录（默认：erp_output）
- `--params_output`：保存估计的相机参数的文件路径（默认：estimated_K.txt）
- `--num_pairs`：用于估计相机参数的图像对数量（默认：5）
- `--num_samples`：要转换为ERP的图像数量（如果不指定，将处理所有图像）
- `--visualize`：是否可视化相机参数的分布（默认：False）

## 输出文件

- `estimated_K.txt`：估计的相机内参和baseline，格式与assets/K.txt相同
- `erp_output/`：包含转换后的ERP全景图像和深度图
  - `*_erp_rgb.png`：ERP全景RGB图像
  - `*_erp_depth.png`：ERP全景深度图
  - `*_erp_mask.png`：ERP全景有效区域掩码
  - `*_erp_params.json`：ERP转换参数
- `erp_output/camera_params_distribution.png`：相机参数分布图（仅高级版本）

## 相机参数格式

估计的相机参数将保存为与assets/K.txt相同的格式：

```
fx 0.0 cx 0.0 fy cy 0.0 0.0 1.0
baseline
```

其中：
- 第一行是相机内参矩阵K的元素（3x3矩阵按行排列）
- 第二行是baseline值

## 注意事项

1. 确保VGGT模型可以正常加载（需要网络连接下载预训练模型）
2. 处理大量图像可能需要较长时间，特别是在CPU上运行时
3. 视差图应与RGB图像具有相同的文件名（扩展名不同）
4. 如果遇到内存不足的问题，可以减少`num_pairs`或`num_samples`的值

## 参考

- VGGT（Visual Geometry Grounded Transformer）：https://github.com/facebookresearch/vggt
- DepthAnyCamera：https://github.com/depth-anything/depth-anything-camera