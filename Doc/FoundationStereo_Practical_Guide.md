# FoundationStereo: Practical Guide

This document provides a practical guide for using the FoundationStereo project, including installation, running demos, and troubleshooting common issues.

## Table of Contents

1. [Installation](#installation)
2. [Dataset and Model Weights](#dataset-and-model-weights)
3. [Running the Demo](#running-the-demo)
4. [Advanced Usage](#advanced-usage)
5. [Troubleshooting](#troubleshooting)
6. [Performance Optimization](#performance-optimization)
7. [Working with Custom Data](#working-with-custom-data)
8. [Integration with Other Systems](#integration-with-other-systems)

## Installation

### Prerequisites

- CUDA-capable GPU (recommended: NVIDIA RTX 3090 or better)
- CUDA Toolkit 11.3 or later
- Python 3.8 or later

### Environment Setup

The easiest way to set up the environment is using Conda:

```bash
# Clone the repository
git clone https://github.com/NVlabs/FoundationStereo.git
cd FoundationStereo

# Create and activate the conda environment
conda env create -f environment.yml
conda activate foundation_stereo
```

If you encounter issues with the Conda installation, you can try installing the dependencies manually using pip:

```bash
pip install -r requirements.txt
```

### Common Installation Issues

#### CUDA Compatibility

If you encounter CUDA compatibility issues, make sure your CUDA toolkit version is compatible with the PyTorch version specified in the requirements. You may need to install a specific version of PyTorch:

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Flash Attention

FoundationStereo uses Flash Attention for efficient transformer operations. If you encounter issues with Flash Attention, you can disable it by making the following change:

```python
# In core/submodule.py, replace the flash_attn_func call with a standard attention implementation
# Original:
attn_output = flash_attn_func(Q, K, V, window_size=window_size)

# Modified:
Q = Q.transpose(1, 2)  # (B, nh, L, d)
K = K.transpose(1, 2)  # (B, nh, L, d)
V = V.transpose(1, 2)  # (B, nh, L, d)
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, L, L)
attn_weights = F.softmax(scores, dim=-1)
attn_output = torch.matmul(attn_weights, V)  # (B, nh, L, d)
attn_output = attn_output.transpose(1, 2).reshape(B, L, -1)  # (B, L, C)
```

## Dataset and Model Weights

### Model Weights

Download the foundation model for zero-shot inference from [here](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing). Put the entire folder (e.g., `23-51-11`) under `./pretrained_models/`.

Available models:

| Model | Description |
| ----- | ----------- |
| 23-51-11 | Best performing model for general use, based on Vit-large |
| 11-33-40 | Slightly lower accuracy but faster inference, based on Vit-small |

### FSD Dataset

The Foundation Stereo Dataset (FSD) is a large-scale synthetic dataset with 1M stereo pairs. You can download the entire dataset [here](https://drive.google.com/drive/folders/1YdC2a0_KTZ9xix_HyqNMPCrClpm0-XFU?usp=sharing) (>1TB).

For a quick peek, you can download a small [sample data](https://drive.google.com/file/d/1dJwK5x8xsaCazz5xPGJ2OKFIWrd9rQT5/view?usp=drive_link) (3GB).

To visualize the dataset:

```bash
python scripts/vis_dataset.py --dataset_path ./DATA/sample/manipulation_v5_realistic_kitchen_2500_1/dataset/data/
```

## Running the Demo

### Basic Usage

To run FoundationStereo on the provided sample stereo images:

```bash
python scripts/run_demo.py --left_file ./assets/left.png --right_file ./assets/right.png --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --out_dir ./test_outputs/
```

This will:
1. Load the stereo images
2. Run the model to compute a disparity map
3. Convert the disparity to a depth map and point cloud
4. Save the results to the specified output directory
5. Open a visualization of the point cloud

### Command-Line Arguments

The `run_demo.py` script accepts several command-line arguments:

| Argument | Description | Default |
| -------- | ----------- | ------- |
| `--left_file` | Path to the left stereo image | `./assets/left.png` |
| `--right_file` | Path to the right stereo image | `./assets/right.png` |
| `--intrinsic_file` | Path to the camera intrinsics file | `./assets/K.txt` |
| `--ckpt_dir` | Path to the pretrained model | `./pretrained_models/23-51-11/model_best_bp2.pth` |
| `--out_dir` | Directory to save results | `./output/` |
| `--scale` | Scale factor to resize input images (must be ≤1) | `1` |
| `--hiera` | Enable hierarchical inference for high-resolution images | `0` |
| `--valid_iters` | Number of refinement iterations | `32` |
| `--get_pc` | Generate point cloud output | `1` |
| `--denoise_cloud` | Apply denoising to the point cloud | `1` |
| `--denoise_nb_points` | Number of points for radius outlier removal | `30` |
| `--denoise_radius` | Radius for outlier removal | `0.03` |

### Output Files

The demo script generates several output files:

- `vis.png`: Visualization of the input image and disparity map
- `depth_meter.npy`: Depth map in meters
- `cloud.ply`: 3D point cloud
- `cloud_denoise.ply`: Denoised 3D point cloud (if `--denoise_cloud` is enabled)

## Advanced Usage

### Hierarchical Inference

For high-resolution images (>1000px), you can enable hierarchical inference for better performance:

```bash
python scripts/run_demo.py --left_file ./assets/left.png --right_file ./assets/right.png --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --out_dir ./test_outputs/ --hiera 1
```

This will:
1. Downsample the input images
2. Run inference on the downsampled images
3. Upsample the resulting disparity map
4. Use the upsampled disparity as initialization for full-resolution inference

### Faster Inference

For faster inference, you can:
1. Reduce the input image resolution using the `--scale` parameter
2. Reduce the number of refinement iterations using the `--valid_iters` parameter

```bash
python scripts/run_demo.py --left_file ./assets/left.png --right_file ./assets/right.png --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --out_dir ./test_outputs/ --scale 0.5 --valid_iters 16
```

### ONNX/TensorRT Inference

For even faster inference, you can convert the model to ONNX and TensorRT:

1. Make the necessary changes to replace flash-attention (see [this issue](https://github.com/NVlabs/FoundationStereo/issues/13#issuecomment-2708791825))

2. Convert to ONNX:
```bash
export XFORMERS_DISABLED=1
python scripts/make_onnx.py --save_path ./output/foundation_stereo.onnx --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --height 480 --width 640 --valid_iters 22
```

3. Convert ONNX to TensorRT:
```bash
trtexec --onnx=./output/foundation_stereo.onnx --saveEngine=./output/foundation_stereo.engine --fp16 --verbose
```

This can provide up to 6X speedup on the same GPU.

## Troubleshooting

### Common Issues and Solutions

#### Conda Installation Issues

If you encounter issues with the Conda installation, try:

```bash
# Update conda
conda update -n base -c defaults conda

# Create environment with specific Python version
conda create -n foundation_stereo python=3.8
conda activate foundation_stereo

# Install dependencies manually
pip install -r requirements.txt
```

#### CUDA Out of Memory

If you encounter CUDA out of memory errors:

1. Reduce the input image resolution using the `--scale` parameter
2. Enable the `low_memory` option in the model
3. Use a smaller model (e.g., `11-33-40` instead of `23-51-11`)

#### Flash Attention Issues

If your GPU doesn't support Flash Attention:

1. Make the changes described in the [Flash Attention](#flash-attention) section
2. Set the environment variable: `export XFORMERS_DISABLED=1`

#### cuDNN Error: CUDNN_STATUS_NOT_SUPPORTED

This error may indicate an out-of-memory issue. Try:

1. Reducing your image resolution
2. Using a GPU with more memory

#### RealSense Integration

For running with RealSense cameras, see [this issue](https://github.com/NVlabs/FoundationStereo/issues/26).

## Performance Optimization

### Memory Usage

To reduce memory usage:

1. Use a smaller model (e.g., `11-33-40` instead of `23-51-11`)
2. Reduce the input image resolution using the `--scale` parameter
3. Enable the `low_memory` option in the model

### Inference Speed

To improve inference speed:

1. Use ONNX/TensorRT inference
2. Reduce the number of refinement iterations using the `--valid_iters` parameter
3. Use a smaller model (e.g., `11-33-40` instead of `23-51-11`)
4. Reduce the input image resolution using the `--scale` parameter

### Quality vs. Speed Tradeoffs

| Setting | Quality | Speed | Memory Usage |
| ------- | ------- | ----- | ------------ |
| Default (23-51-11, scale=1, valid_iters=32) | High | Slow | High |
| 23-51-11, scale=0.5, valid_iters=16 | Medium-High | Medium | Medium |
| 11-33-40, scale=0.5, valid_iters=16 | Medium | Fast | Low |
| TensorRT, 11-33-40, scale=0.5, valid_iters=16 | Medium | Very Fast | Low |

## Working with Custom Data

### Input Requirements

For best results, your stereo images should:

- Be **rectified and undistorted** (no fisheye distortion, horizontal epipolar lines)
- Have the left image from the left-side camera (objects appear more to the right)
- Be in PNG format with no lossy compression
- Have RGB or grayscale color space

### Camera Calibration

To get accurate depth and point cloud information, you need to provide camera intrinsics:

1. Create a text file with the following format:
   - Line 1: Flattened 1x9 intrinsic matrix (fx, 0, cx, 0, fy, cy, 0, 0, 1)
   - Line 2: Baseline (distance between left and right cameras) in meters

Example (`K.txt`):
```
700 0 320 0 700 240 0 0 1
0.12
```

### Using Multiple RGB Cameras

If you have two or more RGB cameras that are not in a stereo setup:

1. Rectify a pair of images using [OpenCV's stereoRectify function](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6)
2. Feed the rectified images into FoundationStereo

### Using RealSense Cameras

For RealSense D4XX series cameras:

1. Capture stereo IR images from the left and right cameras
2. Use the provided intrinsics and baseline from the RealSense SDK
3. Feed the images into FoundationStereo

## Integration with Other Systems

### Python API

You can integrate FoundationStereo into your Python projects:

```python
import torch
from core.foundation_stereo import FoundationStereo
from omegaconf import OmegaConf
from core.utils.utils import InputPadder

# Load configuration
cfg = OmegaConf.load('./pretrained_models/23-51-11/cfg.yaml')
cfg['vit_size'] = 'vitl'
cfg['valid_iters'] = 32

# Initialize model
model = FoundationStereo(cfg)
ckpt = torch.load('./pretrained_models/23-51-11/model_best_bp2.pth')
model.load_state_dict(ckpt['model'])
model.cuda()
model.eval()

# Process images
def process_stereo_pair(left_img, right_img):
    """
    Process a stereo pair and return the disparity map.
    
    Args:
        left_img: Left image as numpy array (H, W, 3)
        right_img: Right image as numpy array (H, W, 3)
        
    Returns:
        Disparity map as numpy array (H, W)
    """
    # Convert to torch tensors
    left_tensor = torch.from_numpy(left_img).cuda().float()[None].permute(0, 3, 1, 2)
    right_tensor = torch.from_numpy(right_img).cuda().float()[None].permute(0, 3, 1, 2)
    
    # Pad images
    padder = InputPadder(left_tensor.shape, divis_by=32, force_square=False)
    left_padded, right_padded = padder.pad(left_tensor, right_tensor)
    
    # Run inference
    with torch.cuda.amp.autocast(True):
        disp = model.forward(left_padded, right_padded, iters=cfg['valid_iters'], test_mode=True)
    
    # Unpad and convert to numpy
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(left_img.shape[0], left_img.shape[1])
    
    return disp
```

### ROS Integration

You can integrate FoundationStereo with ROS (Robot Operating System):

1. Create a ROS node that subscribes to stereo image topics
2. Process the images using FoundationStereo
3. Publish the disparity map and point cloud

Example ROS node (simplified):

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
        # Initialize ROS node
        rospy.init_node('foundation_stereo_node')
        
        # Initialize model
        cfg = OmegaConf.load('./pretrained_models/23-51-11/cfg.yaml')
        cfg['vit_size'] = 'vitl'
        cfg['valid_iters'] = 32
        
        self.model = FoundationStereo(cfg)
        ckpt = torch.load('./pretrained_models/23-51-11/model_best_bp2.pth')
        self.model.load_state_dict(ckpt['model'])
        self.model.cuda()
        self.model.eval()
        
        # Initialize bridge
        self.bridge = CvBridge()
        
        # Subscribe to stereo images
        self.left_sub = rospy.Subscriber('/stereo/left/image_rect', Image, self.left_callback)
        self.right_sub = rospy.Subscriber('/stereo/right/image_rect', Image, self.right_callback)
        
        # Publishers
        self.disp_pub = rospy.Publisher('/stereo/disparity', Image, queue_size=1)
        self.pc_pub = rospy.Publisher('/stereo/points', PointCloud2, queue_size=1)
        
        # Image buffers
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
        
        # Process stereo pair
        # ... (similar to the Python API example)
        
        # Publish results
        # ... (convert disparity to ROS messages and publish)

if __name__ == '__main__':
    node = FoundationStereoNode()
    rospy.spin()
```

### Docker Integration

For easier deployment, you can use Docker:

```dockerfile
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libgl1-mesa-glx \
    libglib2.0-0

# Clone repository
WORKDIR /app
RUN git clone https://github.com/NVlabs/FoundationStereo.git
WORKDIR /app/FoundationStereo

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Download pretrained models
# ... (add commands to download models)

# Set entrypoint
ENTRYPOINT ["python3", "scripts/run_demo.py"]
```

Build and run the Docker container:

```bash
# Build the container
docker build -t foundation_stereo .

# Run the container with GPU support
docker run --gpus all -v /path/to/your/data:/data foundation_stereo \
    --left_file /data/left.png \
    --right_file /data/right.png \
    --out_dir /data/output