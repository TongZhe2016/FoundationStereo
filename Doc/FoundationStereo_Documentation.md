# FoundationStereo: Zero-Shot Stereo Matching

## Overview

FoundationStereo is a foundation model for stereo depth estimation designed to achieve strong zero-shot generalization. This document provides a detailed explanation of the project, its architecture, workflow, and components.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Core Architecture](#core-architecture)
4. [Feature Extraction](#feature-extraction)
5. [Geometry Encoding](#geometry-encoding)
6. [Disparity Update Mechanism](#disparity-update-mechanism)
7. [Workflow](#workflow)
8. [Usage](#usage)
9. [Performance](#performance)

## Introduction

FoundationStereo is a zero-shot stereo matching model that takes a pair of stereo images as input and outputs a dense disparity map, which can be converted to a metric-scale depth map or 3D point cloud. The model was developed by NVIDIA researchers and accepted by CVPR 2025 as an Oral presentation.

The key innovations of FoundationStereo include:
- A large-scale (1M stereo pairs) synthetic training dataset with high photorealism
- A side-tuning feature backbone that adapts rich monocular priors from vision foundation models
- Long-range context reasoning for effective cost volume filtering
- Strong robustness and accuracy across domains without fine-tuning

The model has achieved state-of-the-art performance on the Middlebury and ETH3D leaderboards, establishing a new standard in zero-shot stereo depth estimation.

## Project Structure

The project is organized into several key directories:

- `core/`: Contains the main implementation of the FoundationStereo model
  - `foundation_stereo.py`: Main model implementation
  - `extractor.py`: Feature extraction modules
  - `geometry.py`: Geometry encoding for stereo matching
  - `update.py`: Disparity update mechanisms
  - `submodule.py`: Basic building blocks and utility functions
  - `utils/`: Utility functions
- `depth_anything/`: Integration with Depth Anything model for monocular depth priors
- `dinov2/`: Integration with DINOv2 vision foundation model
- `scripts/`: Scripts for running demos and visualization
  - `run_demo.py`: Main script for running inference on stereo pairs
  - `make_onnx.py`: Script for ONNX/TensorRT conversion
  - `vis_dataset.py`: Script for visualizing the FSD dataset
- `assets/`: Sample stereo images and camera parameters
- `pretrained_models/`: Directory for storing pretrained model weights
- `teaser/`: Sample outputs and visualizations

## Core Architecture

The core of FoundationStereo is implemented in `core/foundation_stereo.py`. The main class `FoundationStereo` inherits from `nn.Module` and implements the stereo matching pipeline.

### Key Components

1. **Feature Extraction**: Uses a combination of CNN-based features and vision foundation models (DINOv2 and Depth Anything)
2. **Cost Volume Construction**: Builds a cost volume using group-wise correlation and concatenation
3. **Cost Volume Filtering**: Uses a 3D hourglass network with attention mechanisms
4. **Disparity Regression**: Initial disparity estimation from the filtered cost volume
5. **Iterative Refinement**: GRU-based iterative updates to refine the disparity map
6. **Upsampling**: Context-aware upsampling to produce the final high-resolution disparity map

### Main Model Class

```python
class FoundationStereo(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, args):
        # Initialize model components
        
    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False, low_memory=False, init_disp=None):
        # Forward pass to compute disparity
        
    def run_hierachical(self, image1, image2, iters=12, test_mode=False, low_memory=False, small_ratio=0.5):
        # Hierarchical inference for high-resolution images
```

## Feature Extraction

Feature extraction is a critical component of FoundationStereo, implemented in `core/extractor.py`. The model uses a hybrid approach that combines:

1. **CNN-based Features**: Using EdgeNext as a backbone
2. **Vision Foundation Models**: Leveraging pre-trained DINOv2 and Depth Anything models

### Feature Class

```python
class Feature(nn.Module):
    def __init__(self, args):
        # Initialize feature extraction components
        
    def forward(self, x):
        # Extract features from input images
```

The feature extraction process:
1. Processes the input images through EdgeNext to get multi-scale features
2. Passes the images through the frozen Depth Anything model to get monocular depth priors
3. Combines these features through a series of deconvolution and concatenation operations
4. Returns a list of multi-scale features and vision transformer features

### DepthAnythingFeature

This class integrates the Depth Anything model for monocular depth priors:

```python
class DepthAnythingFeature(nn.Module):
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }
    
    def __init__(self, encoder='vits'):
        # Initialize Depth Anything model
        
    def forward(self, x):
        # Extract features and depth information
```

## Geometry Encoding

The geometry encoding is implemented in `core/geometry.py`. It creates a combined geometric encoding volume that helps in establishing correspondences between the left and right images.

### Combined_Geo_Encoding_Volume

```python
class Combined_Geo_Encoding_Volume:
    def __init__(self, init_fmap1, init_fmap2, geo_volume, num_levels=2, dx=None):
        # Initialize geometry encoding volume
        
    def __call__(self, disp, coords, low_memory=False):
        # Generate geometry features based on current disparity estimate
        
    @staticmethod
    def corr(fmap1, fmap2):
        # Compute correlation between feature maps
```

This class:
1. Builds a pyramid of correlation volumes at different scales
2. Samples from these volumes based on the current disparity estimate
3. Provides rich geometric features for disparity update

## Disparity Update Mechanism

The disparity update mechanism is implemented in `core/update.py`. It uses a GRU-based approach to iteratively refine the disparity estimate.

### BasicSelectiveMultiUpdateBlock

```python
class BasicSelectiveMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, volume_dim=8):
        # Initialize update block components
        
    def forward(self, net, inp, corr, disp, att):
        # Update disparity estimate
```

This class:
1. Encodes motion features from the current disparity and correlation volume
2. Uses selective GRU units to update the hidden state at multiple scales
3. Predicts a disparity update and a mask for context-aware upsampling

### SelectiveConvGRU

```python
class SelectiveConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, small_kernel_size=1, large_kernel_size=3, patch_size=None):
        # Initialize selective GRU
        
    def forward(self, att, h, *x):
        # Update hidden state based on attention
```

This class implements an attention-guided GRU that selectively applies different kernel sizes based on the attention map.

## Workflow

The overall workflow of FoundationStereo can be summarized as follows:

1. **Input Processing**:
   - Normalize the input stereo images
   - Pad images to be divisible by 32

2. **Feature Extraction**:
   - Extract features from both left and right images using the Feature class
   - Get multi-scale features and vision transformer features

3. **Cost Volume Construction**:
   - Build a group-wise correlation volume
   - Build a concatenation volume
   - Combine these volumes and apply initial filtering

4. **Initial Disparity Estimation**:
   - Apply a classifier to the filtered cost volume
   - Perform disparity regression to get an initial disparity map

5. **Context Network**:
   - Process the left image through a context network
   - Generate hidden states and context information for disparity update

6. **Iterative Refinement**:
   - For a specified number of iterations:
     - Generate geometric features based on current disparity
     - Update the hidden states using the SelectiveConvGRU
     - Predict a disparity update and apply it
     - Upsample the disparity if needed

7. **Final Output**:
   - Return the final disparity map
   - Optionally convert to depth map or point cloud

## Usage

### Basic Usage

To run FoundationStereo on a pair of stereo images:

```bash
python scripts/run_demo.py --left_file ./assets/left.png --right_file ./assets/right.png --ckpt_dir ./pretrained_models/model_best_bp2.pth --out_dir ./test_outputs/
```

### Key Parameters

- `--left_file`, `--right_file`: Paths to the left and right stereo images
- `--intrinsic_file`: Path to the camera intrinsics file
- `--ckpt_dir`: Path to the pretrained model
- `--out_dir`: Directory to save results
- `--scale`: Scale factor to resize input images (must be ≤1)
- `--hiera`: Enable hierarchical inference for high-resolution images
- `--valid_iters`: Number of refinement iterations
- `--get_pc`: Generate point cloud output
- `--denoise_cloud`: Apply denoising to the point cloud

### ONNX/TensorRT Inference

For faster inference, the model can be converted to ONNX and TensorRT:

```bash
# Convert to ONNX
export XFORMERS_DISABLED=1
python scripts/make_onnx.py --save_path ./output/foundation_stereo.onnx --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --height 480 --width 640 --valid_iters 22

# Convert ONNX to TensorRT
trtexec --onnx=./output/foundation_stereo.onnx --saveEngine=./output/foundation_stereo.engine --fp16 --verbose
```

## Performance

FoundationStereo has achieved state-of-the-art performance on several benchmarks:

1. **Middlebury Leaderboard**: 1st place
2. **ETH3D Leaderboard**: 1st place

The model outperforms existing approaches in zero-shot stereo matching tasks across different scenes, including:
- Indoor environments
- Outdoor scenes
- Complex textures and lighting conditions

The key to this performance is the model's ability to generalize to unseen data without fine-tuning, leveraging the rich priors from vision foundation models and the large-scale synthetic training dataset.

## Requirements and Dependencies

The main dependencies include:
- PyTorch
- torchvision
- timm
- flash-attention
- open3d (for point cloud visualization)
- CUDA-capable GPU

A complete list of dependencies can be found in the `environment.yml` and `requirements.txt` files.

## Tips for Best Results

- Input images should be rectified and undistorted
- Do not swap left and right images
- Use PNG files with no lossy compression
- For high-resolution images (>1000px), use hierarchical inference
- For faster inference, reduce input resolution and refinement iterations