# FoundationStereo Training Pipeline

This document describes the comprehensive training pipeline for FoundationStereo based on MoGe's architecture.

## Overview

The training pipeline includes:
- **Main Training Script**: `scripts/train.py` - Distributed training with accelerate
- **DataLoader**: `train/dataloader.py` - Stereo pair loading with augmentations
- **Loss Functions**: `train/losses.py` - Disparity-specific losses
- **Training Utilities**: `train/utils.py` - Optimizer and scheduler setup
- **Configuration**: `configs/train/stereo_v1.json` - Training parameters

## Quick Start

### 1. Prepare Data

Organize your stereo dataset in the following structure:
```
data/
├── left/
│   ├── rgb/
│   │   ├── 0000.jpg
│   │   ├── 0001.jpg
│   │   └── ...
│   └── disparity/
│       ├── 0000.png  # Encoded disparity using depth_uint8_decoding format
│       ├── 0001.png
│       └── ...
└── right/
    └── rgb/
        ├── 0000.jpg
        ├── 0001.jpg
        └── ...
```

### 2. Install Dependencies

```bash
pip install torch torchvision accelerate click tqdm mlflow opencv-python pillow numpy
```

### 3. Run Training

```bash
# Basic training
python scripts/train.py --config configs/train/stereo_v1.json

# Distributed training with custom settings
accelerate launch scripts/train.py \
    --config configs/train/stereo_v1.json \
    --workspace workspace/stereo_experiment \
    --batch_size_forward 4 \
    --enable_mixed_precision true \
    --num_iterations 100000
```

## Configuration

### Model Configuration

```json
{
  "model": {
    "max_disp": 192,                    // Maximum disparity range
    "feature_type": "dinov2",           // Feature extractor type
    "correlation_implementation": "alt", // Correlation volume implementation
    "n_gru_layers": 3,                  // Number of GRU update layers
    "hidden_dims": [128, 128, 128]      // Hidden dimensions for updates
  }
}
```

### Data Configuration

```json
{
  "data": {
    "datasets": [
      {
        "name": "foundation_stereo_train",
        "path": "./data",
        "weight": 1.0,
        "label_type": "stereo",
        "image_augmentation": ["jittering"]
      }
    ],
    "image_sizes": [[512, 384], [640, 480]],  // Variable input sizes
    "max_disparity": 192,
    "stereo_augmentation": true
  }
}
```

### Loss Configuration

```json
{
  "loss": {
    "stereo": {
      "disparity_l1": {
        "function": "disparity_l1_loss",
        "weight": 1.0,
        "params": {"max_disparity": 192.0}
      },
      "multi_scale": {
        "function": "multi_scale_loss",
        "weight": 0.3,
        "params": {
          "weights": [0.5, 0.7, 1.0],
          "loss_type": "smooth_l1"
        }
      }
    }
  }
}
```

## Features

### Data Loading
- **Stereo-consistent augmentations**: Horizontal/vertical flips that maintain stereo geometry
- **Variable input sizes**: Support for different image resolutions during training
- **Disparity encoding**: Uses FoundationStereo's `depth_uint8_decoding` method
- **Multi-threaded loading**: Efficient data pipeline with configurable workers

### Loss Functions
- **L1 Loss**: Basic disparity regression loss
- **Smooth L1 Loss**: Robust loss for outliers
- **Multi-scale Loss**: Pyramid supervision at different resolutions
- **EPE (End-Point Error)**: Standard stereo evaluation metric
- **Gradient Loss**: Smoothness regularization

### Training Features
- **Distributed Training**: Multi-GPU support via Accelerate
- **Mixed Precision**: FP16 training for memory efficiency
- **Gradient Checkpointing**: Reduce memory usage
- **EMA (Exponential Moving Average)**: Model weight averaging
- **MLFlow Integration**: Experiment tracking and logging
- **Checkpoint Management**: Automatic saving and resuming

### Augmentations
- **Color Jittering**: Brightness, contrast, saturation, hue adjustments
- **Stereo Flips**: Horizontal flip with left-right image swapping
- **Geometric Augmentations**: Vertical flips maintaining stereo consistency

## Command Line Options

```bash
python scripts/train.py [OPTIONS]

Options:
  --config TEXT                   Path to config file [default: configs/train/stereo_v1.json]
  --workspace TEXT               Workspace directory [default: workspace/stereo_train]
  --checkpoint TEXT              Checkpoint to load (path, 'latest', or step number)
  --batch_size_forward INTEGER   Batch size per device [default: 4]
  --gradient_accumulation_steps INTEGER  Gradient accumulation [default: 1]
  --enable_gradient_checkpointing BOOLEAN  Use gradient checkpointing [default: True]
  --enable_mixed_precision BOOLEAN  Use FP16 training [default: False]
  --enable_ema BOOLEAN           Use EMA model [default: True]
  --num_iterations INTEGER       Total training iterations [default: 100000]
  --save_every INTEGER           Save checkpoint interval [default: 5000]
  --log_every INTEGER            Logging interval [default: 500]
  --vis_every INTEGER            Visualization interval [default: 0]
  --enable_mlflow BOOLEAN        Enable MLFlow logging [default: True]
  --seed INTEGER                 Random seed [default: 0]
```

## Monitoring

### MLFlow Integration
The training script automatically logs:
- Loss values (L1, Smooth L1, Multi-scale)
- Evaluation metrics (EPE, D1 error, D3 error)
- Learning rates and training statistics
- Model checkpoints and configurations

### Visualization
When `--vis_every > 0`, the script saves:
- Input stereo pairs
- Ground truth disparity visualizations
- Predicted disparity visualizations
- Training progress images

## Checkpointing

The training script saves:
- **Model checkpoints**: `{step:08d}.pt` - Model weights only
- **Optimizer checkpoints**: `{step:08d}_optimizer.pt` - Optimizer and scheduler state
- **EMA checkpoints**: `{step:08d}_ema.pt` - EMA model weights
- **Latest checkpoint**: `latest.pt` - Quick resume pointer

### Resuming Training

```bash
# Resume from latest checkpoint
python scripts/train.py --checkpoint latest

# Resume from specific step
python scripts/train.py --checkpoint 50000

# Resume from specific file
python scripts/train.py --checkpoint /path/to/checkpoint.pt
```

## Evaluation Metrics

The training pipeline computes standard stereo metrics:
- **EPE (End-Point Error)**: Mean absolute disparity error
- **D1 Error**: Percentage of pixels with error > 3 pixels
- **D3 Error**: Percentage of pixels with error > 1 pixel
- **RMSE**: Root mean square error

## Performance Tips

1. **Memory Optimization**:
   - Use `--enable_mixed_precision true` for FP16 training
   - Enable `--enable_gradient_checkpointing true`
   - Adjust `--batch_size_forward` based on GPU memory

2. **Speed Optimization**:
   - Use multiple GPUs with `accelerate launch`
   - Increase `--gradient_accumulation_steps` for larger effective batch size
   - Use SSD storage for faster data loading

3. **Quality Optimization**:
   - Enable EMA with `--enable_ema true`
   - Use multi-scale loss for better convergence
   - Apply appropriate data augmentations

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `batch_size_forward`
   - Enable mixed precision
   - Enable gradient checkpointing

2. **Data Loading Errors**:
   - Check data directory structure
   - Verify disparity encoding format
   - Ensure left/right/disparity files match

3. **Training Instability**:
   - Reduce learning rate
   - Check gradient clipping
   - Verify loss function weights

### Debug Mode

For debugging, use a minimal configuration:
```bash
python scripts/train.py \
    --config configs/train/stereo_v1.json \
    --batch_size_forward 1 \
    --num_iterations 100 \
    --log_every 10 \
    --save_every 50
```

## Integration with FoundationStereo

The training pipeline is designed to work seamlessly with FoundationStereo:
- Uses existing model architecture from `core/foundation_stereo.py`
- Leverages disparity utilities from `Utils.py`
- Maintains compatibility with existing data formats
- Supports variable input sizes via `InputPadder`

## Contributing

When extending the training pipeline:
1. Add new loss functions to `train/losses.py`
2. Extend data augmentations in `train/dataloader.py`
3. Add training utilities to `train/utils.py`
4. Update configuration schema in `configs/train/`
5. Document changes in this README