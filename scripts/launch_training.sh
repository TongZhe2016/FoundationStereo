#!/bin/bash

# FoundationStereo Training Launch Script
# This script demonstrates how to launch training with different configurations

set -e

echo "FoundationStereo Training Pipeline"
echo "=================================="

# Check if accelerate is available
if ! command -v accelerate &> /dev/null; then
    echo "Error: accelerate is not installed. Please install it with:"
    echo "pip install accelerate"
    exit 1
fi

# Default values
CONFIG="configs/train/stereo_v1.json"
WORKSPACE="workspace/stereo_training"
BATCH_SIZE=4
MIXED_PRECISION=false
NUM_ITERATIONS=100000
ENABLE_MLFLOW=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --mixed_precision)
            MIXED_PRECISION="$2"
            shift 2
            ;;
        --num_iterations)
            NUM_ITERATIONS="$2"
            shift 2
            ;;
        --disable_mlflow)
            ENABLE_MLFLOW=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config PATH              Training config file (default: $CONFIG)"
            echo "  --workspace PATH           Workspace directory (default: $WORKSPACE)"
            echo "  --batch_size INT           Batch size per device (default: $BATCH_SIZE)"
            echo "  --mixed_precision BOOL     Enable FP16 training (default: $MIXED_PRECISION)"
            echo "  --num_iterations INT       Number of training iterations (default: $NUM_ITERATIONS)"
            echo "  --disable_mlflow           Disable MLFlow logging"
            echo "  --help                     Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Basic training"
            echo "  $0"
            echo ""
            echo "  # Training with mixed precision"
            echo "  $0 --mixed_precision true --batch_size 8"
            echo ""
            echo "  # Distributed training (run with accelerate)"
            echo "  accelerate launch scripts/train.py --config $CONFIG"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if config file exists
if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Config file not found: $CONFIG"
    echo "Available configs:"
    find configs/train -name "*.json" 2>/dev/null || echo "  No config files found in configs/train/"
    exit 1
fi

# Check if data directory exists
DATA_DIR=$(python -c "import json; print(json.load(open('$CONFIG'))['data']['datasets'][0]['path'])" 2>/dev/null || echo "./data")
if [[ ! -d "$DATA_DIR" ]]; then
    echo "Warning: Data directory not found: $DATA_DIR"
    echo "Please ensure your stereo dataset is properly organized:"
    echo "  $DATA_DIR/"
    echo "  ├── left/"
    echo "  │   ├── rgb/        # Left RGB images (.jpg)"
    echo "  │   └── disparity/  # Disparity maps (.png)"
    echo "  └── right/"
    echo "      └── rgb/        # Right RGB images (.jpg)"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create workspace directory
mkdir -p "$WORKSPACE"

echo "Training Configuration:"
echo "  Config: $CONFIG"
echo "  Workspace: $WORKSPACE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Mixed Precision: $MIXED_PRECISION"
echo "  Iterations: $NUM_ITERATIONS"
echo "  MLFlow: $ENABLE_MLFLOW"
echo ""

# Check if we should use distributed training
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [[ $GPU_COUNT -gt 1 ]]; then
        echo "Detected $GPU_COUNT GPUs. Consider using distributed training:"
        echo "  accelerate config  # Configure accelerate"
        echo "  accelerate launch scripts/train.py [OPTIONS]"
        echo ""
    fi
fi

# Launch training
echo "Starting training..."
echo "Command: python scripts/train.py --config $CONFIG --workspace $WORKSPACE --batch_size_forward $BATCH_SIZE --enable_mixed_precision $MIXED_PRECISION --num_iterations $NUM_ITERATIONS --enable_mlflow $ENABLE_MLFLOW"
echo ""

python scripts/train.py \
    --config "$CONFIG" \
    --workspace "$WORKSPACE" \
    --batch_size_forward "$BATCH_SIZE" \
    --enable_mixed_precision "$MIXED_PRECISION" \
    --num_iterations "$NUM_ITERATIONS" \
    --enable_mlflow "$ENABLE_MLFLOW"