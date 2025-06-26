#!/usr/bin/env python3
"""
Test script for FoundationStereo training pipeline
This script creates dummy data and tests the training components
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image

# Add parent directory to path
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

from train.dataloader import StereoTrainDataLoaderPipeline
from train.losses import disparity_l1_loss, disparity_smooth_l1_loss, multi_scale_loss
from train.utils import build_optimizer, build_lr_scheduler
from Utils import vis_disparity


def create_dummy_data(data_dir: Path, num_samples: int = 10):
    """Create dummy stereo data for testing"""
    print(f"Creating dummy data in {data_dir}")
    
    # Create directory structure
    left_rgb_dir = data_dir / 'left' / 'rgb'
    right_rgb_dir = data_dir / 'right' / 'rgb'
    left_disp_dir = data_dir / 'left' / 'disparity'
    
    for dir_path in [left_rgb_dir, right_rgb_dir, left_disp_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_samples):
        # Create dummy RGB images
        left_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        right_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Create dummy disparity (encoded as uint8)
        disparity = np.random.uniform(0, 64, (480, 640)).astype(np.float32)
        
        # Encode disparity using the expected format
        scale = 1000
        disparity_encoded = (disparity * scale).astype(np.uint32)
        disparity_uint8 = np.zeros((480, 640, 3), dtype=np.uint8)
        disparity_uint8[:, :, 0] = (disparity_encoded // (255 * 255)) % 255
        disparity_uint8[:, :, 1] = (disparity_encoded // 255) % 255
        disparity_uint8[:, :, 2] = disparity_encoded % 255
        
        # Save images
        Image.fromarray(left_img).save(left_rgb_dir / f'{i:04d}.jpg')
        Image.fromarray(right_img).save(right_rgb_dir / f'{i:04d}.jpg')
        Image.fromarray(disparity_uint8).save(left_disp_dir / f'{i:04d}.png')
    
    print(f"Created {num_samples} dummy stereo pairs")


def test_dataloader():
    """Test the stereo dataloader"""
    print("\n=== Testing Dataloader ===")
    
    config = {
        'datasets': [
            {
                'name': 'test_dataset',
                'path': './test_data',
                'weight': 1.0,
                'label_type': 'stereo',
                'image_augmentation': ['jittering']
            }
        ],
        'image_sizes': [[384, 256], [512, 384]],
        'max_disparity': 64,
        'image_augmentation': ['jittering'],
        'stereo_augmentation': True
    }
    
    try:
        dataloader = StereoTrainDataLoaderPipeline(config, batch_size=2)
        
        # Test getting a batch
        with dataloader:
            batch = dataloader.get()
            
        print(f"Batch keys: {batch.keys()}")
        print(f"Left image shape: {batch['left_image'].shape}")
        print(f"Right image shape: {batch['right_image'].shape}")
        print(f"Disparity shape: {batch['disparity'].shape}")
        print(f"Disparity mask shape: {batch['disparity_mask'].shape}")
        print(f"Label types: {batch['label_type']}")
        
        print("✓ Dataloader test passed")
        return True
        
    except Exception as e:
        print(f"✗ Dataloader test failed: {e}")
        return False


def test_losses():
    """Test the loss functions"""
    print("\n=== Testing Loss Functions ===")
    
    try:
        # Create dummy tensors
        pred_disparity = torch.rand(256, 384) * 64
        gt_disparity = torch.rand(256, 384) * 64
        mask = torch.rand(256, 384) > 0.2
        
        # Test L1 loss
        loss_l1, misc_l1 = disparity_l1_loss(pred_disparity, gt_disparity, mask)
        print(f"L1 Loss: {loss_l1.item():.4f}, EPE: {misc_l1['epe']:.4f}")
        
        # Test Smooth L1 loss
        loss_smooth, misc_smooth = disparity_smooth_l1_loss(pred_disparity, gt_disparity, mask)
        print(f"Smooth L1 Loss: {loss_smooth.item():.4f}, EPE: {misc_smooth['epe']:.4f}")
        
        # Test multi-scale loss
        pyramid = [
            torch.rand(128, 192) * 32,  # 1/2 scale
            torch.rand(64, 96) * 16,    # 1/4 scale
            torch.rand(32, 48) * 8      # 1/8 scale
        ]
        loss_multi, misc_multi = multi_scale_loss(pyramid, gt_disparity, mask)
        print(f"Multi-scale Loss: {loss_multi.item():.4f}, EPE: {misc_multi['multi_scale_epe']:.4f}")
        
        print("✓ Loss functions test passed")
        return True
        
    except Exception as e:
        print(f"✗ Loss functions test failed: {e}")
        return False


def test_optimizer_scheduler():
    """Test optimizer and scheduler building"""
    print("\n=== Testing Optimizer and Scheduler ===")
    
    try:
        # Create a dummy model
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 1, 3, padding=1)
        )
        
        # Test optimizer config
        optimizer_config = {
            'type': 'AdamW',
            'params': [
                {
                    'params': {
                        'include': ['*'],
                        'exclude': []
                    },
                    'lr': 1e-4,
                    'weight_decay': 1e-4
                }
            ]
        }
        
        optimizer = build_optimizer(model, optimizer_config)
        print(f"Optimizer: {type(optimizer).__name__}")
        print(f"Parameter groups: {len(optimizer.param_groups)}")
        
        # Test scheduler config
        scheduler_config = {
            'type': 'LambdaLR',
            'params': {
                'lr_lambda': 'min(epoch / 1000, 1.0)'
            }
        }
        
        scheduler = build_lr_scheduler(optimizer, scheduler_config)
        print(f"Scheduler: {type(scheduler).__name__}")
        
        print("✓ Optimizer and scheduler test passed")
        return True
        
    except Exception as e:
        print(f"✗ Optimizer and scheduler test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("FoundationStereo Training Pipeline Test")
    print("=" * 50)
    
    # Create test data directory
    test_data_dir = Path('./test_data')
    if not test_data_dir.exists():
        create_dummy_data(test_data_dir, num_samples=5)
    
    # Run tests
    tests = [
        test_dataloader,
        test_losses,
        test_optimizer_scheduler,
    ]
    
    results = []
    for test_func in tests:
        results.append(test_func())
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Training pipeline is ready.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    # Cleanup
    import shutil
    if test_data_dir.exists():
        shutil.rmtree(test_data_dir)
        print("Cleaned up test data")


if __name__ == '__main__':
    main()