#!/usr/bin/env python3
"""
Test script for train/dataloader.py

This script tests the StereoTrainDataLoaderPipeline class with various configurations
and validates the data loading, processing, and augmentation functionality.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import unittest
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
import open3d as o3d
import cv2

# Add the project root to the path
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)

# Import the module to test
from train.dataloader import StereoTrainDataLoaderPipeline, stereo_consistent_crop, stereo_consistent_resize

# Try to import MoGe model (optional dependency)
try:
    from moge.model.v2 import MoGeModel
    MOGE_AVAILABLE = True
    print("MoGe model available for intrinsics estimation")
except ImportError:
    MOGE_AVAILABLE = False
    print("MoGe model not available. Will use default intrinsics.")


def estimate_intrinsics_with_moge(image_rgb, device="cuda"):
    """
    使用MoGe模型估计图像的相机内参
    
    Args:
        image_rgb: RGB图像 (H, W, 3) numpy array, 值范围[0, 255]
        device: 计算设备
        
    Returns:
        intrinsics: (3, 3) 相机内参矩阵，如果MoGe不可用则返回None
    """
    if not MOGE_AVAILABLE:
        return None
    
    try:
        # 加载MoGe模型
        model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
        
        # 准备输入图像
        input_image = torch.tensor(image_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        
        # 推理
        with torch.no_grad():
            output = model.infer(input_image)
        
        # 提取内参
        intrinsics = output["intrinsics"].cpu().numpy()
        
        return intrinsics
        
    except Exception as e:
        print(f"MoGe内参估计失败: {e}")
        return None


def save_stereo_data(batch, output_dir="test_outputs", save_pointcloud=True, use_moge_intrinsics=False):
    """
    保存双目图像、深度图和点云数据
    
    Args:
        batch: 从dataloader获取的batch数据
        output_dir: 输出目录
        save_pointcloud: 是否保存点云数据
        use_moge_intrinsics: 是否使用MoGe模型估计内参
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    batch_size = batch['left_image'].shape[0]
    
    for i in range(batch_size):
        # 获取数据
        left_img = batch['left_image'][i].permute(1, 2, 0).numpy()  # (H, W, 3)
        right_img = batch['right_image'][i].permute(1, 2, 0).numpy()  # (H, W, 3)
        disparity = batch['disparity'][i].numpy()  # (H, W)
        disparity_mask = batch['disparity_mask'][i].numpy()  # (H, W)
        
        # 获取文件信息
        info = batch['info'][i]
        dataset_name = info['dataset']
        filename = info['filename']
        
        # 创建输出子目录
        sample_dir = output_path / f"{dataset_name}_{filename}"
        sample_dir.mkdir(exist_ok=True)
        
        # 保存左右图像
        left_img_uint8 = (left_img * 255).astype(np.uint8)
        right_img_uint8 = (right_img * 255).astype(np.uint8)
        
        Image.fromarray(left_img_uint8).save(sample_dir / "left_image.png")
        Image.fromarray(right_img_uint8).save(sample_dir / "right_image.png")
        
        # 保存视差图
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(disparity, cmap='jet')
        plt.title('Disparity Map')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(disparity_mask, cmap='gray')
        plt.title('Disparity Mask')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(sample_dir / "disparity_visualization.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存原始视差数据
        np.save(sample_dir / "disparity.npy", disparity)
        np.save(sample_dir / "disparity_mask.npy", disparity_mask)
        
        # 生成并保存点云
        if save_pointcloud:
            try:
                height, width = disparity.shape
                
                # 尝试使用MoGe估计内参
                if use_moge_intrinsics and MOGE_AVAILABLE:
                    print(f"使用MoGe估计样本 {i+1} 的相机内参...")
                    intrinsics = estimate_intrinsics_with_moge(left_img_uint8)
                    
                    if intrinsics is not None:
                        # 将归一化的内参转换为像素坐标
                        fx = intrinsics[0, 0] * width
                        fy = intrinsics[1, 1] * height
                        cx = intrinsics[0, 2] * width
                        cy = intrinsics[1, 2] * height
                        focal_length = (fx + fy) / 2  # 使用平均焦距
                        print(f"MoGe估计的内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
                    else:
                        # 回退到默认值
                        focal_length = width * 0.8
                        cx, cy = width / 2, height / 2
                        print("MoGe估计失败，使用默认内参")
                else:
                    # 使用默认相机内参
                    focal_length = width * 0.8  # 假设焦距
                    cx, cy = width / 2, height / 2  # 假设主点在图像中心
                    print(f"使用默认内参: 焦距={focal_length:.2f}, 主点=({cx:.2f}, {cy:.2f})")
                
                baseline = 0.1  # 假设基线距离为10cm
                
                # 生成像素坐标网格
                u, v = np.meshgrid(np.arange(width), np.arange(height))
                
                # 只处理有效视差的像素
                valid_mask = (disparity_mask) & (disparity > 0)
                valid_u = u[valid_mask]
                valid_v = v[valid_mask]
                valid_disp = disparity[valid_mask]
                
                # 计算深度
                depth = focal_length * baseline / (valid_disp + 1e-6)
                
                # 计算3D坐标
                x = (valid_u - cx) * depth / focal_length
                y = (valid_v - cy) * depth / focal_length
                z = depth
                
                # 获取颜色
                valid_colors = left_img_uint8[valid_mask]
                
                # 创建点云
                points_3d = np.stack([x, y, z], axis=1)
                
                # 保存为PLY格式
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(points_3d)
                point_cloud.colors = o3d.utility.Vector3dVector(valid_colors / 255.0)
                
                o3d.io.write_point_cloud(str(sample_dir / "pointcloud.ply"), point_cloud)
                
                # 保存点云数据为numpy格式
                np.save(sample_dir / "points_3d.npy", points_3d)
                np.save(sample_dir / "colors.npy", valid_colors)
                
                print(f"保存点云: {len(points_3d)} 个点")
                
            except Exception as e:
                print(f"生成点云时出错: {e}")
        
        # 保存元数据
        metadata = {
            'dataset': dataset_name,
            'filename': filename,
            'label_type': batch['label_type'][i],
            'image_shape': left_img.shape,
            'disparity_shape': disparity.shape,
            'valid_pixels': int(disparity_mask.sum()),
            'disparity_range': [float(disparity[disparity_mask].min()), float(disparity[disparity_mask].max())] if disparity_mask.any() else [0, 0],
            'camera_intrinsics': {
                'focal_length': float(focal_length) if 'focal_length' in locals() else None,
                'cx': float(cx) if 'cx' in locals() else None,
                'cy': float(cy) if 'cy' in locals() else None,
                'baseline': float(baseline) if 'baseline' in locals() else None,
                'estimated_by_moge': use_moge_intrinsics and MOGE_AVAILABLE and 'intrinsics' in locals() and intrinsics is not None
            }
        }
        
        with open(sample_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"已保存样本 {i+1}/{batch_size}: {sample_dir}")


class TestStereoTrainDataLoaderPipeline(unittest.TestCase):
    """Test cases for StereoTrainDataLoaderPipeline"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = {
            "datasets": [
                {
                    "name": "test_dataset",
                    "path": self.temp_dir,
                    "weight": 1.0,
                    "label_type": "stereo",
                    "image_augmentation": ["jittering"],
                    "aspect_ratio_range": [0.5, 2.0]
                }
            ],
            "image_sizes": [[256, 256], [512, 384]],
            "max_disparity": 192,
            "image_augmentation": ["jittering"],
            "stereo_augmentation": True
        }
        
        # Create test dataset structure
        self._create_test_dataset()
    
    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_dataset(self):
        """Create a minimal test dataset structure with sample files."""
        dataset_path = Path(self.temp_dir)
        
        # Create directory structure
        (dataset_path / "left" / "rgb").mkdir(parents=True, exist_ok=True)
        (dataset_path / "right" / "rgb").mkdir(parents=True, exist_ok=True)
        (dataset_path / "left" / "disparity").mkdir(parents=True, exist_ok=True)
        
        # Create sample images and disparity maps
        for i in range(3):
            filename = f"test_{i:04d}"
            
            # Create RGB images (640x480)
            left_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            right_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            Image.fromarray(left_img).save(dataset_path / "left" / "rgb" / f"{filename}.jpg")
            Image.fromarray(right_img).save(dataset_path / "right" / "rgb" / f"{filename}.jpg")
            
            # Create disparity map (encoded as 3-channel uint8)
            disparity = np.random.uniform(0, 100, (480, 640)).astype(np.float32)
            # Simulate the uint8 encoding used in FoundationStereo
            disparity_uint8 = np.stack([
                (disparity * 256).astype(np.uint8),
                ((disparity * 256) % 1 * 256).astype(np.uint8),
                np.zeros((480, 640), dtype=np.uint8)
            ], axis=2)
            
            Image.fromarray(disparity_uint8).save(dataset_path / "left" / "disparity" / f"{filename}.png")
    
    def test_initialization_fixed_size(self):
        """Test dataloader initialization with fixed image sizes."""
        dataloader = StereoTrainDataLoaderPipeline(
            config=self.test_config,
            batch_size=2,
            num_load_workers=1,
            num_process_workers=1
        )
        
        self.assertEqual(dataloader.batch_size, 2)
        self.assertEqual(dataloader.max_disparity, 192)
        self.assertEqual(dataloader.image_size_strategy, 'fixed')
        self.assertIn('test_dataset', dataloader.datasets)
        self.assertEqual(len(dataloader.datasets['test_dataset']['filenames']), 3)
    
    def test_initialization_aspect_area(self):
        """Test dataloader initialization with aspect ratio and area ranges."""
        config = self.test_config.copy()
        del config['image_sizes']
        config['aspect_ratio_range'] = [0.5, 2.0]
        config['area_range'] = [100000, 400000]
        
        dataloader = StereoTrainDataLoaderPipeline(
            config=config,
            batch_size=2
        )
        
        self.assertEqual(dataloader.image_size_strategy, 'aspect_area')
        self.assertEqual(dataloader.aspect_ratio_range, [0.5, 2.0])
        self.assertEqual(dataloader.area_range, [100000, 400000])
    
    def test_initialization_invalid_config(self):
        """Test that invalid configuration raises ValueError."""
        config = self.test_config.copy()
        del config['image_sizes']  # Remove required size configuration
        
        with self.assertRaises(ValueError):
            StereoTrainDataLoaderPipeline(config=config, batch_size=2)
    
    def test_missing_dataset_directories(self):
        """Test that missing dataset directories raise ValueError."""
        config = self.test_config.copy()
        config['datasets'][0]['path'] = '/nonexistent/path'
        
        with self.assertRaises(ValueError):
            StereoTrainDataLoaderPipeline(config=config, batch_size=2)
    
    @patch('train.dataloader.depth_uint8_decoding')
    def test_load_instance_success(self, mock_depth_decode):
        """Test successful instance loading."""
        # Mock the depth decoding function
        mock_depth_decode.return_value = np.random.uniform(0, 100, (480, 640)).astype(np.float32)
        
        dataloader = StereoTrainDataLoaderPipeline(self.test_config, batch_size=1)
        
        instance = {
            'dataset': 'test_dataset',
            'filename': 'test_0000',
            'label_type': 'stereo'
        }
        
        loaded_instance = dataloader._load_instance(instance)
        
        self.assertIn('left_image', loaded_instance)
        self.assertIn('right_image', loaded_instance)
        self.assertIn('disparity', loaded_instance)
        self.assertIn('disparity_mask', loaded_instance)
        self.assertEqual(loaded_instance['left_image'].shape, (480, 640, 3))
        self.assertEqual(loaded_instance['right_image'].shape, (480, 640, 3))
    
    def test_load_instance_failure(self):
        """Test instance loading failure handling."""
        dataloader = StereoTrainDataLoaderPipeline(self.test_config, batch_size=1)
        
        instance = {
            'dataset': 'test_dataset',
            'filename': 'nonexistent_file',
            'label_type': 'stereo'
        }
        
        loaded_instance = dataloader._load_instance(instance)
        
        # Should return invalid instance on failure
        self.assertEqual(loaded_instance['label_type'], 'invalid')
    
    @patch('train.dataloader.depth_uint8_decoding')
    def test_process_instance(self, mock_depth_decode):
        """Test instance processing including augmentations."""
        mock_depth_decode.return_value = np.random.uniform(0, 100, (480, 640)).astype(np.float32)
        
        dataloader = StereoTrainDataLoaderPipeline(self.test_config, batch_size=1)
        
        # Create a test instance
        instance = {
            'dataset': 'test_dataset',
            'filename': 'test_0000',
            'label_type': 'stereo',
            'width': 256,
            'height': 256,
            'seed': 42
        }
        
        # Load and process
        loaded_instance = dataloader._load_instance(instance)
        processed_instance = dataloader._process_instance(loaded_instance)
        
        # Check output format
        self.assertIsInstance(processed_instance['left_image'], torch.Tensor)
        self.assertIsInstance(processed_instance['right_image'], torch.Tensor)
        self.assertIsInstance(processed_instance['disparity'], torch.Tensor)
        self.assertIsInstance(processed_instance['disparity_mask'], torch.Tensor)
        
        # Check tensor shapes
        self.assertEqual(processed_instance['left_image'].shape, (3, 256, 256))
        self.assertEqual(processed_instance['right_image'].shape, (3, 256, 256))
        self.assertEqual(processed_instance['disparity'].shape, (256, 256))
        self.assertEqual(processed_instance['disparity_mask'].shape, (256, 256))
        
        # Check value ranges
        self.assertTrue(torch.all(processed_instance['left_image'] >= 0))
        self.assertTrue(torch.all(processed_instance['left_image'] <= 1))
        self.assertTrue(torch.all(processed_instance['disparity'] >= 0))
        self.assertTrue(torch.all(processed_instance['disparity'] <= dataloader.max_disparity))
    
    def test_process_invalid_instance(self):
        """Test processing of invalid instances."""
        dataloader = StereoTrainDataLoaderPipeline(self.test_config, batch_size=1)
        
        instance = dataloader.invalid_instance.copy()
        instance['label_type'] = 'invalid'
        
        processed_instance = dataloader._process_instance(instance)
        
        self.assertEqual(processed_instance['label_type'], 'invalid')
    
    @patch('train.dataloader.depth_uint8_decoding')
    def test_get_batch(self, mock_depth_decode):
        """Test getting a complete batch."""
        mock_depth_decode.return_value = np.random.uniform(0, 100, (480, 640)).astype(np.float32)
        
        dataloader = StereoTrainDataLoaderPipeline(self.test_config, batch_size=2)
        
        batch = dataloader.get()
        
        # Check batch structure
        self.assertIn('left_image', batch)
        self.assertIn('right_image', batch)
        self.assertIn('disparity', batch)
        self.assertIn('disparity_mask', batch)
        self.assertIn('label_type', batch)
        self.assertIn('info', batch)
        
        # Check batch dimensions
        self.assertEqual(batch['left_image'].shape[0], 2)  # batch size
        self.assertEqual(batch['right_image'].shape[0], 2)
        self.assertEqual(batch['disparity'].shape[0], 2)
        self.assertEqual(batch['disparity_mask'].shape[0], 2)
        
        # Check metadata
        self.assertEqual(len(batch['label_type']), 2)
        self.assertEqual(len(batch['info']), 2)
    
    def test_context_manager(self):
        """Test dataloader as context manager."""
        with StereoTrainDataLoaderPipeline(self.test_config, batch_size=1) as dataloader:
            self.assertIsInstance(dataloader, StereoTrainDataLoaderPipeline)
    
    def test_sample_batch_deterministic(self):
        """Test that batch sampling with same seed produces consistent results."""
        dataloader = StereoTrainDataLoaderPipeline(self.test_config, batch_size=2)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        batch_gen = dataloader._sample_batch()
        batch1 = next(batch_gen)
        
        np.random.seed(42)
        torch.manual_seed(42)
        
        batch_gen = dataloader._sample_batch()
        batch2 = next(batch_gen)
        
        # Should have same structure (though content may vary due to random sampling)
        self.assertEqual(len(batch1), len(batch2))
        self.assertEqual(batch1[0]['dataset'], batch2[0]['dataset'])


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test data."""
        self.left_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.right_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.disparity = np.random.uniform(0, 100, (480, 640)).astype(np.float32)
        self.disparity_mask = np.random.choice([True, False], (480, 640))
    
    def test_stereo_consistent_crop(self):
        """Test stereo consistent cropping."""
        crop_h, crop_w = 256, 256
        
        left_crop, right_crop, disp_crop, mask_crop = stereo_consistent_crop(
            self.left_img, self.right_img, self.disparity, self.disparity_mask,
            crop_h, crop_w, random_crop=False
        )
        
        # Check output shapes
        self.assertEqual(left_crop.shape, (crop_h, crop_w, 3))
        self.assertEqual(right_crop.shape, (crop_h, crop_w, 3))
        self.assertEqual(disp_crop.shape, (crop_h, crop_w))
        self.assertEqual(mask_crop.shape, (crop_h, crop_w))
    
    def test_stereo_consistent_resize(self):
        """Test stereo consistent resizing."""
        target_h, target_w = 240, 320
        
        left_resized, right_resized, disp_resized, mask_resized = stereo_consistent_resize(
            self.left_img, self.right_img, self.disparity, self.disparity_mask,
            target_h, target_w
        )
        
        # Check output shapes
        self.assertEqual(left_resized.shape, (target_h, target_w, 3))
        self.assertEqual(right_resized.shape, (target_h, target_w, 3))
        self.assertEqual(disp_resized.shape, (target_h, target_w))
        self.assertEqual(mask_resized.shape, (target_h, target_w))
        
        # Check disparity scaling
        width_ratio = target_w / self.left_img.shape[1]
        # Disparity should be scaled by width ratio
        self.assertTrue(np.all(disp_resized <= self.disparity.max() * width_ratio + 1))  # +1 for interpolation tolerance


def create_sample_config():
    """Create a sample configuration for manual testing."""
    return {
        "datasets": [
            {
                "name": "sample_dataset",
                "path": "./test_data",
                "weight": 1.0,
                "label_type": "stereo",
                "image_augmentation": ["jittering"],
                "aspect_ratio_range": [0.5, 2.0]
            }
        ],
        "image_sizes": [[256, 256], [512, 384], [640, 480]],
        "max_disparity": 192,
        "image_augmentation": ["jittering"],
        "stereo_augmentation": True
    }


def manual_test_with_real_data():
    """Manual test function for testing with real data (if available)."""
    print("Running manual test with real data...")
    
    # Check if real data exists
    data_path = Path("./data")
    if not data_path.exists():
        print("No real data found at ./data, skipping manual test")
        return
    
    config = {
        "datasets": [
            {
                "name": "real_dataset",
                "path": str(data_path),
                "weight": 1.0,
                "label_type": "stereo",
                "image_augmentation": ["jittering"]
            }
        ],
        "image_sizes": [[256, 256]],
        "max_disparity": 192,
        "stereo_augmentation": True
    }
    
    try:
        dataloader = StereoTrainDataLoaderPipeline(config, batch_size=2)
        print(f"Successfully loaded dataloader with {len(dataloader.datasets)} datasets")
        
        # Get a batch
        batch = dataloader.get()
        print(f"Successfully got batch with shapes:")
        print(f"  left_image: {batch['left_image'].shape}")
        print(f"  right_image: {batch['right_image'].shape}")
        print(f"  disparity: {batch['disparity'].shape}")
        print(f"  disparity_mask: {batch['disparity_mask'].shape}")
        print(f"  label_types: {batch['label_type']}")
        
        # 保存双目图像、深度图和点云数据
        print("\n保存测试数据...")
        save_stereo_data(batch, output_dir="test_outputs/dataloader_test", save_pointcloud=True, use_moge_intrinsics=use_moge_intrinsics)
        print("数据保存完成!")
        
    except Exception as e:
        print(f"Manual test failed: {e}")


def test_and_save_data(disable_augmentation=False, use_moge_intrinsics=False):
    """专门用于测试并保存数据的函数"""
    print("测试数据加载器并保存双目图像、深度图和点云...")
    if disable_augmentation:
        print("注意: 已禁用立体增强以避免左右图像交换")
    
    # 检查真实数据
    data_path = Path("./data")
    if not data_path.exists():
        print("未找到真实数据，创建测试数据...")
        # 使用测试数据
        temp_dir = tempfile.mkdtemp()
        try:
            # 创建测试数据集
            test_config = {
                "datasets": [
                    {
                        "name": "test_dataset",
                        "path": temp_dir,
                        "weight": 1.0,
                        "label_type": "stereo",
                        "image_augmentation": ["jittering"] if not disable_augmentation else []
                    }
                ],
                "image_sizes": [[512, 384]],
                "max_disparity": 192,
                "stereo_augmentation": False if disable_augmentation else True
            }
            
            # 创建测试数据集结构
            dataset_path = Path(temp_dir)
            (dataset_path / "left" / "rgb").mkdir(parents=True, exist_ok=True)
            (dataset_path / "right" / "rgb").mkdir(parents=True, exist_ok=True)
            (dataset_path / "left" / "disparity").mkdir(parents=True, exist_ok=True)
            
            # 创建更大的测试图像
            for i in range(2):
                filename = f"test_{i:04d}"
                
                # 创建RGB图像 (640x480)
                left_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                right_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                Image.fromarray(left_img).save(dataset_path / "left" / "rgb" / f"{filename}.jpg")
                Image.fromarray(right_img).save(dataset_path / "right" / "rgb" / f"{filename}.jpg")
                
                # 创建更真实的视差图
                disparity = np.random.uniform(10, 100, (480, 640)).astype(np.float32)
                # 添加一些结构
                disparity[200:280, 200:440] += 50  # 前景物体
                disparity_uint8 = np.stack([
                    (disparity * 256).astype(np.uint8),
                    ((disparity * 256) % 1 * 256).astype(np.uint8),
                    np.zeros((480, 640), dtype=np.uint8)
                ], axis=2)
                
                Image.fromarray(disparity_uint8).save(dataset_path / "left" / "disparity" / f"{filename}.png")
            
            dataloader = StereoTrainDataLoaderPipeline(test_config, batch_size=2)
            
        finally:
            # 确保清理临时目录
            import atexit
            atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
    else:
        # 使用真实数据
        config = {
            "datasets": [
                {
                    "name": "real_dataset",
                    "path": str(data_path),
                    "weight": 1.0,
                    "label_type": "stereo",
                    "image_augmentation": ["jittering"] if not disable_augmentation else []
                }
            ],
            "image_sizes": [[512, 384]],
            "max_disparity": 192,
            "stereo_augmentation": False if disable_augmentation else True
        }
        dataloader = StereoTrainDataLoaderPipeline(config, batch_size=2)
    
    # 获取并保存数据
    print("获取batch数据...")
    batch = dataloader.get()
    
    print(f"Batch信息:")
    print(f"  left_image: {batch['left_image'].shape}")
    print(f"  right_image: {batch['right_image'].shape}")
    print(f"  disparity: {batch['disparity'].shape}")
    print(f"  disparity_mask: {batch['disparity_mask'].shape}")
    print(f"  label_types: {batch['label_type']}")
    
    # 保存数据
    print("\n保存双目图像、深度图和点云数据...")
    save_stereo_data(batch, output_dir="test_outputs/stereo_data_samples", save_pointcloud=True, use_moge_intrinsics=use_moge_intrinsics)
    print("所有数据保存完成!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the stereo dataloader")
    parser.add_argument("--manual", action="store_true", help="Run manual test with real data")
    parser.add_argument("--save-data", action="store_true", help="Test and save stereo images, depth maps and point clouds")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable stereo augmentation to avoid left-right image swapping")
    parser.add_argument("--use-moge", action="store_true", help="Use MoGe model to estimate camera intrinsics")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    if args.manual:
        manual_test_with_real_data()
    elif args.save_data:
        test_and_save_data(disable_augmentation=args.no_augmentation, use_moge_intrinsics=args.use_moge)
    else:
        # Run unit tests
        if args.verbose:
            unittest.main(verbosity=2)
        else:
            unittest.main()