#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
from pathlib import Path

def load_depth_file(file_path):
    """Load depth map file in .npz format"""
    try:
        data = np.load(file_path)
        # Print all keys in the file
        print(f"Keys in file {os.path.basename(file_path)}: {list(data.keys())}")
        
        # Assume depth map is stored in the first key, adjust if needed
        depth_key = list(data.keys())[0]
        depth_map = data[depth_key]
        
        return depth_map
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def analyze_depth_map(depth_map, file_name):
    """Analyze basic statistics of the depth map"""
    if depth_map is None:
        return
    
    # 检查数据类型和形状
    print(f"\nFile: {file_name}")
    print(f"Data Type: {depth_map.dtype}")
    print(f"Shape: {depth_map.shape}")
    
    # 检查是否有NaN或无限值
    nan_count = np.isnan(depth_map).sum()
    inf_count = np.isinf(depth_map).sum()
    print(f"NaN Count: {nan_count}")
    print(f"Infinity Count: {inf_count}")
    
    # 如果没有NaN或无限值，计算基本统计信息
    if nan_count == 0 and inf_count == 0:
        min_val = np.min(depth_map)
        max_val = np.max(depth_map)
        mean_val = np.mean(depth_map)
        std_val = np.std(depth_map)
        
        print(f"Min Value: {min_val:.4f}")
        print(f"Max Value: {max_val:.4f}")
        print(f"Mean Value: {mean_val:.4f}")
        print(f"Standard Deviation: {std_val:.4f}")
    else:
        print("Warning: Data contains NaN or infinite values, cannot calculate accurate statistics")
    
    return {
        "min": np.min(depth_map) if nan_count == 0 and inf_count == 0 else None,
        "max": np.max(depth_map) if nan_count == 0 and inf_count == 0 else None,
        "mean": np.mean(depth_map) if nan_count == 0 and inf_count == 0 else None,
        "std": np.std(depth_map) if nan_count == 0 and inf_count == 0 else None,
        "nan_count": nan_count,
        "inf_count": inf_count
    }

def visualize_depth_map(depth_map, file_name, stats=None, save_path=None):
    """Visualize the depth map"""
    if depth_map is None:
        return
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 子图1: 深度图 (线性刻度)
    plt.subplot(2, 2, 1)
    im = plt.imshow(depth_map, cmap='viridis', norm=None)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f'Depth Map (Linear): {file_name}')
    
    # 子图2: 深度图 (对数刻度)
    plt.subplot(2, 2, 2)
    # 确保所有值都是正的，对于对数刻度
    min_positive = np.min(depth_map[depth_map > 0]) if np.any(depth_map > 0) else 0.1
    log_depth = np.copy(depth_map)
    log_depth[log_depth <= 0] = min_positive  # 替换零或负值
    
    from matplotlib.colors import LogNorm
    im = plt.imshow(log_depth, cmap='jet', norm=LogNorm(vmin=min_positive, vmax=np.max(log_depth)))
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('Depth Map (Log Scale)')
    
    # 子图3: 深度直方图 (对数刻度)
    plt.subplot(2, 2, 3)
    if stats and stats["nan_count"] == 0 and stats["inf_count"] == 0:
        # 使用对数刻度的直方图
        plt.hist(depth_map.flatten(), bins=100)
        plt.xscale('log')  # 对数X轴
        plt.yscale('log')  # 对数Y轴
        plt.title('Depth Value Histogram (Log Scale)')
        plt.xlabel('Depth Value (log)')
        plt.ylabel('Frequency (log)')
    else:
        plt.text(0.5, 0.5, 'Cannot generate histogram\n(Data contains NaN or infinite values)',
                 horizontalalignment='center', verticalalignment='center')
    
    # 子图4: 3D表面图 (对数Z轴)
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(2, 2, 4, projection='3d')
    if stats and stats["nan_count"] == 0 and stats["inf_count"] == 0:
        # 为了性能考虑，使用更大的降采样率
        sample_rate = max(1, depth_map.shape[0] // 50, depth_map.shape[1] // 50)
        y, x = np.mgrid[0:depth_map.shape[0]:sample_rate, 0:depth_map.shape[1]:sample_rate]
        
        # 对深度值应用对数变换 (确保所有值都是正的)
        z = depth_map[::sample_rate, ::sample_rate]
        z_positive = np.copy(z)
        z_positive[z_positive <= 0] = min_positive  # 替换零或负值
        z_log = np.log10(z_positive)  # 对数变换
        
        surf = ax.plot_surface(x, y, z_log, cmap='viridis', linewidth=0, antialiased=False)
        plt.colorbar(surf, fraction=0.046, pad=0.04)
        ax.set_title('3D Surface Plot (Log Z Scale)')
    else:
        plt.text(0.5, 0.5, 'Cannot generate 3D surface plot\n(Data contains NaN or infinite values)',
                 horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    
    # 保存图像（如果指定了保存路径）
    if save_path:
        plt.savefig(save_path)
        print(f"Image saved to: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize depth map files')
    parser.add_argument('--dir', type=str, default='OmniDatasets/Deep360/ep1_500frames/testing/depth',
                        help='Directory containing depth map files')
    parser.add_argument('--file', type=str, default=None,
                        help='Specific file to visualize (if not specified, random files will be selected)')
    parser.add_argument('--num', type=int, default=3,
                        help='Number of random files to visualize (when specific file is not specified)')
    parser.add_argument('--save', action='store_true',
                        help='Whether to save visualization results')
    parser.add_argument('--save_dir', type=str, default='depth_visualization',
                        help='Directory to save visualization results')
    
    args = parser.parse_args()
    
    # 确保目录存在
    depth_dir = Path(args.dir)
    if not depth_dir.exists():
        print(f"Error: Directory {args.dir} does not exist")
        return
    
    # 获取所有.npz文件
    depth_files = list(depth_dir.glob('*_depth.npz'))
    if not depth_files:
        print(f"Error: No depth map files found in directory {args.dir}")
        return
    
    print(f"Found {len(depth_files)} depth map files in directory {args.dir}")
    
    # 创建保存目录（如果需要）
    if args.save:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        print(f"Visualization results will be saved to: {save_dir}")
    
    # 选择要可视化的文件
    files_to_visualize = []
    if args.file:
        file_path = depth_dir / args.file
        if file_path.exists():
            files_to_visualize.append(file_path)
        else:
            print(f"Warning: File {args.file} does not exist, will select random files instead")
            files_to_visualize = random.sample(depth_files, min(args.num, len(depth_files)))
    else:
        files_to_visualize = random.sample(depth_files, min(args.num, len(depth_files)))
    
    # 处理每个文件
    for file_path in files_to_visualize:
        file_name = file_path.name
        print(f"\nProcessing file: {file_name}")
        
        # 加载深度图
        depth_map = load_depth_file(file_path)
        
        if depth_map is not None:
            # 分析深度图
            stats = analyze_depth_map(depth_map, file_name)
            
            # 可视化深度图
            save_path = Path(args.save_dir) / f"{file_name.replace('.npz', '.png')}" if args.save else None
            visualize_depth_map(depth_map, file_name, stats, save_path)

if __name__ == "__main__":
    main()