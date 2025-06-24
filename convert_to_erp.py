#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将pinhole相机数据转换为ERP（等距矩形投影）全景图像的脚本

这个脚本使用depth_any_camera/dac/utils/erp_geometry.py中的函数
将pinhole相机数据（左右图像和视差图）转换为ERP全景图像。

使用方法:
    python convert_to_erp.py --data_dir data --output_dir erp_output --params_file estimated_K.txt

输出:
    - ERP全景RGB图像
    - ERP全景深度图
    - ERP全景有效区域掩码
"""

import os
import argparse
import numpy as np
import cv2
import torch
import math
from pathlib import Path
from glob import glob
from tqdm import tqdm
import json

# 导入ERP几何变换函数
from depth_any_camera.dac.utils.erp_geometry import cam_to_erp_patch_fast


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='将pinhole相机数据转换为ERP全景图像')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='包含pinhole相机数据的目录（默认：data）')
    parser.add_argument('--output_dir', type=str, default='erp_output',
                        help='保存ERP数据的目录（默认：erp_output）')
    parser.add_argument('--params_file', type=str, default='estimated_K.txt',
                        help='相机参数文件路径（默认：estimated_K.txt）')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='要转换的图像数量（如果不指定，将处理所有图像）')
    return parser.parse_args()


def read_camera_params(params_file):
    """
    读取相机参数文件
    
    格式：
    fx 0.0 cx 0.0 fy cy 0.0 0.0 1.0
    baseline
    """
    with open(params_file, 'r') as f:
        lines = f.readlines()
        
    if len(lines) < 2:
        raise ValueError(f"相机参数文件格式不正确: {params_file}")
    
    # 解析内参矩阵
    intrinsics = lines[0].strip().split()
    if len(intrinsics) != 9:
        raise ValueError(f"内参矩阵格式不正确: {intrinsics}")
    
    fx = float(intrinsics[0])
    cx = float(intrinsics[2])
    fy = float(intrinsics[4])
    cy = float(intrinsics[5])
    
    # 解析baseline
    baseline = float(lines[1].strip())
    
    # 创建相机参数字典
    cam_params = {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'baseline': baseline
    }
    
    return cam_params


def find_image_pairs(data_dir, num_samples=None):
    """
    查找左右图像对和对应的视差图
    
    返回：
        left_rgb_files: 左相机RGB图像文件列表
        right_rgb_files: 右相机RGB图像文件列表
        disparity_files: 视差图文件列表
    """
    data_dir = Path(data_dir)
    
    # 检查目录结构
    left_rgb_dir = data_dir / 'left' / 'rgb'
    right_rgb_dir = data_dir / 'right' / 'rgb'
    left_disp_dir = data_dir / 'left' / 'disparity'
    
    if not left_rgb_dir.exists():
        raise FileNotFoundError(f"左相机RGB图像目录不存在: {left_rgb_dir}")
    if not right_rgb_dir.exists():
        raise FileNotFoundError(f"右相机RGB图像目录不存在: {right_rgb_dir}")
    if not left_disp_dir.exists():
        raise FileNotFoundError(f"视差图目录不存在: {left_disp_dir}")
    
    # 查找所有左相机RGB图像
    left_rgb_files = sorted(glob(str(left_rgb_dir / '*.jpg')) + glob(str(left_rgb_dir / '*.png')))
    
    if not left_rgb_files:
        raise FileNotFoundError(f"在 {left_rgb_dir} 中未找到图像")
    
    # 限制处理的图像数量
    if num_samples is not None:
        left_rgb_files = left_rgb_files[:num_samples]
    
    # 查找对应的右相机RGB图像和视差图
    right_rgb_files = []
    disparity_files = []
    
    for left_rgb_file in left_rgb_files:
        # 提取文件名（不含路径和扩展名）
        filename = os.path.basename(left_rgb_file)
        name, _ = os.path.splitext(filename)
        
        # 查找对应的右相机RGB图像
        right_rgb_file = str(right_rgb_dir / filename)
        if not os.path.exists(right_rgb_file):
            # 尝试不同的扩展名
            alt_right_rgb_file = str(right_rgb_dir / f"{name}.png")
            if os.path.exists(alt_right_rgb_file):
                right_rgb_file = alt_right_rgb_file
            else:
                print(f"警告: 未找到对应的右相机RGB图像: {right_rgb_file}")
                continue
        
        # 查找对应的视差图
        disparity_file = str(left_disp_dir / f"{name}.png")
        if not os.path.exists(disparity_file):
            print(f"警告: 未找到对应的视差图: {disparity_file}")
            continue
        
        right_rgb_files.append(right_rgb_file)
        disparity_files.append(disparity_file)
    
    # 确保所有列表长度相同
    min_len = min(len(left_rgb_files), len(right_rgb_files), len(disparity_files))
    left_rgb_files = left_rgb_files[:min_len]
    right_rgb_files = right_rgb_files[:min_len]
    disparity_files = disparity_files[:min_len]
    
    return left_rgb_files, right_rgb_files, disparity_files


def disparity_to_depth(disparity, baseline, focal_length):
    """
    将视差图转换为深度图
    
    参数:
        disparity: 视差图
        baseline: 双目相机的基线长度
        focal_length: 相机的焦距
    
    返回:
        depth: 深度图
    """
    # 避免除以零
    valid_mask = disparity > 0
    depth = np.zeros_like(disparity, dtype=np.float32)
    
    # 视差到深度的转换公式: depth = baseline * focal_length / disparity
    depth[valid_mask] = baseline * focal_length / disparity[valid_mask]
    
    # 创建有效深度掩码
    mask_valid_depth = valid_mask.astype(np.float32)
    
    return depth, mask_valid_depth


def convert_to_erp(img, depth, mask_valid_depth, cam_params, erp_size=(1024, 2048)):
    """
    将pinhole相机图像和深度图转换为ERP全景图像
    
    参数:
        img: 输入图像
        depth: 深度图
        mask_valid_depth: 有效深度掩码
        cam_params: 相机参数
        erp_size: ERP图像的大小 (高度, 宽度)
    
    返回:
        erp_img: ERP全景图像
        erp_depth: ERP全景深度图
        erp_mask_valid: ERP全景有效区域掩码
    """
    # 准备相机参数
    cam_params_for_erp = {
        'fx': cam_params['fx'],
        'fy': cam_params['fy'],
        'cx': cam_params['cx'],
        'cy': cam_params['cy']
    }
    
    # 确保深度图和掩码的形状正确
    if len(depth.shape) == 2:
        depth = depth[..., np.newaxis]
    if len(mask_valid_depth.shape) == 2:
        mask_valid_depth = mask_valid_depth[..., np.newaxis]
    
    # ERP参数
    erp_h, erp_w = erp_size
    
    # 创建多个不同phi值的ERP图像，以便在两极区域有更多的内容
    # 我们将生成5个不同phi值的图像：赤道、北半球中纬度、南半球中纬度、北极附近和南极附近
    phi_values = [0, math.pi/6, -math.pi/6, math.pi/3, -math.pi/3]  # 0°, 30°, -30°, 60°, -60°
    theta = 0  # 经度（本初子午线）
    
    # 初始化合并后的ERP图像、深度图和掩码
    combined_erp_img = np.zeros((erp_h, erp_w, 3), dtype=np.float32)
    combined_erp_depth = np.zeros((erp_h, erp_w), dtype=np.float32)
    combined_erp_mask_valid = np.zeros((erp_h, erp_w), dtype=np.float32)
    combined_mask_active = np.zeros((erp_h, erp_w), dtype=np.float32)
    
    # 为每个phi值生成ERP图像，并将它们合并
    for phi in phi_values:
        # 转换为ERP
        erp_img, erp_depth, erp_mask_valid, mask_active, lat_grid, lon_grid = cam_to_erp_patch_fast(
            img=img,
            depth=depth,
            mask_valid_depth=mask_valid_depth,
            theta=theta,
            phi=phi,
            patch_h=erp_h,
            patch_w=erp_w,
            erp_h=erp_h,
            erp_w=erp_w,
            cam_params=cam_params_for_erp
        )
        
        # 将当前ERP图像合并到combined_erp_img中
        # 只在mask_active为1的区域进行合并
        mask_active_3d = np.repeat(mask_active[:, :, np.newaxis], 3, axis=2)
        combined_erp_img = np.where(mask_active_3d > 0, erp_img, combined_erp_img)
        
        # 合并深度图和掩码
        combined_erp_depth = np.where(mask_active > 0, erp_depth, combined_erp_depth)
        combined_erp_mask_valid = np.where(mask_active > 0, erp_mask_valid, combined_erp_mask_valid)
        combined_mask_active = np.maximum(combined_mask_active, mask_active)
    
    return combined_erp_img, combined_erp_depth, combined_erp_mask_valid, combined_mask_active


def save_erp_outputs(output_dir, name, erp_img, erp_depth, erp_mask_valid, side):
    """
    保存ERP输出
    
    参数:
        output_dir: 输出目录
        name: 文件名（不含扩展名）
        erp_img: ERP全景图像
        erp_depth: ERP全景深度图
        erp_mask_valid: ERP全景有效区域掩码
        side: 'left' 或 'right'
    """
    # 创建输出目录
    rgb_dir = os.path.join(output_dir, side, 'rgb')
    depth_dir = os.path.join(output_dir, side, 'depth')
    mask_dir = os.path.join(output_dir, side, 'mask')
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    # 保存ERP全景图像
    rgb_path = os.path.join(rgb_dir, f"{name}_erp.png")
    cv2.imwrite(rgb_path, (erp_img * 255).astype(np.uint8))
    
    # 保存ERP全景深度图（归一化到0-65535）
    depth_path = os.path.join(depth_dir, f"{name}_erp.png")
    depth_min = np.min(erp_depth[erp_depth > 0]) if np.any(erp_depth > 0) else 0
    depth_max = np.max(erp_depth) if np.any(erp_depth > 0) else 1
    depth_normalized = np.zeros_like(erp_depth)
    if depth_max > depth_min:
        depth_normalized = (erp_depth - depth_min) / (depth_max - depth_min) * 65535
    depth_normalized = depth_normalized.astype(np.uint16)
    cv2.imwrite(depth_path, depth_normalized)
    
    # 保存ERP全景有效区域掩码
    mask_path = os.path.join(mask_dir, f"{name}_erp.png")
    cv2.imwrite(mask_path, (erp_mask_valid * 255).astype(np.uint8))
    
    # 保存深度范围信息
    depth_info = {
        'depth_min': float(depth_min),
        'depth_max': float(depth_max)
    }
    info_path = os.path.join(depth_dir, f"{name}_depth_info.json")
    with open(info_path, 'w') as f:
        json.dump(depth_info, f, indent=4)


def process_image_pair(left_rgb_file, right_rgb_file, disparity_file, cam_params, output_dir, erp_size=(1024, 2048)):
    """
    处理一对图像和对应的视差图
    
    参数:
        left_rgb_file: 左相机RGB图像文件路径
        right_rgb_file: 右相机RGB图像文件路径
        disparity_file: 视差图文件路径
        cam_params: 相机参数
        output_dir: 输出目录
        erp_size: ERP图像的大小 (高度, 宽度)
    """
    # 读取图像
    left_img = cv2.imread(left_rgb_file)
    right_img = cv2.imread(right_rgb_file)
    disparity = cv2.imread(disparity_file, cv2.IMREAD_UNCHANGED)
    
    # 确保视差图是单通道的
    if len(disparity.shape) > 2:
        disparity = disparity[:, :, 0]
    
    # 将视差图转换为深度图
    depth, mask_valid_depth = disparity_to_depth(
        disparity, 
        cam_params['baseline'], 
        cam_params['fx']
    )
    
    # 提取文件名（不含路径和扩展名）
    name = os.path.basename(left_rgb_file)
    name, _ = os.path.splitext(name)
    
    # 转换左图像为ERP
    left_erp_img, left_erp_depth, left_erp_mask_valid, left_mask_active = convert_to_erp(
        left_img, 
        depth, 
        mask_valid_depth, 
        cam_params, 
        erp_size
    )
    
    # 转换右图像为ERP（使用与左图像相同的深度图，以保持一致性）
    right_erp_img, right_erp_depth, right_erp_mask_valid, right_mask_active = convert_to_erp(
        right_img, 
        depth, 
        mask_valid_depth, 
        cam_params, 
        erp_size
    )
    
    # 保存ERP输出
    save_erp_outputs(output_dir, name, left_erp_img, left_erp_depth, left_erp_mask_valid, 'left')
    save_erp_outputs(output_dir, name, right_erp_img, right_erp_depth, right_erp_mask_valid, 'right')


def main():
    """主函数"""
    args = parse_args()
    
    # 读取相机参数
    cam_params = read_camera_params(args.params_file)
    print(f"已读取相机参数: fx={cam_params['fx']}, fy={cam_params['fy']}, cx={cam_params['cx']}, cy={cam_params['cy']}, baseline={cam_params['baseline']}")
    
    # 查找图像对
    left_rgb_files, right_rgb_files, disparity_files = find_image_pairs(args.data_dir, args.num_samples)
    print(f"找到 {len(left_rgb_files)} 对图像")
    
    # 创建输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每对图像
    for i, (left_rgb_file, right_rgb_file, disparity_file) in enumerate(zip(left_rgb_files, right_rgb_files, disparity_files)):
        print(f"处理图像对 {i+1}/{len(left_rgb_files)}: {os.path.basename(left_rgb_file)}")
        
        try:
            process_image_pair(
                left_rgb_file, 
                right_rgb_file, 
                disparity_file, 
                cam_params, 
                output_dir
            )
        except Exception as e:
            print(f"处理图像对时出错: {e}")
    
    print(f"转换完成，输出保存在 {output_dir}")


if __name__ == "__main__":
    main()