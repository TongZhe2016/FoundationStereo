#!/usr/bin/env python
"""
使用VGGT从双目图像对中估计相机内参和baseline的高级脚本，
支持批量处理多对图像，并将pinhole相机数据转换为ERP全景图像。
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from PIL import Image
import glob
import tqdm
import json
from pathlib import Path
import random
import matplotlib.pyplot as plt

# 添加VGGT路径
sys.path.append("vggt/")

# 添加depth_any_camera路径，使Python能找到dac模块
sys.path.append("depth_any_camera")

# 导入VGGT相关模块
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# 导入ERP转换相关模块
from depth_any_camera.process_pinhole_to_erp import process_image_to_erp, convert_disparity_to_depth


def load_model(device=None):
    """加载VGGT模型"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 从预训练模型加载
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    
    model.eval()
    model = model.to(device)
    return model, device


def find_matching_image_pairs(left_dir, right_dir, num_pairs=5):
    """查找匹配的左右图像对"""
    left_images = sorted(glob.glob(os.path.join(left_dir, "*.jpg")))
    right_images = sorted(glob.glob(os.path.join(right_dir, "*.jpg")))
    
    # 如果图像数量太少，使用所有可用的图像
    if len(left_images) < num_pairs or len(right_images) < num_pairs:
        num_pairs = min(len(left_images), len(right_images))
    
    # 随机选择图像对
    indices = random.sample(range(min(len(left_images), len(right_images))), num_pairs)
    
    pairs = []
    for i in indices:
        if i < len(left_images) and i < len(right_images):
            pairs.append((left_images[i], right_images[i]))
    
    return pairs


def estimate_camera_params_from_pairs(image_pairs, model, device):
    """从多对图像中估计相机内参和baseline"""
    all_intrinsics = []
    all_baselines = []
    
    for left_image_path, right_image_path in tqdm.tqdm(image_pairs, desc="估计相机参数"):
        # 加载和预处理图像
        image_paths = [left_image_path, right_image_path]
        images = load_and_preprocess_images(image_paths).to(device)
        
        # 运行推理
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                predictions = model(images)
        
        # 转换pose_enc为相机参数
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        
        # 将tensors转换为numpy数组
        extrinsic = extrinsic.cpu().numpy().squeeze(0)  # 移除batch维度
        intrinsic = intrinsic.cpu().numpy().squeeze(0)  # 移除batch维度
        
        # 计算baseline（两个相机之间的距离）
        baseline = np.linalg.norm(extrinsic[1, :3, 3] - extrinsic[0, :3, 3])
        
        # 使用第一个相机的内参作为结果
        camera_intrinsic = intrinsic[0]
        
        all_intrinsics.append(camera_intrinsic)
        all_baselines.append(baseline)
    
    # 计算平均值
    avg_intrinsic = np.mean(all_intrinsics, axis=0)
    avg_baseline = np.mean(all_baselines)
    
    # 计算标准差
    std_intrinsic = np.std(all_intrinsics, axis=0)
    std_baseline = np.std(all_baselines)
    
    print(f"内参矩阵平均值:\n{avg_intrinsic}")
    print(f"内参矩阵标准差:\n{std_intrinsic}")
    print(f"Baseline平均值: {avg_baseline}")
    print(f"Baseline标准差: {std_baseline}")
    
    return avg_intrinsic, avg_baseline, all_intrinsics, all_baselines


def save_camera_params(camera_intrinsic, baseline, output_path):
    """将相机内参和baseline保存为与assets/K.txt相同的格式"""
    # 提取内参矩阵的元素
    fx = camera_intrinsic[0, 0]
    fy = camera_intrinsic[1, 1]
    cx = camera_intrinsic[0, 2]
    cy = camera_intrinsic[1, 2]
    
    # 按行排列内参矩阵的元素
    intrinsic_line = f"{fx} 0.0 {cx} 0.0 {fy} {cy} 0.0 0.0 1.0"
    baseline_line = f"{baseline}"
    
    # 写入文件
    with open(output_path, 'w') as f:
        f.write(intrinsic_line + '\n')
        f.write(baseline_line + '\n')
    
    print(f"相机参数已保存到: {output_path}")
    return intrinsic_line, baseline_line


def visualize_camera_params(all_intrinsics, all_baselines, output_dir):
    """可视化相机参数的分布"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取内参矩阵的元素
    fx_values = [intrinsic[0, 0] for intrinsic in all_intrinsics]
    fy_values = [intrinsic[1, 1] for intrinsic in all_intrinsics]
    cx_values = [intrinsic[0, 2] for intrinsic in all_intrinsics]
    cy_values = [intrinsic[1, 2] for intrinsic in all_intrinsics]
    
    # 创建图表
    plt.figure(figsize=(12, 10))
    
    # 绘制fx和fy的分布
    plt.subplot(2, 2, 1)
    plt.hist(fx_values, bins=10, alpha=0.7, label='fx')
    plt.hist(fy_values, bins=10, alpha=0.7, label='fy')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Focal Length Distribution')
    plt.legend()
    
    # 绘制cx和cy的分布
    plt.subplot(2, 2, 2)
    plt.hist(cx_values, bins=10, alpha=0.7, label='cx')
    plt.hist(cy_values, bins=10, alpha=0.7, label='cy')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Principal Point Distribution')
    plt.legend()
    
    # 绘制baseline的分布
    plt.subplot(2, 2, 3)
    plt.hist(all_baselines, bins=10)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Baseline Distribution')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'camera_params_distribution.png'))
    plt.close()
    
    print(f"相机参数分布图已保存到: {os.path.join(output_dir, 'camera_params_distribution.png')}")


def convert_to_erp(data_dir, output_dir, camera_params_path, num_samples=None):
    """使用估计的参数将pinhole相机数据转换为ERP全景图像"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取相机参数
    with open(camera_params_path, 'r') as f:
        lines = f.readlines()
        intrinsic_values = list(map(float, lines[0].strip().split()))
        baseline = float(lines[1].strip())
    
    # 构建相机参数字典
    cam_params = {
        "dataset": "custom",
        "fx": intrinsic_values[0],
        "fy": intrinsic_values[4],
        "cx": intrinsic_values[2],
        "cy": intrinsic_values[5],
        "camera_model": "PINHOLE"
    }
    
    # 获取左相机RGB图像列表
    rgb_dir = os.path.join(data_dir, "left/rgb")
    disparity_dir = os.path.join(data_dir, "left/disparity")
    
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
    
    # 如果指定了样本数量，随机选择图像
    if num_samples is not None and num_samples < len(rgb_files):
        rgb_files = random.sample(rgb_files, num_samples)
    
    # 处理每个图像
    successful_count = 0
    for rgb_file in tqdm.tqdm(rgb_files, desc="转换为ERP"):
        rgb_path = os.path.join(rgb_dir, rgb_file)
        disparity_file = os.path.splitext(rgb_file)[0] + '.png'
        disparity_path = os.path.join(disparity_dir, disparity_file)
        
        # 检查视差图是否存在
        if not os.path.exists(disparity_path):
            print(f"警告: 视差文件 {disparity_path} 不存在，跳过。")
            continue
        
        # 处理图像并转换为ERP
        try:
            erp_rgb_path, erp_depth_path, erp_mask_path, erp_params_path = process_image_to_erp(
                rgb_path, disparity_path, output_dir, cam_params,
                erp_h=1024, erp_w=2048, random_params=False
            )
            successful_count += 1
        except Exception as e:
            print(f"处理 {rgb_file} 时出错: {str(e)}")
    
    print(f"成功处理了 {successful_count} 个图像（共 {len(rgb_files)} 个）")


def main():
    parser = argparse.ArgumentParser(description="从双目图像对中估计相机内参和baseline，然后将pinhole相机数据转换为ERP全景图像")
    parser.add_argument("--data_dir", type=str, default="data", help="包含pinhole相机数据的目录")
    parser.add_argument("--output_dir", type=str, default="erp_output", help="保存ERP数据的目录")
    parser.add_argument("--params_output", type=str, default="estimated_K.txt", help="保存估计的相机参数的文件路径")
    parser.add_argument("--num_pairs", type=int, default=5, help="用于估计相机参数的图像对数量")
    parser.add_argument("--num_samples", type=int, default=None, help="要转换为ERP的图像数量（如果不指定，将处理所有图像）")
    parser.add_argument("--visualize", action="store_true", help="是否可视化相机参数的分布")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model, device = load_model()
    
    # 查找匹配的图像对
    left_rgb_dir = os.path.join(args.data_dir, "left/rgb")
    right_rgb_dir = os.path.join(args.data_dir, "right/rgb")
    image_pairs = find_matching_image_pairs(left_rgb_dir, right_rgb_dir, args.num_pairs)
    
    if not image_pairs:
        raise ValueError("未找到匹配的图像对")
    
    print(f"找到 {len(image_pairs)} 对匹配的图像")
    
    # 估计相机内参和baseline
    avg_intrinsic, avg_baseline, all_intrinsics, all_baselines = estimate_camera_params_from_pairs(
        image_pairs, model, device
    )
    
    # 保存相机参数
    save_camera_params(avg_intrinsic, avg_baseline, args.params_output)
    
    # 可视化相机参数的分布
    if args.visualize and len(image_pairs) > 1:
        visualize_camera_params(all_intrinsics, all_baselines, args.output_dir)
    
    # 将pinhole相机数据转换为ERP全景图像
    convert_to_erp(args.data_dir, args.output_dir, args.params_output, args.num_samples)
    
    print(f"处理完成！ERP全景图像已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()