#!/usr/bin/env python
"""
使用VGGT从双目图像对中估计相机内参和baseline，然后将pinhole相机数据转换为ERP全景图像。
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


def estimate_camera_params(left_image_path, right_image_path, model, device):
    """使用VGGT从左右图像对中估计相机内参和baseline"""
    # 加载和预处理图像
    image_paths = [left_image_path, right_image_path]
    images = load_and_preprocess_images(image_paths).to(device)
    
    # 运行推理
    print(f"正在处理图像对: {os.path.basename(left_image_path)} 和 {os.path.basename(right_image_path)}")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            predictions = model(images)
    
    # 转换pose_enc为相机参数
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    
    # 将tensors转换为numpy数组
    extrinsic = extrinsic.cpu().numpy().squeeze(0)  # 移除batch维度
    intrinsic = intrinsic.cpu().numpy().squeeze(0)  # 移除batch维度
    
    # 计算baseline（两个相机之间的距离）
    # 假设第一个相机是参考相机，第二个相机是相对于第一个相机的
    baseline = np.linalg.norm(extrinsic[1, :3, 3] - extrinsic[0, :3, 3])
    
    # 使用第一个相机的内参作为结果
    camera_intrinsic = intrinsic[0]
    
    return camera_intrinsic, baseline


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


def convert_to_erp(data_dir, output_dir, camera_params_path):
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
    
    # 处理每个图像
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
            print(f"已处理: {rgb_file} -> {os.path.basename(erp_rgb_path)}")
        except Exception as e:
            print(f"处理 {rgb_file} 时出错: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="从双目图像对中估计相机内参和baseline，然后将pinhole相机数据转换为ERP全景图像")
    parser.add_argument("--data_dir", type=str, default="data", help="包含pinhole相机数据的目录")
    parser.add_argument("--output_dir", type=str, default="erp_output", help="保存ERP数据的目录")
    parser.add_argument("--params_output", type=str, default="estimated_K.txt", help="保存估计的相机参数的文件路径")
    parser.add_argument("--left_image", type=str, help="用于估计相机参数的左图像路径（如果不指定，将使用data/left/rgb中的第一个图像）")
    parser.add_argument("--right_image", type=str, help="用于估计相机参数的右图像路径（如果不指定，将使用data/right/rgb中的第一个图像）")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model, device = load_model()
    
    # 如果未指定图像，则使用默认图像
    if args.left_image is None:
        left_images = sorted(glob.glob(os.path.join(args.data_dir, "left/rgb/*.jpg")))
        if not left_images:
            raise ValueError("未找到左图像，请指定--left_image参数")
        args.left_image = left_images[0]
    
    if args.right_image is None:
        # 尝试找到匹配的右图像（相同的文件名）
        left_basename = os.path.basename(args.left_image)
        right_image_path = os.path.join(args.data_dir, "right/rgb", left_basename)
        if os.path.exists(right_image_path):
            args.right_image = right_image_path
        else:
            # 如果没有找到匹配的右图像，使用第一个右图像
            right_images = sorted(glob.glob(os.path.join(args.data_dir, "right/rgb/*.jpg")))
            if not right_images:
                raise ValueError("未找到右图像，请指定--right_image参数")
            args.right_image = right_images[0]
    
    print(f"使用左图像: {args.left_image}")
    print(f"使用右图像: {args.right_image}")
    
    # 估计相机内参和baseline
    camera_intrinsic, baseline = estimate_camera_params(args.left_image, args.right_image, model, device)
    
    # 保存相机参数
    save_camera_params(camera_intrinsic, baseline, args.params_output)
    
    # 将pinhole相机数据转换为ERP全景图像
    convert_to_erp(args.data_dir, args.output_dir, args.params_output)
    
    print(f"处理完成！ERP全景图像已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()