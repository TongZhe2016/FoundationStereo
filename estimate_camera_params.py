#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
估计相机内参和baseline的脚本

这个脚本使用VGGT（Visual Geometry Grounded Transformer）从双目图像对中
估计相机内参和baseline，并将结果保存为标准格式。

使用方法:
    python estimate_camera_params.py --data_dir data --params_output estimated_K.txt
    
    或者直接指定左右图像:
    python estimate_camera_params.py --left_image data/left/rgb/image.jpg --right_image data/right/rgb/image.jpg

输出格式:
    fx 0.0 cx 0.0 fy cy 0.0 0.0 1.0
    baseline
    
    其中:
    - fx, fy: 焦距
    - cx, cy: 主点坐标
    - baseline: 双目相机的基线长度
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from glob import glob



# 导入VGGT相关模块
sys.path.append("vggt/")
sys.path.append("depth_any_camera")
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='从双目图像对中估计相机内参和baseline')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='包含pinhole相机数据的目录（默认：data）')
    parser.add_argument('--params_output', type=str, default='estimated_K.txt',
                        help='保存估计的相机参数的文件路径（默认：estimated_K.txt）')
    parser.add_argument('--left_image', type=str, default=None,
                        help='用于估计相机参数的左图像路径（如果不指定，将使用data/left/rgb中的第一个图像）')
    parser.add_argument('--right_image', type=str, default=None,
                        help='用于估计相机参数的右图像路径（如果不指定，将使用data/right/rgb中的第一个图像）')
    return parser.parse_args()


def find_first_image(directory):
    """查找目录中的第一个图像文件（按字母顺序排序）"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob(os.path.join(directory, ext)))
    
    if all_images:
        # 按字母顺序排序
        all_images.sort()
        return all_images[0]
    return None


def save_camera_params(intrinsic, baseline, output_path):
    """
    将相机内参和baseline保存为标准格式
    
    格式：
    fx 0.0 cx 0.0 fy cy 0.0 0.0 1.0
    baseline
    """
    # 检查内参矩阵的形状
    if intrinsic.shape != (3, 3):
        raise ValueError(f"内参矩阵形状不正确: {intrinsic.shape}，应为 (3, 3)")
    
    # 提取内参矩阵的元素
    fx = intrinsic[0, 0].item()
    fy = intrinsic[1, 1].item()
    cx = intrinsic[0, 2].item()
    cy = intrinsic[1, 2].item()
    
    # 检查参数是否合理
    if fx <= 0 or fy <= 0:
        print(f"警告: 焦距可能不正确 (fx={fx}, fy={fy})")
    
    if baseline <= 0:
        print(f"警告: baseline可能不正确 ({baseline})")
    
    # 打印估计的相机参数
    print("\n估计的相机参数:")
    print(f"焦距: fx={fx}, fy={fy}")
    print(f"主点: cx={cx}, cy={cy}")
    print(f"Baseline: {baseline}")
    
    # 按照标准格式写入文件
    with open(output_path, 'w') as f:
        f.write(f"{fx} 0.0 {cx} 0.0 {fy} {cy} 0.0 0.0 1.0\n")
        f.write(f"{baseline}\n")
    
    print(f"\n相机参数已保存到 {output_path}")


def estimate_baseline(extrinsics, intrinsic=None, scale_factor=None):
    """
    从双目相机的外参矩阵中估计baseline
    
    假设：
    - extrinsics[0, 0] 是左相机的外参矩阵
    - extrinsics[0, 1] 是右相机的外参矩阵
    - baseline是两个相机中心之间的距离
    - intrinsic是相机内参矩阵（可选）
    - scale_factor是图像缩放比例（可选），用于将内参还原到原始图像尺寸
    
    注意：VGGT返回的外参矩阵形状为 [batch_size, sequence_length, 3, 4]
    """
    # 提取左右相机的平移向量
    left_translation = extrinsics[0, 0, :3, 3]
    right_translation = extrinsics[0, 1, :3, 3]
    
    # 计算两个相机中心之间的欧氏距离
    baseline_euclidean = torch.norm(right_translation - left_translation).item()
    
    # 对于标准双目设置，baseline通常是x轴方向的距离
    baseline_x = abs(right_translation[0] - left_translation[0]).item()
    
    # 计算相机之间的相对旋转
    left_rotation = extrinsics[0, 0, :3, :3]
    right_rotation = extrinsics[0, 1, :3, :3]
    relative_rotation = torch.matmul(right_rotation, left_rotation.transpose(-2, -1))
    
    # 检查相机是否大致平行（标准双目设置）
    rotation_trace = relative_rotation[0, 0] + relative_rotation[1, 1] + relative_rotation[2, 2]
    is_parallel = rotation_trace > 2.9  # 接近3表示旋转矩阵接近单位矩阵
    
    # 如果提供了缩放比例，则将baseline还原到原始图像尺寸
    if scale_factor is not None:
        baseline_x = baseline_x / scale_factor
        baseline_euclidean = baseline_euclidean / scale_factor
    
    if is_parallel:
        print("检测到标准双目设置（相机大致平行）")
        if scale_factor is not None:
            print(f"X轴方向的baseline（已还原到原始图像尺寸）: {baseline_x}")
            print(f"欧氏距离baseline（已还原到原始图像尺寸）: {baseline_euclidean}")
        else:
            print(f"X轴方向的baseline: {baseline_x}")
            print(f"欧氏距离baseline: {baseline_euclidean}")
        
        # 如果提供了内参矩阵，则输出相机内参
        if intrinsic is not None:
            # 提取内参矩阵的元素
            fx = intrinsic[0, 0].item()
            fy = intrinsic[1, 1].item()
            cx = intrinsic[0, 2].item()
            cy = intrinsic[1, 2].item()
            
            # 如果提供了缩放比例，则将内参还原到原始图像尺寸
            if scale_factor is not None:
                fx = fx * scale_factor
                fy = fy * scale_factor
                cx = cx * scale_factor
                cy = cy * scale_factor
                print("\n标准双目设置下的相机内参（已还原到原始图像尺寸）:")
            else:
                print("\n标准双目设置下的相机内参（基于处理后的图像尺寸）:")
                
            print(f"焦距: fx={fx}, fy={fy}")
            print(f"主点: cx={cx}, cy={cy}")
        
        return baseline_x
    else:
        print("检测到非标准双目设置（相机不平行）")
        if scale_factor is not None:
            print(f"欧氏距离baseline（已还原到原始图像尺寸）: {baseline_euclidean}")
        else:
            print(f"欧氏距离baseline: {baseline_euclidean}")
        return baseline_euclidean


def validate_camera_params(intrinsic, baseline, image_shape):
    """
    验证估计的相机参数是否合理
    
    参数:
        intrinsic: 相机内参矩阵
        baseline: 双目相机的基线长度
        image_shape: 图像形状 (高度, 宽度)
    
    返回:
        bool: 参数是否合理
    """
    height, width = image_shape
    
    # 检查焦距是否在合理范围内（通常在0.5到2倍图像宽度之间）
    fx = intrinsic[0, 0].item()
    fy = intrinsic[1, 1].item()
    if fx < 0.5 * width or fx > 2.0 * width or fy < 0.5 * height or fy > 2.0 * height:
        print(f"警告: 焦距可能不合理 (fx={fx}, fy={fy})")
        print(f"图像尺寸: {width}x{height}")
        return False
    
    # 检查主点是否接近图像中心
    cx = intrinsic[0, 2].item()
    cy = intrinsic[1, 2].item()
    if abs(cx - width/2) > width/4 or abs(cy - height/2) > height/4:
        print(f"警告: 主点可能不合理 (cx={cx}, cy={cy})")
        print(f"图像中心: ({width/2}, {height/2})")
        return False
    
    # 检查baseline是否在合理范围内（通常在0.01到0.2米之间）
    if baseline < 0.01 or baseline > 0.2:
        print(f"警告: baseline可能不合理 ({baseline})")
        return False
    
    return True


def main():
    """主函数"""
    args = parse_args()
    
    # 检查数据目录是否存在
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录 {data_dir} 不存在")
    
    # 确定左右图像路径
    if args.left_image:
        left_image_path = args.left_image
    else:
        left_rgb_dir = data_dir / 'left' / 'rgb'
        if not left_rgb_dir.exists():
            raise FileNotFoundError(f"左相机RGB图像目录 {left_rgb_dir} 不存在")
        left_image_path = find_first_image(str(left_rgb_dir))
        if not left_image_path:
            raise FileNotFoundError(f"在 {left_rgb_dir} 中未找到图像")
    
    if args.right_image:
        right_image_path = args.right_image
    else:
        right_rgb_dir = data_dir / 'right' / 'rgb'
        if not right_rgb_dir.exists():
            raise FileNotFoundError(f"右相机RGB图像目录 {right_rgb_dir} 不存在")
        right_image_path = find_first_image(str(right_rgb_dir))
        if not right_image_path:
            raise FileNotFoundError(f"在 {right_rgb_dir} 中未找到图像")
    
    print(f"使用左图像: {left_image_path}")
    print(f"使用右图像: {right_image_path}")
    
    # 获取原始图像尺寸
    original_left_img = Image.open(left_image_path)
    original_width, original_height = original_left_img.size
    print(f"原始图像尺寸: {original_width}x{original_height}")
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 设置数据类型
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    # 初始化模型并加载预训练权重
    print("初始化VGGT模型...")
    try:
        model = VGGT().to(device)
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    except Exception as e:
        raise RuntimeError(f"加载VGGT模型失败: {e}")
    
    # 加载和预处理图像
    print("加载和预处理图像...")
    try:
        # 确保图像文件存在
        for img_path in [left_image_path, right_image_path]:
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"图像文件不存在: {img_path}")
        
        image_paths = [left_image_path, right_image_path]
        # 注意：load_and_preprocess_images函数默认mode为'crop'，会将图像等比例缩放到宽为518。
        # 在后续处理内参外参的时候需要注意图像的实际尺寸，将内参外参还原到原始尺寸。
        images = load_and_preprocess_images(image_paths).to(device)
        print(f"处理后图像形状: {images.shape}")
        
        # 计算缩放比例（原始宽度与处理后宽度的比值）
        processed_width = images.shape[-1]
        scale_factor = original_width / processed_width
        print(f"图像缩放比例: {scale_factor}")
    except Exception as e:
        raise RuntimeError(f"加载和预处理图像失败: {e}")
    
    # 使用模型估计相机参数
    print("使用VGGT估计相机参数...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # 预测相机参数
            predictions = model(images)
            pose_enc = predictions["pose_enc"]
            
            # 转换为外参和内参矩阵
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    
    # 打印相机参数形状，帮助调试
    print(f"外参矩阵形状: {extrinsic.shape}")
    print(f"内参矩阵形状: {intrinsic.shape}")
    
    # 提取内参矩阵（使用左相机的内参）
    left_intrinsic = intrinsic[0, 0]
    
    # 估计baseline，同时传递内参矩阵和缩放比例
    baseline = estimate_baseline(extrinsic, left_intrinsic, scale_factor)
    
    # 验证估计的参数
    is_valid = validate_camera_params(left_intrinsic, baseline, images.shape[-2:])
    if not is_valid:
        print("警告: 估计的相机参数可能不准确，请检查输入图像和模型")
    
    # 保存相机参数（将内参还原到原始图像尺寸）
    try:
        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(os.path.abspath(args.params_output))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 创建还原到原始尺寸的内参矩阵
        scaled_intrinsic = left_intrinsic.clone()
        scaled_intrinsic[0, 0] = left_intrinsic[0, 0] * scale_factor  # fx
        scaled_intrinsic[1, 1] = left_intrinsic[1, 1] * scale_factor  # fy
        scaled_intrinsic[0, 2] = left_intrinsic[0, 2] * scale_factor  # cx
        scaled_intrinsic[1, 2] = left_intrinsic[1, 2] * scale_factor  # cy
        
        save_camera_params(scaled_intrinsic, baseline, args.params_output)
    except Exception as e:
        raise RuntimeError(f"保存相机参数失败: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)