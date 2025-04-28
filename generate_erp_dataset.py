#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成ERP全景双目深度数据集的脚本

这个脚本整合了estimate_camera_params.py和convert_to_erp.py的功能，
提供一个完整的工作流程，从估计相机参数到生成ERP全景数据集。

使用方法:
    python generate_erp_dataset.py --data_dir data --output_dir erp_output

输出:
    - estimated_K.txt: 估计的相机内参和baseline
    - erp_output/: 包含转换后的ERP全景图像、深度图和有效区域掩码
    - 如果指定了--visualize参数，还会生成可视化结果
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from tqdm import tqdm
import logging
import time
import json
from datetime import datetime

# 导入相机参数估计模块
from estimate_camera_params import (
    find_first_image, 
    save_camera_params, 
    estimate_baseline, 
    validate_camera_params
)


# 导入ERP转换模块
from convert_to_erp import (
    read_camera_params,
    find_image_pairs,
    disparity_to_depth,
    convert_to_erp,
    save_erp_outputs,
    process_image_pair
)

# 导入ERP几何变换函数
from depth_any_camera.dac.utils.erp_geometry import cam_to_erp_patch_fast


# 导入VGGT相关模块
sys.path.append("vggt/")
sys.path.append("depth_any_camera")
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def setup_logger(log_file=None):
    """
    设置日志记录器
    
    参数:
        log_file: 日志文件路径（如果为None，则只输出到控制台）
    
    返回:
        logger: 日志记录器
    """
    logger = logging.getLogger('generate_erp_dataset')
    logger.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 如果指定了日志文件，创建文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='生成ERP全景双目深度数据集')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='包含pinhole相机数据的目录（默认：data）')
    parser.add_argument('--output_dir', type=str, default='erp_output',
                        help='保存ERP数据的目录（默认：erp_output）')
    parser.add_argument('--params_output', type=str, default='estimated_K.txt',
                        help='保存估计的相机参数的文件路径（默认：estimated_K.txt）')
    parser.add_argument('--num_pairs', type=int, default=5,
                        help='用于估计相机参数的图像对数量（默认：5）')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='要转换为ERP的图像数量（如果不指定，将处理所有图像）')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化结果（默认：False）')
    parser.add_argument('--skip_estimation', action='store_true',
                        help='跳过相机参数估计步骤，直接使用已有的参数文件（默认：False）')
    parser.add_argument('--erp_size', type=str, default='1024,2048',
                        help='ERP图像的大小，格式为"高度,宽度"（默认：1024,2048）')
    parser.add_argument('--log_file', type=str, default=None,
                        help='日志文件路径（如果不指定，则只输出到控制台）')
    return parser.parse_args()


def estimate_camera_parameters(data_dir, params_output, num_pairs=5, logger=None):
    """
    估计相机内参和baseline
    
    参数:
        data_dir: 包含pinhole相机数据的目录
        params_output: 保存估计的相机参数的文件路径
        num_pairs: 用于估计相机参数的图像对数量
        logger: 日志记录器
    
    返回:
        cam_params: 相机参数字典
    """
    if logger is None:
        logger = logging.getLogger('generate_erp_dataset')
    
    logger.info("开始估计相机参数...")
    
    # 检查数据目录是否存在
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录 {data_dir} 不存在")
    
    # 查找左右图像目录
    left_rgb_dir = data_dir / 'left' / 'rgb'
    right_rgb_dir = data_dir / 'right' / 'rgb'
    
    if not left_rgb_dir.exists():
        raise FileNotFoundError(f"左相机RGB图像目录 {left_rgb_dir} 不存在")
    if not right_rgb_dir.exists():
        raise FileNotFoundError(f"右相机RGB图像目录 {right_rgb_dir} 不存在")
    
    # 查找所有左相机RGB图像
    left_rgb_files = sorted(glob(str(left_rgb_dir / '*.jpg')) + glob(str(left_rgb_dir / '*.png')))
    
    if not left_rgb_files:
        raise FileNotFoundError(f"在 {left_rgb_dir} 中未找到图像")
    
    # 限制处理的图像对数量
    left_rgb_files = left_rgb_files[:num_pairs]
    
    # 查找对应的右相机RGB图像
    right_rgb_files = []
    
    for left_rgb_file in left_rgb_files:
        # 提取文件名（不含路径和扩展名）
        filename = os.path.basename(left_rgb_file)
        name, ext = os.path.splitext(filename)
        
        # 查找对应的右相机RGB图像
        right_rgb_file = str(right_rgb_dir / filename)
        if not os.path.exists(right_rgb_file):
            # 尝试不同的扩展名
            for possible_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                alt_right_rgb_file = str(right_rgb_dir / f"{name}{possible_ext}")
                if os.path.exists(alt_right_rgb_file):
                    right_rgb_file = alt_right_rgb_file
                    break
            else:
                logger.warning(f"未找到对应的右相机RGB图像: {right_rgb_file}")
                continue
        
        right_rgb_files.append(right_rgb_file)
    
    # 确保至少有一对图像
    if not right_rgb_files:
        raise FileNotFoundError("未找到有效的图像对")
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 设置数据类型
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    # 初始化模型并加载预训练权重
    logger.info("初始化VGGT模型...")
    try:
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    except Exception as e:
        logger.warning(f"从Hugging Face加载模型失败: {e}")
        logger.info("尝试从URL直接加载...")
        try:
            model = VGGT().to(device)
            _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
            model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        except Exception as e:
            raise RuntimeError(f"加载VGGT模型失败: {e}")
    
    # 处理每对图像并累积结果
    all_intrinsics = []
    all_baselines = []
    
    for i, (left_rgb_file, right_rgb_file) in enumerate(zip(left_rgb_files, right_rgb_files)):
        logger.info(f"处理图像对 {i+1}/{len(right_rgb_files)}: {os.path.basename(left_rgb_file)}")
        
        try:
            # 加载和预处理图像
            image_paths = [left_rgb_file, right_rgb_file]
            images = load_and_preprocess_images(image_paths).to(device)
            
            # 使用模型估计相机参数
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    # 预测相机参数
                    predictions = model(images)
                    pose_enc = predictions["pose_enc"]
                    
                    # 转换为外参和内参矩阵
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            
            # 提取内参矩阵（使用左相机的内参）
            left_intrinsic = intrinsic[0, 0]
            
            # 估计baseline
            baseline = estimate_baseline(extrinsic)
            
            # 验证估计的参数
            is_valid = validate_camera_params(left_intrinsic, baseline, images.shape[-2:])
            if is_valid:
                all_intrinsics.append(left_intrinsic)
                all_baselines.append(baseline)
            else:
                logger.warning(f"图像对 {i+1} 的估计参数无效，已跳过")
        
        except Exception as e:
            logger.error(f"处理图像对 {i+1} 时出错: {e}")
    
    # 确保至少有一组有效的参数
    if not all_intrinsics:
        raise RuntimeError("未能估计出有效的相机参数")
    
    # 计算平均内参和baseline
    avg_intrinsic = torch.stack(all_intrinsics).mean(dim=0)
    avg_baseline = sum(all_baselines) / len(all_baselines)
    
    # 保存相机参数
    try:
        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(os.path.abspath(params_output))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        save_camera_params(avg_intrinsic, avg_baseline, params_output)
        logger.info(f"相机参数已保存到 {params_output}")
    except Exception as e:
        raise RuntimeError(f"保存相机参数失败: {e}")
    
    # 创建相机参数字典
    fx = avg_intrinsic[0, 0].item()
    fy = avg_intrinsic[1, 1].item()
    cx = avg_intrinsic[0, 2].item()
    cy = avg_intrinsic[1, 2].item()
    
    cam_params = {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'baseline': avg_baseline
    }
    
    return cam_params
def convert_to_erp_dataset(data_dir, output_dir, cam_params, num_samples=None, erp_size=(1024, 2048), logger=None):
    """
    将pinhole相机数据转换为ERP全景数据集
    
    参数:
        data_dir: 包含pinhole相机数据的目录
        output_dir: 保存ERP数据的目录
        cam_params: 相机参数字典
        num_samples: 要转换的图像数量
        erp_size: ERP图像的大小 (高度, 宽度)
        logger: 日志记录器
    
    返回:
        processed_pairs: 处理的图像对数量
    """
    if logger is None:
        logger = logging.getLogger('generate_erp_dataset')
    
    logger.info("开始转换为ERP全景数据集...")
    
    # 查找图像对
    left_rgb_files, right_rgb_files, disparity_files = find_image_pairs(data_dir, num_samples)
    logger.info(f"找到 {len(left_rgb_files)} 对图像")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每对图像
    processed_pairs = 0
    for i, (left_rgb_file, right_rgb_file, disparity_file) in enumerate(zip(left_rgb_files, right_rgb_files, disparity_files)):
        logger.info(f"处理图像对 {i+1}/{len(left_rgb_files)}: {os.path.basename(left_rgb_file)}")
        
        try:
            process_image_pair(
                left_rgb_file, 
                right_rgb_file, 
                disparity_file, 
                cam_params, 
                output_dir,
                erp_size
            )
            processed_pairs += 1
        except Exception as e:
            logger.error(f"处理图像对时出错: {e}")
    
    logger.info(f"转换完成，成功处理 {processed_pairs} 对图像，输出保存在 {output_dir}")
    return processed_pairs


def validate_erp_dataset(output_dir, logger=None):
    """
    验证生成的ERP全景数据集
    
    参数:
        output_dir: ERP数据集目录
        logger: 日志记录器
    
    返回:
        is_valid: 数据集是否有效
    """
    if logger is None:
        logger = logging.getLogger('generate_erp_dataset')
    
    logger.info("开始验证ERP全景数据集...")
    
    # 检查左右图像目录
    left_rgb_dir = os.path.join(output_dir, 'left', 'rgb')
    right_rgb_dir = os.path.join(output_dir, 'right', 'rgb')
    left_depth_dir = os.path.join(output_dir, 'left', 'depth')
    right_depth_dir = os.path.join(output_dir, 'right', 'depth')
    left_mask_dir = os.path.join(output_dir, 'left', 'mask')
    right_mask_dir = os.path.join(output_dir, 'right', 'mask')
    
    # 检查目录是否存在
    for dir_path in [left_rgb_dir, right_rgb_dir, left_depth_dir, right_depth_dir, left_mask_dir, right_mask_dir]:
        if not os.path.exists(dir_path):
            logger.error(f"目录不存在: {dir_path}")
            return False
    
    # 查找所有左相机ERP图像
    left_erp_files = sorted(glob(os.path.join(left_rgb_dir, '*_erp.png')))
    
    if not left_erp_files:
        logger.error(f"在 {left_rgb_dir} 中未找到ERP图像")
        return False
    
    # 验证每对ERP图像
    valid_pairs = 0
    total_pairs = 0
    
    for left_erp_file in left_erp_files:
        # 提取文件名（不含路径和扩展名）
        filename = os.path.basename(left_erp_file)
        
        # 查找对应的右相机ERP图像
        right_erp_file = os.path.join(right_rgb_dir, filename)
        if not os.path.exists(right_erp_file):
            logger.warning(f"未找到对应的右相机ERP图像: {right_erp_file}")
            continue
        
        # 查找对应的深度图和掩码
        left_depth_file = os.path.join(left_depth_dir, filename)
        right_depth_file = os.path.join(right_depth_dir, filename)
        left_mask_file = os.path.join(left_mask_dir, filename)
        right_mask_file = os.path.join(right_mask_dir, filename)
        
        # 检查文件是否存在
        if not all(os.path.exists(f) for f in [left_depth_file, right_depth_file, left_mask_file, right_mask_file]):
            logger.warning(f"缺少对应的深度图或掩码: {filename}")
            continue
        
        # 读取图像
        left_erp = cv2.imread(left_erp_file)
        right_erp = cv2.imread(right_erp_file)
        left_depth = cv2.imread(left_depth_file, cv2.IMREAD_UNCHANGED)
        right_depth = cv2.imread(right_depth_file, cv2.IMREAD_UNCHANGED)
        left_mask = cv2.imread(left_mask_file, cv2.IMREAD_UNCHANGED)
        right_mask = cv2.imread(right_mask_file, cv2.IMREAD_UNCHANGED)
        
        # 检查图像尺寸是否一致
        shapes = [left_erp.shape, right_erp.shape, left_depth.shape[:2], right_depth.shape[:2], 
                  left_mask.shape[:2], right_mask.shape[:2]]
        if len(set(str(s) for s in shapes)) > 1:
            logger.warning(f"图像尺寸不一致: {filename}")
            continue
        
        # 检查有效区域掩码的重叠度
        if left_mask.ndim > 2:
            left_mask = left_mask[:, :, 0]
        if right_mask.ndim > 2:
            right_mask = right_mask[:, :, 0]
        
        left_valid = left_mask > 0
        right_valid = right_mask > 0
        overlap = np.logical_and(left_valid, right_valid)
        overlap_ratio = np.sum(overlap) / max(1, min(np.sum(left_valid), np.sum(right_valid)))
        
        # 检查深度图的一致性
        if overlap_ratio > 0.5:
            # 在重叠区域比较深度值
            left_depth_float = left_depth.astype(np.float32)
            right_depth_float = right_depth.astype(np.float32)
            
            # 计算深度差异
            depth_diff = np.abs(left_depth_float - right_depth_float)
            depth_diff[~overlap] = 0
            
            # 计算相对深度差异
            max_depth = max(np.max(left_depth_float), np.max(right_depth_float))
            if max_depth > 0:
                relative_diff = np.mean(depth_diff[overlap]) / max_depth
                
                if relative_diff < 0.1:  # 相对差异小于10%
                    valid_pairs += 1
                else:
                    logger.warning(f"深度图不一致: {filename}, 相对差异: {relative_diff:.2f}")
            else:
                valid_pairs += 1  # 如果深度全为0，也认为是有效的
        else:
            logger.warning(f"有效区域重叠度不足: {filename}, 重叠率: {overlap_ratio:.2f}")
        
        total_pairs += 1
    
    # 计算有效率
    if total_pairs > 0:
        valid_ratio = valid_pairs / total_pairs
        logger.info(f"数据集验证完成: {valid_pairs}/{total_pairs} 对有效 ({valid_ratio:.2%})")
        
        # 如果有效率大于80%，认为数据集有效
        is_valid = valid_ratio > 0.8
        if is_valid:
            logger.info("数据集验证通过")
        else:
            logger.warning("数据集验证未通过，有效率低于80%")
        
        return is_valid
    else:
        logger.error("未找到有效的ERP图像对")
        return False
def visualize_results(data_dir, output_dir, num_samples=3, logger=None):
    """
    可视化原始图像和转换后的ERP图像
    
    参数:
        data_dir: 包含pinhole相机数据的目录
        output_dir: 保存ERP数据的目录
        num_samples: 要可视化的样本数量
        logger: 日志记录器
    """
    if logger is None:
        logger = logging.getLogger('generate_erp_dataset')
    
    logger.info("开始可视化结果...")
    
    # 查找原始图像
    left_rgb_dir = os.path.join(data_dir, 'left', 'rgb')
    left_rgb_files = sorted(glob(os.path.join(left_rgb_dir, '*.jpg')) + glob(os.path.join(left_rgb_dir, '*.png')))
    
    if not left_rgb_files:
        logger.error(f"在 {left_rgb_dir} 中未找到图像")
        return
    
    # 查找ERP图像
    left_erp_dir = os.path.join(output_dir, 'left', 'rgb')
    left_erp_files = sorted(glob(os.path.join(left_erp_dir, '*_erp.png')))
    
    if not left_erp_files:
        logger.error(f"在 {left_erp_dir} 中未找到ERP图像")
        return
    
    # 创建可视化输出目录
    vis_dir = os.path.join(output_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 限制样本数量
    num_samples = min(num_samples, len(left_rgb_files), len(left_erp_files))
    
    # 为了可视化，选择均匀分布的样本
    if len(left_rgb_files) > num_samples:
        indices = np.linspace(0, len(left_rgb_files) - 1, num_samples, dtype=int)
        left_rgb_files = [left_rgb_files[i] for i in indices]
    
    # 查找对应的ERP图像
    selected_pairs = []
    for left_rgb_file in left_rgb_files:
        # 提取文件名（不含路径和扩展名）
        filename = os.path.basename(left_rgb_file)
        name, _ = os.path.splitext(filename)
        
        # 查找对应的ERP图像
        erp_file = None
        for f in left_erp_files:
            if name in f:
                erp_file = f
                break
        
        if erp_file:
            selected_pairs.append((left_rgb_file, erp_file))
    
    # 可视化每对图像
    for i, (rgb_file, erp_file) in enumerate(selected_pairs):
        try:
            # 读取图像
            rgb_img = cv2.imread(rgb_file)
            erp_img = cv2.imread(erp_file)
            
            # 转换为RGB（OpenCV默认是BGR）
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            erp_img = cv2.cvtColor(erp_img, cv2.COLOR_BGR2RGB)
            
            # 创建图像
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # 显示原始图像
            axes[0].imshow(rgb_img)
            axes[0].set_title('原始Pinhole图像')
            axes[0].axis('off')
            
            # 显示ERP图像
            axes[1].imshow(erp_img)
            axes[1].set_title('ERP全景图像')
            axes[1].axis('off')
            
            # 保存图像
            name = os.path.basename(rgb_file).split('.')[0]
            fig.suptitle(f'样本 {i+1}: {name}')
            plt.tight_layout()
            
            # 保存可视化结果
            vis_file = os.path.join(vis_dir, f'vis_{name}.png')
            plt.savefig(vis_file, dpi=150)
            plt.close()
            
            logger.info(f"可视化结果已保存: {vis_file}")
        
        except Exception as e:
            logger.error(f"可视化图像时出错: {e}")
    
    logger.info(f"可视化完成，结果保存在 {vis_dir}")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志记录器
    if args.log_file is None:
        # 如果未指定日志文件，创建一个默认的日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        args.log_file = os.path.join(log_dir, f"generate_erp_dataset_{timestamp}.log")
    
    logger = setup_logger(args.log_file)
    
    # 记录开始时间
    start_time = time.time()
    logger.info(f"开始生成ERP全景双目深度数据集...")
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    
    try:
        # 解析ERP图像大小
        erp_size = tuple(map(int, args.erp_size.split(',')))
        if len(erp_size) != 2:
            raise ValueError(f"ERP图像大小格式不正确: {args.erp_size}，应为'高度,宽度'")
        
        # 步骤1：估计相机内参和baseline
        if args.skip_estimation and os.path.exists(args.params_output):
            logger.info(f"跳过相机参数估计，使用已有的参数文件: {args.params_output}")
            cam_params = read_camera_params(args.params_output)
            logger.info(f"已读取相机参数: fx={cam_params['fx']}, fy={cam_params['fy']}, cx={cam_params['cx']}, cy={cam_params['cy']}, baseline={cam_params['baseline']}")
        else:
            logger.info(f"使用 {args.num_pairs} 对图像估计相机参数...")
            cam_params = estimate_camera_parameters(args.data_dir, args.params_output, args.num_pairs, logger)
        
        # 步骤2：将pinhole相机数据转换为ERP全景图像
        logger.info(f"开始转换为ERP全景图像，ERP大小: {erp_size}...")
        processed_pairs = convert_to_erp_dataset(args.data_dir, args.output_dir, cam_params, args.num_samples, erp_size, logger)
        
        # 步骤3：验证生成的数据集
        if processed_pairs > 0:
            logger.info("开始验证生成的数据集...")
            is_valid = validate_erp_dataset(args.output_dir, logger)
            
            # 步骤4：如果指定了--visualize参数，显示原始图像和转换后的ERP图像
            if args.visualize and is_valid:
                logger.info("开始可视化结果...")
                visualize_results(args.data_dir, args.output_dir, 3, logger)
        else:
            logger.warning("未处理任何图像对，跳过验证和可视化步骤")
        
        # 记录总耗时
        elapsed_time = time.time() - start_time
        logger.info(f"处理完成，总耗时: {elapsed_time:.2f} 秒")
        
    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())