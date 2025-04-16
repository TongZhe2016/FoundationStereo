#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将鱼眼相机图像转换为ERP（等距矩形投影）图像

此脚本实现了从四个鱼眼相机图像到全景ERP图像的转换。
步骤：
1. 读取四个鱼眼相机的图像和参数
2. 使用cam2world函数将像素坐标映射到3D空间单位向量
3. 考虑相机的旋转（忽略平移）
4. 将四个鱼眼相机的图像拼接成一个完整的球面
5. 将球面投影为ERP图像

使用方法：
python fisheye_to_erp.py [数据集路径] [帧索引]

例如：
python fisheye_to_erp.py OmniHouse 1
"""

import numpy as np
import cv2
import os
import yaml
import sys
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def read_ocam_model(ocam_file):
    """读取OCamCalib标定文件，获取相机模型参数"""
    with open(ocam_file, 'r') as f:
        lines = f.readlines()
    
    # 读取多项式系数 (ss)
    ss_line = lines[2].strip().split()
    ss = np.array([float(x) for x in ss_line[1:]])
    
    # 读取逆多项式系数 (invpol)
    invpol_line = lines[6].strip().split()
    invpol = np.array([float(x) for x in invpol_line[1:]])
    
    # 读取中心点坐标
    center_line = lines[10].strip().split()
    xc = float(center_line[0])
    yc = float(center_line[1])
    
    # 读取仿射参数
    affine_line = lines[14].strip().split()
    c = float(affine_line[0])
    d = float(affine_line[1])
    e = float(affine_line[2])
    
    # 读取图像尺寸
    size_line = lines[18].strip().split()
    height = int(size_line[0])
    width = int(size_line[1])
    
    return {
        'ss': ss,
        'invpol': invpol,
        'xc': xc,
        'yc': yc,
        'c': c,
        'd': d,
        'e': e,
        'height': height,
        'width': width
    }

def cam2world(m, ocam_model):
    """
    将像素坐标映射到3D空间单位向量
    
    参数:
    m: 像素坐标，形状为(2, N)的numpy数组，第一行是u坐标，第二行是v坐标
    ocam_model: 相机模型参数
    
    返回:
    M: 3D空间单位向量，形状为(3, N)的numpy数组
    """
    n_points = m.shape[1]
    
    # 提取参数
    ss = ocam_model['ss']
    xc = ocam_model['xc']
    yc = ocam_model['yc']
    c = ocam_model['c']
    d = ocam_model['d']
    e = ocam_model['e']
    
    # 构建仿射矩阵
    A = np.array([[c, d], [e, 1]])
    A_inv = np.linalg.inv(A)
    
    # 构建平移向量
    T = np.tile(np.array([[xc], [yc]]), (1, n_points))
    
    # 坐标系转换
    m_prime = A_inv @ (m - T)
    
    # 计算距离
    rho = np.sqrt(np.sum(m_prime**2, axis=0))
    
    # 计算z坐标
    z_prime = np.zeros_like(rho)
    for i in range(len(ss)):
        z_prime += ss[i] * rho**i
    
    # 构建3D向量
    w = np.vstack((m_prime, z_prime))
    
    # 单位化
    norm = np.sqrt(np.sum(w**2, axis=0))
    M = w / norm
    
    return M

def read_config(config_file):
    """读取配置文件"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def world2cam(M, ocam_model):
    """
    将3D空间单位向量映射到像素坐标
    
    参数:
    M: 3D空间单位向量，形状为(3, N)的numpy数组
    ocam_model: 相机模型参数
    
    返回:
    m: 像素坐标，形状为(2, N)的numpy数组
    """
    # 提取参数
    invpol = ocam_model['invpol']
    xc = ocam_model['xc']
    yc = ocam_model['yc']
    c = ocam_model['c']
    d = ocam_model['d']
    e = ocam_model['e']
    
    # 计算向量的范数
    norm = np.sqrt(M[0, :]**2 + M[1, :]**2)
    
    # 处理z轴上的点
    z_axis_points = norm == 0
    norm[z_axis_points] = np.finfo(float).eps  # 避免除以零
    
    # 计算theta角（与z轴的夹角）
    theta = np.arctan2(M[2, :], norm)
    
    # 使用逆多项式计算rho（到图像中心的距离）
    rho = np.zeros_like(theta)
    for i in range(len(invpol)):
        rho += invpol[i] * theta**i
    
    # 计算图像坐标
    x = M[0, :] / norm * rho
    y = M[1, :] / norm * rho
    
    # 应用仿射变换
    m = np.zeros((2, M.shape[1]))
    m[0, :] = x * c + y * d + xc
    m[1, :] = x * e + y + yc
    
    return m

def create_erp_image(cameras, image_files, output_size=(1024, 2048)):
    """
    创建ERP图像
    
    参数:
    cameras: 相机参数列表
    image_files: 图像文件列表
    output_size: 输出ERP图像的大小，默认为(1024, 2048)
    
    返回:
    erp_image: ERP图像
    """
    height, width = output_size
    erp_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 为每个ERP像素创建对应的3D方向向量
    phi = np.linspace(-np.pi, np.pi, width)
    theta = np.linspace(np.pi, 0, height)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    
    # 将球面坐标转换为笛卡尔坐标（单位向量）
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    
    # 将坐标堆叠为(3, H*W)的数组
    xyz = np.stack((x.flatten(), y.flatten(), z.flatten()))
    
    # 读取图像
    images = [cv2.imread(img_file) for img_file in image_files]
    
    # 创建一个掩码，记录每个ERP像素是否已被填充
    filled_mask = np.zeros((height, width), dtype=bool)
    
    # 为每个像素找到对应的相机和图像坐标
    for cam_idx, camera in enumerate(cameras):
        # 提取相机参数
        ocam_model = camera['ocam_model']
        pose = camera['pose']
        
        # 创建旋转矩阵
        rot = R.from_euler('xyz', pose[:3], degrees=False)
        rot_matrix = rot.as_matrix()
        
        # 将全局坐标转换到相机坐标系
        cam_xyz = rot_matrix.T @ xyz
        
        # 找出朝向相机的点（z坐标为负）
        valid_points = cam_xyz[2, :] < 0
        
        # 对于有效点，计算图像坐标
        if np.any(valid_points):
            # 提取有效的相机坐标
            cam_xyz_valid = cam_xyz[:, valid_points]
            
            # 使用world2cam函数计算图像坐标
            uv = world2cam(cam_xyz_valid, ocam_model)
            
            # 将坐标转换为整数
            u = np.round(uv[0, :]).astype(int)
            v = np.round(uv[1, :]).astype(int)
            
            # 过滤掉超出图像范围的点
            valid_uv = (u >= 0) & (u < ocam_model['width']) & (v >= 0) & (v < ocam_model['height'])
            
            if np.any(valid_uv):
                # 获取有效的坐标和对应的ERP索引
                u_valid = u[valid_uv]
                v_valid = v[valid_uv]
                erp_indices = np.where(valid_points)[0][valid_uv]
                
                # 从图像中获取颜色
                colors = images[cam_idx][v_valid, u_valid]
                
                # 填充ERP图像
                erp_y = erp_indices // width
                erp_x = erp_indices % width
                
                # 只填充尚未填充的像素
                unfilled = ~filled_mask[erp_y, erp_x]
                erp_image[erp_y[unfilled], erp_x[unfilled]] = colors[unfilled]
                filled_mask[erp_y, erp_x] = True
    
    return erp_image

def main():
    # 解析命令行参数
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
    else:
        dataset_dir = 'OmniHouse'
    
    if len(sys.argv) > 2:
        frame_idx = int(sys.argv[2])
    else:
        frame_idx = 1
    
    print(f"处理数据集: {dataset_dir}, 帧索引: {frame_idx}")
    
    # 确定配置文件路径
    if dataset_dir == 'OmniHouse':
        config_file = os.path.join(dataset_dir, 'house_config.yaml')
    elif dataset_dir == 'OmniThing':
        config_file = os.path.join(dataset_dir, 'things_config.yaml')
    elif 'OmniUrban' in dataset_dir:
        config_file = os.path.join(dataset_dir, 'urban_config.yaml')
    else:
        config_file = os.path.join(dataset_dir, 'house_config.yaml')  # 默认
    
    print(f"使用配置文件: {config_file}")
    
    # 读取配置文件
    config = read_config(config_file)
    
    # 读取相机参数
    cameras = []
    for cam_config in config['cameras']:
        cam_id = cam_config['cam_id']
        ocam_file = os.path.join(dataset_dir, f'ocam{cam_id}.txt')
        
        print(f"读取相机参数: {ocam_file}")
        ocam_model = read_ocam_model(ocam_file)
        
        # 提取相机姿态
        pose = np.array(cam_config['pose'])
        print(f"相机 {cam_id} 姿态: {pose}")
        
        cameras.append({
            'cam_id': cam_id,
            'ocam_model': ocam_model,
            'pose': pose
        })
    
    # 选择要处理的图像
    image_files = []
    for cam_id in range(1, 5):
        img_file = os.path.join(dataset_dir, f'cam{cam_id}', f'{frame_idx:04d}.png')
        if not os.path.exists(img_file):
            print(f"警告: 图像文件不存在: {img_file}")
            print(f"尝试其他格式...")
            # 尝试其他格式
            img_file = os.path.join(dataset_dir, f'cam{cam_id}', f'{frame_idx:04d}.jpg')
            if not os.path.exists(img_file):
                print(f"错误: 图像文件不存在: {img_file}")
                sys.exit(1)
        
        print(f"处理图像: {img_file}")
        image_files.append(img_file)
    
    # 创建ERP图像
    print("创建ERP图像...")
    erp_image = create_erp_image(cameras, image_files, output_size=(1024, 2048))
    
    # 保存结果
    output_file = os.path.join(dataset_dir, f'erp_{frame_idx:04d}.png')
    print(f"保存结果到: {output_file}")
    cv2.imwrite(output_file, erp_image)
    
    # 显示结果
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(erp_image, cv2.COLOR_BGR2RGB))
    plt.title(f'ERP Image - Frame {frame_idx}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_dir, f'erp_{frame_idx:04d}_display.png'))
    print("完成!")

if __name__ == '__main__':
    main()