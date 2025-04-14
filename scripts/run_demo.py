# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os, sys
import shutil
import numpy as np
import datetime
import argparse
import logging
import cv2
import imageio
import torch
import open3d as o3d

# Add parent directory to path for imports
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

# Import project modules
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *

# Configure matplotlib backend
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


if __name__=="__main__":
  # Get the directory of the current script
  code_dir = os.path.dirname(os.path.realpath(__file__))
  
  # Set up command line argument parser
  parser = argparse.ArgumentParser(description='FoundationStereo demo script for stereo depth estimation')
  
  # Input/output arguments
  parser.add_argument('--left_file', default=f'{code_dir}/../assets/left.png', type=str, 
                      help='Path to the left image')
  parser.add_argument('--right_file', default=f'{code_dir}/../assets/right.png', type=str, 
                      help='Path to the right image')
  parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K.txt', type=str, 
                      help='Path to camera intrinsic matrix and baseline file')
  parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/model_best_bp2.pth', type=str, 
                      help='Path to pretrained model')
  parser.add_argument('--out_dir', default=f'{code_dir}/../output/', type=str, 
                      help='Directory to save results')
  
  # Camera type argument
  parser.add_argument('--camera_type', type=str, default='pinhole', choices=['pinhole', 'panorama'],
                      help='Type of camera: pinhole (standard perspective) or panorama (equirectangular projection)')
  
  # Processing arguments
  parser.add_argument('--scale', default=1, type=float, 
                      help='Downsize the image by scale, must be <=1')
  parser.add_argument('--hiera', default=0, type=int, 
                      help='Use hierarchical inference (only needed for high-resolution images (>1K))')
  parser.add_argument('--z_far', default=10, type=float, 
                      help='Maximum depth to clip in point cloud')
  parser.add_argument('--valid_iters', type=int, default=32, 
                      help='Number of flow-field updates during forward pass')
  
  # Point cloud arguments
  parser.add_argument('--get_pc', type=int, default=1, 
                      help='Save point cloud output')
  parser.add_argument('--remove_invisible', default=1, type=int, 
                      help='Remove non-overlapping observations between left and right images from point cloud')
  parser.add_argument('--denoise_cloud', type=int, default=1, 
                      help='Whether to denoise the point cloud')
  parser.add_argument('--denoise_nb_points', type=int, default=30, 
                      help='Number of points to consider for radius outlier removal')
  parser.add_argument('--denoise_radius', type=float, default=0.03, 
                      help='Radius to use for outlier removal')
  
  # Parse arguments
  args = parser.parse_args()
  
  # Set default paths based on camera type
  if args.camera_type == 'panorama':
    # Default paths for panorama (equirectangular) images
    args.left_file = f'{code_dir}/../assets/blender/up_erp.png'
    args.right_file = f'{code_dir}/../assets/blender/down_erp.png'
    args.intrinsic_file = f'{code_dir}/../assets/blender/K.txt'
  else:  # pinhole camera
    # Default paths for pinhole camera images
    args.left_file = f'{code_dir}/../assets/left.png'
    args.right_file = f'{code_dir}/../assets/right.png'
    args.intrinsic_file = f'{code_dir}/../assets/K.txt'
  
  # Create output directory with timestamp and camera type
  args.out_dir = f'{code_dir}/../test_outputs/{args.camera_type}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)
  os.makedirs(args.out_dir, exist_ok=True)
  
  # 复制输入文件到输出文件夹，以便后期回顾
  shutil.copy(args.left_file, os.path.join(args.out_dir, 'left.png'))
  shutil.copy(args.right_file, os.path.join(args.out_dir, 'right.png'))
  shutil.copy(args.intrinsic_file, os.path.join(args.out_dir, 'K.txt'))
  logging.info(f"Copied input files to {args.out_dir} for future reference")

  ckpt_dir = args.ckpt_dir
  cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
  if 'vit_size' not in cfg:
    cfg['vit_size'] = 'vitl'
  for k in args.__dict__:
    cfg[k] = args.__dict__[k]
  args = OmegaConf.create(cfg)
  logging.info(f"args:\n{args}")
  logging.info(f"Using pretrained model from {ckpt_dir}")

  model = FoundationStereo(args)

  ckpt = torch.load(ckpt_dir)
  logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
  model.load_state_dict(ckpt['model'])

  model.cuda()
  model.eval()

  code_dir = os.path.dirname(os.path.realpath(__file__))
  img0 = imageio.imread(args.left_file)
  img1 = imageio.imread(args.right_file)

  if len(img0.shape)==2:
    img0 = img0[:,:,None]
    img1 = img1[:,:,None]
    img0 = np.concatenate([img0,img0,img0], axis=2)
    img1 = np.concatenate([img1,img1,img1], axis=2)
  if img0.shape[2]==4:
    img0 = img0[:,:,:3]
    img1 = img1[:,:,:3]
  # img1 = np.random.permutation(img1)
  # tmp = img1.copy()
  # img1[:-2] = tmp[2:]
  # img1[-2:] = 0
  # img1 *= 0

  scale = args.scale
  assert scale<=1, "scale must be <=1"
  img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
  img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
  H,W = img0.shape[:2]
  img0_ori = img0.copy()
  logging.info(f"img0: {img0.shape}")

  img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
  img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
  padder = InputPadder(img0.shape, divis_by=32, force_square=False)
  img0, img1 = padder.pad(img0, img1)

  with torch.cuda.amp.autocast(True):
    if not args.hiera:
      disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
    else:
      disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
  disp = padder.unpad(disp.float())
  disp = disp.data.cpu().numpy().reshape(H,W)
  vis = vis_disparity(disp)
  vis = np.concatenate([img0_ori, vis], axis=1)
  imageio.imwrite(f'{args.out_dir}/vis.png', vis)
  logging.info(f"Output saved to {args.out_dir}")

  # Remove non-overlapping observations if specified
  if args.remove_invisible:
    yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
    us_right = xx - disp
    invalid = us_right < 0
    disp[invalid] = np.inf

  # Process point cloud based on camera type
  if args.camera_type == 'panorama':
    # Process for panorama (equirectangular) camera
    logging.info("Processing panorama (equirectangular) camera data")
    
    # Define field of view parameters for equirectangular projection
    half_fov_lat = np.pi * 90 / 180  # Latitude (vertical) field of view in radians
    half_fov_lon = np.pi * 180 / 180  # Longitude (horizontal) field of view in radians

    # Convert pixel coordinates to normalized coordinates (-1 to 1)
    sx_up = yy * 2 / H - 1
    sy_up = xx * 2 / W - 1
    
    # Convert normalized coordinates to spherical coordinates
    lon_up = sx_up * half_fov_lon
    lat_up = sy_up * half_fov_lat

    # Calculate coordinates for the down image
    vs_down = us_right.copy()
    sy_down = vs_down * 2 / W - 1
    lat_down = sy_down * half_fov_lat

    # Calculate angular disparity and depth
    ang_disp = disp * 2 * half_fov_lon / W
    
    # Read baseline from file or use default
    with open(args.intrinsic_file, 'r') as f:
      lines = f.readlines()
      baseline = float(lines[1])
    
    # Calculate radius based on angular disparity
    tr = baseline * np.cos(lat_down) / np.sin(ang_disp)

    # Convert spherical coordinates to Cartesian coordinates
    tx = np.sin(lat_up)
    tz = np.cos(lat_up) * np.sin(lon_up)
    ty = -np.cos(lat_up) * np.cos(lon_up)

    # Create 3D point cloud
    point_up = np.stack([tx * tr, ty * tr, tz * tr], axis=-1)
    
  else:  # pinhole camera
    # Process for pinhole camera
    logging.info("Processing pinhole camera data")
    
    # Read camera intrinsic matrix and baseline from file
    with open(args.intrinsic_file, 'r') as f:
      lines = f.readlines()
      K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3, 3)
      baseline = float(lines[1])
    
    # Scale intrinsic matrix if image was resized
    K[:2] *= scale
    
    # Calculate depth from disparity
    depth = K[0, 0] * baseline / disp
    
    # Convert depth to 3D point cloud
    xyz_map = depth2xyzmap(depth, K)
    point_up = xyz_map
    invalid = np.isinf(disp)

  # Create Open3D point cloud object
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(point_up[~invalid].astype(np.float64))
  pcd.colors = o3d.utility.Vector3dVector(img0_ori[~invalid] / 255.0)
  # Filter points by depth if needed (only for pinhole camera)
  if args.camera_type == 'pinhole':
    logging.info("[Pinhole cam only] Filtering point cloud by depth...")
    keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (np.asarray(pcd.points)[:, 2] <= args.z_far)
    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)
  
  
  # Save point cloud to file
  o3d.io.write_point_cloud(f'{args.out_dir}/cloud.ply', pcd)
  logging.info(f"Point cloud saved to {args.out_dir}/cloud.ply")

  # Save depth map for pinhole camera
  if args.camera_type == 'pinhole':
    # Save depth map in meters
    np.save(f'{args.out_dir}/depth_meter.npy', depth)
    logging.info(f"Depth map saved to {args.out_dir}/depth_meter.npy")

  # Save the original point cloud for visualization
  original_pcd = o3d.geometry.PointCloud()
  original_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
  original_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
  
  # Denoise point cloud if specified
  if args.denoise_cloud:
    logging.info("[Optional step] denoise point cloud...")
    cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
    inlier_cloud = pcd.select_by_index(ind)
    o3d.io.write_point_cloud(f'{args.out_dir}/cloud_denoise.ply', inlier_cloud)
    logging.info(f"Denoised point cloud saved to {args.out_dir}/cloud_denoise.ply")

  # Visualize the original (not denoised) point cloud
  logging.info("Visualizing original point cloud. Press ESC to exit.")
  vis = o3d.visualization.Visualizer()
  vis.create_window()
  vis.add_geometry(original_pcd)
  vis.get_render_option().point_size = 1.0
  vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
  vis.run()
  vis.destroy_window()
