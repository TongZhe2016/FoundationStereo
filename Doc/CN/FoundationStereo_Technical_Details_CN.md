# FoundationStereo：技术细节与实现

本文档提供了FoundationStereo项目的深入技术解释，重点关注实现细节、架构组件和数据流。

## 目录

1. [模型架构](#模型架构)
2. [特征提取流程](#特征提取流程)
3. [成本体积构建与处理](#成本体积构建与处理)
4. [视差估计与细化](#视差估计与细化)
5. [分层推理](#分层推理)
6. [点云生成](#点云生成)
7. [实现细节](#实现细节)
8. [代码解析](#代码解析)

## 模型架构

FoundationStereo建立在结合传统立体匹配技术与现代深度学习方法的混合架构上。该模型由几个关键组件组成：

### 主要组件

1. **特征提取网络**
   - 基于EdgeNext的CNN骨干网络
   - 与Depth Anything集成，用于单目深度先验
   - 多尺度特征提取

2. **成本体积处理**
   - 分组相关性体积
   - 连接体积
   - 带有注意力机制的3D沙漏网络

3. **视差估计**
   - 初始视差回归
   - 基于GRU的迭代细化
   - 上下文感知上采样

4. **后处理**
   - 视差到深度的转换
   - 点云生成
   - 可选的点云去噪

### 数据流

FoundationStereo的整体数据流如下：

```
立体图像 → 特征提取 → 成本体积构建 → 
初始视差估计 → 迭代细化 → 
上采样 → 最终视差图 → (可选) 深度图/点云
```

## 特征提取流程

特征提取流程是FoundationStereo最关键的组件之一，在`core/extractor.py`中实现。

### Feature类

`Feature`类是特征提取的主要入口点：

```python
class Feature(nn.Module):
    def __init__(self, args):
        super(Feature, self).__init__()
        self.args = args
        model = timm.create_model('edgenext_small', pretrained=True, features_only=False)
        self.stem = model.stem
        self.stages = model.stages
        chans = [48, 96, 160, 304]
        self.chans = chans
        self.dino = DepthAnythingFeature(encoder=self.args.vit_size)
        self.dino = freeze_model(self.dino)
        vit_feat_dim = DepthAnythingFeature.model_configs[self.args.vit_size]['features']//2
        
        # 用于多尺度特征融合的反卷积层
        self.deconv32_16 = Conv2x_IN(chans[3], chans[2], deconv=True, concat=True)
        self.deconv16_8 = Conv2x_IN(chans[2]*2, chans[1], deconv=True, concat=True)
        self.deconv8_4 = Conv2x_IN(chans[1]*2, chans[0], deconv=True, concat=True)
        self.conv4 = nn.Sequential(
          BasicConv(chans[0]*2+vit_feat_dim, chans[0]*2+vit_feat_dim, kernel_size=3, stride=1, padding=1, norm='instance'),
          ResidualBlock(chans[0]*2+vit_feat_dim, chans[0]*2+vit_feat_dim, norm_fn='instance'),
          ResidualBlock(chans[0]*2+vit_feat_dim, chans[0]*2+vit_feat_dim, norm_fn='instance'),
        )
        
        self.patch_size = 14
        self.d_out = [chans[0]*2+vit_feat_dim, chans[1]*2, chans[2]*2, chans[3]]
```

### 特征提取过程

特征提取过程包括几个步骤：

1. **输入处理**：
   - 将输入图像调整为与patch大小兼容
   - 通过Depth Anything模型获取单目深度特征

2. **CNN特征提取**：
   - 通过EdgeNext的stem和stages处理图像
   - 在不同分辨率提取多尺度特征

3. **特征融合**：
   - 使用反卷积层组合不同尺度的特征
   - 将视觉transformer特征与CNN特征集成

4. **输出**：
   - 返回多尺度特征列表和视觉transformer特征

### DepthAnythingFeature

`DepthAnythingFeature`类集成了Depth Anything模型以获取单目深度先验：

```python
class DepthAnythingFeature(nn.Module):
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }
    
    def __init__(self, encoder='vits'):
        super().__init__()
        from depth_anything.dpt import DepthAnything
        self.encoder = encoder
        depth_anything = DepthAnything(self.model_configs[encoder])
        self.depth_anything = depth_anything
        
        self.intermediate_layer_idx = {   #!NOTE For V2
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }
```

该类：
1. 使用指定的编码器大小初始化Depth Anything模型
2. 定义用于特征提取的中间层索引
3. 从输入图像提取特征和深度信息

## 成本体积构建与处理

成本体积是一个4D张量，编码左右图像中像素之间的匹配成本。FoundationStereo使用两种类型的成本体积：

### 分组相关性体积

分组相关性体积使用`core/submodule.py`中的`build_gwc_volume`函数构建：

```python
def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups, stride=1):
    """
    @refimg_fea: 左图像特征
    @targetimg_fea: 右图像特征
    """
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume
```

该函数：
1. 创建一个零初始化的体积张量
2. 对于每个视差级别，计算左右特征之间的分组相关性
3. 根据视差级别水平移动右侧特征

### 连接体积

连接体积使用`build_concat_volume`函数构建：

```python
def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, :] = refimg_fea[:, :, :, :]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume
```

该函数：
1. 创建一个零初始化的体积张量
2. 对于每个视差级别，连接左特征与移位的右特征

### 成本体积处理

成本体积使用带有注意力机制的3D沙漏网络处理：

```python
class hourglass(nn.Module):
    def __init__(self, cfg, in_channels, feat_dims=None):
        super().__init__()
        self.cfg = cfg
        # 用于下采样的3D卷积层
        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17))
        # ... 更多层 ...
        
        # 特征注意力机制
        self.feature_att_8 = FeatureAtt(in_channels*2, feat_dims[1])
        self.feature_att_16 = FeatureAtt(in_channels*4, feat_dims[2])
        self.feature_att_32 = FeatureAtt(in_channels*6, feat_dims[3])
        self.feature_att_up_16 = FeatureAtt(in_channels*4, feat_dims[2])
        self.feature_att_up_8 = FeatureAtt(in_channels*2, feat_dims[1])
```

该网络：
1. 通过一系列3D卷积处理成本体积
2. 在不同尺度应用特征注意力机制
3. 使用跳跃连接保留空间信息
4. 输出用于视差估计的过滤后成本体积

## 视差估计与细化

视差估计和细化过程是FoundationStereo的关键创新。

### 初始视差估计

初始视差使用`disparity_regression`函数估计：

```python
def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.reshape(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)
```

该函数：
1. 创建从0到maxdisp的视差值张量
2. 使用成本体积中的softmax概率执行加权和

### 视差细化

视差使用基于GRU的更新机制进行细化：

```python
class BasicSelectiveMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, volume_dim=8):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args, volume_dim)
        
        # 不同尺度的GRU层
        if args.n_gru_layers == 3:
            self.gru16 = SelectiveConvGRU(hidden_dim, hidden_dim * 2)
        if args.n_gru_layers >= 2:
            self.gru08 = SelectiveConvGRU(hidden_dim, hidden_dim * (args.n_gru_layers == 3) + hidden_dim * 2)
        self.gru04 = SelectiveConvGRU(hidden_dim, hidden_dim * (args.n_gru_layers > 1) + hidden_dim * 2)
        self.disp_head = DispHead(hidden_dim, 256)
        self.mask = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            )
```

该类：
1. 从当前视差和相关体积编码运动特征
2. 使用选择性GRU单元在多个尺度更新隐藏状态
3. 预测视差更新和用于上下文感知上采样的掩码

### 上下文感知上采样

最终视差使用上下文感知上采样进行上采样：

```python
def upsample_disp(self, disp, mask_feat_4, stem_2x):
    with autocast(enabled=self.args.mixed_precision):
        xspx = self.spx_2_gru(mask_feat_4, stem_2x)   # 1/2分辨率
        spx_pred = self.spx_gru(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)
    return up_disp.float()
```

该函数：
1. 处理掩码特征和stem特征以生成上采样权重
2. 对视差图应用上下文感知上采样
3. 返回上采样后的视差图

## 分层推理

对于高分辨率图像，FoundationStereo使用分层推理方法：

```python
def run_hierachical(self, image1, image2, iters=12, test_mode=False, low_memory=False, small_ratio=0.5):
    B,_,H,W = image1.shape
    # 下采样图像
    img1_small = F.interpolate(image1, scale_factor=small_ratio, align_corners=False, mode='bilinear')
    img2_small = F.interpolate(image2, scale_factor=small_ratio, align_corners=False, mode='bilinear')
    padder = InputPadder(img1_small.shape[-2:], divis_by=32, force_square=False)
    img1_small, img2_small = padder.pad(img1_small, img2_small)
    
    # 在下采样图像上运行推理
    disp_small = self.forward(img1_small, img2_small, test_mode=True, iters=iters, low_memory=low_memory)
    disp_small = padder.unpad(disp_small.float())
    
    # 上采样视差并用作全分辨率推理的初始化
    disp_small_up = F.interpolate(disp_small, size=(H,W), mode='bilinear', align_corners=True) * 1/small_ratio
    disp_small_up = disp_small_up.clip(0, None)
    
    padder = InputPadder(image1.shape[-2:], divis_by=32, force_square=False)
    image1, image2, disp_small_up = padder.pad(image1, image2, disp_small_up)
    disp_small_up += padder._pad[0]
    init_disp = F.interpolate(disp_small_up, scale_factor=0.25, mode='bilinear', align_corners=True) * 0.25
    
    # 使用初始化在全分辨率图像上运行推理
    disp = self.forward(image1, image2, iters=iters, test_mode=test_mode, low_memory=low_memory, init_disp=init_disp)
    disp = padder.unpad(disp.float())
    return disp
```

该函数：
1. 下采样输入图像
2. 在下采样图像上运行推理
3. 上采样结果视差图
4. 使用上采样视差作为全分辨率推理的初始化
5. 返回最终视差图

## 点云生成

视差图可以使用相机内参转换为点云：

```python
if args.get_pc:
    with open(args.intrinsic_file, 'r') as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
        baseline = float(lines[1])
    K[:2] *= scale
    depth = K[0,0]*baseline/disp
    np.save(f'{args.out_dir}/depth_meter.npy', depth)
    xyz_map = depth2xyzmap(depth, K)
    pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
    keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.z_far)
    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)
    o3d.io.write_point_cloud(f'{args.out_dir}/cloud.ply', pcd)
```

该代码：
1. 读取相机内参和基线
2. 使用公式将视差转换为深度：depth = focal_length * baseline / disparity
3. 将深度转换为3D点云
4. 过滤掉具有无效深度值的点
5. 将点云保存为PLY文件

## 实现细节

### 混合精度训练

FoundationStereo使用混合精度训练以减少内存使用并提高速度：

```python
with autocast(enabled=self.args.mixed_precision):
    # 前向传播
```

### Flash注意力

该模型使用Flash注意力进行高效的transformer操作：

```python
class FlashMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim必须能被num_heads整除"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, attn_mask=None, window_size=(-1,-1)):
        """
        @query: (B,L,C)
        """
        B,L,C = query.shape
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        Q = Q.view(Q.size(0), Q.size(1), self.num_heads, self.head_dim)
        K = K.view(K.size(0), K.size(1), self.num_heads, self.head_dim)
        V = V.view(V.size(0), V.size(1), self.num_heads, self.head_dim)
        
        attn_output = flash_attn_func(Q, K, V, window_size=window_size)
        
        attn_output = attn_output.reshape(B,L,-1)
        output = self.out_proj(attn_output)
        
        return output
```

### 内存优化

对于大图像，模型包括一个low_memory选项，以较小的块处理相关体积：

```python
geo_feat = geo_fn(disp, coords, low_memory=low_memory)
```

## 代码解析

### 主要前向传播

`FoundationStereo`中的主要前向传播如下：

```python
def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False, low_memory=False, init_disp=None):
    """ 估计帧对之间的视差 """
    B = len(image1)
    low_memory = low_memory or (self.args.get('low_memory', False))
    image1 = normalize_image(image1)
    image2 = normalize_image(image2)
    with autocast(enabled=self.args.mixed_precision):
        # 提取特征
        out, vit_feat = self.feature(torch.cat([image1, image2], dim=0))
        vit_feat = vit_feat[:B]
        features_left = [o[:B] for o in out]
        features_right = [o[B:] for o in out]
        stem_2x = self.stem_2(image1)
        
        # 构建成本体积
        gwc_volume = build_gwc_volume(features_left[0], features_right[0], self.args.max_disp//4, self.cv_group)
        left_tmp = self.proj_cmb(features_left[0])
        right_tmp = self.proj_cmb(features_right[0])
        concat_volume = build_concat_volume(left_tmp, right_tmp, maxdisp=self.args.max_disp//4)
        del left_tmp, right_tmp
        comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)
        comb_volume = self.corr_stem(comb_volume)
        comb_volume = self.corr_feature_att(comb_volume, features_left[0])
        comb_volume = self.cost_agg(comb_volume, features_left)
        
        # 初始视差估计
        prob = F.softmax(self.classifier(comb_volume).squeeze(1), dim=1)
        if init_disp is None:
          init_disp = disparity_regression(prob, self.args.max_disp//4)
        
        # 上下文网络
        cnet_list = self.cnet(image1, vit_feat=vit_feat, num_layers=self.args.n_gru_layers)
        cnet_list = list(cnet_list)
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [self.cam(x) * x for x in inp_list]
        att = [self.sam(x) for x in inp_list]
    
    # 几何编码
    geo_fn = Combined_Geo_Encoding_Volume(features_left[0].float(), features_right[0].float(), comb_volume.float(), num_levels=self.args.corr_levels, dx=self.dx)
    b, c, h, w = features_left[0].shape
    coords = torch.arange(w, dtype=torch.float, device=init_disp.device).reshape(1,1,w,1).repeat(b, h, 1, 1)
    disp = init_disp.float()
    disp_preds = []
    
    # GRU迭代更新视差
    for itr in range(iters):
        disp = disp.detach()
        geo_feat = geo_fn(disp, coords, low_memory=low_memory)
        with autocast(enabled=self.args.mixed_precision):
          net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, att)
        
        disp = disp + delta_disp.float()
        if test_mode and itr < iters-1:
            continue
        
        # 上采样预测
        disp_up = self.upsample_disp(disp.float(), mask_feat_4.float(), stem_2x.float())
        disp_preds.append(disp_up)
    
    if test_mode:
        return disp_up
    
    return init_disp, disp_preds
```

该函数：
1. 归一化输入图像
2. 从两个图像提取特征
3. 构建并处理成本体积
4. 估计初始视差图
5. 使用基于GRU的更新机制迭代细化视差
6. 上采样视差图
7. 返回最终视差图或视差预测列表

### 演示脚本

`run_demo.py`脚本展示了如何使用模型进行推理：

```python
if __name__=="__main__":
    # 解析参数
    parser = argparse.ArgumentParser()
    # ... 参数定义 ...
    args = parser.parse_args()
    
    # 加载模型
    model = FoundationStereo(args)
    ckpt = torch.load(ckpt_dir)
    model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()
    
    # 加载并预处理图像
    img0 = imageio.imread(args.left_file)
    img1 = imageio.imread(args.right_file)
    scale = args.scale
    img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
    img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
    H,W = img0.shape[:2]
    img0_ori = img0.copy()
    
    img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
    img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)
    
    # 运行推理
    with torch.cuda.amp.autocast(True):
        if not args.hiera:
            disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
        else:
            disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H,W)
    
    # 可视化结果
    vis = vis_disparity(disp)
    vis = np.concatenate([img0_ori, vis], axis=1)
    imageio.imwrite(f'{args.out_dir}/vis.png', vis)
    
    # 生成点云
    if args.get_pc:
        # ... 点云生成代码 ...
```

该脚本：
1. 解析命令行参数
2. 加载预训练模型
3. 加载并预处理输入立体图像
4. 运行推理获取视差图
5. 可视化结果
6. 可选地生成点云