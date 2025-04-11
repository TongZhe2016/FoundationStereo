# FoundationStereo: Technical Details and Implementation

This document provides an in-depth technical explanation of the FoundationStereo project, focusing on the implementation details, architecture components, and data flow.

## Table of Contents

1. [Model Architecture](#model-architecture)
2. [Feature Extraction Pipeline](#feature-extraction-pipeline)
3. [Cost Volume Construction and Processing](#cost-volume-construction-and-processing)
4. [Disparity Estimation and Refinement](#disparity-estimation-and-refinement)
5. [Hierarchical Inference](#hierarchical-inference)
6. [Point Cloud Generation](#point-cloud-generation)
7. [Implementation Details](#implementation-details)
8. [Code Walkthrough](#code-walkthrough)

## Model Architecture

FoundationStereo is built on a hybrid architecture that combines traditional stereo matching techniques with modern deep learning approaches. The model consists of several key components:

### Main Components

1. **Feature Extraction Network**
   - EdgeNext-based CNN backbone
   - Integration with Depth Anything for monocular depth priors
   - Multi-scale feature extraction

2. **Cost Volume Processing**
   - Group-wise correlation volume
   - Concatenation volume
   - 3D hourglass network with attention mechanisms

3. **Disparity Estimation**
   - Initial disparity regression
   - GRU-based iterative refinement
   - Context-aware upsampling

4. **Post-processing**
   - Disparity-to-depth conversion
   - Point cloud generation
   - Optional point cloud denoising

### Data Flow

The overall data flow in FoundationStereo is as follows:

```
Stereo Images → Feature Extraction → Cost Volume Construction → 
Initial Disparity Estimation → Iterative Refinement → 
Upsampling → Final Disparity Map → (Optional) Depth Map/Point Cloud
```

## Feature Extraction Pipeline

The feature extraction pipeline is one of the most critical components of FoundationStereo, implemented in `core/extractor.py`.

### Feature Class

The `Feature` class is the main entry point for feature extraction:

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
        
        # Deconvolution layers for multi-scale feature fusion
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

### Feature Extraction Process

The feature extraction process involves several steps:

1. **Input Processing**:
   - Resize input images to be compatible with the patch size
   - Pass through the Depth Anything model to get monocular depth features

2. **CNN Feature Extraction**:
   - Process images through EdgeNext stem and stages
   - Extract multi-scale features at different resolutions

3. **Feature Fusion**:
   - Combine features from different scales using deconvolution layers
   - Integrate vision transformer features with CNN features

4. **Output**:
   - Return a list of multi-scale features and vision transformer features

### DepthAnythingFeature

The `DepthAnythingFeature` class integrates the Depth Anything model for monocular depth priors:

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

This class:
1. Initializes the Depth Anything model with the specified encoder size
2. Defines intermediate layer indices for feature extraction
3. Extracts features and depth information from input images

## Cost Volume Construction and Processing

The cost volume is a 4D tensor that encodes the matching costs between pixels in the left and right images. FoundationStereo uses two types of cost volumes:

### Group-wise Correlation Volume

The group-wise correlation volume is built using the `build_gwc_volume` function in `core/submodule.py`:

```python
def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups, stride=1):
    """
    @refimg_fea: left image feature
    @targetimg_fea: right image feature
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

This function:
1. Creates a zero-initialized volume tensor
2. For each disparity level, computes the group-wise correlation between left and right features
3. Shifts the right features horizontally based on the disparity level

### Concatenation Volume

The concatenation volume is built using the `build_concat_volume` function:

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

This function:
1. Creates a zero-initialized volume tensor
2. For each disparity level, concatenates the left features with shifted right features

### Cost Volume Processing

The cost volume is processed using a 3D hourglass network with attention mechanisms:

```python
class hourglass(nn.Module):
    def __init__(self, cfg, in_channels, feat_dims=None):
        super().__init__()
        self.cfg = cfg
        # 3D convolution layers for downsampling
        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17))
        # ... more layers ...
        
        # Feature attention mechanisms
        self.feature_att_8 = FeatureAtt(in_channels*2, feat_dims[1])
        self.feature_att_16 = FeatureAtt(in_channels*4, feat_dims[2])
        self.feature_att_32 = FeatureAtt(in_channels*6, feat_dims[3])
        self.feature_att_up_16 = FeatureAtt(in_channels*4, feat_dims[2])
        self.feature_att_up_8 = FeatureAtt(in_channels*2, feat_dims[1])
```

This network:
1. Processes the cost volume through a series of 3D convolutions
2. Applies feature attention mechanisms at different scales
3. Uses skip connections to preserve spatial information
4. Outputs a filtered cost volume for disparity estimation

## Disparity Estimation and Refinement

The disparity estimation and refinement process is a key innovation in FoundationStereo.

### Initial Disparity Estimation

The initial disparity is estimated using the `disparity_regression` function:

```python
def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.reshape(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)
```

This function:
1. Creates a tensor of disparity values from 0 to maxdisp
2. Performs a weighted sum using the softmax probabilities from the cost volume

### Disparity Refinement

The disparity is refined using a GRU-based update mechanism:

```python
class BasicSelectiveMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, volume_dim=8):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args, volume_dim)
        
        # GRU layers for different scales
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

This class:
1. Encodes motion features from the current disparity and correlation volume
2. Updates hidden states at multiple scales using selective GRU units
3. Predicts a disparity update and a mask for context-aware upsampling

### Context-Aware Upsampling

The final disparity is upsampled using context-aware upsampling:

```python
def upsample_disp(self, disp, mask_feat_4, stem_2x):
    with autocast(enabled=self.args.mixed_precision):
        xspx = self.spx_2_gru(mask_feat_4, stem_2x)   # 1/2 resolution
        spx_pred = self.spx_gru(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)
    return up_disp.float()
```

This function:
1. Processes mask features and stem features to generate upsampling weights
2. Applies context-aware upsampling to the disparity map
3. Returns the upsampled disparity map

## Hierarchical Inference

For high-resolution images, FoundationStereo uses a hierarchical inference approach:

```python
def run_hierachical(self, image1, image2, iters=12, test_mode=False, low_memory=False, small_ratio=0.5):
    B,_,H,W = image1.shape
    # Downsample images
    img1_small = F.interpolate(image1, scale_factor=small_ratio, align_corners=False, mode='bilinear')
    img2_small = F.interpolate(image2, scale_factor=small_ratio, align_corners=False, mode='bilinear')
    padder = InputPadder(img1_small.shape[-2:], divis_by=32, force_square=False)
    img1_small, img2_small = padder.pad(img1_small, img2_small)
    
    # Run inference on downsampled images
    disp_small = self.forward(img1_small, img2_small, test_mode=True, iters=iters, low_memory=low_memory)
    disp_small = padder.unpad(disp_small.float())
    
    # Upsample disparity and use as initialization for full-resolution inference
    disp_small_up = F.interpolate(disp_small, size=(H,W), mode='bilinear', align_corners=True) * 1/small_ratio
    disp_small_up = disp_small_up.clip(0, None)
    
    padder = InputPadder(image1.shape[-2:], divis_by=32, force_square=False)
    image1, image2, disp_small_up = padder.pad(image1, image2, disp_small_up)
    disp_small_up += padder._pad[0]
    init_disp = F.interpolate(disp_small_up, scale_factor=0.25, mode='bilinear', align_corners=True) * 0.25
    
    # Run inference on full-resolution images with initialization
    disp = self.forward(image1, image2, iters=iters, test_mode=test_mode, low_memory=low_memory, init_disp=init_disp)
    disp = padder.unpad(disp.float())
    return disp
```

This function:
1. Downsamples the input images
2. Runs inference on the downsampled images
3. Upsamples the resulting disparity map
4. Uses the upsampled disparity as initialization for full-resolution inference
5. Returns the final disparity map

## Point Cloud Generation

The disparity map can be converted to a point cloud using the camera intrinsics:

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

This code:
1. Reads the camera intrinsics and baseline
2. Converts disparity to depth using the formula: depth = focal_length * baseline / disparity
3. Converts depth to a 3D point cloud
4. Filters out points with invalid depth values
5. Saves the point cloud to a PLY file

## Implementation Details

### Mixed Precision Training

FoundationStereo uses mixed precision training to reduce memory usage and increase speed:

```python
with autocast(enabled=self.args.mixed_precision):
    # Forward pass
```

### Flash Attention

The model uses Flash Attention for efficient transformer operations:

```python
class FlashMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
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

### Memory Optimization

For large images, the model includes a low_memory option that processes the correlation volume in smaller chunks:

```python
geo_feat = geo_fn(disp, coords, low_memory=low_memory)
```

## Code Walkthrough

### Main Forward Pass

The main forward pass in `FoundationStereo` is as follows:

```python
def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False, low_memory=False, init_disp=None):
    """ Estimate disparity between pair of frames """
    B = len(image1)
    low_memory = low_memory or (self.args.get('low_memory', False))
    image1 = normalize_image(image1)
    image2 = normalize_image(image2)
    with autocast(enabled=self.args.mixed_precision):
        # Extract features
        out, vit_feat = self.feature(torch.cat([image1, image2], dim=0))
        vit_feat = vit_feat[:B]
        features_left = [o[:B] for o in out]
        features_right = [o[B:] for o in out]
        stem_2x = self.stem_2(image1)
        
        # Build cost volumes
        gwc_volume = build_gwc_volume(features_left[0], features_right[0], self.args.max_disp//4, self.cv_group)
        left_tmp = self.proj_cmb(features_left[0])
        right_tmp = self.proj_cmb(features_right[0])
        concat_volume = build_concat_volume(left_tmp, right_tmp, maxdisp=self.args.max_disp//4)
        del left_tmp, right_tmp
        comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)
        comb_volume = self.corr_stem(comb_volume)
        comb_volume = self.corr_feature_att(comb_volume, features_left[0])
        comb_volume = self.cost_agg(comb_volume, features_left)
        
        # Initial disparity estimation
        prob = F.softmax(self.classifier(comb_volume).squeeze(1), dim=1)
        if init_disp is None:
          init_disp = disparity_regression(prob, self.args.max_disp//4)
        
        # Context network
        cnet_list = self.cnet(image1, vit_feat=vit_feat, num_layers=self.args.n_gru_layers)
        cnet_list = list(cnet_list)
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [self.cam(x) * x for x in inp_list]
        att = [self.sam(x) for x in inp_list]
    
    # Geometry encoding
    geo_fn = Combined_Geo_Encoding_Volume(features_left[0].float(), features_right[0].float(), comb_volume.float(), num_levels=self.args.corr_levels, dx=self.dx)
    b, c, h, w = features_left[0].shape
    coords = torch.arange(w, dtype=torch.float, device=init_disp.device).reshape(1,1,w,1).repeat(b, h, 1, 1)
    disp = init_disp.float()
    disp_preds = []
    
    # GRU iterations to update disparity
    for itr in range(iters):
        disp = disp.detach()
        geo_feat = geo_fn(disp, coords, low_memory=low_memory)
        with autocast(enabled=self.args.mixed_precision):
          net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, att)
        
        disp = disp + delta_disp.float()
        if test_mode and itr < iters-1:
            continue
        
        # Upsample predictions
        disp_up = self.upsample_disp(disp.float(), mask_feat_4.float(), stem_2x.float())
        disp_preds.append(disp_up)
    
    if test_mode:
        return disp_up
    
    return init_disp, disp_preds
```

This function:
1. Normalizes the input images
2. Extracts features from both images
3. Builds and processes cost volumes
4. Estimates an initial disparity map
5. Iteratively refines the disparity using a GRU-based update mechanism
6. Upsamples the disparity map
7. Returns the final disparity map or a list of disparity predictions

### Demo Script

The `run_demo.py` script shows how to use the model for inference:

```python
if __name__=="__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # ... argument definitions ...
    args = parser.parse_args()
    
    # Load model
    model = FoundationStereo(args)
    ckpt = torch.load(ckpt_dir)
    model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()
    
    # Load and preprocess images
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
    
    # Run inference
    with torch.cuda.amp.autocast(True):
        if not args.hiera:
            disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
        else:
            disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H,W)
    
    # Visualize results
    vis = vis_disparity(disp)
    vis = np.concatenate([img0_ori, vis], axis=1)
    imageio.imwrite(f'{args.out_dir}/vis.png', vis)
    
    # Generate point cloud
    if args.get_pc:
        # ... point cloud generation code ...
```

This script:
1. Parses command-line arguments
2. Loads the pretrained model
3. Loads and preprocesses the input stereo images
4. Runs inference to get the disparity map
5. Visualizes the results
6. Optionally generates a point cloud