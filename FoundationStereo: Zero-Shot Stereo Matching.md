# FoundationStereo: Zero-Shot Stereo Matching 

Bowen Wen Matthew Trepte Joseph Aribido Jan Kautz Orazio Gallo Stan Birchfield

NVIDIA

![img-0.jpeg](img-0.jpeg)

Figure 1. Zero-shot prediction on in-the-wild images. Our method generalizes to diverse scenarios (indoor / outdoor), objects of challenging properties (textureless / reflective / translucent / thin-structured), complex illuminations (shadow / strong exposure), various viewing perspectives and sensing ranges.


#### Abstract

Tremendous progress has been made in deep stereo matching to excel on benchmark datasets through per-domain fine-tuning. However, achieving strong zero-shot generalization - a hallmark of foundation models in other computer vision tasks - remains challenging for stereo matching. We introduce FoundationStereo, a foundation model for stereo depth estimation designed to achieve strong zeroshot generalization. To this end, we first construct a largescale (1M stereo pairs) synthetic training dataset featuring large diversity and high photorealism, followed by an automatic self-curation pipeline to remove ambiguous samples. We then design a number of network architecture components to enhance scalability, including a side-tuning feature backbone that adapts rich monocular priors from vision foundation models to mitigate the sim-to-real gap, and long-range context reasoning for effective cost volume filtering. Together, these components lead to strong robustness and accuracy across domains, establishing a new standard in zero-shot stereo depth estimation. Project page: https://nvlabs.github.io/FoundationStereo/


## 1. Introduction

Since the advent of the first stereo matching algorithm nearly half a century ago [42], we have come a long way.

Recent stereo algorithms can achieve amazing results, almost saturating the most challenging benchmarks-thanks to the proliferation of training datasets and advances in deep neural network architectures. Yet, fine-tuning on the dataset of the target domain is still the method of choice to get competitive results. Given the zero-shot generalization ability shown on other problems within computer vision via the scaling law [32, 46, 78, 79], what prevents stereo matching algorithms from achieving a similar level of generalization?

Leading stereo networks [11, 41, 53, 54, 73, 80] construct cost volumes from the unary features and leverage 3D CNNs for cost filtering. Refinement-based methods [14, 21, 27, 34, 36, 60, 67, 86] iteratively refine the disparity map based on recurrent modules such as Gated Recurrent Units (GRU). Despite their success on public benchmarks under per-domain fine-tuning setup, however, they struggle to gather non-local information to effectively scale to larger datasets. Other methods [35, 68] explore transformer architectures for unary feature extraction, while lacking the specialized structure afforded by cost volumes and iterative refinement to achieve high accuracy.

Such limitations have, to date, hindered the development of a stereo network that generalizes well to other domains. While it is true that cross-domain generalization has been explored by some prior works [10, 17, 37, 49, 82, 84], such 

![img-1.jpeg](img-1.jpeg)

approaches have not achieved results that are competitive with those obtained by fine-tuning on the target domain, either due to insufficient structure in the network architecture, impoverished training data, or both. These networks are generally experimented on Scene Flow [43], a rather small dataset with only 40 K annotated training image pairs. As a result, none of these methods can be used as an off-the-shelf solution, as opposed to the strong generalizability of vision foundation models that have emerged in other tasks.

To address these limitations, we propose FoundationStereo, a large foundation model for stereo depth estimation that achieves strong zero-shot generalization without perdomain fine-tuning. We train the network on a large-scale (1M image pairs) high-fidelity synthetic training dataset with high diversity and photorealism. An automatic selfcuration pipeline is developed to eliminate the ambiguous samples that are inevitably introduced during the domain randomized data generation process, improving both the dataset quality and model robustness over iterate updates. To mitigate the sim-to-real gap, we propose a side-tuning feature backbone that adapts internet-scale rich priors from DepthAnythingV2 [79] that is trained on real monocular images to the stereo setup. To effectively leverage these rich monocular priors embedded into the 4D cost volume, we then propose an Attentive Hybrid Cost Volume (AHCF) module, consisting of 3D Axial-Planar Convolution (APC) filtering that decouples standard 3D convolution into two separate spatial- and disparity-oriented 3D convolutions, enhancing the receptive fields for volume feature aggregation; and a Disparity Transformer (DT) that performs selfattention over the entire disparity space within the cost volume, providing long range context for global reasoning. Together, these innovations significantly enhance the representation, leading to better disparity initialization, as well as more powerful features for the subsequent iterative refinement process.

Our contributions can be summarized as follows:

- We present FoundationStereo, a zero-shot generalizable stereo matching model that achieves comparable or even more favorable results to prior works fine-tuned on a target domain; it also significantly outperforms existing methods when applied to in-the-wild data.
- We create a large-scale (1M) high-fidelity synthetic dataset for stereo learning with high diversity and photorealism; and a self-curation pipeline to ensure that bad samples are pruned.
- To harness internet-scale knowledge containing rich semantic and geometric priors, we propose a Side-Tuning Adapter (STA) that adapts the ViT-based monocular depth estimation model [79] to the stereo setup.
- We develop Attentive Hybrid Cost Filtering (AHCF), which includes an hourglass module with 3D AxialPlanar Convolution (APC), and a Disparity Transformer
(DT) module that performs full self-attention over the disparity dimension.


## 2. Related Work

Deep Stereo Matching. Recent advances in stereo matching have been driven by deep learning, significantly enhancing accuracy and generalization. Cost volume aggregation methods construct cost volumes from unary features and perform 3D CNN for volume filtering [11, 41, 53, 54, 73, 80], though the high memory consumption prevents direct application to high resolution images. Iterative refinement methods, inspired by RAFT [57], bypasses the costly 4D volume construction and filtering by recurrently refining the disparity $[14,21,27,34,36,60,67,86]$. While they generalize well to various disparity range, the recurrent updates are often time-consuming, and lack long-range context reasoning. Recent works [71, 72] thus combine the strengths of cost filtering and iterative refinement. With the tremendous progress made by vision transformers, another line of research $[23,35,68]$ introduces transformer architecture to stereo matching, particularly in the unary feature extraction stage. Despite their success on per-domain fine-tuning setup, zero-shot generalization still remains challenging. To tackle this problem, $[10,17,37,49,82,84]$ explore learning domain-invariant features for cross-domain generalization, with a focus on training on Scene Flow [43] dataset. Concurrent work [3] achieves remarkable zero-shot generalization with monocular prior enhanced correlation volumes. However, the strong generalizability of vision foundation models emerged in other tasks that is supported by scaling law has yet to be fully realized in stereo matching for practical applications.
Stereo Matching Training Data. Training data is essential for deep learning models. KITTI 12 [20] and KITTI 15 [45] provide hundreds of training pairs on driving scenarios. DrivingStereo [76] further scales up to 180K stereo pairs. Nevertheless, the sparse ground-truth disparity obtained by LiDAR sensors hinders learning accurate and dense stereo matching. Middlebury [51] and ETH3D [52] develop a low number of training data covering both indoor and outdoor scenarios beyond driving. Booster [48] presents a real-world dataset focusing on transparent objects. InStereo2K [2] presents a larger training dataset consisting of 2 K stereo pairs with denser ground-truth disparity obtained with structured light system. However, challenges of scarce data size, imperfect ground-truth disparity and lack of collection scalability in real-world have driven the widespread adoption of synthetic data for training. This includes Scene Flow [43], Sintel [6], CREStereo [34], IRS [64], TartanAir [66], FallingThings [61], Virtual KITTI 2 [7], CARLA HR-VS [75], Dynamic Replica [28]. In Tab. 1, we compare our proposed FoundationStereo dataset (FSD) with commonly used synthetic training datasets for

| Properties | Simpl [8] | Sceneflow [43] | CREStereo [34] | IRS [64] | TartanAir [66] | FallingThings [61] | UnrealStereo+IR [59] | Spring [44] | FSD (Ours) |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| Flying Objects | $\#$ | $c$ | $c$ | $\#$ | $\#$ | $\#$ | $\#$ | $\#$ | $c$ |
| Indoor | $\#$ | $\#$ | $\#$ | $c$ | $c$ | $c$ | $c$ | $\#$ | $c$ |
| Outdoor | $\#$ | $c$ | $\#$ | $\#$ | $c$ | $c$ | $c$ | $\#$ | $c$ |
| Driving | $\#$ | $c$ | $\#$ | $\#$ | $\#$ | $\#$ | $\#$ | $\#$ | $c$ |
| Movie | $c$ | $c$ | $c$ | $\#$ | $\#$ | $\#$ | $\#$ | $c$ | $\#$ |
| Simulator | Blender | Blender | Blender | Unreal Engine | Unreal Engine | Unreal Engine | Unreal Engine | Blender | NVIDIA Omnivert |
| Rendering Realism | High | Low | High | High | High | High | High | High | High |
| Scenes | 10 | 9 | 0 | 4 | 18 | 3 | 8 | 47 | 12 |
| Layout Realism | Medium | Low | Low | High | High | Medium | High | High | High |
| Stereo Pairs | 1K | 40K | 200K | 103K | 306K | 62K | 7.7K | 8K | 1000K |
| Resolution | 1024 × 436 | 960 × 540 | 1920 × 1080 | 960 × 540 | 640 × 480 | 960 × 540 | 2640 × 2160 | 1920 × 1080 | 1280 × 720 |
| Reflections | $\#$ | $\#$ | $c$ | $c$ | $c$ | $c$ | $c$ | $\#$ | $c$ |
| Camera Params | Constant | Constant | Constant | Constant | Constant | Constant | Constant | $\begin{aligned} & \text { Unreal } \\ & \text { Simpl } \end{aligned}$ | Varying baseline and intrinsics |

Table 1. Synthetic datasets for training stereo algorithms (excluding test images with inaccessible ground truth). $^{\text {I }}$ Indicates reduced diversity, caused by including many similar frames from video sequences.
stereo matching. Our dataset encompasses a wide range of scenarios, features the largest data volume to date, includes diverse 3D assets, captures stereo images under diversely randomized camera parameters, and achieves high fidelity in both rendering and spatial layouts.
Vision Foundation Models. Vision foundation models have significantly advanced across various vision tasks in 2D, 3D and multi-modal alignment. CLIP [47] leverages large-scale image-text pair training to align visual and textual modalities, enabling zero-shot classification and facilitating cross-modal applications. DINO series [8, 38, 46] employ self-supervised learning for dense representation learning, effectively capturing detailed features critical for segmentation and recognition tasks. SAM series [32, 50, 77] demonstrate high versatility in segmentation driven by various prompts such as points, bounding boxes, language. Similar advancements also appear in 3D vision tasks. DUSt3R [65] and MASt3R [33] present generalizable frameworks for dense 3D reconstruction from uncalibrated and unposed cameras. FoundationPose [69] develops a unified framework of 6D object pose estimation and tracking for novel objects. More closely related to this work, a number of efforts [4, 29, 78, 79] demonstrated strong generalization in monocular depth estimation task and multi-view stereo [26]. Together, these approaches exemplify under the scaling law, how foundation models in vision are evolving to support robust applications across diverse scenarios without tedious per-domain fine-tuning.

## 3. Approach

The overall network architecture is shown in Fig. 2. The rest of this section describes the various components.

### 3.1. Monocular Foundation Model Adaptation

To mitigate the sim-to-real gap when the stereo network is primarily trained on synthetic dataset, we leverage the recent advancements on monocular depth estimation trained on internet-scale real data [5, 79]. We use a CNN network to adapt the ViT-based monocular depth estimation network to the stereo setup, thus synergizing the strengths of both CNN and ViT architectures.

We explored multiple design choices for combining CNN and ViT approaches, as outlined in Fig. 3 (left). In particular, (a) directly uses the feature pyramids from the DPT head in a frozen DepthAnythingV2 [79] without using CNN features. (b) resembles ViT-Adapter [12] by exchanging features between CNN and ViT. (c) applies a $4 \times 4$ convolution with stride 4 to downscale the feature before the DepthAnythingV2 final output head. The feature is then concatenated with the same level CNN feature to obtain a hybrid feature at $1 / 4$ scale. The side CNN network is thus learned to adapt the ViT features [83] to stereo matching task. Surprisingly, while being simple, we found (c) significantly surpasses the alternative choices on the stereo matching task, as shown in the experiments (Sec. 4.5). As a result, we adopt (c) as the main design of STA module.

Formally, given a pair of left and right images $I_{l}, I_{r} \in$ $\mathbb{R}^{H \times W \times 3}$, we employ EdgeNeXt-S [40] as the CNN module within STA to extract multi-level pyramid features, where the $1 / 4$ level feature is equipped with DepthAnythingV2 feature: $f_{l}^{(i)}, f_{r}^{(i)} \in \mathbb{R}^{C_{i} \times \frac{H}{4} \times \frac{W}{4}}, i \in$ $\{4,8,16,32\}$. EdgeNeXt-S [40] is chosen for its memory efficiency and because larger CNN backbones did not yield additional benefits in our investigation. When forwarding to DepthAnythingV2, we first resize the image to be divisible by 14 , to be consistent with its pretrained patch size. The STA weights are shared when applied to $I_{l}, I_{r}$.

Similarly, we employ STA to extract context feature, with the difference that the CNN module is designed with a sequence of residual blocks [25] and down-sampling layers. It generates context features of multiple scales: $f_{c}^{(i)} \in$ $\mathbb{R}^{C_{i} \times \frac{H}{4} \times \frac{W}{4}}, i \in\{4,8,16\}$, as in [36]. $f_{c}$ participates in initializing the hidden state of the ConvGRU block and inputting to the ConvGRU block at each iteration, effectively guiding the iterative process with progressively refined contextual information.

Fig. 3 visualizes the power of rich monocular prior that helps to reliably predict on ambiguous regions which is challenging to deal with by naive correspondence search along the epipolar line. Instead of using the raw monocular depth from DepthAnythingV2 which has scale ambiguity, we use its latent feature as geometric priors extracted from

![img-2.jpeg](img-2.jpeg)

Figure 2. Overview of our proposed FoundationStereo. The Side-Tuning Adapter (STA) adapts the rich monocular priors from a frozen DepthAnythingV2 [79], while combined with fine-grained high-frequency features from multi-level CNN for unary feature extraction. Attentive Hybrid Cost Filtering (AHCF) combines the strengths of the Axial-Planar Convolution (APC) filtering and a Disparity Transformer (DT) module to effectively aggregate the features along spatial and disparity dimensions over the 4D hybrid cost volume. An initial disparity is then predicted from the filtered cost volume, and subsequently refined through GRU blocks. At each refinement step, the latest disparity is used to look up features from both filtered hybrid cost volume and correlation volume to guide the next refinement. The iteratively refined disparity becomes the final output.

![img-3.jpeg](img-3.jpeg)

Figure 3. Left: Design choices for STA module. Right: Effects of the proposed STA and AHCF modules. "W/o STA" only uses CNN to extract features. "W/o AHCF" uses conventional 3D CNN-based hourglass network for cost volume filtering. Results are obtained via zero-shot inference without finetuning on target dataset. STA leverages rich monocular prior to reliably predict the lamp region with inconsistent lighting and dark guitar sound hole. AHCF effectively aggregates the spatial and long-range disparity context to accurately predict over thin repetitive structures.
both stereo images and compared through cost filtering as described next.

### 3.2. Attentive Hybrid Cost Filtering

Hybrid Cost Volume Construction. Given unary features at $1 / 4$ scale $f_{t}^{4}, f_{c}^{4}$ extracted from previous step, we construct the cost volume $\mathbf{V}_{\mathbf{C}} \in \mathbb{R}^{C \times \frac{D}{4} \times \frac{D}{4} \times \frac{W}{4}}$ with a combination of group-wise correlation and concatenation [24]:

$$
\begin{aligned}
& \mathbf{V}_{\mathrm{gwc}}(g, d, h, w)=\left\langle\widetilde{f}_{l, g}^{(4)}(h, w), \widetilde{f}_{r, g}^{(4)}(h, w-d)\right\rangle \\
& \mathbf{V}_{\mathrm{cat}}(d, h, w)=\left[\operatorname{Conv}\left(f_{t}^{(4)}\right)(h, w), \operatorname{Conv}\left(f_{c}^{(4)}\right)(h, w-d)\right] \\
& \mathbf{V}_{\mathbf{C}}(d, h, w)=\left[\mathbf{V}_{\mathrm{gwc}}(d, h, w), \mathbf{V}_{\mathrm{cat}}(d, h, w)\right]
\end{aligned}
$$

where $\widetilde{f}$ denotes $L_{2}$ normalized feature for better training stability; $\langle\cdot, \cdot\rangle$ represents dot product; $g \in\{1,2, \ldots, G\}$ is the group index among the total $G=8$ feature groups that we evenly divide the total features into; $d \in\left\{1,2, \ldots, \frac{D}{4}\right\}$ is the disparity index. $[\cdot, \cdot]$ denotes concatenation along channel dimension. The group-wise correlation $\mathbf{V}_{\text {gwc }}$ harnesses the strengths of conventional correlation-based matching
costs, offering a diverse set of similarity measurement features from each group. $\mathbf{V}_{\text {cat }}$ preserves unary features including the rich monocular priors by concatenating left and right features at shifted disparity. To reduce memory consumption, we linearly downsize the unary feature dimension to 14 using a convolution of kernel size 1 (weights are shared between $f_{t}^{4}$ and $f_{c}^{4}$ ) before concatenation. Next, we describe two sub-modules for effective cost volume filtering.

Axial-Planar Convolution (APC) Filtering. An hourglass network consisting of 3D convolutions, with three downsampling blocks and three up-sampling blocks with residual connections, is leveraged for cost volume filtering [1, 71]. While 3D convolutions of kernel size $3 \times 3 \times 3$ are commonly used for relatively small disparity sizes [9, 24, 71], we observe it struggles with larger disparities when applied to high resolution images, especially since the disparity dimension is expected to model the probability distribution for the initial disparity prediction. However, it is impractical to naively increase the kernel size, due to the intensive memory consumption. In fact, even when setting kernel size to $5 \times 5 \times 5$ we observe unmanageable memory usage on an 

![img-4.jpeg](img-4.jpeg)

80 GB GPU. This drastically limits the model's representation power when scaling up with large amount of training data. We thus develop "Axial-Planar Convolution" which decouples a single $3 \times 3 \times 3$ convolution into two separate convolutions: one over spatial dimensions (kernel size $K_{s} \times K_{s} \times 1$ ) and the other over disparity $\left(1 \times 1 \times K_{d}\right)$, each followed by BatchNorm and ReLU. APC can be regarded as a 3D version of Separable Convolution [16] with the difference that we only separate the spatial and disparity dimensions without subdividing the channel into groups which sacrifices representation power. The disparity dimension is specially treated due to its uniquely encoded feature comparison within the cost volume. We use APC wherever possible in the hourglass network except for the downsampling and up-sampling layers.
Disparity Transformer (DT). While prior works [35, 68] introduced transformer architecture to unary feature extraction step to scale up stereo training, the cost filtering process is often overlooked, which remains an essential step in achieving accurate stereo matching by encapsulating correspondence information. Therefore, we introduce DT to further enhance the long-range context reasoning within the 4 D cost volume. Given $\mathbf{V}_{\mathbf{C}}$ obtained in Eq. (1), we first apply a 3D convolution of kernel size $4 \times 4 \times 4$ with stride 4 to downsize the cost volume. We then reshape the volume into a batch of token sequences, each with length of disparity. We apply position encoding before feeding it to a series ( 4 in our case) of transformer encoder blocks, where FlashAttention [18] is leveraged to perform multi-head selfattention [63]. The process can be written as:

$$
\begin{aligned}
& \mathbf{Q}_{0}=\operatorname{PE}\left(\mathbf{R}\left(\operatorname{Conv}_{4 \times 4 \times 4}\left(\mathbf{V}_{\mathbf{C}}\right)\right)\right) \in \mathbb{R}\left(\frac{H}{16} \times \frac{W}{16}\right) \times C \times \frac{C}{16} \\
& \text { MultiHead }(\mathbf{Q}, \mathbf{K}, \mathbf{V})=\left[\text { head }_{1}, \ldots, \text { head }_{h}\right] \mathbf{W}_{O} \\
& \quad \text { where head }_{i}=\text { FlashAttention }\left(\mathbf{Q}_{i}, \mathbf{K}_{i}, \mathbf{V}_{i}\right) \\
& \mathbf{Q}_{1}=\operatorname{Norm}\left(\operatorname{MultiHead}\left(\mathbf{Q}_{0}, \mathbf{Q}_{0}, \mathbf{Q}_{0}\right)+\mathbf{Q}_{0}\right) \\
& \mathbf{Q}_{2}=\operatorname{Norm}\left(\operatorname{FFN}\left(\mathbf{Q}_{1}\right)+\mathbf{Q}_{1}\right)
\end{aligned}
$$

where $\mathbf{R}(\cdot)$ denotes reshape operation; $\mathrm{PE}(\cdot)$ represents position encoding; $[\cdot, \cdot]$ denotes concatenation along the channel dimension; $\mathbf{W}_{O}$ is linear weights. The number of heads is $h=4$ in our case. Finally, the DT output is up-sampled to the same size as $\mathbf{V}_{\mathbf{C}}$ using trilinear interpolation and summed with hourglass output, as shown in Fig. 2.
Initial Disparity Prediction. We apply soft-argmin [30] to the filtered volume $\mathbf{V}_{\mathbf{C}}^{\prime}$ to produce an initial disparity:

$$d_{0}=\sum_{d=0}^{D-1} d \cdot \operatorname{Softmax}\left(\mathbf{V}_{\mathbf{C}}^{\prime}\right)(d)$$

where $d_{0}$ is at $1 / 4$ scale of the original image resolution.

### 3.3. Iterative Refinement

Given $d_{0}$, we perform iterative GRU updates to progressively refine disparity, which helps to avoid local optimum and accelerate convergence [71]. In general, the k-th update
can be formulated as:

$$
\begin{aligned}
& \mathbf{V}_{\text {corr }}\left(w^{\prime}, h, w\right)=\left\langle f_{l}^{(4)}(h, w), f_{c}^{(4)}\left(h, w^{\prime}\right)\right\rangle \\
& \mathbf{F}_{\mathbf{V}}(h, w)=\left[\mathbf{V}_{\mathbf{C}}^{\prime}\left(d_{k}, h, w\right), \mathbf{V}_{\text {corr }}\left(w-d_{k}, h, w\right)\right] \\
& x_{k}=\left[\operatorname{Conv}_{v}\left(\mathbf{F}_{\mathbf{V}}\right), \operatorname{Conv}_{d}\left(d_{k}\right), d_{k}, c\right] \\
& z_{k}=\sigma\left(\operatorname{Conv}_{z}\left(\left[h_{k-1}, x_{k}\right]\right)\right) \\
& r_{k}=\sigma\left(\operatorname{Conv}_{r}\left(\left[h_{k-1}, x_{k}\right]\right)\right) \\
& \hat{h}_{k}=\tanh \left(\operatorname{Conv}_{h}\left(\left[r_{k} \odot h_{k-1}, x_{k}\right]\right)\right) \\
& h_{k}=\left(1-z_{k}\right) \odot h_{k-1}+z_{k} \odot \hat{h}_{k} \\
& d_{k+1}=d_{k}+\operatorname{Conv}_{\Delta}\left(h_{k}\right)
\end{aligned}
$$

where $\odot$ denotes element-wise product; $\sigma$ denotes sigmoid; $\mathbf{V}_{\text {corr }} \in \mathbb{R}^{\frac{W}{4} \times \frac{W}{4} \times \frac{W}{4}}$ is the pair-wise correlation volume; $\mathbf{F}_{\mathbf{V}}$ represents the looked up volume features using latest disparity; $c=\operatorname{ReLU}\left(f_{c}\right)$ encodes the context feature from left image, including STA adapted features (Sec. 3.1) which effectively guide the refinement process leveraging rich monocular priors.

We use three levels of GRU blocks to perform coarse-to-fine hidden state update in each iteration, where the initial hidden states are produced from context features $h_{0}^{(i)}=$ $\tanh \left(f_{c}^{(i)}\right), i \in\{4,8,16\}$. At each level, attention-based selection mechanism [67] is leveraged to capture information at different frequencies. Finally, $d_{k}$ is up-sampled to the full resolution using convex sampling [57].

### 3.4. Loss Function

The model is trained with the following objective:

$$\mathcal{L}=\left|d_{0}-\bar{d}\right|_{\text {smooth }}+\sum_{k=1}^{K} \gamma^{K-k}\left\|d_{k}-\bar{d}\right\|_{1}$$

where $\bar{d}$ represents ground-truth disparity; $|\cdot|_{\text {smooth }}$ denotes smooth $L_{1}$ loss; $k$ is the iteration number; $\gamma$ is set to 0.9 , and we apply exponentially increasing weights [36] to supervise the iteratively refined disparity.

### 3.5. Synthetic Training Dataset

We created a large scale synthetic training dataset with NVIDIA Omniverse. This FoundationStereo Dataset (FSD) accounts for crucial stereo matching challenges such as reflections, low-texture surfaces, and severe occlusions. We perform domain randomization [58] to augment dataset diversity, including random stereo baseline, focal length, camera perspectives, lighting conditions and object configurations. Meanwhile, high-quality 3D assets with abundant textures and path-tracing rendering are leveraged to enhance realism in rendering and layouts. Fig. 4 displays some samples from our dataset including both structured indoor and outdoor scenarios, as well as more diversely randomized flying objects with various geometries and textures under complex yet realistic lighting. See the appendix for details.
Iterative Self-Curation. While synthetic data generation

![img-5.jpeg](img-5.jpeg)

Figure 4. Left: Samples from our FoundationStereo dataset (FSD), which consists of synthetic stereo images with structured indoor / outdoor scenes (top), as well as more randomized scenes with challenging flying objects and higher geometry and texture diversity (bottom). Right: The iterative self-curation process removes ambiguous samples inevitably produced from the domain randomized synthetic data generation process. Example ambiguities include severe texture repetition, ubiquitous reflections with limited surrounding context, and pure color under improper lighting.
in theory can produce unlimited amount of data and achieve large diversity through randomization, ambiguities can be inevitably introduced especially for less structured scenes with flying objects, which confuses the learning process. To eliminate those samples, we design an automatic iterative self-curation strategy. Fig. 4 demonstrates this process and detected ambiguous samples. We start with training an initial version of FoundationStereo on FSD, after which it is evaluated on FSD. Samples where BP-2 (Sec. 4.2) is larger than $60 \%$ are regarded as ambiguous samples and replaced by regenerating new ones. The training and curation processes are alternated to iteratively (twice in our case) update both FSD and FoundationStereo.

## 4. Experiments

### 4.1. Implementation Details

We implement FoundationStereo in PyTorch. The foundation model is trained on a mixed dataset consisting of our proposed FSD, together with Scene Flow [43], Sintel [6], CREStereo [34], FallingThings [61], InStereo2K [2] and Virtual KITTI 2 [7]. We train FoundationStereo using AdamW optimizer [39] for 200 K steps with a total batch size of 128 evenly distributed over 32 NVIDIA A100 GPUs. The learning rate starts at 1e-4 and decays by 0.1 at 0.8 of the entire training process. Images are randomly cropped to $320 \times 736$ before feeding to the network. Data augmentations similar to [36] are performed. During training, 22 iterations are used in GRU updates. In the following, unless otherwise mentioned, we use the same foundation model for zero-shot inference with 32 refinement iterations and 416 for maximum disparity.

### 4.2. Benchmark Datasets and Metric

Datasets. We consider five commonly used public datasets for evaluation: Scene Flow [43] is a synthetic dataset including three subsets: FlyingThings3D, Driving, and Monkaa. Middlebury [51] consists of indoor stereo image pairs with high-quality ground-truth disparity captured via structured light. Unless otherwise mentioned, evaluations are performed on half resolution and non-occluded regions. ETH3D [52] provides grayscale stereo image pairs cover-

| Methods | Middlebury | ETH3D | KITTI-12 | KITTI-15 |
| :-- | :--: | :--: | :--: | :--: |
|  | BP-2 | BP-1 | D1 | D1 |
| CREStereo++ [27] | 14.8 | 4.4 | 4.7 | 5.2 |
| DSMNet [82] | 13.8 | 6.2 | 6.2 | 6.5 |
| Mask-CFNet [49] | 13.7 | 5.7 | 4.8 | 5.8 |
| HVT-RAFT [10] | 10.4 | 3.0 | 3.7 | 5.2 |
| RAFT-Stereo [36] | 12.6 | 3.3 | 4.7 | 5.5 |
| Selective-IGEV [67] | 9.2 | 5.7 | 4.5 | 5.6 |
| IGEV [36] | 8.8 | 4.0 | 5.2 | 5.7 |
| Former-RAFT-DAM [84] | 8.1 | 3.3 | 3.9 | 5.1 |
| IGEV++ [72] | 7.8 | 4.1 | 5.1 | 5.9 |
| NMRF [22] | 7.5 | 3.8 | 4.2 | 5.1 |
| Ours (Scene Flow) | $\mathbf{5 . 5}$ | $\mathbf{1 . 8}$ | $\mathbf{3 . 2}$ | $\mathbf{4 . 9}$ |
| Selective-IGEV* [67] | 7.5 | 3.4 | 3.2 | 4.5 |
| Ours | $\mathbf{1 . 1}$ | $\mathbf{0 . 5}$ | $\mathbf{2 . 3}$ | $\mathbf{2 . 8}$ |

Table 2. Zero-shot generalization results on four public datasets. The most commonly used metrics for each dataset were adopted. In the first block, all methods were trained only on Scene Flow. In the second block, methods are allowed to train on any existing datasets excluding the four target domains. The weights and parameters are fixed for evaluation.
ing both indoor and outdoor scenarios. KITTI 2012 [20] and KITTI 2015 [45] datasets feature real-world driving scenes, where sparse ground-truth disparity maps are provided, which are derived from LIDAR sensors.
Metrics. "EPE" computes average per-pixel disparity error. "BP-X" computes the percentage of pixels where the disparity error is larger than X pixels. "D1" computes the percentage of pixels whose disparity error is larger than 3 pixels and $5 \%$ of the ground-truth disparity.

### 4.3. Zero-Shot Generalization Comparison

Benchmark Evaluation. Tab. 2 exhibits quantitative comparison of zero-shot generalization results on four public real-world datasets. Even when trained solely on Scene Flow, our method outperforms the comparison methods consistently across all datasets, thanks to the efficacy of adapting rich monocular priors from vision foundation models. We further evaluate in a more realistic setup, allowing methods to train on any available dataset while excluding the target domain, to achieve optimal zero-shot inference results as required in practical applications.
In-the-Wild Generalization. We compare our foundation model against recent approaches that released their checkpoints trained on a mixture of datasets, to resemble the practical zero-shot application on in-the-wild images. Com-

![img-6.jpeg](img-6.jpeg)

Figure 5. Qualitative comparison of zero-shot inference on in-the-wild images. For each comparison method we select the best performing checkpoint from their public release, which has been trained on a mixture of public datasets. These images exhibit challenging reflection, translucency, repetitive textures, complex illuminations and thin-structures, revealing the importance of our network architecture and large-scale training.

| Method | LEAStereo [15] | GANet [81] | ACVNet [70] | IGEV-Stereo [71] | NMRF [22] | MoCha-Stereo [14] | Selective-IGEV [67] | Ours |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| EPE | 0.78 | 0.84 | 0.48 | 0.47 | 0.45 | 0.41 | 0.44 | $\mathbf{0 . 3 4}$ |

Table 3. Comparison of methods trained / tested on the Scene Flow train / test sets, respectively.

| Method | Zero-Shot | BP-0.5 | BP-1.0 | EPE |
| :-- | --: | --: | --: | --: |
| GMStereo [74] | $\#$ | 5.94 | 1.83 | 0.19 |
| HITNet [56] | $\#$ | 7.83 | 2.79 | 0.20 |
| EAI-Stereo [85] | $\#$ | 5.21 | 2.31 | 0.21 |
| RAFT-Stereo [36] | $\#$ | 7.04 | 2.44 | 0.18 |
| CREStereo [34] | $\#$ | 3.58 | 0.98 | 0.13 |
| IGEV-Stereo [71] | $\#$ | 3.52 | 1.12 | 0.14 |
| CroCo-Stereo [68] | $\#$ | 3.27 | 0.99 | 0.14 |
| MoCha-Stereo [14] | $\#$ | 3.20 | 1.41 | 0.13 |
| Selective-IGEV [67] | $\#$ | 3.06 | 1.23 | 0.12 |
| Ours (finetuned) | $\#$ | $\mathbf{1 . 2 6}$ | $\mathbf{0 . 2 6}$ | $\mathbf{0 . 0 9}$ |
| Ours | $\checkmark$ | 2.31 | 1.52 | 0.13 |

Table 4. Results on ETH3D leaderboard (test set). All methods except for the last row have used ETH3D training set for fine-tuning. Our fine-tuned version ranks 1st on leaderboard at the time of submission. Last row is obtained via zero-shot inference from our foundation model.
parison methods include CroCo v2 [68], CREStereo [34], IGEV [71] and Selective-IGEV [67]. For each method, we select the best performing checkpoint from their public release. In this evaluation, the four real-world benchmark datasets [20, 45, 51, 52] have been used for training comparison methods, whereas they are not used in our fixed foundation model. Fig. 5 displays qualitative comparison on various scenarios, including a robot scene from DROID [31] dataset and custom captures covering indoor and outdoor.

### 4.4. In-Domain Comparison

Tab. 3 presents quantitative comparison on Scene Flow, where all methods are following the same officially divided train and test split. Our FoundationStereo model outperforms the comparison methods by a large margin, reducing the previous best EPE from 0.41 to 0.33. Although indomain training is not the focus of this work, the results reflect the effectiveness of our model design.

Tab. 4 exhibits quantitative comparison on ETH3D leaderboard (test set). For our approach, we perform evaluations in two settings. First, we fine-tune our foundation model on a mixture of the default training dataset (Sec. 4.1) and ETH3D training set for another 50K steps, using the same learning rate schedule and data augmentation. Our model significantly surpasses the previous best approach by reducing more than half of the error rates and ranks 1st on leaderboard at the time of submission. This indicates great potential of transferring capability from our foundation model if in-domain fine-tuning is desired. Second, we also evaluated our foundation model without using any data from ETH3D. Remarkably, our foundation model's zeroshot inference achieves comparable or even better results than leading approaches that perform in-domain training.

In addition, our finetuned model also ranks 1st on the Middlebury leaderboard. See appendix for details.

### 4.5. Ablation Study

We investigate different design choices for our model and dataset. Unless otherwise mentioned, we train on a randomly subsampled version (100K) of FSD to make the experiment scale more affordable. Given Middlebury dataset's high quality ground-truth, results are evaluated on its training set to reflect zero-shot generalization. Since the focus of this work is to build a stereo matching foundation

| Row | Variations | BP-2 |
| :--: | :--: | :--: |
| 1 | DINOv2-L [46] | 2.46 |
| 2 | DepthAnythingV2-S [79] | 2.22 |
| 3 | DepthAnythingV2-B [79] | 2.11 |
| 4 | DepthAnythingV2-L [79] | 1.97 |
| 5 | STA (a) | 6.48 |
| 6 | STA (b) | 2.22 |
| 7 | STA (c) | 1.97 |
| 8 | Unfreeze ViT | 3.94 |
| 9 | Freeze ViT | 1.97 |

Table 5. Ablation study of STA module. Variations (a-c) correspond to Fig. 3. The choices adopted in our full model are highlighted in green.

| Row | Variations | BP-2 | Row | Variations | BP-2 |
| :--: | :--: | :--: | :--: | :--: | :--: |
| 1 | RoPE | 2.19 | 10 | $(3,3,1),(1,1,5)$ | 2.10 |
| 2 | Cosine | 1.97 | 11 | $(3,3,1),(1,1,9)$ | 2.06 |
| 3 | $1 / 32$ | 2.06 | 12 | $(3,3,1),(1,1,13)$ | 2.01 |
| 4 | $1 / 16$ | 1.97 | 13 | $(3,3,1),(1,1,17)$ | 1.97 |
| 5 | Full | 2.25 | 14 | $(3,3,1),(1,1,21)$ | 1.98 |
| 6 | Disparity | 1.97 | 15 | $(7,7,1),(1,1,17)$ | 1.99 |
| 7 | Pre-hourglass | 2.06 |  |  |  |
| 8 | Post-hourglass | 2.20 |  |  |  |
| 9 | Parallel | 1.97 |  |  |  |

Table 6. Ablation study of AHCF module. Left corresponds to DT, while right corresponds to APC. The choices adopted in our full model are highlighted in green.
model with strong generalization, we do not deliberately limit model size while pursuing better performance.
STA Design Choices. As shown in Tab. 5, we first compare different vision foundation models for adapting rich monocular priors, including different model sizes of DepthAnythingV2 [79] and DINOv2-Large [46]. While DINOv2 previously exhibited promising results in correspondence matching [19], it is not as effective as DepthAnythingV2 in the stereo matching task, possibly due to its less taskrelevance and its limited resolution to reason high-precision pixel-level correspondence. We then study different design choices from Fig. 3. Surprisingly, while being simple, we found (c) significantly surpasses the alternatives. We hypothesize the latest feature before the final output head preserves high-resolution and fine-grained semantic and geometric priors that are suitable for subsequent cost volume construction and filtering process. We also experimented whether to freeze the adapted ViT model. As expected, unfreezing ViT corrupts the pretrained monocular priors, leading to degraded performance.
AHCF Design Choices. As shown in Tab. 6, for DT module we study different position embedding (row 1-2); different feature scale to perform transformer (row 3-4); transformer over the full cost-volume or only along the disparity dimension (row 5-6); different placements of DT module relative to the hourglass network (row 7-9). Specifically, RoPE [55] encodes relative distances between tokens instead of absolute positions, making it more adaptive to vary-

| Row | STA | AHCF |  | BP2 | Row | FSD | BP2 |
| :--: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  | APC | DT |  | 1 | $\#$ | 2.34 |
| 1 |  |  |  | 2.48 | 2 | $\checkmark$ | 1.15 |
| 2 | $\checkmark$ |  |  | 2.21 |  |  |  |
| 3 | $\checkmark$ | $\checkmark$ |  | 2.16 |  |  |  |
| 4 | $\checkmark$ |  | $\checkmark$ | 2.05 |  |  |  |
| 5 | $\checkmark$ | $\checkmark$ | $\checkmark$ | 1.97 |  |  |  |

Table 7. Left: Ablation study of proposed network modules. Right: Ablation study of whether to use FSD dataset when training the foundation model described in Sec. 4.1. The choices adopted in our full model are highlighted in green.
ing sequence lengths. However, it does not outperform cosine position embedding, probably due to the constant disparity size in 4D cost volume. While in theory, full volume attention provides larger receptive field, it is less effective than merely applying over the disparity dimension of the cost volume. We hypothesize the extremely large space of 4D cost volume makes it less tractable, whereas attention over disparity provides sufficient context for a better initial disparity prediction and subsequent volume feature lookup during GRU updates. Next, we compare different kernel sizes in APC (row 10-15), where the last dimension in each parenthesis corresponds to disparity dimension. We observe increasing benefits when enlarging disparity kernel size until it saturates at around 17.
Effects of Proposed Modules. The quantitative effects are shown in Tab. 7 (left). STA leverages rich monocular priors which greatly enhances generalization to real images for ambiguous regions. DT and APC effectively aggregate cost volume features along spatial and disparity dimensions, leading to improved context for disparity initialization and subsequent volume feature look up during GRU updates. Fig. 3 further visualizes the resulting effects.
Effects of FoundationStereo Dataset. We study whether to include FSD dataset with the existing public datasets for training our foundation model described in Sec. 4.1. Results are shown in Tab. 7 (right).

## 5. Conclusion

We introduced FoundationStereo, a foundation model for stereo depth estimation that achieves strong zero-shot generalization across various domains without fine-tuning. We envision such a foundation model can facilitate broader adoption of stereo estimation models in practical applications. Despite its remarkable generalization, it has several limitations. First, our model is not yet optimized for efficiency, which takes 0.7 s on image size of $375 \times 1242$ on NVIDIA A100 GPU. Future work could explore adapting distillation and pruning techniques applied to other vision foundation models [13, 87]. Second, our dataset FSD includes a limited collection of transparent objects. Robustness could be further enhanced by augmenting with a larger diversity of fully transparent objects during training.

## References

[1] Antyanta Bangunharcana, Jae Won Cho, Seokju Lee, In So Kweon, Kyung-Soo Kim, and Soohyun Kim. Correlate-andexcite: Real-time stereo matching via guided cost volume excitation. In 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 3542-3548. IEEE, 2021. 4
[2] Wei Bao, Wei Wang, Yuhua Xu, Yulan Guo, Siyu Hong, and Xiaohu Zhang. InStereo2k: a large real dataset for stereo matching in indoor scenes. Science China Information Sciences, 63:1-11, 2020. 2, 6
[3] Luca Bartolomei, Fabio Tosi, Matteo Poggi, and Stefano Mattoccia. Stereo anywhere: Robust zero-shot deep stereo matching even where either stereo or mono fail. arXiv preprint arXiv:2412.04472, 2024. 2
[4] Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias Müller. ZoeDepth: Zero-shot transfer by combining relative and metric depth. arXiv preprint arXiv:2302.12288, 2023. 3
[5] Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R Richter, and Vladlen Koltun. Depth Pro: Sharp monocular metric depth in less than a second. arXiv preprint arXiv:2410.02073, 2024. 3
[6] Daniel J Butler, Jonas Wulff, Garrett B Stanley, and Michael J Black. A naturalistic open source movie for optical flow evaluation. In Proceedings of the European Conference on Computer Vision (ECCV), pages 611-625, 2012. 2, 3, 6
[7] Yohann Cabon, Naila Murray, and Martin Humenberger. Virtual KITTI 2. arXiv preprint arXiv:2001.10773, 2020. 2, 6
[8] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 9650-9660, 2021. 3
[9] Jia-Ren Chang and Yong-Sheng Chen. Pyramid stereo matching network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5410-5418, 2018. 4
[10] Tianyu Chang, Xun Yang, Tianzhu Zhang, and Meng Wang. Domain generalized stereo matching via hierarchical visual transformation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9559-9568, 2023. 1, 2, 6
[11] Liyan Chen, Weihan Wang, and Philippos Mordohai. Learning the distribution of errors in stereo matching for joint disparity and uncertainty estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 17235-17244, 2023. 1, 2
[12] Zhe Chen, Yuchen Duan, Wenhai Wang, Junjun He, Tong Lu, Jifeng Dai, and Yu Qiao. Vision transformer adapter for dense predictions. ICLR, 2023. 3
[13] Zigeng Chen, Gongfan Fang, Xinyin Ma, and Xinchao Wang. 0.1% data makes segment anything slim. NeurIPS, 2023. 8
[14] Ziyang Chen, Wei Long, He Yao, Yongjun Zhang, Bingshu Wang, Yongbin Qin, and Jia Wu. Mocha-stereo: Motif channel attention network for stereo matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 27768-27777, 2024. 1, 2, 7
[15] Xuelian Cheng, Yiran Zhong, Mehrtash Harandi, Yuchao Dai, Xiaojun Chang, Hongdong Li, Tom Drummond, and Zongyuan Ge. Hierarchical neural architecture search for deep stereo matching. Proceedings of Neural Information Processing Systems (NeurIPS), 33:22158-22169, 2020. 7
[16] François Chollet. Xception: Deep learning with depthwise separable convolutions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1251-1258, 2017. 5
[17] WeiQin Chuah, Ruwan Tennakoon, Reza Hoseinnezhad, Alireza Bab-Hadiashar, and David Suter. ITSA: An information-theoretic approach to automatic shortcut avoidance and domain generalization in stereo matching networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13022-13032, 2022. 1, 2
[18] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with io-awareness. Proceedings of Neural Information Processing Systems (NeurIPS), 35:16344-16359, 2022. 5
[19] Mohamed El Banani, Amit Raj, Kevis-Kokitsi Maninis, Abhishek Kar, Yuanzhen Li, Michael Rubinstein, Deqing Sun, Leonidas Guibas, Justin Johnson, and Varan Jampani. Probing the 3D awareness of visual foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21795-21806, 2024. 8
[20] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the KITTI vision benchmark suite. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 33543361, 2012. 2, 6, 7
[21] Rui Gong, Weide Liu, Zaiwang Gu, Xulei Yang, and Jun Cheng. Learning intra-view and cross-view geometric knowledge for stereo matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20752-20762, 2024. 1, 2
[22] Tongfan Guan, Chen Wang, and Yun-Hui Liu. Neural markov random field for stereo matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5459-5469, 2024. 6, 7
[23] Weiyu Guo, Zhaoshuo Li, Yongkui Yang, Zheng Wang, Russell H Taylor, Mathias Unberath, Alan Yuille, and Yingwei Li. Context-enhanced stereo transformer. In Proceedings of the European Conference on Computer Vision (ECCV), pages 263-279, 2022. 2
[24] Xiaoyang Guo, Kai Yang, Wukui Yang, Xiaogang Wang, and Hongsheng Li. Group-wise correlation stereo network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3273-3282, 2019. 4
[25] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 770-778, 2016. 3
[26] Sergio Izquierdo, Mohamed Sayed, Michael Firman, Guillermo Garcia-Hernando, Daniyar Turmukhambetov, Javier Civera, Oisin Mac Aodha, Gabriel J. Brostow, and Jamie Watson. MVSAnywhere: Zero shot multi-view stereo. In CVPR, 2025. 3
[27] Junpeng Jing, Jiankun Li, Pengfei Xiong, Jiangyu Liu, Shuaicheng Liu, Yichen Guo, Xin Deng, Mai Xu, Lai Jiang, and Leonid Sigal. Uncertainty guided adaptive warping for robust and efficient stereo matching. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 3318-3327, 2023. 1, 2, 6
[28] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. DynamicStereo: Consistent dynamic depth from stereo videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1322913239, 2023. 2
[29] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9492-9502, 2024. 3
[30] Alex Kendall, Hayk Martirosyan, Saumitro Dasgupta, Peter Henry, Ryan Kennedy, Abraham Bachrach, and Adam Bry. End-to-end learning of geometry and context for deep stereo regression. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 66-75, 2017. 5
[31] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. DROID: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024. 7
[32] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 4015-4026, 2023. 1, 3
[33] Vincent Leroy, Yohann Cabon, and Jérôme Revaud. Grounding image matching in 3D with MASt3R. arXiv preprint arXiv:2406.09756, 2024. 3
[34] Jiankun Li, Peisen Wang, Pengfei Xiong, Tao Cai, Ziwei Yan, Lei Yang, Jiangyu Liu, Haoqiang Fan, and Shuaicheng Liu. Practical stereo matching via cascaded recurrent network with adaptive correlation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16263-16272, 2022. 1, 2, 3, 6, 7
[35] Zhaoshuo Li, Xingtong Liu, Nathan Drenkow, Andy Ding, Francis X Creighton, Russell H Taylor, and Mathias Unberath. Revisiting stereo depth estimation from a sequence-to-sequence perspective with transformers. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 6197-6206, 2021. 1, 2, 5
[36] Lahav Lipson, Zachary Teed, and Jia Deng. RAFT-Stereo: Multilevel recurrent field transforms for stereo matching. In International Conference on 3D Vision (3DV), pages 218227, 2021. 1, 2, 3, 5, 6, 7
[37] Biyang Liu, Huimin Yu, and Guodong Qi. GraftNet: Towards domain generalized stereo matching with a broadspectrum and task-oriented feature. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13012-13021, 2022. 1, 2
[38] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection. In Proceedings of the European Conference on Computer Vision (ECCV), 2024. 3
[39] I Loshchilov. Decoupled weight decay regularization. ICLR, 2019. 6
[40] Muhammad Maaz, Abdelrahman Shaker, Hisham Cholakkal, Salman Khan, Syed Waqas Zamir, Rao Muhammad Anwer, and Fahad Shahbaz Khan. EdgeNeXt: Efficiently amalgamated cnn-transformer architecture for mobile vision applications. In Proceedings of the European Conference on Computer Vision (ECCV), pages 3-20, 2022. 3
[41] Yamin Mao, Zhihua Liu, Weiming Li, Yuchao Dai, Qiang Wang, Yun-Tae Kim, and Hong-Seok Lee. UASNet: Uncertainty adaptive sampling network for deep stereo matching. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 6311-6319, 2021. 1, 2
[42] D. Marr and T. Poggio. Cooperative computation of stereo disparity. Science, 194:283-287, 1976. 1
[43] Nikolaus Mayer, Eddy Ilg, Philip Hausser, Philipp Fischer, Daniel Cremers, Alexey Dosovitskiy, and Thomas Brox. A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4040-4048, 2016. 2, 3, 6
[44] Lukas Mehl, Jenny Schmalfuss, Azin Jahedi, Yaroslava Nalivayko, and Andrés Bruhn. Spring: A high-resolution highdetail dataset and benchmark for scene flow, optical flow and stereo. In Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 3
[45] Moritz Menze and Andreas Geiger. Object scene flow for autonomous vehicles. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3061-3070, 2015. 2, 6, 7
[46] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. DINOv2: Learning robust visual features without supervision. TMLR, 2024. 1, 3, 8
[47] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), pages 8748-8763, 2021. 3
[48] Pierluigi Zama Ramirez, Alex Costanzino, Fabio Tosi, Matteo Poggi, Samuele Salti, and Stefano Mattoccia. Booster: A benchmark for depth from images of specular and transparent surfaces. IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2023. 2
[49] Zhibo Rao, Bangshu Xiong, Mingyi He, Yuchao Dai, Renjie He, Zhelun Shen, and Xing Li. Masked representation learning for domain generalized stereo matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5435-5444, 2023. 1, 2, 6
[50] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. SAM 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 3
[51] Daniel Scharstein, Heiko Hirschmüller, York Kitajima, Greg Krathwohl, Nera Nešić, Xi Wang, and Porter Westling. High-resolution stereo datasets with subpixel-accurate ground truth. In Pattern Recognition: 36th German Conference, GCPR 2014, Münster, Germany, September 2-5, 2014, Proceedings 36, pages 31-42. Springer, 2014. 2, 6, 7
[52] Thomas Schops, Johannes L Schonberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. A multi-view stereo benchmark with highresolution images and multi-camera videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3260-3269, 2017. 2, 6, 7
[53] Zhelun Shen, Yuchao Dai, and Zhibo Rao. CFNet: Cascade and fused cost volume for robust stereo matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13906-13915, 2021. 1, 2
[54] Zhelun Shen, Yuchao Dai, Xibin Song, Zhibo Rao, Dingfu Zhou, and Liangjun Zhang. PCW-Net: Pyramid combination and warping cost volume for stereo matching. In Proceedings of the European Conference on Computer Vision (ECCV), pages 280-297, 2022. 1, 2
[55] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. 8
[56] Vladimir Tankovich, Christian Hane, Yinda Zhang, Adarsh Kowdle, Sean Fanello, and Sofien Bouaziz. HITNet: Hierarchical iterative tile refinement network for real-time stereo matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14362-14372, 2021. 7
[57] Zachary Teed and Jia Deng. RAFT: Recurrent all-pairs field transforms for optical flow. In Proceedings of the European Conference on Computer Vision (ECCV), pages 402-419, 2020. 2, 5
[58] Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel. Domain randomization for transferring deep neural networks from simulation to the real world. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 23-30, 2017. 5
[59] Fabio Tosi, Yiyi Liao, Carolin Schmitt, and Andreas Geiger. Smd-nets: Stereo mixture density networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8942-8952, 2021. 3
[60] Fabio Tosi, Filippo Aleotti, Pierluigi Zama Ramirez, Matteo Poggi, Samuele Salti, Stefano Mattoccia, and Luigi Di Stefano. Neural disparity refinement. IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2024. 1, 2
[61] Jonathan Tremblay, Thang To, and Stan Birchfield. Falling things: A synthetic dataset for 3d object detection and pose estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pages 2038-2041, 2018. 2, 3, 6
[62] Jonathan Tremblay, Thang To, Balakumar Sundaralingam, Yu Xiang, Dieter Fox, and Stan Birchfield. Deep object pose estimation for semantic robotic grasping of household objects. In Conference on Robot Learning (CoRL), pages 306316, 2018. 3
[63] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems (NeurIPS), 30, 2017. 5
[64] Qiang Wang, Shizhen Zheng, Qingsong Yan, Fei Deng, Kaiyong Zhao, and Xiaowen Chu. IRS: A large naturalistic indoor robotics stereo dataset to train deep models for disparity and surface normal estimation. In IEEE International Conference on Multimedia and Expo (ICME), 2021. 2, 3
[65] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. DUSi3R: Geometric 3D vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20697-20709, 2024. 3
[66] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. TartanAir: A dataset to push the limits of visual slam. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 4909-4916, 2020. 2, 3
[67] Xianqi Wang, Gangwei Xu, Hao Jia, and Xin Yang. Selective-Stereo: Adaptive frequency information selection for stereo matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19701-19710, 2024. 1, 2, 5, 6, 7
[68] Philippe Weinzaepfel, Thomas Lucas, Vincent Leroy, Yohann Cabon, Vaibhav Arora, Romain Brégier, Gabriela Csurka, Leonid Antsfeld, Boris Chidlovskii, and Jérôme Revaud. CroCo v2: Improved cross-view completion pretraining for stereo matching and optical flow. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 17969-17980, 2023. 1, 2, 5, 7
[69] Bowen Wen, Wei Yang, Jan Kautz, and Stan Birchfield. FoundationPose: Unified 6D pose estimation and tracking of novel objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 17868-17879, 2024. 3
[70] Gangwei Xu, Junda Cheng, Peng Guo, and Xin Yang. Attention concatenation volume for accurate and efficient stereomatching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12981-12990, 2022. 7
[71] Gangwei Xu, Xianqi Wang, Xiaohuan Ding, and Xin Yang. Iterative geometry encoding volume for stereo matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21919-21928, 2023. 2, 4, 5, 7
[72] Gangwei Xu, Xianqi Wang, Zhaoxing Zhang, Junda Cheng, Chunyuan Liao, and Xin Yang. IGEV++: Iterative multirange geometry encoding volumes for stereo matching. arXiv preprint arXiv:2409.00638, 2024. 2, 6
[73] Haofei Xu and Juyong Zhang. AANet: Adaptive aggregation network for efficient stereo matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1959-1968, 2020. 1, 2
[74] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, Fisher Yu, Dacheng Tao, and Andreas Geiger. Unifying flow, stereo and depth estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2023. 7
[75] Gengshan Yang, Joshua Manela, Michael Happold, and Deva Ramanan. Hierarchical deep stereo matching on highresolution images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5515-5524, 2019. 2
[76] Guorun Yang, Xiao Song, Chaoqin Huang, Zhidong Deng, Jianping Shi, and Bolei Zhou. DrivingStereo: A large-scale dataset for stereo matching in autonomous driving scenarios. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 899908, 2019. 2
[77] Jinyu Yang, Mingqi Gao, Zhe Li, Shang Gao, Fangjing Wang, and Feng Zheng. Track anything: Segment anything meets videos. arXiv preprint arXiv:2304.11968, 2023. 3
[78] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10371-10381, 2024. 1, 3
[79] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. In Proceedings of Neural Information Processing Systems (NeurIPS), 2024. 1, 2, 3, 4, 8
[80] Menglong Yang, Fangrui Wu, and Wei Li. WaveletStereo: Learning wavelet coefficients of disparity map in stereo matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12885-12894, 2020. 1, 2
[81] Feihu Zhang, Victor Prisacariu, Ruigang Yang, and Philip HS Torr. GA-Net: Guided aggregation net for end-to-end stereo matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 185-194, 2019. 7
[82] Feihu Zhang, Xiaojuan Qi, Ruigang Yang, Victor Prisacariu, Benjamin Wah, and Philip Torr. Domain-invariant stereo matching networks. In Proceedings of the European Conference on Computer Vision (ECCV), pages 420-439, 2020. 1, 2, 6
[83] Jeffrey O Zhang, Alexander Sax, Amir Zamir, Leonidas Guibas, and Jitendra Malik. Side-tuning: a baseline for network adaptation via additive side networks. In Proceedings of the European Conference on Computer Vision (ECCV), pages 698-714, 2020. 3
[84] Yongjian Zhang, Longguang Wang, Kunhong Li, Yun Wang, and Yulan Guo. Learning representations from foundation models for domain generalized stereo matching. In European Conference on Computer Vision, pages 146-162. Springer, 2024. 1, 2, 6
[85] Haoliang Zhao, Huizhou Zhou, Yongjun Zhang, Yong Zhao, Yitong Yang, and Ting Ouyang. EAI-Stereo: Error aware iterative network for stereo matching. In Proceedings of the Asian Conference on Computer Vision (ACCV), pages 315332, 2022. 7
[86] Haoliang Zhao, Huizhou Zhou, Yongjun Zhang, Jie Chen, Yitong Yang, and Yong Zhao. High-frequency stereo matching network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1327-1336, 2023. 1, 2
[87] Xu Zhao, Wenchao Ding, Yongqi An, Yinglong Du, Tao Yu, Min Li, Ming Tang, and Jinqiao Wang. Fast segment anything. arXiv preprint arXiv:2306.12156, 2023. 8