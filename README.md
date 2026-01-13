<h1 align="center">Vision-Spatiotemporal-Fusion-in-TF</h1>

This repository tracks the latest advances in Vision-Spatiotemporal Fusion for Traffic Forecasting (TF) and serves as the official repository for **Towards Vision-Spatiotemporal Fusion in Traffic Forecasting: A Survey on Cross-Modal Alignment**.

## Challenges and Contributions

### 🎯 Challenges
The integration of real-time visual information with spatiotemporal data has become a crucial direction for enhancing traffic forecasting capabilities. However, this cross-modal fusion still faces fundamental bottlenecks. Among them, the semantic gap makes it difficult to transform raw pixels into understandable traffic states and events, while geometric inconsistency hinders the accurate spatial registration of 2D visual observations with 3D traffic networks. Existing research largely focuses on designing fusion model architectures but often overlooks in-depth solutions to these underlying alignment issues, resulting in systems with limited interpretability and generalization in complex scenarios.

### 🌟 Contributions
To address these challenges, this survey systematically restructures and prospects the field with cross-modal alignment as its central theme. Its main contributions include: (1) proposing a three-tier classification framework based on alignment granularity (Feature-level, Semantic-level, Task-level), which clearly outlines the evolution of methods from shallow association to deep collaboration; specifically focusing on the critical challenge of geometric alignment, systematically analyzing core issues such as cross-view object association and spatial mapping; and (2) critically examining existing datasets and evaluation systems, highlighting their limitations in assessing alignment quality, thereby charting a path toward building spatially intelligent, interpretable, and robust traffic world models. This survey not only establishes a unified analytical framework for the field but also lays the theoretical foundation for realizing next-generation transportation systems that truly integrate perception and decision-making.

To clarify key concepts, this survey employs two schematic diagrams for explanation. **Figure 1** demonstrates the integration levels of visual and spatiotemporal information from the perspective of data alignment. **Figure 2** points out the two main semantic and geometric gaps that need to be addressed in building a traffic world model from the perspective of model construction. 

<div align="center">
  <img src="https://github.com/user-attachments/assets/cc36d6c3-35df-42e8-bda2-c05be5f1fb3f" width="65%">
  <br>
  <small><b>Figure 1:</b> Levels of alignment for vision and spatiotemporal data</small>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/980e7570-2a0b-4ad3-8e8e-1641fb4e341d" width="65%">
  <br>
  <small><b>Figure 2:</b> Dual pillars of a traffic world model: semantic and geometric gaps</small>
</div>
<br><br>

**Figure 1** illustrates the classification framework of cross-modal alignment granularity using a hierarchical structure, categorizing existing methods into three progressive stages: feature-level, semantic-level, and task-level. It visually presents the technological development path from low-level feature association to high-level task coordination. 

**Figure 2** depicts the two core challenges in traffic world models: semantic alignment and geometric alignment. The upper part demonstrates the semantic parsing pipeline from raw video to traffic concepts, while the lower part focuses on the geometric alignment process involving cross-view association and 3D spatial mapping, clearly revealing the key technical layers that must be addressed when constructing a world model.


## Representative Methods for Vision-Spatiotemporal Alignment in Traffic Forecasting

| Method | Alignment Granularity | Visual Input | S-T Input | Mechanism | Traffic Task | Key Insight |
|:------:|:--------------------:|:------------:|:---------:|:---------:|:------------:|:-----------:|
| VisionTS [[Chen et al., 2025](https://icml.cc/virtual/2025/poster/46441)] | • (F) | Time-series image (GAF) | Traffic time series | Visual MAE pre-training | Zero-shot forecasting | Transforms signals to images; bridges vision pre-training to traffic forecasting |
| Dual-Encoder [[Gong et al., 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Gong_Bi-Level_Alignment_for_Cross-Domain_Crowd_Counting_CVPR_2022_paper.pdf)] | • (F) | Traffic camera image | Graph traffic state | Bi-level adversarial alignment | Multi-modal forecasting | Task-aware data and refined feature alignment for domain adaptation |
| Semantic Extractor [[Nie et al., 2023](https://arxiv.org/abs/2211.14730)] | ∘ (S) | Traffic video | Graph node states | Pixel-to-concept parsing | Explainable event forecast | Guides reconstruction with traffic semantics (instances, events) |
| Scene Graph Anticipation [[Peddi et al., 2024](https://link.springer.com/chapter/10.1007/978-3-031-73223-2_10)] | ∘ (S) | Traffic scene images | Graph-structured events | Neural ODE/SDE modeling | Scene understanding | Models continuous dynamics of traffic entity interactions |
| Gated Fusion [[Yi et al., 2024](https://nips.cc/media/neurips-2024/Slides/97948_PJHhMuP.pdf)] | <span style="font-size:0.8em">▪</span> (T) | Visual feature sequence | Spatiotemporal graph features | Multi-dimensional spatiotemporal interaction | Multi-task forecasting | Captures cross-interactions for continuous multi-task learning |
| Unified ViT-GNN Transformer [[Lee et al., 2022](https://link.springer.com/chapter/10.1007/978-3-031-25072-9_41)] | <span style="font-size:0.8em">▪</span> (T) | Camera image patches | Traffic graph nodes | Cross-modality attention fusion | Pedestrian detection | Explores modality-specific features for traffic perception |

<sup>**Legend:** Alignment Granularity (•: Feature-level (F), ∘: Semantic-level (S), <span style="font-size:0.8em">▪</span>: Task-level (T)). S-T Input: Spatiotemporal Input</sup>

## Core Multimodal Datasets for Vision-Spatiotemporal Traffic Forecasting Research

| Dataset | Visual Modality | Spatiotemporal Modality | Temporal Sync. | Primary Alignment Challenge |
|:---------:|:----------------:|:-------------------------:|:-------------:|:----------------------------:|
| nuScenes [[Caesar et al., 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.html)] | Multi-camera, 360° views | 3D object tracks, HD map elements | Yes | Large-scale geometric registration in complex intersections |
| Waymo Open [[Sun et al., 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Scalability_in_Perception_for_Autonomous_Driving_Waymo_Open_Dataset_CVPR_2020_paper.pdf)] | Multi-camera, LiDAR streams | Detailed vehicle trajectories, map data | Yes | Fine-grained spatial grounding for traffic participants |
| CityFlow [[Tang et al., 2019](https://ieeexplore.ieee.org/document/8954067)] | City-scale traffic camera network | Vehicle trajectories, ReID labels | Approximate | Cross-view association for network-wide traffic flow analysis |
| CARLA (Synthetic) [[Dosovitskiy et al., 2017](https://proceedings.mlr.press/v78/dosovitskiy17a.html)] | Rendered traffic camera views | Perfect vehicle state/logs | Perfect | Sim2Real domain gap for traffic scenarios |
| Time-Series Benchmarks | Generated images from traffic signals | Traffic flow/speed series | N/A | Representation conversion fidelity for traffic forecasting |

## List of References
### 📅 2025
- [2025] [TKDE] Adaptive traffic forecasting on daily basis: A spatio-temporal context learning approach [[paper](https://ieeexplore.ieee.org/document/11012680)]
- [2025] [ICCV] Self-supervised sparse sensor fusion for long range perception [[paper](https://iccv.thecvf.com/virtual/2025/poster/744)]
- [2025] [ACM Comput. Surv.] Understanding world or predicting future? A comprehensive survey of world models [[paper](https://dl.acm.org/doi/full/10.1145/3746449)]
- [2025] [IJCAI] Words over pixels? Rethinking vision in multimodal large language models [[paper](https://www.ijcai.org/proceedings/2025/1164)]
- [2025] [CVPR] How to Merge your multimodal models over time? [[paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Dziadzio_How_to_Merge_Your_Multimodal_Models_Over_Time_CVPR_2025_paper.pdf)]
- [2025] [Pattern Recogn.] Multimodal urban traffic flow prediction based on multi-scale time series imaging [[paper](https://www.sciencedirect.com/science/article/pii/S0031320325001591)]
- [2025] [IJCAI] Harnessing vision models for time series analysis: A survey [[paper](https://arxiv.org/abs/2502.08869)]
- [2025] [IJCAI] Towards cross-modality modeling for time series analytics: A survey in the LLM era [[paper](https://arxiv.org/abs/2505.02583)]
- [2025] [ICCV] SafeRoute: Enhancing traffic scene understanding via a unified deep learning and multimodal LLM [[paper](https://openaccess.thecvf.com/content/ICCV2025W/WDFM-AD/papers/Shaw_SafeRoute_Enhancing_Traffic_Scene_Understanding_via_a_Unified_Deep_Learning_ICCVW_2025_paper.pdf)]
- [2025] [SIGKDD] Are vision llms road-ready? A comprehensive benchmark for safety-critical driving video understanding [[paper](https://dl.acm.org/doi/abs/10.1145/3711896.3737396)]
- [2025] [ICCV] TinyBEV: Cross-modal knowledge distillation for efficient multi-task bird's-eye-view perception and planning [[paper](https://arxiv.org/abs/2509.18372)]
- [2025] [TIP] MAFS: Masked autoencoder for infrared-visible image fusion and semantic segmentation [[paper](https://ieeexplore.ieee.org/document/11178208)]
- [2025] [SIGKDD] Improving open-world continual learning under the constraints of scarce labeled data [[paper](https://dl.acm.org/doi/abs/10.1145/3711896.3737004)]
- [2025] [ICML] VisionTS: Visual masked autoencoders are free-lunch zero-shot time series forecasters [[paper](https://icml.cc/virtual/2025/poster/46441)]
- [2025] [TNNLS] Structure-preserved self-attention for fusion image information in multiple color spaces [[paper](https://ieeexplore.ieee.org/abstract/document/10750905)]
- [2025] [SIGMM] SeqVLM: Proposal-guided multi-view sequences reasoning via VLM for zero-shot 3D visual grounding [[paper](https://arxiv.org/abs/2508.20758)]
- [2025] [ICCV] Unbiased missing-modality multimodal learning [[paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Dai_Unbiased_Missing-modality_Multimodal_Learning_ICCV_2025_paper.pdf)]
- [2025] [ICRA] Zeroscd: Zero-shot street scene change detection [[paper](https://arxiv.org/abs/2409.15255)]
- [2025] [ICCV] Interpretable decision-making for end-to-end autonomous driving [[paper](https://arxiv.org/abs/2508.18898)]
- [2025] [CVPR] Patchcontrast: Self-supervised pre-training for 3d object detection [[paper](https://openaccess.thecvf.com/content/CVPR2025W/WAD/html/Shrout_PatchContrast_Self-Supervised_Pre-Training_for_3D_Object_Detection_CVPRW_2025_paper.html)]
- [2025] [ICCV] TrafficInternVL: Understanding traffic scenarios with vision-language models [[paper](https://openaccess.thecvf.com/content/ICCV2025W/AICity/papers/Wu_TrafficInternVL_Understanding_Traffic_Scenarios_with_Vision-Language_Models_ICCVW_2025_paper.pdf)]
- [2025] [SIGKDD] Tsfm-bench: A comprehensive and unified benchmark of foundation models for time series forecasting [[paper](https://arxiv.org/abs/2410.11802)]
- [2025] [TPAMI] Out-of-distribution generalization on graphs: A survey [[paper](https://ieeexplore.ieee.org/document/11106188)]
- [2025] [TITS] A 3d convolution-incorporated dimension preserved decomposition model for traffic data prediction [[paper](https://ieeexplore.ieee.org/abstract/document/10745874)]
- [2025] [TKDE] Adaptive Hyper-Box Granulation With Justifiable Granularity for Feature Selection [[paper](https://ieeexplore.ieee.org/abstract/document/11192594)]

### 📅 2024
- [2024] [ECCV] Towards multimodal in-context learning for vision and language models [[paper](https://link.springer.com/chapter/10.1007/978-3-031-93806-1_19)]
- [2024] [ECCV] Towards scene graph anticipation [[paper](https://link.springer.com/chapter/10.1007/978-3-031-73223-2_10)]
- [2024] [NeurIPS] Multimodal task vectors enable many-shot multimodal in-context learning [[paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/27571b74d6cd650b8eb6cf1837953ae8-Paper-Conference.pdf)]
- [2024] [TITS] Unraveling urban mobility: A domain knowledge-free trajectory classification using gramian angular fields [[paper](https://ieeexplore.ieee.org/abstract/document/11258590)]
- [2024] [NeurIPS] SeeClear: Semantic distillation enhances pixel condensation for video super-resolution [[paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f358b2a880adf34939d2d6f926e54d2a-Abstract-Conference.html)]
- [2024] [NeurIPS] Get rid of isolation: A continuous multi-task spatio-temporal learning framework [[paper](https://nips.cc/media/neurips-2024/Slides/97948_PJHhMuP.pdf)]
- [2024] [Inform. Fusion] A multi-modal spatial--temporal model for accurate motion forecasting with visual fusion [[paper](https://www.sciencedirect.com/science/article/pii/S1566253523003627)]
- [2024] [AAAI] Far3d: Expanding the horizon for surround-view 3d object detection [[paper](https://arxiv.org/abs/2308.09616)]
- [2024] [TITS] Highway visibility level prediction using geometric and visual features driven dual-branch fusion network [[paper](https://ieeexplore.ieee.org/document/10440153)]
- [2024] [ECCV] Wts: A pedestrian-centric traffic video dataset for fine-grained spatial-temporal understanding [[paper](https://link.springer.com/chapter/10.1007/978-3-031-73116-7_1)]
- [2024] [CVPR] Benchmarking implicit neural representation and geometric rendering in real-time rgb-d slam [[paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Hua_Benchmarking_Implicit_Neural_Representation_and_Geometric_Rendering_in_Real-Time_RGB-D_CVPR_2024_paper.pdf)]
- [2024] [SIGGRAPH] Path-space differentiable rendering of implicit surfaces [[paper](https://dl.acm.org/doi/10.1145/3641519.3657473)]
- [2024] [NeurIPS] Evaluating the world model implicit in a generative model [[paper](https://arxiv.org/abs/2406.03689)]

### 📅 2023
- [2023] [Artif. Intell.] AutoSTG+: An automatic framework to discover the optimal network for spatio-temporal graph prediction [[paper](https://www.sciencedirect.com/science/article/pii/S0004370223000450)]
- [2023] [ICLR] Timesnet: Temporal 2d-variation modeling for general time series analysis [[paper](https://arxiv.org/abs/2210.02186)]
- [2023] [SIGMM] Improving anomaly segmentation with multi-granularity cross-domain alignment [[paper](https://dl.acm.org/doi/10.1145/3581783.3611849)]
- [2023] [SIGKDD] A study of situational reasoning for traffic understanding [[paper](https://dl.acm.org/doi/10.1145/3580305.3599246)]
- [2023] [ICLR] A Time Series is Worth 64Words: Long-term Forecasting with Transformers [[paper](https://arxiv.org/abs/2211.14730)]

### 📅 2022
- [2022] [ICCV] Bi-level alignment for cross-domain crowd counting [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Gong_Bi-Level_Alignment_for_Cross-Domain_Crowd_Counting_CVPR_2022_paper.pdf)]
- [2022] [ECCV] Cross-modality attention and multimodal fusion transformer for pedestrian detection [[paper](https://link.springer.com/chapter/10.1007/978-3-031-25072-9_41)]

### 📅 2021
- [2021] [ICCV] Projecting your view attentively: Monocular road scene layout estimation via cross-view transformation [[paper](https://ieeexplore.ieee.org/document/9578824)]

### 📅 2020
- [2020] [CVPR] Nuscenes: A multimodal dataset for autonomous driving [[paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.html)]
- [2020] [CVPR] Scalability in perception for autonomous driving: Waymo open dataset [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Scalability_in_Perception_for_Autonomous_Driving_Waymo_Open_Dataset_CVPR_2020_paper.pdf)]

### 📅 Before 2020
- [2019] [CVPR] Cityflow: A city-scale benchmark for multi-target multi-camera vehicle tracking and re-identification [[paper](https://ieeexplore.ieee.org/document/8954067)]
- [2017] [PMLR] CARLA: An open urban driving simulator [[paper](https://proceedings.mlr.press/v78/dosovitskiy17a.html)]
