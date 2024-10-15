# OOD Machine Learning: Detection, Robustness, and Generalization

<img width="1665" alt="image" src="https://github.com/user-attachments/assets/ebb0d89c-e929-4c7f-af5e-d7e39547c487">

<p align="center">
    <a href="https://discord.gg/gdap3n9jZ2">
        <img src="https://img.shields.io/badge/Discord-5865F2.svg?style=for-the-badge&logo=Discord&logoColor=white" alt="Discord">
    </a>
    <a href="https://github.com/continuousml/Awesome-Out-Of-Distribution-Detection">
        <img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=for-the-badge" alt="Maintenance">
    </a>
</p>

This repo aims to provide the most comprehensive, up-to-date, high-quality resource for **OOD detection, robustness, and generalization** in Machine Learning/Deep Learning. Your one-stop shop for everything OOD is here.


---

## Table of Contents
- [Researchers](#researchers)
- [Articles](#articles)
- [Talks](#talks)
- [Benchmarks, libraries, datasets, etc](#benchmarks-libraries-datasets-etc)
- [Surveys](#surveys)
- [Theses](#theses)
- [Papers](#papers)
  - [OOD Detection](#ood-detection)
  - [OOD Robustness](#ood-robustness)
  - [OOD Generalization](#ood-generalization)
  - [OOD Everything else](#ood-everything-else)

---

# Researchers

TODO: Update list of researcher

# Articles

[(2022) Data Distribution Shifts and Monitoring](https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html) by Chip Huyen

[(2020) Out-of-Distribution Detection in Deep Neural Networks](https://medium.com/analytics-vidhya/out-of-distribution-detection-in-deep-neural-networks-450da9ed7044) by Neeraj Varshney


# Talks

[(2023) How to detect Out-of-Distribution data in the wild?](https://www.youtube.com/watch?v=Ga09-9JItxs) by Sharon Yixuan Li

[(2022) Anomaly detection for OOD and novel category detection](https://www.youtube.com/watch?v=jFQUW2n8gpA&t=888s) by Thomas G. Dietterich

[(2022) Reliable Open-World Learning Against Out-of-distribution Data](https://www.youtube.com/watch?v=zaXiHljOl9Y&t=22s) by Sharon Yixuan Li

[(2022) Challenges and Opportunities in Out-of-distribution Detection](https://www.youtube.com/watch?v=X8XTOiNin0I&t=523s) by Sharon Yixuan Li

[(2022) Exploring the limits of out-of-distribution detection in vision and biomedical applications](https://www.youtube.com/watch?v=cSkBcqBhKVY) by Jie Ren

[(2021) Understanding the Failure Modes of Out-of-distribution Generalization](https://www.youtube.com/watch?v=DhPMq_550OE) by Vaishnavh Nagarajan

[(2020) Uncertainty and Out-of-Distribution Robustness in Deep Learning](https://www.youtube.com/watch?v=ssD7jNDIL2c&t=2293s) by Balaji Lakshminarayanan, Dustin Tran, and Jasper Snoek

# Benchmarks, libraries, datasets, etc

## Benchmarks

[OpenOOD v1.5: Benchmarking Generalized OOD Detection](https://github.com/Jingkang50/OpenOOD)

[RoboDepth: Robust Out-of-distribution Depth Estimation Under Corruptions](https://github.com/ldkong1205/RoboDepth)

[OOD NLP: Revisiting Out-of-distribution Robustness in NLP: Benchmark, Analysis, and LLMs Evaluations](https://github.com/lifan-yuan/OOD_NLP) 

[OOD-CV : A Benchmark for Robustness to Out-of-Distribution Shifts of Individual Nuisances in Natural Images](https://www.ood-cv.org/)

[Photorealistic Unreal Graphics (PUG)](https://pug.metademolab.com/) by Meta AI

> "Abstract: Synthetic image datasets offer unmatched advantages for designing and evaluating deep neural networks: they make it possible to (i) render as many data samples as needed, (ii) precisely control each scene and yield granular ground truth labels (and captions), (iii) precisely control distribution shifts between training and testing to isolate variables of interest for sound experimentation. Despite such promise, the use of synthetic image data is still limited -- and often played down -- mainly due to their lack of realism. Most works therefore rely on datasets of real images, which have often been scraped from public images on the internet, and may have issues with regards to privacy, bias, and copyright, while offering little control over how objects precisely appear. In this work, we present a path to democratize the use of photorealistic synthetic data: we develop a new generation of interactive environments for representation learning research, that offer both controllability and realism. We use the Unreal Engine, a powerful game engine well known in the entertainment industry, to produce PUG (Photorealistic Unreal Graphics) environments and datasets for representation learning. In this paper, we demonstrate the potential of PUG to enable more rigorous evaluations of vision models."

[OODRobustBench: a Benchmark and Large-Scale Analysis of Adversarial Robustness under Distribution Shift](https://github.com/OODRobustBench/OODRobustBench)

[A Noisy Elephant in the Room: Is Your Out-of-Distribution Detector Robust to Label Noise?](https://github.com/glhr/ood-labelnoise)

[Benchmarking Out-of-Distribution Generalization and Adaptation for Electromyography]()

[Benchmarking Out-of-Distribution Generalization Capabilities of DNN-based Encoding Models for the Ventral Visual Cortex]()

[Vision Transformer Neural Architecture Search for Out-of-Distribution Generalization: Benchmark and Insights]()

## Libraries

[PyTorch Out-of-Distribution Detection](https://github.com/kkirchheim/pytorch-ood)

[FrOoDo: Framework for Out-of-Distribution Detection](https://github.com/MECLabTUDA/FrOoDo?tab=readme-ov-file)


# Surveys

[Generalized Out-of-Distribution Detection and Beyond in Vision Language Model Era: A Survey](https://arxiv.org/pdf/2407.21794) by Miyai et al.

[Generalized Out-of-Distribution Detection: A Survey](https://arxiv.org/pdf/2110.11334.pdf) by Yang et al

[A Unified Survey on Anomaly, Novelty, Open-Set, and Out of-Distribution Detection: Solutions and Future Challenges](https://arxiv.org/pdf/2110.14051.pdf) by Salehi et al.

(TMLR 2023) [A Survey on Out-of-distribution Detection in NLP](https://arxiv.org/pdf/2305.03236v2.pdf) by Lang et al.

# Theses

[Robust Out-of-Distribution Detection in Deep Classifiers](https://ub01.uni-tuebingen.de/xmlui/bitstream/handle/10900/141438/Dissertation.pdf?sequence=2&isAllowed=y) by Alexander Meinke

[Out of Distribution Generalization in Machine Learning](https://arxiv.org/pdf/2103.02667.pdf) by Martin Arjovsky

# Papers

> "Know thy literature"

## OOD Detection

(NeurIPS 2024) [Revisiting Score Propagation in Graph Out-of-Distribution Detection]() by Ma et al.

(NeurIPS 2024) [Human-Assisted Out-of-Distribution Generalization and Detection]() by Bai et al.

(NeurIPS 2024) [Long-Tailed Out-of-Distribution Detection via Normalized Outlier Distribution Adaptation]() by Miao et al.

(NeurIPS 2024) [Trajectory Volatility for Out-of-Distribution Detection in Mathematical Reasoning]() by Wang et al.

(NeurIPS 2024) [Rethinking Out-of-Distribution Detection on Imbalanced Data Distribution]() by Liu et al.

(NeurIPS 2024) [Learning to Shape In-distribution Feature Space for Out-of-distribution Detection]() by Zhang et al.

(NeurIPS 2024) [Self-Calibrated Tuning of Vision-Language Models for Out-of-Distribution Detection]() by Yu et al.

(NeurIPS 2024) [FOOGD: Federated Collaboration for Both Out-of-distribution Generalization and Detection]() by Liao et al.

(NeurIPS 2024) [Diffusion-based Layer-wise Semantic Reconstruction for Unsupervised Out-of-Distribution Detection]() by Yang et al.

(NeurIPS 2024) [Out-of-Distribution Detection with a Single Unconditional Diffusion Model]() by Heng et al.

(NeurIPS 2024) [Expecting The Unexpected: Towards Broad Out-Of-Distribution Detection]() by Guille-Escuret et al.

(NeurIPS 2024) [Kernel PCA for Out-of-Distribution Detection]() by Fang et al.

(NeurIPS 2024) [SeTAR: Out-of-Distribution Detection with Selective Low-Rank Approximation]() by Li et al.

(NeurIPS 2024) [Rethinking the Evaluation of Out-of-Distribution Detection: A Sorites Paradox]() by Long et al.

(NeurIPS 2024) [Energy-based Hopfield Boosting for Out-of-Distribution Detection]() by Hofmann et al.

(NeurIPS 2024) [The Best of Both Worlds: On the Dilemma of Out-of-distribution Detection]() by Zhang et al.

(NeurIPS 2024) [Out-Of-Distribution Detection with Diversification (Provably)]() by Yao et al.

(NeurIPS 2024) [Hyper-opinion Evidential Deep Learning for Out-of-Distribution Detection]() by Qu et al.

(NeurIPS 2024) [MultiOOD: Scaling Out-of-Distribution Detection for Multiple Modalities](https://arxiv.org/pdf/2405.17419) [Code](https://github.com/donghao51/MultiOOD) by Dong et al.

(ECCV 2024) [Gradient-Regularized Out-of-Distribution Detection](https://arxiv.org/pdf/2404.12368) by Sharifi, Entesari, and Safaei et al.

(AISTATS 2024) [Taming False Positives in Out-of-Distribution Detection with Human Feedback](https://arxiv.org/pdf/2404.16954) [[Code]](https://github.com/2454511550Lin/TameFalsePositives-OOD) by Vishwakarma et al.

(ICML 2024) [OODRobustBench: a Benchmark and Large-Scale Analysis of Adversarial Robustness under Distribution Shift](https://openreview.net/pdf?id=kAFevjEYsz) [[Code]](https://github.com/OODRobustBench/OODRobustBench) by Li et al.

(ICML 2024) [A Geometric Explanation of the Likelihood OOD Detection Paradox](https://openreview.net/pdf?id=EVMzCKLpdD) [[Code]](https://github.com/layer6ai-labs/dgm_ood_detection) by Kamkari et al.

(ICML 2024) [ODIM: Outlier Detection via Likelihood of Under-Fitted Generative Models](https://openreview.net/pdf?id=R8nbccD7kv) [[Code]](https://github.com/jshwang0311/ODIM) by Kim and Hwang et al.

(ICML 2024) [A Provable Decision Rule for Out-of-Distribution Detection](https://openreview.net/pdf?id=SPygKwms0X) by Ma et al.

(ICML 2024) [When and How Does In-Distribution Label Help Out-of-Distribution Detection?](https://openreview.net/pdf?id=knhbhDLdry) [[Code]](https://github.com/deeplearning-wisc/id_label) by Du et al.

(ICML 2024) [Graph Out-of-Distribution Detection Goes Neighborhood Shaping](https://openreview.net/pdf?id=pmcusTywXO) by Bao et al.

(ICML 2024) [Out-of-Distribution Detection via Deep Multi-Comprehension Ensemble](https://openreview.net/pdf?id=HusShERjlc) by Xu et al.

(ICML 2024) [Bounded and Uniform Energy-based Out-of-distribution Detection for Graphs](https://openreview.net/pdf?id=mjh7AOWozN) [[Code]](https://github.com/ShenzhiYang2000/NODESAFE-Bounded-and-Uniform-Energy-based-Out-of-distribution-Detection-for-Graphs) by Yang et al.

(ICML 2024) [Fast Decision Boundary based Out-of-Distribution Detector]() [[Code]](https://github.com/litianliu/fDBD-OOD) by Liu et al.

(ICML 2024) [DeCoOp: Robust Prompt Tuning with Out-of-Distribution Detection](https://openreview.net/pdf?id=MoTUdh9ZCc) [[Code]](https://github.com/WNJXYK/DeCoOp) by Zhou et al.

(ICML 2024) [Envisioning Outlier Exposure by Large Language Models for Out-of-Distribution Detection](https://openreview.net/pdf?id=xZO7SmM12y) [[Code]](https://github.com/tmlr-group/EOE) by Cao et al.

(ICML 2024) [Prometheus: Out-of-distribution Fluid Dynamics Modeling with Disentangled Graph ODE](https://openreview.net/pdf?id=JsPvL6ExK8) by Wu et al.

(ICML 2024) [Optimal Ridge Regularization for Out-of-Distribution Prediction](https://openreview.net/pdf?id=bvPYroQgc3) [[Code]](https://github.com/jaydu1/ood-ridge) by Patil et al.

(CVPR 2024) [Long-Tailed Anomaly Detection with Learnable Class Names](https://openaccess.thecvf.com/content/CVPR2024/papers/Ho_Long-Tailed_Anomaly_Detection_with_Learnable_Class_Names_CVPR_2024_paper.pdf) by Ho et al.

(CVPR 2024) [Enhancing the Power of OOD Detection via Sample-Aware Model Selection](https://openaccess.thecvf.com/content/CVPR2024/papers/Xue_Enhancing_the_Power_of_OOD_Detection_via_Sample-Aware_Model_Selection_CVPR_2024_paper.pdf) by Xue et al.

(CVPR 2024) [Test-Time Linear Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2024/papers/Fan_Test-Time_Linear_Out-of-Distribution_Detection_CVPR_2024_paper.pdf) [[Code]](https://github.com/kfan21/RTL) by Fan et al.

(CVPR 2024) [ID-like Prompt Learning for Few-Shot Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2024/papers/Bai_ID-like_Prompt_Learning_for_Few-Shot_Out-of-Distribution_Detection_CVPR_2024_paper.pdf) [[Code]](https://github.com/ycfate/ID-like) by Bai et al.

(CVPR 2024) [YolOOD: Utilizing Object Detection Concepts for Multi-Label Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2024/papers/Zolfi_YolOOD_Utilizing_Object_Detection_Concepts_for_Multi-Label_Out-of-Distribution_Detection_CVPR_2024_paper.pdf) [[Code]](https://github.com/AlonZolfi/YolOOD) by Zolfi et al.

(CVPR 2024) [Learning Transferable Negative Prompts for Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Learning_Transferable_Negative_Prompts_for_Out-of-Distribution_Detection_CVPR_2024_paper.pdf) [[Code]](https://github.com/mala-lab/negprompt) by Li et al.

(CVPR 2024) [A Noisy Elephant in the Room: Is Your Out-of-Distribution Detector Robust to Label Noise?](https://openaccess.thecvf.com/content/CVPR2024/papers/Humblot-Renaux_A_Noisy_Elephant_in_the_Room_Is_Your_Out-of-Distribution_Detector_CVPR_2024_paper.pdf) [[Code]](https://github.com/glhr/ood-labelnoise) by Humblot-Renaux et al.

(CVPR 2024) [Discriminability-Driven Channel Selection for Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2024/papers/Yuan_Discriminability-Driven_Channel_Selection_for_Out-of-Distribution_Detection_CVPR_2024_paper.pdf) by Yuan et al.

(CVPR 2024) [CORES: Convolutional Response-based Score for Out-of-distribution Detection](https://openaccess.thecvf.com/content/CVPR2024/papers/Tang_CORES_Convolutional_Response-based_Score_for_Out-of-distribution_Detection_CVPR_2024_paper.pdf) by Tang and Hou et al.

(ICLR 2024 Reject) [Detecting Out-Of-Distribution Samples Via Conditional Distribution Entropy With Optimal Transport](https://openreview.net/pdf?id=YnaGcMJQ0M) by Feng et al.

(ICLR 2024) [CONJNORM: Tractable Density Estimation for Out-of-distribution Detection](https://arxiv.org/pdf/2402.17888.pdf) by Peng and Luo et al.

(ICLR 2024) [Learning With Mixture Of Prototypes For Out-Of-Distribution Detection](https://openreview.net/pdf?id=uNkKaD3MCs) by Lu et al.

(ICLR 2024) [How Does Unlabled Data Provably Help Out-Of-Distribution Detection?](https://openreview.net/pdf?id=jlEjB8MVGa) [[Code]](https://github.com/deeplearning-wisc/sal) by Du and Fang et al.

(ICLR 2024) [HYPO: Hyperspherical Out-Of-Distribution Generalization](https://openreview.net/pdf?id=VXak3CZZGC) [[Code]](https://github.com/deeplearning-wisc/hypo) by Bai and Ming et al.

(ICLR 2024) [ImageNet-OOD: Deciphering Modern Out-Of-Distribution Detection Algorithms](https://openreview.net/pdf?id=VTYg5ykEGS) by Yang and Zhang et al.

(ICLR 2024) [Towards Optimal Feature-Shaping Methods For Out-Of-Distribution Detection](https://openreview.net/pdf?id=dm8e7gsH0d) by Zhao et al.

(ICLR 2024) [Out-Of-Distribution Detection With Negative Prompts](https://openreview.net/pdf?id=nanyAujl6e) by Nie et al.

(ICLR 2024) [DOS: Diverse Outlier Sampling For Out-Of-Distribution Detection](https://openreview.net/pdf?id=iriEqxFB4y) by Jiang et al.

(ICLR 2024) [NECO: Neural Collapse Based Out-Of-Distribution Detection](https://openreview.net/pdf?id=9ROuKblmi7) [[Code]](https://gitlab.com/drti/neco) by Ammar et al.

(ICLR 2024) [Plugin Estimators For Selective Classification With Out-Of-Distribution Detection](https://openreview.net/pdf?id=DASh78rJ7g) by Narasimhan et al.

(ICLR 2024) [Image Background Servers As Good Proxy For Out-Of-Distribution Data](https://openreview.net/pdf?id=ym0ubZrsmm) by Sen Pei

(ICLR 2024) [Out-Of-Distribution Detection By Leveraging Between-Layer Transformation Smoothness](https://openreview.net/pdf?id=AcRfzLS6se) by Jelenić et al.

(ICLR 2024) [Scaling For Training-Time and Posthoc Out-Of-Distribution Detection Enhancement](https://openreview.net/pdf?id=RDSTjtnqCg) by Xu et al.

(ICLR 2024) [Neuron Activation Coverage: Rethinking Out-Of-Distribution Detection and Generalization](https://openreview.net/pdf?id=SNGXbZtK6Q) by Liu et al.

(AAAI 2024) [How To Overcome Curse-of-Dimensionality for Out-of-distribution Detection?](https://arxiv.org/pdf/2312.14452v1.pdf) by Ghosal and Sun et al.

(NeurIPS 2023) [GradOrth: A Simple Yet Efficient Out-of-distribution Detection with Orthogonal Projection of Gradients](https://proceedings.neurips.cc/paper_files/paper/2023/file/77cf940349218069bbc230fc2c9c8a21-Paper-Conference.pdf) by Behypour et al.

(NeurIPS 2023) [Characterizing Out-of-distribution Error via Optimal Transport](https://proceedings.neurips.cc/paper_files/paper/2023/file/38fd51cf36f28566230a93a5fbeaabbf-Paper-Conference.pdf) [[Code]](https://github.com/luyuzhe111/COT) by Lu and Qin et al.

(NeurIPS 2023) [On the Importancce of Feature Separability in Predicting Out-of-distribution Error](https://proceedings.neurips.cc/paper_files/paper/2023/file/585e9cf25585612ac27b535457116513-Paper-Conference.pdf) by Xie et al.

(NeurIPS 2023) [ATTA: Anomaly-aware Test-Time Adaptation for Out-of-distribution Detection in Segmentation](https://proceedings.neurips.cc/paper_files/paper/2023/file/8dcc306a2522c60a78f047ab8739e631-Paper-Conference.pdf) [[Code]](https://github.com/gaozhitong/ATTA) by Gao et al.

(NeurIPS 2023) [Diversified Outlier Exposure for Out-of-distribution Detection via Informative Extrapolation](https://proceedings.neurips.cc/paper_files/paper/2023/file/46d943bc6a15a57c923829efc0db7c7a-Paper-Conference.pdf) [[Code]](https://github.com/tmlr-group/DivOE) by Zhu et al.

(NeurIPS 2023) [Optimal Parameter and Neuron Pruning for Out-of-distribution Detection](https://proceedings.neurips.cc/paper_files/paper/2023/file/a4316bb210a59fb7aafeca5dd21c2703-Paper-Conference.pdf) by Chen et al.

(NeurIPS 2023) [VRA: Variational Rectified Activation for Out-of-distribution Detection](https://proceedings.neurips.cc/paper_files/paper/2023/file/5c20c00504e0c049ec2370d0cceaf3c4-Paper-Conference.pdf) by Chen and Li et al.

(NeurIPS 2023) [GAIA: Delving into Gradient-based Attribution Abnormality for Out-of-distribution Detection](https://proceedings.neurips.cc/paper_files/paper/2023/file/fcdccd419c4dc471fa3b73ec97b53789-Paper-Conference.pdf) [[Code]](https://github.com/zeroQiaoba/VRA) by Xu et al.

(NeurIPS 2023) [CADet: Fully Self-supervised Anomaly Detection With Contrastive Learning](https://proceedings.neurips.cc/paper_files/paper/2023/file/1700ad4e6252e8f2955909f96367b34d-Paper-Conference.pdf) [[Code]](https://github.com/charlesGE/OpenOOD-CADet) by Guille-Escuret et al.

(NeurIPS 2023) [RoboDepth: Robust Out-of-distribution Depth Estimation Under Corruptions](https://proceedings.neurips.cc/paper_files/paper/2023/file/43119db5d59f07cc08fca7ba6820179a-Paper-Datasets_and_Benchmarks.pdf) by Kong et al.

(NeurIPS 2023) [Diversify & Conquer: Outcome-directed Curriculum RL via Out-of-distribution Disagreement](https://proceedings.neurips.cc/paper_files/paper/2023/file/a815fe7cad6af20a6c118f2072a881d2-Paper-Conference.pdf) by Cho et al.

(NeurIPS 2023) [LoCoOp: Few-Shot Out-of-distribution Detection via Prompt Learning](https://proceedings.neurips.cc/paper_files/paper/2023/file/f0606b882692637835e8ac981089eccd-Paper-Conference.pdf) [[Code]](https://github.com/AtsuMiyai/LoCoOp) by Miyai et al.

(NeurIPS 2023) [Category-Extensible Out-of-distribution Detection via Hierarchical Context Descriptions](https://proceedings.neurips.cc/paper_files/paper/2023/file/695b6f9490d27d852e439e35c56e73e3-Paper-Conference.pdf) by Liu et al.

(NeurIPS 2023) [Out-of-distribution Detection Learning With Unreliable Out-of-distribution Sources](https://proceedings.neurips.cc/paper_files/paper/2023/file/e43f900f571de6c96a70d5724a0fb565-Paper-Conference.pdf) [[Code]](https://github.com/tmlr-group/ATOL) by Zheng and Wang et al.

(NeurIPS 2023) [Dream the Impossible: Outlier Imagination with Diffusion Models](https://arxiv.org/pdf/2309.13415.pdf) by Du et al.

(NeurIPS 2023) [Learning To Augment Distributions For Out-of-distribution Detection](https://proceedings.neurips.cc/paper_files/paper/2023/file/e812af67a942c21dd0104bd929f99da1-Paper-Conference.pdf) [[Code]](https://github.com/tmlr-group/DAL) by Wang et al.

(UAI 2023)  [In- or Out-of-Distribution Detection via Dual Divergence Estimation](https://proceedings.mlr.press/v216/garg23b/garg23b.pdf) by Garg et al.

(ICCV 2023) [Nearest Neighbor Guidance for Out-of-Distribution Detection](https://openaccess.thecvf.com/content/ICCV2023/papers/Park_Nearest_Neighbor_Guidance_for_Out-of-Distribution_Detection_ICCV_2023_paper.pdf) [[Code]](https://github.com/roomo7time/nnguide) by Park et al.

(ICCV 2023) [DIFFGUARD: Semantic Mismatch-Guided Out-of-Distribution Detection using Pre-trained Diffusion Models](https://openaccess.thecvf.com/content/ICCV2023/papers/Gao_DIFFGUARD_Semantic_Mismatch-Guided_Out-of-Distribution_Detection_Using_Pre-Trained_Diffusion_Models_ICCV_2023_paper.pdf) [[Code]](https://github.com/cure-lab/DiffGuard) by Gao et al.

(ICCV 2023) [Understanding the Feature Norm for Out-of-Distribution Detection](https://openaccess.thecvf.com/content/ICCV2023/papers/Park_Understanding_the_Feature_Norm_for_Out-of-Distribution_Detection_ICCV_2023_paper.pdf) by Park et al.

(ICCV 2023) [SAFE: Sensitivity-Aware Features for Out-of-Distribution Object Detection](https://openaccess.thecvf.com/content/ICCV2023/papers/Wilson_SAFE_Sensitivity-Aware_Features_for_Out-of-Distribution_Object_Detection_ICCV_2023_paper.pdf) [[Code]](https://github.com/SamWilso/SAFE_Official) by Wilson et al.

(ICCV 2023) [Residual Pattern Learning for Pixel-wise Out-of-Distribution Detection in Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Residual_Pattern_Learning_for_Pixel-Wise_Out-of-Distribution_Detection_in_Semantic_Segmentation_ICCV_2023_paper.pdf) [[Code]](https://github.com/yyliu01/RPL) by Liu et al.

(ICCV 2023) [Simple and Effective Out-of-Distribution Detection via Cosine-based Softmax Loss](https://openaccess.thecvf.com/content/ICCV2023/papers/Noh_Simple_and_Effective_Out-of-Distribution_Detection_via_Cosine-based_Softmax_Loss_ICCV_2023_paper.pdf) by Noh et al.

(ICCV 2023) [Deep Feature Deblurring Diffusion for Detecting Out-of-Distribution Objects](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Deep_Feature_Deblurring_Diffusion_for_Detecting_Out-of-Distribution_Objects_ICCV_2023_paper.pdf) [[Code]](https://github.com/AmingWu/DFDD-OOD) by Wu et al.

(ICCV 2023) [Revisit PCA-based technique for Out-of-Distribution Detection](https://openaccess.thecvf.com/content/ICCV2023/papers/Guan_Revisit_PCA-based_Technique_for_Out-of-Distribution_Detection_ICCV_2023_paper.pdf) [[Code]](ttps://github.com/SYSUMIA-GROUP/pca-based-out-of-distribution-detection) by Guan and Liu et al.

(ICCV 2023) [WDiscOOD: Out-of-Distribution Detection via Whitened Linear Discriminant Analysis](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_WDiscOOD_Out-of-Distribution_Detection_via_Whitened_Linear_Discriminant_Analysis_ICCV_2023_paper.pdf) [[Code]](https://github.com/ivalab/WDiscOOD) by Chen et al.

(ICCV 2023) [Anomaly Detection under Distribution Shift](https://openaccess.thecvf.com/content/ICCV2023/papers/Cao_Anomaly_Detection_Under_Distribution_Shift_ICCV_2023_paper.pdf) [[Code]](https://github.com/mala-lab/ADShift) by Cao et al.

(ICCV 2023) [Out-of-Distribution Detection for Monocular Depth Estimation](https://openaccess.thecvf.com/content/ICCV2023/papers/Hornauer_Out-of-Distribution_Detection_for_Monocular_Depth_Estimation_ICCV_2023_paper.pdf) [[Code]](https://github.com/jhornauer/mde_ood) by Hornauer et al.

(ICCV 2023) [Unified Out-Of-Distribution Detection: A Model-Specific Perspective](https://openaccess.thecvf.com/content/ICCV2023/papers/Averly_Unified_Out-Of-Distribution_Detection_A_Model-Specific_Perspective_ICCV_2023_paper.pdf) by Averly et al.

(ICCV 2023) [CLIPN for Zero-Shot OOD Detection: Teaching CLIP to Say No](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_CLIPN_for_Zero-Shot_OOD_Detection_Teaching_CLIP_to_Say_No_ICCV_2023_paper.pdf) [[Code]](https://github.com/xmed-lab/CLIPN) by Wang et al.

(CVPR 2023) [Distribution Shift Inversion for Out-of-Distribution Prediction](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Distribution_Shift_Inversion_for_Out-of-Distribution_Prediction_CVPR_2023_paper.pdf) [[Code]](https://github.com/yu-rp/Distribution-Shift-Iverson) by Yu et al.

(CVPR 2023) [Uncertainty-Aware Optimal Transport for Semantically Coherent Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2023/papers/Lu_Uncertainty-Aware_Optimal_Transport_for_Semantically_Coherent_Out-of-Distribution_Detection_CVPR_2023_paper.pdf) [[Code]](https://github.com/LuFan31/ET-OOD) by Lu et al.

(CVPR 2023) [GEN: Pushing the Limits of Softmax-Based Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_GEN_Pushing_the_Limits_of_Softmax-Based_Out-of-Distribution_Detection_CVPR_2023_paper.pdf) [[Video]](https://www.youtube.com/watch?v=v5f8_fme9aY) [[Code]](https://github.com/XixiLiu95/GEN) by Liu et al.

(CVPR 2023) [(NAP) Detection of Out-of-Distribution Samples Using Binary Neuron Activation Patterns](https://openaccess.thecvf.com/content/CVPR2023/papers/Olber_Detection_of_Out-of-Distribution_Samples_Using_Binary_Neuron_Activation_Patterns_CVPR_2023_paper.pdf) [[Code]](https://github.com/safednn-group/nap-ood) by Olber et al.

(CVPR 2023) [Decoupling MaxLogit for Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Decoupling_MaxLogit_for_Out-of-Distribution_Detection_CVPR_2023_paper.pdf) by Zhang and Xiang

(CVPR 2023) [Balanced Energy Regularization Loss for Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2023/papers/Choi_Balanced_Energy_Regularization_Loss_for_Out-of-Distribution_Detection_CVPR_2023_paper.pdf) [[Code]](https://github.com/hyunjunChhoi/Balanced_Energy) by Choi et al.

(CVPR 2023) [Rethinking Out-of-Distribution (OOD) Detection: Masked Image Modeling Is All You Need](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Rethinking_Out-of-Distribution_OOD_Detection_Masked_Image_Modeling_Is_All_You_CVPR_2023_paper.pdf) [[Code]](https://github.com/JulietLJY/MOOD) by Li et al.

(CVPR 2023) [LINe: Out-of-Distribution Detection by Leveraging Important Neurons](https://openaccess.thecvf.com/content/CVPR2023/papers/Ahn_LINe_Out-of-Distribution_Detection_by_Leveraging_Important_Neurons_CVPR_2023_paper.pdf) [[Code]](https://github.com/YongHyun-Ahn/LINe-Out-of-Distribution-Detection-by-Leveraging-Important-Neurons) by Ahn et al.

(EMNLP 2023) [A Critical Analysis of Out-of-distribution Detection for Document Understanding](https://openreview.net/pdf?id=IHGnybgLo1Z) by Gu et al.

(ICLR 2023) ⭐⭐⭐⭐⭐ [A framework for benchmarking Class-out-of-distribution detection and its application to ImageNet](https://openreview.net/pdf?id=Iuubb9W6Jtk) [[Code]](https://github.com/mdabbah/COOD_benchmarking) by Galil et al.

(ICLR 2023) [Energy-based Out-of-Distribution Detection for Graph Neural Networks](https://openreview.net/pdf?id=zoz7Ze4STUL) [[Code]](https://github.com/qitianwu/GraphOOD-GNNSafe) by Wu et al.

(ICLR 2023) [The Tilted Variational Autoencoder: Improving Out-of-Distribution Detection](https://openreview.net/pdf?id=YlGsTZODyjz) [[Code]](https://github.com/anonconfsubaccount/tilted_prior) by Floto et al.

(ICLR 2023) [Out-of-Distribution Detection based on In-Distribution Data Patterns Memorization with Modern Hopfield Energy](https://openreview.net/pdf?id=KkazG4lgKL) by Zhang et al.

(ICLR 2023) [Out-of-Distribution Detection and Selective Generation for Conditional Language Models](https://openreview.net/pdf?id=kJUS5nD0vPB) by Ren et al.

(ICLR 2023) [Turning the Curse of Heterogeneity in Federated Learning into a Blessing for Out-of-Distribution Detection](https://openreview.net/pdf?id=mMNimwRb7Gr) by Yu et al.

(ICLR 2023) [Non-Parametric Outlier Synthesis](https://arxiv.org/pdf/2303.02966.pdf) [[Code]](https://github.com/deeplearning-wisc/npos) by Tao et al.

(ICLR 2023) [Out-of-distribution Detection with Implicit Outlier Transformation](https://arxiv.org/pdf/2303.05033.pdf) by Wang et al.

(UAI 2023) [A Constrained Bayesian Approach to Out-of-Distribution Prediction](https://proceedings.mlr.press/v216/wang23f/wang23f.pdf) by Wang and Yuan et al.

(UAI 2023) [Why Out-of-Distribution Detection Experiments Are Not Reliable - Subtle Experimental Details Muddle the OOD Detector Rankings](https://proceedings.mlr.press/v216/szyc23a/szyc23a.pdf) by Szyc et al.

(ICML 2023) [Unsupervised Out-of-Distribution Detection with Diffusion Inpainting](https://openreview.net/pdf?id=HiX1ybkFMl) by Liu et al.

(ICML 2023) [Generative Causal Representation Learning for Out-of-Distribution Motion Forecasting](https://openreview.net/pdf?id=Kw90j2pNSt) by Bagi et al.

(ICML 2023) [Model Ratatouille: Recycling Diverse Models for Out-of-Distribution Generalization](https://openreview.net/pdf?id=6x15tarUo9) [[Code]](https://github.com/facebookresearch/ModelRatatouille) by Ramé et al.

(ICML 2023) [Out-of-Distribution Generalization of Federated Learning via Implicit Invariant Relationships](https://proceedings.mlr.press/v202/guo23b/guo23b.pdf) by Guo et al.

(ICML 2023) [Feed Two Birds with One Scone: Exploiting Wild Data for Both Out-of-Distribution Generalization and Detection](https://arxiv.org/pdf/2306.09158.pdf) [[Video]](https://www.youtube.com/watch?v=4qMY-pLe638) by Bai et al.

(ICML 2023) [Concept-based Explanations for Out-of-Distribution Detectors](https://openreview.net/pdf?id=a33IYBCFey) by Choi et al.

(ICML 2023) [Hybrid Energy Based Model in the Feature Space for Out-of-Distribution Detection](https://openreview.net/pdf?id=61MLtEM-3gw) by Lafon et al.

(ICML 2023) [Detecting Out-of-distribution Data through In-distribution Class Prior](https://openreview.net/pdf?id=charggEv8v) by Jiang et al.

(ICML 2023) [Unleashing Mask: Explore the Intrinsic Out-of-Distribution Detection Capability](https://arxiv.org/pdf/2306.03715.pdf) [[Code]](https://github.com/ZFancy/Unleashing-Mask) by Zhu et al

(ICML 2023) [In or Out? Fixing ImageNet Out-of-Distribution Detection Evaluation](https://arxiv.org/pdf/2306.00826.pdf) [[Code]](https://github.com/j-cb/NINCO) by Bitterwolf et al.

(AAAI 2023) [READ: Aggregating Reconstruction Error into Out-of-Distribution Detection](https://arxiv.org/pdf/2206.07459.pdf) by Jiang et al.

(AAAI 2023) [Towards In-Distribution Compatible Out-of-Distribution Detection](https://ojs.aaai.org/index.php/AAAI/article/view/26230/26002) by Wu et al.

(AAAI 2023) [Robustness to Spurious Correlations Improves Semantic Out-of-Distribution Detection](https://ojs.aaai.org/index.php/AAAI/article/view/26785) by Zhang and Ranganath

(arXiv 2023)[OpenOOD v1.5: Enhanced Benchmark for Out-of-distribution Detection](https://arxiv.org/pdf/2306.09301.pdf) by Zhang et al.

(MIDL 2023) [Know Your Space: Inlier and Outlier Construction for Calibrating Medical OOD Detectors](https://openreview.net/pdf?id=RU7fr0-M8N) [[Project Page]](https://software.llnl.gov/OODmedic/) by Narayanaswamy, Mubarka et al.

(TMLR 2022) [Linking Neural Collapse and L2 Normalization with Improved Out-of-Distribution Detection in Deep Neural Networks](https://openreview.net/pdf?id=fjkN5Ur2d6) by Haas et al.

(CVPR 2022) [ViM: Out-Of-Distribution with Virtual-logit Matching](https://arxiv.org/pdf/2203.10807.pdf) [[Project Page]](https://ooddetection.github.io/#comp-l0p561in) by Wang et al.

(CVPR 2022) [Neural Mean Discrepancy for Efficient Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2022/papers/Dong_Neural_Mean_Discrepancy_for_Efficient_Out-of-Distribution_Detection_CVPR_2022_paper.pdf) by Dong et al.

(CVPR 2022) [Deep Hybrid Models for Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2022/papers/Cao_Deep_Hybrid_Models_for_Out-of-Distribution_Detection_CVPR_2022_paper.pdf) by Cao and Zhang

(CVPR 2022) [Rethinking Reconstruction Autoencoder-Based Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Rethinking_Reconstruction_Autoencoder-Based_Out-of-Distribution_Detection_CVPR_2022_paper.pdf) by Yibo Zhou

(CVPR 2022) [Unknown-Aware Object Detection: Learning What You Don't Know from Videos in the Wild](https://arxiv.org/pdf/2203.03800.pdf) [[Code]](https://github.com/deeplearning-wisc/stud) by Du et al.

(NeurIPS 2022) ⭐⭐⭐⭐⭐ [OpenOOD: Benchmarking Generalized Out-of-Distribution Detection](https://arxiv.org/pdf/2210.07242.pdf) [[Code]](https://github.com/Jingkang50/OpenOOD) by Yang et al.

(NeurIPS 2022) [Boosting Out-of-distribution Detection with Typical Features](https://arxiv.org/pdf/2210.04200.pdf) by Zhu et al.

(NeurIPS 2022) [GraphDE: A Generative Framework for Debiased Learning and Out-of-Distribution Detection on Graphs](https://proceedings.neurips.cc/paper_files/paper/2022/file/c34262c35aa5f8c1a091822cbb2020c2-Paper-Conference.pdf) [[Code]](https://github.com/Emiyalzn/GraphDE) by Li et al.

(NeurIPS 2022) [Out-of-Distribution Detection via Conditional Kernel Independence Model](https://proceedings.neurips.cc/paper_files/paper/2022/file/ec14daa5c50745f83fb27f685f8dfc22-Paper-Conference.pdf) by Wang et al.

(NeurIPS 2022) [Your Out-of-Distribution Detection Method is Not Robust!](https://proceedings.neurips.cc/paper_files/paper/2022/file/1f6591cc41be737e9ba4cc487ac8082d-Paper-Conference.pdf) [[Code]](https://github.com/rohban-lab/ATD) by Azizmalayeri et al.

(NeurIPS 2022) [Out-of-Distribution Detection with An Adaptive Likelihood Ratio on Informative Hierarchical VAE](https://proceedings.neurips.cc/paper_files/paper/2022/file/3066f60a91d652f4dc690637ac3a2f8c-Paper-Conference.pdf) by Li et al.

(NeurIPS 2022) [GOOD: A Graph Out-of-Distribution Benchmark](https://proceedings.neurips.cc/paper_files/paper/2022/file/0dc91de822b71c66a7f54fa121d8cbb9-Paper-Datasets_and_Benchmarks.pdf) [[Code]](https://github.com/divelab/GOOD) by Gui et al.

(NeurIPS 2022) ⭐⭐⭐⭐⭐ [Is Out-of-Distribution Detection Learnable?](https://proceedings.neurips.cc/paper_files/paper/2022/file/f0e91b1314fa5eabf1d7ef6d1561ecec-Paper-Conference.pdf) by Fang et al.

(NeurIPS 2022) [Towards Out-of-Distribution Sequential Event Prediction: A Causal Treatment](https://proceedings.neurips.cc/paper_files/paper/2022/file/8e69a97cbdd91ac0808603fa589d6c17-Paper-Conference.pdf) by Yang et al.

(NeurIPS 2022) [Delving into Out-of-Distribution Detection with Vision-Language Representations](https://arxiv.org/pdf/2211.13445.pdf) [[Video]](https://www.youtube.com/watch?v=ZZlxBgGalVA) [[Code]](https://github.com/deeplearning-wisc/MCM) by Ming et al.

(NeurIPS 2022) [Beyond Mahalanobis Distance for Textual OOD Detection](https://proceedings.neurips.cc/paper_files/paper/2022/file/70fa5df8e3300dc30bf19bee44a56155-Paper-Conference.pdf) by Colombo et al.

(NeurIPS 2022) [Density-driven Regularization for Out-of-distribution Detection](https://proceedings.neurips.cc/paper_files/paper/2022/file/05b69cc4c8ff6e24c5de1ecd27223d37-Paper-Conference.pdf) by Huang et al.

(NeurIPS 2022) [SIREN: Shaping Representations for Detecting Out-of-Distribution Objects](https://proceedings.neurips.cc/paper_files/paper/2022/file/804dbf8d3b8eee1ef875c6857efc64eb-Paper-Conference.pdf) [[Code]](https://github.com/deeplearning-wisc/siren) by Du et al.

(ICML 2022) [Mitigating Neural Network Overconfidence with Logit Normalization](https://arxiv.org/pdf/2205.09310.pdf) [[Code]](https://github.com/hongxin001/logitnorm_ood) by Hsu et al.

(ICML 2022) [Scaling Out-of-Distribution Detection for Real-World Settings](https://arxiv.org/pdf/1911.11132.pdf) [[Code]](https://github.com/hendrycks/anomaly-seg) by Hendrycks et al.

(ICML 2022) [Open-Sampling: Exploring Out-of-Distribution data for Re-balancing Long-tailed datasets](https://proceedings.mlr.press/v162/wei22c/wei22c.pdf) by Wei et al.

(ICML 2022) [Model Agnostic Sample Reweighting for Out-of-Distribution Learning](https://proceedings.mlr.press/v162/zhou22d/zhou22d.pdf) [[Code]](https://github.com/x-zho14/MAPLE) by Zhou et al.

(ICML 2022) [Partial and Asymmetric Contrastive Learning for Out-of-Distribution Detection in Long-Tailed Recognition](https://proceedings.mlr.press/v162/wang22aq/wang22aq.pdf) [[Code]](https://github.com/amazon-science/long-tailed-ood-detection) by Wang et al.

(ICML 2022) [Breaking Down Out-of-Distribution Detection: Many Methods Based on OOD Training Data Estimate a Combination of the Same Core Quantities](https://proceedings.mlr.press/v162/bitterwolf22a/bitterwolf22a.pdf) [[Code]](https://github.com/j-cb/Breaking_Down_OOD_Detection) by Bitterwolf et al.

(ICML 2022) [Predicting Out-of-Distribution Error with the Projection Norm](https://proceedings.mlr.press/v162/yu22i/yu22i.pdf) [[Code]](https://github.com/yaodongyu/ProjNorm) by Yu et al.

(ICML 2022) [POEM: Out-of-Distribution Detection with Posterior Sampling](https://proceedings.mlr.press/v162/ming22a/ming22a.pdf) [[Code]](https://github.com/deeplearning-wisc/poem) by Ming et al.

(ICML 2022) [(kNN) Out-of-Distribution Detection with Deep Nearest Neighbors](https://arxiv.org/pdf/2204.06507.pdf) [[Code]](https://github.com/deeplearning-wisc/knn-ood) by Sun et al.

(ICML 2022) [Training OOD Detectors in their Natural Habitats](https://proceedings.mlr.press/v162/katz-samuels22a/katz-samuels22a.pdf) by Katz-Samuels et al.

(ICLR 2022) [A Statistical Framework For Efficient Out-of-distribution Detection in Deep Neural Networks](https://arxiv.org/pdf/2102.12967) by Haroush and Frostig et al.

(ICLR 2022) [Extremely Simple Activation Shaping for Out-of-Distribution Detection](https://arxiv.org/pdf/2209.09858.pdf) [[Code]](https://github.com/andrijazz/ash) by Djurisic et al.

(ICLR 2022) [Revisiting flow generative models for Out-of-distribution detection](https://openreview.net/pdf?id=6y2KBh-0Fd9) by Jiang et al.

(ICLR 2022) [PI3NN: Out-of-distribution-aware Prediction Intervals from Three Neural Networks](https://openreview.net/pdf?id=NoB8YgRuoFU) [[Code]](https://github.com/liusiyan/UQnet) by Liu et al.

(ICLR 2022) [(ATC) Leveraging unlabeled data to predict out-of-distribution performance](https://openreview.net/pdf?id=o_HsiMPYh_x) by Garg et al.

(ICLR 2022) [Igeood: An Information Geometry Approach to Out-of-Distribution Detection](https://arxiv.org/pdf/2203.07798.pdf) [[Code]](https://github.com/igeood/Igeood) by Gomes et al.

(ICLR 2022) [How to Exploit Hyperspherical Embeddings for Out-of-Distribution Detection?](https://arxiv.org/pdf/2203.04450.pdf) [[Code]](https://github.com/deeplearning-wisc/cider) by Ming et al.

(ICLR 2022) [VOS: Learning What You Don't Know by Virtual Outlier Synthesis](https://arxiv.org/pdf/2202.01197.pdf) [[Code]](https://github.com/deeplearning-wisc/vos) by Du et al.

(UAI 2022) [Variational- and Metric-based Deep Latent Space for Out-Of-Distribution Detection](https://openreview.net/pdf?id=ScLeuUUi9gq) [[Code]](https://github.com/BGU-CS-VIL/vmdls) by Dinari and Freifeld

(ECCV 2022) [ Out-of-Distribution Identification: Let Detector Tell Which I Am Not Sure](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700631.pdf) by Li et al.

(ECCV 2022) [Out-of-distribution Detection with Boundary Aware Learning](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840232.pdf) [[Code]](https://github.com/ForeverPs/BAL) by Pei et al.

(ECCV 2022) [ Out-of-Distribution Detection with Semantic Mismatch under Masking](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840369.pdf) [[Code]](https://github.com/cure-lab/MOODCat) by Yang et al.

(ECCV 2022) [Data Invariants to Understand Unsupervised Out-of-Distribution Detection](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910129.pdf) [[Code]](https://github.com/LarsDoorenbos/Data-invariants) by Doorenbos et al.

(ECCV 2022) [Embedding contrastive unsupervised features to cluster in- and out-of-distribution noise in corrupted image datasets](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910389.pdf) [[Code]](github.com/PaulAlbert31/SNCF) by Albert et al.

(ECCV 2022) [DICE: Leveraging Sparsification for Out-of-Distribution Detection](https://arxiv.org/pdf/2111.09805.pdf) [[Code]](https://github.com/deeplearning-wisc/dice) by Sun and Li

(AAAI 2022) [On the Impact of Spurious Correlation for Out-of-distribution Detection](https://arxiv.org/pdf/2109.05642.pdf) [[Code]](https://github.com/deeplearning-wisc/Spurious_OOD) by Ming et al.

(AAAI 2022) [iDECODe: In-Distribution Equivariance for Conformal Out-of-Distribution Detection](https://www.aaai.org/AAAI22Papers/AAAI-12912.KaurR.pdf) by Kaur et al.

(AAAI 2022) [Provable Guarantees for Understanding Out-of-distribution Detection](https://arxiv.org/pdf/2112.00787.pdf) [[Code]](https://github.com/AlexMeinke/Provable-OOD-Detection) by Morteza and Li

(AAAI 2022) [Learning Modular Structures That Generalize Out-of-Distribution (Student Abstract)](https://www.aaai.org/AAAI22Papers/SA-00398-AshokA.pdf) by Ashok et al.

(AAAI 2022) [Exploiting Mixed Unlabeled Data for Detecting Samples of Seen and Unseen Out-of-Distribution Classes](https://www.aaai.org/AAAI22Papers/AAAI-6640.SunY.pdf) by Sun and Wang

(CVPR 2021) [Out-of-Distribution Detection Using Union of 1-Dimensional Subspaces](https://openaccess.thecvf.com/content/CVPR2021/papers/Zaeemzadeh_Out-of-Distribution_Detection_Using_Union_of_1-Dimensional_Subspaces_CVPR_2021_paper.pdf) [[Code]](https://github.com/zaeemzadeh/OOD) by Zaeemzadeh et al.

(CVPR 2021) [MOOD: Multi-level Out-of-distribution Detection](https://openaccess.thecvf.com/content/CVPR2021/papers/Lin_MOOD_Multi-Level_Out-of-Distribution_Detection_CVPR_2021_paper.pdf) [[Code]](https://github.com/deeplearning-wisc/MOOD) by Lin et al.

(CVPR 2021) [MOS: Towards Scaling Out-of-distribution Detection for Large Semantic Space](https://arxiv.org/pdf/2105.01879.pdf) [[Code]](https://github.com/deeplearning-wisc/large_scale_ood) by Huang and Li

(NeurIPS 2021) [Single Layer Predictive Normalized Maximum Likelihood for Out-of-Distribution Detection](https://arxiv.org/pdf/2110.09246.pdf) [[Code]](https://github.com/kobybibas/pnml_ood_detection) by Bibas et al.

(NeurIPS 2021) [STEP: Out-of-Distribution Detection in the Presence of Limited In-Distribution Labeled Data](https://openreview.net/pdf?id=p9dySshcS0q) by Zhou et al.

(NeurIPS 2021) [Exploring the Limits of Out-of-Distribution Detection](https://openreview.net/pdf?id=j5NrN8ffXC) [[Code]](https://github.com/stanislavfort/exploring_the_limits_of_OOD_detection) by Fort et al.

(NeurIPS 2021) [Learning Causal Semantic Representation for Out-of-Distribution Prediction](https://openreview.net/pdf?id=-msETI57gCH) [[Code]](https://github.com/changliu00/causal-semantic-generative-model) by Liu et al.

(NeurIPS 2021) [Towards optimally abstaining from prediction with OOD test examples](https://openreview.net/pdf?id=P9_gOq5w7Eb) by Kalai and Kanade

(NeurIPS 2021) [Locally Most Powerful Bayesian Test for Out-of-Distribution Detection using Deep Generative Models](https://openreview.net/pdf?id=-nLW4nhdkO) [[Code]](https://github.com/keunseokim91/LMPBT) by Kim et al.

(NeurIPS 2021) [RankFeat: Rank-1 Feature Removal for Out-of-distribution Detection](https://arxiv.org/pdf/2209.08590.pdf) [[Code]](https://github.com/KingJamesSong/RankFeat) by Song et al.

(NeurIPS 2021) ⭐⭐⭐⭐⭐ [ReAct: Out-of-distribution Detection With Rectified Activations](https://arxiv.org/pdf/2111.12797.pdf) [[Code]](https://github.com/deeplearning-wisc/react) by Sun et al.

(NeurIPS 2021) ⭐⭐⭐⭐⭐ [(GradNorm) On the Importance of Gradients for Detecting Distributional Shifts in the Wild](https://arxiv.org/pdf/2110.00218.pdf) [[Code]](https://github.com/deeplearning-wisc/gradnorm_ood) by Huang et al.

(NeurIPS 2021) [Watermarking for Out-of-distribution Detection](https://arxiv.org/pdf/2210.15198.pdf) by Wang et al.

(NeurIPS 2021) [Can multi-label classification networks know what they don't know?](https://arxiv.org/pdf/2109.14162.pdf) [[Code]](https://github.com/deeplearning-wisc/multi-label-ood) by Wang et al.

(ICLR 2021) [SSD: A Unified Framework for Self-Supervised Outlier Detection](https://arxiv.org/pdf/2103.12051.pdf) [[Code]](https://github.com/inspire-group/SSD) by Sehwag et al.

(ICLR 2021) [Multiscale Score Matching for Out-of-Distribution Detection](https://openreview.net/pdf?id=xoHdgbQJohv) [[Code]](https://github.com/ahsanMah/msma) by Mahmood et al.

(UAI 2021) [Sketching Curvature for Efficient Out-of-Distribution Detection for Deep Neural Networks](https://proceedings.mlr.press/v161/sharma21a/sharma21a.pdf) [[Code]](https://github.com/StanfordASL/SCOD) by Sharma et al.

(UAI 2021) [Know Your Limits: Uncertainty Estimation with ReLU Classifiers Fails at Reliable OOD Detection](https://proceedings.mlr.press/v161/ulmer21a/ulmer21a.pdf) [[Code]](https://github.com/Kaleidophon/know-your-limits) by Ulmer and Cinà

(ICML 2021) [Understanding Failures in Out-of-Distribution Detection with Deep Generative Models](http://proceedings.mlr.press/v139/zhang21g/zhang21g.pdf) by Zhang et al.

(AISTATS 2021) [Density of States Estimation for Out of Distribution Detection](https://proceedings.mlr.press/v130/morningstar21a/morningstar21a.pdf) by Morningstar et al.

(ICCV 2021) [Semantically Coherent Out-of-Distribution Detection](https://arxiv.org/pdf/2108.11941.pdf) [[Project Page]](https://jingkang50.github.io/projects/scood) [[Code]](https://github.com/jingkang50/ICCV21_SCOOD) by Yang et al.

(ICCV 2021) [CODEs: Chamfer Out-of-Distribution Examples against Overconfidence Issue](https://arxiv.org/pdf/2108.06024.pdf) by Tang et al.

(CVPR 2020) [Deep Residual Flow for Out of Distribution Detection](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zisselman_Deep_Residual_Flow_for_Out_of_Distribution_Detection_CVPR_2020_paper.pdf) [[Code]](https://github.com/EvZissel/Residual-Flow) by Zisselman and Tamar

(CVPR 2020) [Generalized ODIN: Detecting Out-of-Distribution Image Without Learning From Out-of-Distribution Data](https://arxiv.org/pdf/2002.11297.pdf) [[Code]](https://github.com/sayakpaul/Generalized-ODIN-TF)  by Hsu et al.

(NeurIPS 2020) [CSI: Novelty Detection via Contrastive Learning on Distributionally Shifted Instances](https://arxiv.org/pdf/2007.08176.pdf) [[Code]](https://github.com/alinlab/CSI) by Tack et al.

(NeurIPS 2020) ⭐⭐⭐⭐⭐ [Energy-based Out-of-distribution Detection](https://arxiv.org/pdf/2010.03759.pdf) [[Code]](https://github.com/wetliu/energy_ood) by Liu et al.

(NeurIPS 2020) [OOD-MAML: Meta-Learning for Few-Shot Out-of-Distribution Detection and Classification](https://proceedings.neurips.cc/paper/2020/file/28e209b61a52482a0ae1cb9f5959c792-Paper.pdf) [[Video]](https://slideslive.com/38935997/oodmaml-metalearning-for-fewshot-outofdistribution-detection-and-classification) by Jeong and Kim

(NeurIPS 2020) [Towards Maximizing the Representation Gap between In-Domain & Out-of-Distribution Examples](https://proceedings.neurips.cc/paper/2020/file/68d3743587f71fbaa5062152985aff40-Paper.pdf) [[Code]](https://github.com/jayjaynandy/maximize-representation-gap) by Nandy et al.

(NeurIPS 2020) [Likelihood Regret: An Out-of-Distribution Detection Score For Variational Auto-encoder](https://proceedings.neurips.cc/paper/2020/file/eddea82ad2755b24c4e168c5fc2ebd40-Paper.pdf) [[Code]](https://github.com/XavierXiao/Likelihood-Regret) by Xiao et al.

(NeurIPS 2020) ⭐⭐⭐⭐⭐ [Why Normalizing Flows Fail to Detect Out-of-Distribution Data](https://proceedings.neurips.cc/paper/2020/file/ecb9fe2fbb99c31f567e9823e884dbec-Paper.pdf) [[Code]](https://github.com/PolinaKirichenko/flows_ood) by Kirichenko et al.

(ICLR 2020) [Towards Neural Networks That Provably Know When They Don't Know](https://openreview.net/pdf?id=ByxGkySKwH) [[Code]](https://github.com/AlexMeinke/certified-certain-uncertainty) by Meinke et al.

(ICML 2020) [Detecting Out-of-Distribution Examples with Gram Matrices](http://proceedings.mlr.press/v119/sastry20a/sastry20a.pdf) [[Code]](https://github.com/VectorInstitute/gram-ood-detection) by Sastry and Oore

(CVPR 2019) [Out-Of-Distribution Detection for Generalized Zero-Shot Action Recognition](https://openaccess.thecvf.com/content_CVPR_2019/papers/Mandal_Out-Of-Distribution_Detection_for_Generalized_Zero-Shot_Action_Recognition_CVPR_2019_paper.pdf) [[Code]](https://github.com/naraysa/gzsl-od) by Mandal et al.

(NeurIPS 2019) [Likelihood Ratios for Out-of-Distribution Detection](https://arxiv.org/pdf/1906.02845.pdf) [[Video]](https://www.youtube.com/watch?v=-FduW9ZWAR4) by Ren et al.

(ICCV 2019) [Unsupervised Out-of-Distribution Detection by Maximum Classifier Discrepancy](https://arxiv.org/pdf/1908.04951.pdf) [[Code]](https://github.com/Mephisto405/Unsupervised-Out-of-Distribution-Detection-by-Maximum-Classifier-Discrepancy) by Yu and Aizawa

(arXiv 2019) [WAIC, but Why? Generative Ensembles for Robust Anomaly Detection](https://arxiv.org/pdf/1810.01392) by Choi and Jang et al.

(NeurIPS 2018) ⭐⭐⭐⭐⭐ [(Mahalanobis) A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks](https://arxiv.org/pdf/1807.03888.pdf) [[Code]](https://github.com/pokaxpoka/deep_Mahalanobis_detector) by Lee et al.

(NeurIPS 2018) [Out-of-Distribution Detection using Multiple Semantic Label Representations](https://arxiv.org/pdf/1808.06664.pdf) by Shalev et al.

(NeurIPS 2018) [Why ReLU Networks Yield High-Confidence Predictions Far Away From the Training Data and How to Mitigate the Problem](https://arxiv.org/pdf/1812.05720.pdf) [[Code]](https://github.com/max-andr/relu_networks_overconfident) by Hein et al.

(ICLR 2018) [Do Deep Generative Models Know What They Don't Know?](https://arxiv.org/pdf/1810.09136.pdf) [[Slides]](https://fernandoperezc.github.io/Advanced-Topics-in-Machine-Learning-and-Data-Science/Jia.pdf) by Nalisnick et al.

(ICLR 2018) ⭐⭐⭐⭐⭐ [(OE) Deep Anomaly Detection with Outlier Exposure](https://arxiv.org/pdf/1812.04606.pdf) [[Code]](https://github.com/hendrycks/outlier-exposure) by Hendrycks et al.

(ICLR 2018) ⭐⭐⭐⭐⭐ [(ODIN) Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks](https://arxiv.org/pdf/1706.02690.pdf) [[Code]](https://github.com/facebookresearch/odin) by Liang et al.

(ICLR 2018) [Training Confidence-calibrated Classifiers for Detecting Out-of-Distribution Samples](https://arxiv.org/pdf/1711.09325.pdf) [[Code]](https://github.com/alinlab/Confident_classifier) by Lee et al.

(ECCV 2018) [Out-of-Distribution Detection Using an Ensemble of Self-Supervised Leave-out Classifiers](https://arxiv.org/pdf/1809.03576.pdf) [[Code]](https://github.com/YU1ut/Ensemble-of-Leave-out-Classifiers) by Vyas et al.

(ArXiv 2018) [Learning Confidence for Out-of-Distribution Detection in Neural Networks](https://arxiv.org/pdf/1802.04865.pdf) [[Code]](https://github.com/uoguelph-mlrg/confidence_estimation) by DeVries and Taylor

(ICLR 2017) ⭐⭐⭐⭐⭐ [A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks](https://arxiv.org/pdf/1610.02136.pdf) [[Code]](https://github.com/hendrycks/error-detection) by Hendrycks and Gimpel

## OOD Robustness

(NeurIPS 2024) [Reconstruct and Match: Out-of-Distribution Robustness via Topological Homogeneity]() by Chen et al.

(CVPR 2024) [A Bayesian Approach to OOD Robustness in Image Classification](https://openaccess.thecvf.com/content/CVPR2024/papers/Kaushik_A_Bayesian_Approach_to_OOD_Robustness_in_Image_Classification_CVPR_2024_paper.pdf) by Kaushik et al.

(NeurIPS 2023) [Distilling Out-of-distribution Robustness from Vision-Language Foundation Models](https://proceedings.neurips.cc/paper_files/paper/2023/file/67f30132d98e758f7b4e28c36091d86e-Paper-Conference.pdf) by Zhou et al.

(NeurIPS 2023) [Revisiting Out-of-distribution Robustness in NLP: Benchmark, Analysis, and LLMs Evaluations](https://proceedings.neurips.cc/paper_files/paper/2023/file/b6b5f50a2001ad1cbccca96e693c4ab4-Paper-Datasets_and_Benchmarks.pdf) by Yuan et al.

(ICLR 2023) [Diversify and Disambiguate: Out-of-Distribution Robustness via Disagreement](https://openreview.net/pdf?id=RVTOp3MwT3n) by Lee et al.

(ICML 2023) [Learning Unforeseen Robustness from Out-of-distribution Data Using Equivariant Domain Translator](https://proceedings.mlr.press/v202/zhu23a/zhu23a.pdf) by Zhu et al.

(ICML 2023) [Out-of-Domain Robustness via Targeted Augmentations](https://arxiv.org/pdf/2302.11861.pdf) [[Code]](https://github.com/i-gao/targeted-augs) by Gao et al.

(TMLR 2022) [The Evolution of Out-of-Distribution Robustness Throughout Fine-Tuning](https://openreview.net/pdf?id=Qs3EfpieOh) by Andreassen et al.

(NeurIPS 2022) [Using Mixup as a Regularizer Can Surprisingly Improve Accuracy & Out-of-Distribution Robustness](https://proceedings.neurips.cc/paper_files/paper/2022/file/5ddcfaad1cb72ce6f1a365e8f1ecf791-Paper-Conference.pdf) by Pinto et al.

(NeurIPS 2022) [Provably Adversarially Robust Detection of Out-of-Distribution Data (Almost) for Free](https://proceedings.neurips.cc/paper_files/paper/2022/file/c2c62117283dda155db754e54dbe8d71-Paper-Conference.pdf) [[Code]](https://github.com/AlexMeinke/Provable-OOD-Detection) by Meinke et al.

(ICML 2022) [Improving Out-of-Distribution Robustness via Selective Augmentation](https://proceedings.mlr.press/v162/yao22b/yao22b.pdf) [[Video]](https://www.youtube.com/watch?v=jaLkGVoun_4) [[Code]](https://github.com/huaxiuyao/LISA) by Yao et al.

(NeurIPS 2021) [A Winning Hand: Compressing Deep Networks Can Improve Out-of-Distribution Robustness](https://openreview.net/pdf?id=YygA0yppTR) by Diffenderfer et al.

(ICLR 2021) [In-N-Out: Pre-Training and Self-Training using Auxiliary Information for Out-of-Distribution Robustness](https://openreview.net/pdf?id=jznizqvr15J) [[Code]](https://github.com/p-lambda/in-n-out) by Xie et al.

(NeurIPS 2020) [Certifiably Adversarially Robust Detection of Out-of-Distribution Data](https://proceedings.neurips.cc/paper/2020/file/b90c46963248e6d7aab1e0f429743ca0-Paper.pdf) [[Code]](https://github.com/j-cb/GOOD) by Bitterwolf et al.

## OOD Generalization


(NeurIPS 2024) [Bridging Multicalibration and Out-of-distribution Generalization Beyond Covariate Shift]() by Wu et al.

(NeurIPS 2024) [Human-Assisted Out-of-Distribution Generalization and Detection]() by Bai et al.

(NeurIPS 2024) [Vision Transformer Neural Architecture Search for Out-of-Distribution Generalization: Benchmark and Insights]() by Ho et al.

(NeurIPS 2024) [What Variables Affect Out-Of-Distribution Generalization in Pretrained Models?]() by Harun et al.

(NeurIPS 2024) [FOOGD: Federated Collaboration for Both Out-of-distribution Generalization and Detection]() by Liao et al.

(NeurIPS 2024) [Benchmarking Out-of-Distribution Generalization Capabilities of DNN-based Encoding Models for the Ventral Visual Cortex]() by Madan et al.

(NeurIPS 2024) [Neural Collapse Inspired Feature Alignment for Out-of-Distribution Generalization]() by Chen et al.

(NeurIPS 2024) [Benchmarking Out-of-Distribution Generalization and Adaptation for Electromyography]() by Yang et al.

(NeurIPS 2024) [WikiDO: Evaluating Out-of-Distribution Generalization of Vision-Language Models in Cross-Modal Retrieval]() by Tankala et al.

(Nature) [Out-of-distribution generalization for learning quantum dynamics](https://www.nature.com/articles/s41467-023-39381-w) by Caro et al.

(ICML 2024) [CRoFT: Robust Fine-Tuning with Concurrent Optimization for OOD Generalization and Open-Set OOD Detection](https://openreview.net/pdf?id=xFDJBzPhci) [[Code]](https://github.com/LinLLLL/CRoFT) by Zhu et al.

(ICML 2024) [Time-Series Forecasting for Out-of-Distribution Generalization Using Invariant Learning](https://openreview.net/pdf?id=SMUXPVKUBg) [[Code]](https://github.com/AdityaLab/FOIL) by Liu et al.

(CVPR 2024) [Improving Out-of-Distribution Generalization in Graphs via Hierarchical Semantic Environments](https://openaccess.thecvf.com/content/CVPR2024/papers/Piao_Improving_Out-of-Distribution_Generalization_in_Graphs_via_Hierarchical_Semantic_Environments_CVPR_2024_paper.pdf) by Piao et al.

(ICLR 2024) [Unraveling The Key Components Of OOD Generalization Via Diversification](https://openreview.net/pdf?id=Lvf7GnaLru) by Benoit, Jiang, Atanov et al.

(ICLR 2024) [Maxmimum Likelihood Estimation Is All You Need For Well-Specified Covariate Shift](https://openreview.net/pdf?id=eoTCKKOgIs) by Ge and Tang et al.

(ICLR 2024) [Towards Robust Out-Of-Distribution Generalization Bounds via Sharpness](https://openreview.net/pdf?id=tPEwSYPtAC) by Zou et al.

(ICLR 2024) [Spurious Feature Diversification Improves Out-Of-Distribution Generalization](https://openreview.net/pdf?id=d6H4RBi7RH) by Yong and Tan et al.

(ICLR 2024) [HYPO: Hyperspherical Out-of-distribution Generalization](https://arxiv.org/pdf/2402.07785.pdf) Bai and Ming et al.

(NeurIPS 2023) [On the Adversarial Robustness of Out-of-distribution Generalization Models](https://proceedings.neurips.cc/paper_files/paper/2023/file/d9888cc7baa04c2e44e8115588133515-Paper-Conference.pdf) [[Code]](https://github.com/ZouXinn/OOD-Adv) by Zou and Liu

(NeurIPS 2023) [Joint Learning of Label and Environment Causal Independence for Graph Out-of-distribution Generalization](https://proceedings.neurips.cc/paper_files/paper/2023/file/0c6c92a0c5237761168eafd4549f1584-Paper-Conference.pdf) [[Code]](https://github.com/divelab/LECI) by Gui et al.

(NeurIPS 2023) [Environment-Aware Dynamic Graph Leaning for Out-of-distribution Generalization](https://proceedings.neurips.cc/paper_files/paper/2023/file/9bf12308ece130daa083fb21f7faf1b6-Paper-Conference.pdf) by Yuan et al.

(NeurIPS 2023) [Secure Out-of-distribution Task Generalization with Energy-based Models](https://proceedings.neurips.cc/paper_files/paper/2023/file/d39e3ae9a11b79691709a7a6e06a63d9-Paper-Conference.pdf) by Chen et al.

(NeurIPS 2023) [Understanding and Improving Feature Learning for Out-of-distribution Generalization](https://proceedings.neurips.cc/paper_files/paper/2023/file/d73d5645ddbb9ada6c862116435574f6-Paper-Conference.pdf) [[Code]](https://github.com/LFhase/FeAT) by Chen, Huang, and Zhou et al.

(ICLR 2023) [Improving Out-of-distribution Generalization with Indirection Representations](https://openreview.net/pdf?id=0f-0I6RFAch) by Pham et al.

(ICLR 2023) [Topology-aware Robust Optimization for Out-of-Distribution Generalization](https://openreview.net/pdf?id=ylMq8MBnAp) [[Code]](https://github.com/joffery/TRO) by Qiao and Peng

(ICLR 2023) ⭐⭐⭐⭐⭐[Modeling the Data-Generating Process is Necessary for Out-of-Distribution Generalization](https://openreview.net/pdf?id=uyqks-LILZX) by Kaur et al.

(ICCV 2023) [Distilling Large Vision-Language Model with Out-of-Distribution Generalizability](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Distilling_Large_Vision-Language_Model_with_Out-of-Distribution_Generalizability_ICCV_2023_paper.pdf) [[Code]](https://github.com/xuanlinli17/large_vlm_distillation_ood) by Li and Fang et al.

(ICML 2023) [Feed Two Birds with One Scone: Exploiting Wild Data for Both Out-of-Distribution Generalization and Detection](https://arxiv.org/pdf/2306.09158.pdf) [[Video]](https://www.youtube.com/watch?v=4qMY-pLe638) by Bai et al.

(AAAI 2023) [On the Connection between Invariant Learning and Adversarial Training for Out-of-Distribution Generalization](https://ojs.aaai.org/index.php/AAAI/article/view/26250/26022) by Xin et al.

(AAAI 2023) [Certifiable Out-of-Distribution Generalization](https://ojs.aaai.org/index.php/AAAI/article/view/26295/26067) by Ye et al.

(AAAI 2023) [Bayesian Cross-Modal Alignment Learning for Few-Shot Out-of-Distribution Generalization](https://ojs.aaai.org/index.php/AAAI/article/view/26355/26127) by Zhu et al.

(AAAI 2023) [Out-of-Distribution Generalization by Neural-Symbolic Joint Training](https://ojs.aaai.org/index.php/AAAI/article/view/26444/26216) by Liu et al.

(CVPR 2022) [Out-of-Distribution Generalization With Causal Invariant Transformations](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Out-of-Distribution_Generalization_With_Causal_Invariant_Transformations_CVPR_2022_paper.pdf) by Wang et al.

(CVPR 2022) [OoD-Bench: Quantifying and Understanding Two Dimensions of Out-of-Distribution Generalization](https://openaccess.thecvf.com/content/CVPR2022/papers/Ye_OoD-Bench_Quantifying_and_Understanding_Two_Dimensions_of_Out-of-Distribution_Generalization_CVPR_2022_paper.pdf) [[Video]](https://www.youtube.com/watch?v=txwI9f5Bfio) [[Code]](https://github.com/ynysjtu/ood_bench) by Ye et al.

(NeurIPS 2022) [Learning Invariant Graph Representations for Out-of-Distribution Generalization](https://proceedings.neurips.cc/paper_files/paper/2022/file/4d4e0ab9d8ff180bf5b95c258842d16e-Paper-Conference.pdf) by Li et al.

(NeurIPS 2022) [Improving Out-of-Distribution Generalization by Adversarial Training with Structured Priors](https://proceedings.neurips.cc/paper_files/paper/2022/file/adc98a266f45005c403b8311ca7e8bd7-Paper-Conference.pdf) by Wang et al.

(NeurIPS 2022) [Functional Indirection Neural Estimator for Better Out-of-distribution Generalization](https://proceedings.neurips.cc/paper_files/paper/2022/file/13b8d8fb8d05369480c2c344f2ce3f25-Paper-Conference.pdf) by Pham et al.

(NeurIPS 2022) [Multi-Instance Causal Representation Learning for Instance Label Prediction and Out-of-Distribution Generalization](https://proceedings.neurips.cc/paper_files/paper/2022/file/e261e92e1cfb820da930ad8c38d0aead-Paper-Conference.pdf) [[Code]](https://github.com/weijiazhang24/causalmil) by Zhang et al.

(NeurIPS 2022) [Assaying Out-Of-Distribution Generalization in Transfer Learning](https://proceedings.neurips.cc/paper_files/paper/2022/file/2f5acc925919209370a3af4eac5cad4a-Paper-Conference.pdf) [[Code]](https://github.com/amazon-science/assaying-ood) by Wenzel et al.

(NeurIPS 2022) [Learning Causally Invariant Representations for Out-of-Distribution Generalization on Graphs](https://proceedings.neurips.cc/paper_files/paper/2022/file/8b21a7ea42cbcd1c29a7a88c444cce45-Paper-Conference.pdf) [[Code]](https://github.com/LFhase/CIGA) by Chen et al.

(NeurIPS 2022) [Diverse Weight Averaging for Out-of-Distribution Generalization](https://proceedings.neurips.cc/paper_files/paper/2022/file/46108d807b50ad4144eb353b5d0e8851-Paper-Conference.pdf) [[Code]](https://github.com/alexrame/diwa) by Ramé et al.

(NeurIPS 2022) [ZooD: Exploiting Model Zoo for Out-of-Distribution Generalization](https://proceedings.neurips.cc/paper_files/paper/2022/file/cd305fdee96836d5cc1de94577d71b61-Paper-Conference.pdf) by Dong et al.

(ICML 2022) [Certifying Out-of-Domain Generalization for Blackbox Functions](https://proceedings.mlr.press/v162/weber22a/weber22a.pdf) [[Code]](https://github.com/DS3Lab/certified-generalization) by Weber et al.

(NeurIPS 2022) [LOG: Active Model Adaptation for Label-Efficient OOD Generalization](https://proceedings.neurips.cc/paper_files/paper/2022/file/4757094e8ccc17e3e25b40efaf06c746-Paper-Conference.pdf) by Shao et al.

(ICML 2022) [Fishr: Invariant Gradient Variances for Out-of-Distribution Generalization](https://proceedings.mlr.press/v162/rame22a/rame22a.pdf) [[Code]](https://github.com/alexrame/fishr) by Ramé et al.

(ICLR 2022) [Out-of-distribution Generalization in the Presence of Nuisance-Induced Spurious Correlations](https://openreview.net/pdf?id=12RoR2o32T) [[Code]](https://github.com/rajesh-lab/nurd-code-public) by Puli et al.

(ICLR 2022) [Uncertainty Modeling for Out-of-Distribution Generalization](https://openreview.net/pdf?id=6HN7LHyzGgC) [[Code]](https://github.com/lixiaotong97/DSU) by Li et al.

(ICLR 2022) [Invariant Causal Representation Learning for Out-of-Distribution Generalization](https://openreview.net/pdf?id=-e4EXDWXnSn) by Lu et al.

(ECCV 2022) [Out-of-Distribution Detection Using an Ensemble of Self Supervised Leave-out Classifiers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850089.pdf) [[Code]](https://github.com/simpleshinobu/IRMCon) by Qi et al.

(ECCV 2022) [Learning to Balance Specificity and Invariance for In and Out of Domain Generalization](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540290.pdf) [[Code]](https://github.com/prithv1/DMG) by Chattopadhyay et al.

(AAAI 2022) [VITA: A Multi-Source Vicinal Transfer Augmentation Method for Out-of-Distribution Generalization](https://www.aaai.org/AAAI22Papers/AAAI-733.ChenM.pdf) by Chen et al.

(CVPR 2021) [Deep Stable Learning for Out-of-Distribution Generalization](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Deep_Stable_Learning_for_Out-of-Distribution_Generalization_CVPR_2021_paper.pdf) by Zhang et al.

(NeurIPS 2021) [Invariance Principle Meets Information Bottleneck for Out-of-Distribution Generalization](https://openreview.net/pdf?id=jlchsFOLfeF) [[Video]](https://www.youtube.com/watch?v=g7SkcvMjVeI) by Ahuja et al.

(NeurIPS 2021) [On the Out-of-distribution Generalization of Probabilistic Image Modelling](https://openreview.net/pdf?id=q1yLPNF0UFV) by Zhang et al.

(NeurIPS 2021) [On Calibration and Out-of-Domain Generalization](https://openreview.net/pdf?id=XWYJ25-yTRS) [[Video]](https://www.youtube.com/watch?v=MidtYmDEhoA) by Wald et al.

(NeurIPS 2021) [Towards a Theoretical Framework of Out-of-Distribution Generalization](https://openreview.net/pdf?id=kFJoj7zuDVi) [[Slides]](https://haotianye.com/files/NeurIPS21/slides_NeurIPS21_OOD.pdf) by Ye et al.

(NeurIPS 2021) [Out-of-Distribution Generalization in Kernel Regression](https://openreview.net/pdf?id=-h6Ldc0MO-) by Canatar et al.

(NeurIPS 2021) [Characterizing Generalization under Out-Of-Distribution Shifts in Deep Metric Learning](https://openreview.net/pdf?id=_KqWSCu566) [[Code]](https://github.com/Confusezius/Characterizing_Generalization_in_DeepMetricLearning) by Millbich et al.

(ICLR 2021) [Understanding the failure modes of out-of-distribution generalization](https://openreview.net/pdf?id=fSTD6NFIW_b) [[Video]](https://www.youtube.com/watch?v=DhPMq_550OE) by Nagarajan et al.

(ICML 2021) [Accuracy on the Line: on the Strong Correlation Between Out-of-Distribution and In-Distribution Generalization](http://proceedings.mlr.press/v139/miller21b/miller21b.pdf) [[Code]](https://github.com/millerjohnp/linearfits_app) by Miller et al.

(ICML 2021) [Out-of-Distribution Generalization via Risk Extrapolation (REx)](http://proceedings.mlr.press/v139/krueger21a/krueger21a.pdf) by Krueger et al.

(ICML 2021) [Can Subnetwork Structure Be the Key to Out-of-Distribution Generalization?](http://proceedings.mlr.press/v139/zhang21a/zhang21a.pdf) [[Slides]](https://zdhnarsil.github.io/files/icml2021_invsubnet_slides.pdf) by Zhang et al.

(ICML 2021) [Graph Convolution for Semi-Supervised Classification: Improved Linear Separability and Out-of-Distribution Generalization](http://proceedings.mlr.press/v139/baranwal21a/baranwal21a.pdf) [[Code]](https://github.com/opallab/Graph-Convolution-for-Semi-Supervised-Classification-Improved-Linear-Separability-and-OoD-Gen.) by Baranwal et al.

## OOD Everything else

(NeurIPS 2024) [Adaptive Labeling for Efficient Out-of-distribution Model Evaluation]() by Mittal et al.

(NeurIPS 2024) [TAIA: Large Language Models are Out-of-Distribution Data Learners]() by Jiang et al.

(NeurIPS 2024) [PURE: Prompt Evolution with Graph ODE for Out-of-distribution Fluid Dynamics Modeling]() by Wu et al.

(NeurIPS 2024) [Scanning Trojaned Models Using Out-of-Distribution Samples]() by Mirzaei et al.

(ICML 2024 Reject) [Split-Ensemble: Efficient OOD-aware Ensemble via Task and Model Splitting](https://openreview.net/pdf?id=SLA7VOqwwT) by Chen et al.

(ICML 2024) [Context-Guided Diffusion for Out-of-Distribution Molecular and Protein Design](https://openreview.net/pdf/b7a45cf1733be5651f5bddabeece506fb72a174c.pdf) [[Code]](https://github.com/leojklarner/context-guided-diffusion) by Klarner et al.

(ICML 2024) [A Generative Approach for Treatment Effect Estimation under Collider Bias: From an Out-of-Distribution Perspective](https://openreview.net/pdf?id=kUj9b2CezT) [[Code]](https://github.com/ZJUBaohongLi/C2GAM) by Li et al.

(CVPR 2024) [Unexplored Faces of Robustness and Out-of-Distribution: Covariate Shifts in Environment and Sensor Domains](https://openaccess.thecvf.com/content/CVPR2024/papers/Baek_Unexplored_Faces_of_Robustness_and_Out-of-Distribution_Covariate_Shifts_in_Environment_CVPR_2024_paper.pdf) [[Code]](https://github.com/Edw2n/ImageNet-ES) by Baek et al.

(CVPR 2024) [Label-Efficient Group Robustness via Out-of-Distribution Concept Curation](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Label-Efficient_Group_Robustness_via_Out-of-Distribution_Concept_Curation_CVPR_2024_paper.pdf) by Yang et al.

(CVPR 2024) [Descriptor and Word Soups: Overcoming the Parameter Efficiency Accuracy Tradeoff for Out-of-Distribution Few-shot Learning](https://openaccess.thecvf.com/content/CVPR2024/papers/Liao_Descriptor_and_Word_Soups_Overcoming_the_Parameter_Efficiency_Accuracy_Tradeoff_CVPR_2024_paper.pdf) [[Code]](https://github.com/Chris210634/word_soups) by Liao et al.

(CVPR 2024) [Segment Every Out-of-Distribution Object](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Segment_Every_Out-of-Distribution_Object_CVPR_2024_paper.pdf) [[Code]](https://github.com/WenjieZhao1/S2M) by Zhao et al.

(ICCV 2023) [Adaptive Calibrator Ensemble: Navigating Test Set Difficulty in Out-of-Distribution Scenarios](https://openaccess.thecvf.com/content/ICCV2023/papers/Zou_Adaptive_Calibrator_Ensemble_Navigating_Test_Set_Difficulty_in_Out-of-Distribution_Scenarios_ICCV_2023_paper.pdf) [[Code]](https://github.com/insysgroup/Adaptive-Calibrator-Ensemble) by Zou and Deng et al.

(NeurIPS 2023) [Not All Out-of-distribution Data Are Harmful to Open-Set Active Learning](https://proceedings.neurips.cc/paper_files/paper/2023/file/2c8d9636f74d0207ff4f65956010f450-Paper-Conference.pdf) by Yang et al.

(NeurIPS 2023) [AlberDICE: Addressing Out-of-distribution Joint Actions in Offline Multi-Agent RL via Alternating Stationary Distribution Correction Estimation](https://proceedings.neurips.cc/paper_files/paper/2023/file/e5b6eb1dbabff82838d5e99f62de37c8-Paper-Conference.pdf) by Matsunaga and Lee et al.

(ICLR 2023) [Harnessing Out-Of-Distribution Examples via Augmenting Content and Style](https://openreview.net/pdf?id=boNyg20-JDm) by Huang et al.

(ICLR 2023) [Pareto Invariant Risk Minimization: Towards Mitigating the Optimization Dilemma in Out-of-Distribution Generalization](https://openreview.net/pdf?id=esFxSb_0pSL) [[Code]](https://github.com/LFhase/PAIR) by Chen et al.

(ICLR 2023) [On the Effectiveness of Out-of-Distribution Data in Self-Supervised Long-Tail Learning](https://openreview.net/pdf?id=v8JIQdiN9Sh) by Bai et al.

(ICLR 2023) [Out-of-distribution Representation Learning for Time Series Classification](https://openreview.net/pdf?id=gUZWOE42l6Q) by Lu et al.

(ICML 2023) [Exploring Chemical Space with Score-based Out-of-distribution Generation](https://openreview.net/pdf?id=45TeQUJw9tn) [[Code]](https://github.com/SeulLee05/MOOD) by Lee et al.

(ICML 2023) [The Value of Out-of-Distribution Data](https://proceedings.mlr.press/v202/de-silva23a/de-silva23a.pdf) by Silva et al.

(ICML 2023) [CLIPood: Generalizing CLIP to Out-of-Distributions](https://proceedings.mlr.press/v202/shu23a/shu23a.pdf) by Shu et al.

(ICRA 2023) [Unsupervised Road Anomaly Detection with Language Anchors](https://ieeexplore.ieee.org/document/10160470) by Tian et al.

(NeurIPS 2022) [GenerSpeech: Towards Style Transfer for Generalizable Out-Of-Domain Text-to-Speech](https://arxiv.org/pdf/2205.07211.pdf) [[Code]](https://github.com/Rongjiehuang/GenerSpeech) by Huang et al.

(NeurIPS 2022) [Learning Substructure Invariance for Out-of-Distribution Molecular Representations](https://openreview.net/pdf?id=2nWUNTnFijm) [[Code]](https://github.com/yangnianzu0515/MoleOOD) by Yang et al.

(NeurIPS 2022) [Evaluating Out-of-Distribution Performance on Document Image Classifiers](https://openreview.net/pdf?id=uDlkiCI5N7Y) by Larson et al.

(NeurIPS 2022) [OOD Link Prediction Generalization Capabilities of Message-Passing GNNs in Larger Test Graphs](https://openreview.net/pdf?id=q_AeTuxv02D) by Zhou et al.

(ICLR 2022) [Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution](https://arxiv.org/pdf/2202.10054.pdf) by Kumar et al.

(ICML 2022) [Improved StyleGAN-v2 based Inversion for Out-of-Distribution Images](https://proceedings.mlr.press/v162/subramanyam22a/subramanyam22a.pdf) by Subramanyam et al.

(NeurIPS 2021) [The Out-of-Distribution Problem in Explainability and Search Methods for Feature Importance Explanations](https://openreview.net/pdf?id=HCrp4pdk2i) [[Slides]](https://peterbhase.github.io/files/OODProblemAndSearchUberAI.pdf) by Hase et al.

(NeurIPS 2021) [POODLE: Improving Few-shot Learning via Penalizing Out-of-Distribution Samples](https://proceedings.neurips.cc/paper/2021/file/c91591a8d461c2869b9f535ded3e213e-Paper.pdf) [[Code]](https://github.com/lehduong/poodle) by Le et al.

(NeurIPS 2021) [Task-Agnostic Undesirable Feature Deactivation Using Out-of-Distribution Data](https://proceedings.neurips.cc/paper_files/paper/2021/file/21186d7b1482412ab14f0332b8aee119-Paper.pdf) [[Code]](https://github.com/kaist-dmlab/TAUFE) by Park et al.

(ICLR 2021) [Removing Undesirable Feature Contributions Using Out-of-Distribution Data](https://openreview.net/pdf?id=eIHYL6fpbkA) by Lee et al.

(ECCV 2020) [A Boundary Based Out-of-Distribution Classifier for Generalized Zero-Shot Learning](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690562.pdf) [[Code]](https://github.com/Chenxingyu1990/A-Boundary-Based-Out-of-Distribution-Classifier-for-Generalized-Zero-Shot-Learning) by Chen et al.
