# OOD Detection, Robustness, and Generalization [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/continuousml/Awesome-Out-Of-Distribution-Detection)


This repo aims to provide the most comprehensive, up-to-date, high-quality resource for **OOD detection, robustness, and generalization** in Deep Learning. Your one-stop shop for everything OOD is here. If you spot errors or omissions, please open an issue or contact me at continuousml@gmail.com.

[![Discord](https://img.shields.io/badge/Discord-5865F2.svg?style=for-the-badge&logo=Discord&logoColor=white)](https://discord.gg/WEzyEx36)


---

## Table of Contents
- [Researchers](#introduction)
- [Articles](#installation)
- [Talks](#usage)
- [Benchmarks, libraries, etc](#benchmarks-libraries-etc)
- [Surveys](#surveys)
- [Papers](#papers)
  - [OOD Detection](#ood-detection)
  - [OOD Robustness](#ood-robustness)
  - [OOD Generalization](#ood-generalization)
  - [OOD Everything else](#ood-everything-else)

---

# Researchers

- Dan Hendrycks [[Twitter]](https://twitter.com/DanHendrycks) [[Scholar]](https://scholar.google.com/citations?user=czyretsAAAAJ&hl=en)

- Sharon Y. Li [[Twitter]](https://twitter.com/SharonYixuanLi) [[Scholar]](https://scholar.google.com/citations?user=QSTd1oUAAAAJ)

- Yiyou Sun [[Twitter]](https://twitter.com/YiyouSun) [[Scholar]](https://scholar.google.com/citations?user=IKqlQo4AAAAJ)
  

# Articles

[(2020) Out-of-Distribution Detection in Deep Neural Networks](https://medium.com/analytics-vidhya/out-of-distribution-detection-in-deep-neural-networks-450da9ed7044) by Neeraj Varshney


# Talks

[(2022) Anomaly detection for OOD and novel category detection](https://www.youtube.com/watch?v=jFQUW2n8gpA&t=888s) by Thomas G. Dietterich

[(2022) Reliable Open-World Learning Against Out-of-distribution Data](https://www.youtube.com/watch?v=zaXiHljOl9Y&t=22s) by Sharon Yixuan Li

[(2022) Challenges and Opportunities in Out-of-distribution Detection](https://www.youtube.com/watch?v=X8XTOiNin0I&t=523s) by Sharon Yixuan Li

[(2022) Exploring the limits of out-of-distribution detection in vision and biomedical applications](https://www.youtube.com/watch?v=cSkBcqBhKVY) by Jie Ren

[(2021) Understanding the Failure Modes of Out-of-distribution Generalization](https://www.youtube.com/watch?v=DhPMq_550OE) by Vaishnavh Nagarajan

[(2020) Uncertainty and Out-of-Distribution Robustness in Deep Learning](https://www.youtube.com/watch?v=ssD7jNDIL2c&t=2293s) by Balaji Lakshminarayanan, Dustin Tran, and Jasper Snoek

# Benchmarks, libraries etc

[OpenOOD: Benchmarking Generalized OOD Detection](https://github.com/Jingkang50/OpenOOD)

[PyTorch Out-of-Distribution Detection](https://github.com/kkirchheim/pytorch-ood)

# Surveys

[Generalized Out-of-Distribution Detection: A Survey](https://arxiv.org/pdf/2110.11334.pdf) by Yang et al

[A Unified Survey on Anomaly, Novelty, Open-Set, and Out of-Distribution Detection: Solutions and Future Challenges](https://arxiv.org/pdf/2110.14051.pdf) by Salehi et al.

# Papers

> "Know thy literature"

## OOD Detection

(ArXiv 2023) [Neuron Activation Coverage: Rethinking Out-of-distribution Detection and Generalization](https://arxiv.org/pdf/2306.02879.pdf) by Liu et al.

(ArXiv 2023) [Characterizing Out-of-Distribution Error via Optimal Transport](https://arxiv.org/pdf/2305.15640.pdf) by Lu et al.

(CVPR 2023) [Distribution Shift Inversion for Out-of-Distribution Prediction](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Distribution_Shift_Inversion_for_Out-of-Distribution_Prediction_CVPR_2023_paper.pdf) [[Code]](https://github.com/yu-rp/Distribution-Shift-Iverson) by Yu et al.

(CVPR 2023) [Uncertainty-Aware Optimal Transport for Semantically Coherent Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2023/papers/Lu_Uncertainty-Aware_Optimal_Transport_for_Semantically_Coherent_Out-of-Distribution_Detection_CVPR_2023_paper.pdf) [[Code]](https://github.com/LuFan31/ET-OOD) by Lu et al.

(CVPR 2023) [GEN: Pushing the Limits of Softmax-Based Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_GEN_Pushing_the_Limits_of_Softmax-Based_Out-of-Distribution_Detection_CVPR_2023_paper.pdf) [[Video]](https://www.youtube.com/watch?v=v5f8_fme9aY) [[Code]](https://github.com/XixiLiu95/GEN) by Liu et al.

(CVPR 2023) [(NAP) Detection of Out-of-Distribution Samples Using Binary Neuron Activation Patterns](https://openaccess.thecvf.com/content/CVPR2023/papers/Olber_Detection_of_Out-of-Distribution_Samples_Using_Binary_Neuron_Activation_Patterns_CVPR_2023_paper.pdf) [[Code]](https://github.com/safednn-group/nap-ood) by Olber et al.

(CVPR 2023) [Decoupling MaxLogit for Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Decoupling_MaxLogit_for_Out-of-Distribution_Detection_CVPR_2023_paper.pdf) by Zhang and Xiang

(CVPR 2023) [Balanced Energy Regularization Loss for Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2023/papers/Choi_Balanced_Energy_Regularization_Loss_for_Out-of-Distribution_Detection_CVPR_2023_paper.pdf) [[Code]](https://github.com/hyunjunChhoi/Balanced_Energy) by Choi et al.

(CVPR 2023) [Rethinking Out-of-Distribution (OOD) Detection: Masked Image Modeling Is All You Need](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Rethinking_Out-of-Distribution_OOD_Detection_Masked_Image_Modeling_Is_All_You_CVPR_2023_paper.pdf) [[Code]](https://github.com/JulietLJY/MOOD) by Li et al.

(CVPR 2023) [LINe: Out-of-Distribution Detection by Leveraging Important Neurons](https://openaccess.thecvf.com/content/CVPR2023/papers/Ahn_LINe_Out-of-Distribution_Detection_by_Leveraging_Important_Neurons_CVPR_2023_paper.pdf) [[Code]](https://github.com/YongHyun-Ahn/LINe-Out-of-Distribution-Detection-by-Leveraging-Important-Neurons) by Ahn et al.

(ICLR 2023) ⭐⭐⭐⭐⭐ [A framework for benchmarking Class-out-of-distribution detection and its application to ImageNet](https://openreview.net/pdf?id=Iuubb9W6Jtk) [[Code]](https://github.com/mdabbah/COOD_benchmarking) by Galil et al.

(ICLR 2023) [Energy-based Out-of-Distribution Detection for Graph Neural Networks](https://openreview.net/pdf?id=zoz7Ze4STUL) [[Code]](https://github.com/qitianwu/GraphOOD-GNNSafe) by Wu et al.

(ICLR 2023) [The Tilted Variational Autoencoder: Improving Out-of-Distribution Detection](https://openreview.net/pdf?id=YlGsTZODyjz) [[Code]](https://github.com/anonconfsubaccount/tilted_prior) by Floto et al.

(ICLR 2023) [Out-of-Distribution Detection based on In-Distribution Data Patterns Memorization with Modern Hopfield Energy](https://openreview.net/pdf?id=KkazG4lgKL) by Zhang et al.

(ICLR 2023) [Out-of-Distribution Detection and Selective Generation for Conditional Language Models](https://openreview.net/pdf?id=kJUS5nD0vPB) by Ren et al.

(ICLR 2023) [Turning the Curse of Heterogeneity in Federated Learning into a Blessing for Out-of-Distribution Detection](https://openreview.net/pdf?id=mMNimwRb7Gr) by Yu et al.

(ICLR 2023) [Non-Parametric Outlier Synthesis](https://arxiv.org/pdf/2303.02966.pdf) [[Code]](https://github.com/deeplearning-wisc/npos) by Tao et al.

(ICLR 2023) [Out-of-distribution Detection with Implicit Outlier Transformation](https://arxiv.org/pdf/2303.05033.pdf) by Wang et al.

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

(ICLR 2022) [Extremely Simple Activation Shaping for Out-of-Distribution Detection](https://arxiv.org/pdf/2209.09858.pdf) [[Code]](https://github.com/andrijazz/ash) by Djurisic et al.

(ICLR 2022) [Revisiting flow generative models for Out-of-distribution detection](https://openreview.net/pdf?id=6y2KBh-0Fd9) by Jiang et al.

(ICLR 2022) [PI3NN: Out-of-distribution-aware Prediction Intervals from Three Neural Networks](https://openreview.net/pdf?id=NoB8YgRuoFU) [[Code]](https://github.com/liusiyan/UQnet) by Liu et al.

(ICLR 2022) [(ATC) Leveraging unlabeled data to predict out-of-distribution performance](https://openreview.net/pdf?id=o_HsiMPYh_x) by Garg et al.

(ICLR 2022) [Igeood: An Information Geometry Approach to Out-of-Distribution Detection](https://arxiv.org/pdf/2203.07798.pdf) [[Code]](https://github.com/igeood/Igeood) by Gomes et al.

(ICLR 2022) [How to Exploit Hyperspherical Embeddings for Out-of-Distribution Detection?](https://arxiv.org/pdf/2203.04450.pdf) [[Code]](https://github.com/deeplearning-wisc/cider) by Ming et al.

(ICLR 2022) [VOS: Learning What You Don't Know by Virtual Outlier Synthesis](https://arxiv.org/pdf/2202.01197.pdf) [[Code]](https://github.com/deeplearning-wisc/vos) by Du et al.

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

(ICML 2021) [Understanding Failures in Out-of-Distribution Detection with Deep Generative Models](http://proceedings.mlr.press/v139/zhang21g/zhang21g.pdf) by Zhang et al.

(ICCV 2021) [Semantically Coherent Out-of-Distribution Detection](https://arxiv.org/pdf/2108.11941.pdf) [[Project Page]](https://jingkang50.github.io/projects/scood) [[Code]](https://github.com/jingkang50/ICCV21_SCOOD) by Yang et al.

(ICCV 2021) [CODEs: Chamfer Out-of-Distribution Examples against Overconfidence Issue](https://arxiv.org/pdf/2108.06024.pdf) by Tang et al.

(ECCV 2021) [DICE: Leveraging Sparsification for Out-of-Distribution Detection](https://arxiv.org/pdf/2111.09805.pdf) [[Code]](https://github.com/deeplearning-wisc/dice) by Sun and Li

(CVPR 2020) [Deep Residual Flow for Out of Distribution Detection](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zisselman_Deep_Residual_Flow_for_Out_of_Distribution_Detection_CVPR_2020_paper.pdf) [[https://github.com/EvZissel/Residual-Flow]](Code) by Zisselman and Tamar

(CVPR 2020) [Generalized ODIN: Detecting Out-of-Distribution Image Without Learning From Out-of-Distribution Data] [[Code]](https://github.com/sayakpaul/Generalized-ODIN-TF) (https://arxiv.org/pdf/2002.11297.pdf) by Hsu et al.

(NeurIPS 2020) [CSI: Novelty Detection via Contrastive Learning on Distributionally Shifted Instances](https://arxiv.org/pdf/2007.08176.pdf) [[Code]](https://github.com/alinlab/CSI) by Tack et al.

(NeurIPS 2020) ⭐⭐⭐⭐⭐ [Energy-based Out-of-distribution Detection](https://arxiv.org/pdf/2010.03759.pdf) [[Code]](https://github.com/wetliu/energy_ood) by Liu et al.

(NeurIPS 2020) [OOD-MAML: Meta-Learning for Few-Shot Out-of-Distribution Detection and Classification](https://proceedings.neurips.cc/paper/2020/file/28e209b61a52482a0ae1cb9f5959c792-Paper.pdf) [[Video]](https://slideslive.com/38935997/oodmaml-metalearning-for-fewshot-outofdistribution-detection-and-classification) by Jeong and Kim

(NeurIPS 2020) [Towards Maximizing the Representation Gap between In-Domain & Out-of-Distribution Examples](https://proceedings.neurips.cc/paper/2020/file/68d3743587f71fbaa5062152985aff40-Paper.pdf) [[Code]](https://github.com/jayjaynandy/maximize-representation-gap) by Nandy et al.

(NeurIPS 2020) [Likelihood Regret: An Out-of-Distribution Detection Score For Variational Auto-encoder](https://proceedings.neurips.cc/paper/2020/file/eddea82ad2755b24c4e168c5fc2ebd40-Paper.pdf) [[Code]](https://github.com/XavierXiao/Likelihood-Regret) by Xiao et al.

(NeurIPS 2020) ⭐⭐⭐⭐⭐ [Why Normalizing Flows Fail to Detect Out-of-Distribution Data](https://proceedings.neurips.cc/paper/2020/file/ecb9fe2fbb99c31f567e9823e884dbec-Paper.pdf) [[Code]](https://github.com/PolinaKirichenko/flows_ood) by Kirichenko et al.

(ICML 2020) [Detecting Out-of-Distribution Examples with Gram Matrices](http://proceedings.mlr.press/v119/sastry20a/sastry20a.pdf) [[Code]](https://github.com/VectorInstitute/gram-ood-detection) by Sastry and Oore

(CVPR 2019) [Out-Of-Distribution Detection for Generalized Zero-Shot Action Recognition](https://openaccess.thecvf.com/content_CVPR_2019/papers/Mandal_Out-Of-Distribution_Detection_for_Generalized_Zero-Shot_Action_Recognition_CVPR_2019_paper.pdf) [[Code]](https://github.com/naraysa/gzsl-od) by Mandal et al.

(NeurIPS 2019) [Likelihood Ratios for Out-of-Distribution Detection](https://arxiv.org/pdf/1906.02845.pdf) [[Video]](https://www.youtube.com/watch?v=-FduW9ZWAR4) by Ren et al.

(ICCV 2019) [Unsupervised Out-of-Distribution Detection by Maximum Classifier Discrepancy](https://arxiv.org/pdf/1908.04951.pdf) [[Code]](https://github.com/Mephisto405/Unsupervised-Out-of-Distribution-Detection-by-Maximum-Classifier-Discrepancy) by Yu and Aizawa

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

(ICLR 2023) [Improving Out-of-distribution Generalization with Indirection Representations](https://openreview.net/pdf?id=0f-0I6RFAch) by Pham et al.

(ICLR 2023) [Topology-aware Robust Optimization for Out-of-Distribution Generalization](https://openreview.net/pdf?id=ylMq8MBnAp) [[Code]](https://github.com/joffery/TRO) by Qiao and Peng

(ICLR 2023) ⭐⭐⭐⭐⭐[Modeling the Data-Generating Process is Necessary for Out-of-Distribution Generalization](https://openreview.net/pdf?id=uyqks-LILZX) by Kaur et al.

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
