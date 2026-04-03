# OOD Machine Learning: Detection, Robustness, and Generalization

<img width="1665" alt="image" src="https://github.com/user-attachments/assets/ebb0d89c-e929-4c7f-af5e-d7e39547c487">

<p align="center">
    <a href="https://discord.gg/gdap3n9jZ2">
        <img src="https://img.shields.io/badge/Discord-5865F2.svg?style=for-the-badge&logo=Discord&logoColor=white" alt="Discord">
    </a>
    <a href="https://github.com/continuousml/Awesome-Out-Of-Distribution-Detection">
        <img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=for-the-badge" alt="Maintenance">
    </a>
    <a href="https://doi.org/10.5281/zenodo.17527926">
        <img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17527926-blue.svg?style=for-the-badge" alt="DOI">
    </a>
</p>

This repository aims to provide the most comprehensive, up-to-date, high-quality resource for **OOD detection, robustness, and generalization** in Machine Learning/Deep Learning. Your one-stop shop for everything OOD is here.

---

# Citation

If you use this repository, please consider citing:

```bibtex
@software{tran2023ood,
  author       = {Huy Tran},
  title        = {OOD Machine Learning: Detection, Robustness, and Generalization},
  year         = {2023},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17527926},
  url          = {https://github.com/huytransformer/Awesome-Out-Of-Distribution-Detection}
}
```

---

# Primer🌟

Out-of-distribution (OOD) problems present critical challenges in modern machine learning systems. 🤖 As AI models are increasingly deployed in real-world applications, they often encounter data that differs significantly from what they were trained on. This can lead to unexpected and potentially dangerous failures ⚠️. The field of OOD research encompasses several key areas:

- 🔍 ***OOD detection*** aims to identify when a model is presented with inputs that deviate from its training distribution. This allows systems to flag unusual cases for human review or fallback strategies.
- 💪 ***OOD robustness*** focuses on maintaining reliable performance even when test data differs from training data in important ways.
- 🌐 ***OOD generalization*** seeks to develop models and techniques that can successfully extrapolate to novel domains and tasks without requiring retraining.

Together, these interconnected areas of research are crucial for building AI systems that are safe, reliable, and adaptable in open-world environments. 🌍 Recent work has made significant progress, but major challenges remain in areas such as handling complex input spaces, scaling to large models, and providing theoretical guarantees, among others. Advancing the state-of-the-art in OOD machine learning is essential as AI becomes more pervasive in high-stakes domains like healthcare 🏥, autonomous vehicles 🚗, and critical infrastructure 🏙️.

<p align="center">
  <img src="https://github.com/user-attachments/assets/3797dd2e-2587-4898-b2da-3fa8208a4d6c" alt="image" width="400px">
</p>

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

Selected researchers with sustained contributions to OOD detection, open-world learning, robustness under shift, and OOD generalization:

- [Sharon Yixuan Li](https://pages.cs.wisc.edu/~sharonli/)
- [Dan Hendrycks](https://danhendrycks.com/)
- [Thomas G. Dietterich](https://web.engr.oregonstate.edu/~tgd/)
- [Balaji Lakshminarayanan](https://www.gatsby.ucl.ac.uk/~balaji/index.html)
- [Yarin Gal](https://www.cs.ox.ac.uk/people/yarin.gal/website/)
- [Martin Arjovsky](https://arxiv.org/abs/2103.02667)
- [Pang Wei Koh](https://koh.pw/)
- [Peng Cui](https://pengcui.thumedialab.com/)
- [Jie Ren](https://jessieren.github.io/)
- [Alexander Meinke](https://www.alexmeinke.de/)
- [Jingkang Yang](https://jingkangyang.com/)
- [Kimin Lee](https://openreview.net/profile?id=~Kimin_Lee1)
- [Yao Qin](https://www.ece.ucsb.edu/people/faculty/yao-qin)
- [Kaiyang Zhou](https://kaiyangzhou.github.io/)
- [David Lopez-Paz](https://openreview.net/profile?id=~David_Lopez-Paz1)
- [Vaishnavh Nagarajan](https://vaishnavh.github.io/home/index.html)

# Articles

[(2022) Data Distribution Shifts and Monitoring](https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html) by Chip Huyen

[(2023) OpenOOD v1.5 Methods & Benchmarks Overview](https://github.com/Jingkang50/OpenOOD/wiki/OpenOOD-v1.5-methods-%26-benchmarks-overview) by the OpenOOD team

[(2020) Adapting on the Fly to Test-Time Distribution Shift](https://bair.berkeley.edu/blog/2020/11/05/arm/) by BAIR

[(2022) Keeping Learning-Based Control Safe by Regulating Distributional Shift](https://bair.berkeley.edu/blog/2022/09/19/ldm-control/) by BAIR

[(2021) Machine Learning Under Distributional Shifts](https://ai.stanford.edu/?courses=machine-learning-under-distributional-shifts) by Stanford AI Lab


# Talks

[(2024) NeurIPS Tutorial: Out-of-Distribution Generalization: Shortcuts, Spuriousness, and Stability](https://neurips.cc/virtual/2024/tutorial/99523)

[(2024) Lec 17. Generalization: Out-of-Distribution (OOD)](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/resources/mit6_7960f24_lec17_mp4/) by Phillip Isola, Sara Beery, and Jeremy Bernstein

[(2024) Intro to Out-of-Distribution Detection](https://uw-madison-datascience.github.io/ML-X-Nexus/Learn/Videos/OOD-detection.html) by Sharon Yixuan Li

[(2023) How to Detect Out-of-Distribution Data in the Wild?](https://www.youtube.com/watch?v=Ga09-9JItxs) by Sharon Yixuan Li

[(2022) Anomaly Detection for OOD and Novel Category Detection](https://www.youtube.com/watch?v=jFQUW2n8gpA&t=888s) by Thomas G. Dietterich

[(2022) Reliable Open-World Learning Against Out-of-Distribution Data](https://www.youtube.com/watch?v=zaXiHljOl9Y&t=22s) by Sharon Yixuan Li

[(2022) Challenges and Opportunities in Out-of-Distribution Detection](https://www.youtube.com/watch?v=X8XTOiNin0I&t=523s) by Sharon Yixuan Li

[(2022) Exploring the Limits of Out-of-Distribution Detection in Vision and Biomedical Applications](https://www.youtube.com/watch?v=cSkBcqBhKVY) by Jie Ren

[(2021) Understanding the Failure Modes of Out-of-distribution Generalization](https://www.youtube.com/watch?v=DhPMq_550OE) by Vaishnavh Nagarajan

[(2020) Practical Uncertainty Estimation and Out-of-Distribution Robustness in Deep Learning](https://neurips.cc/virtual/2020/tutorial/16649) by Dustin Tran, Balaji Lakshminarayanan, and Jasper Snoek

# Benchmarks, libraries, datasets, etc

## Benchmarks

[OpenOOD](https://github.com/Jingkang50/OpenOOD): canonical benchmark suite for generalized OOD detection across classic image settings and methods

[OpenOOD-VLM](https://github.com/YBZh/OpenOOD-VLM): benchmark suite for generalized OOD detection in the vision-language model setting

[WILDS](https://wilds.stanford.edu/): canonical real-world benchmark for distribution shift across vision, text, graphs, and biology

[DomainBed](https://github.com/facebookresearch/DomainBed): standard evaluation suite for domain generalization and out-of-domain robustness

[GOOD](https://github.com/divelab/GOOD): leading benchmark suite for graph out-of-distribution and graph domain generalization

[DrugOOD](https://drugood.github.io/): benchmark and platform for out-of-distribution generalization in AI-aided drug discovery

[TableShift](https://tableshift.org/): benchmark and toolkit for real-world tabular distribution shift

[OODRobustBench](https://oodrobustbench.github.io/): benchmark for adversarial robustness under natural distribution shift

[OOD NLP](https://github.com/lifan-yuan/OOD_NLP): benchmark suite for out-of-distribution robustness and evaluation in NLP

[NINCO](https://github.com/j-cb/NINCO): ImageNet-scale near-OOD dataset for modern large-scale visual evaluation

[WOODS](https://woods-benchmarks.github.io/): benchmark suite for out-of-distribution generalization in time-series tasks

[OpenMIBOOD](https://github.com/remic-othr/OpenMIBOOD): medical imaging benchmark suite for OOD detection under covariate, near-OOD, and far-OOD shifts

[Semantic Shift Benchmark (SSB)](https://openreview.net/forum?id=2ql76f4zE3): benchmark for semantic-shift, open-set, and class-level OOD evaluation

## Libraries

[OpenOOD](https://github.com/Jingkang50/OpenOOD): end-to-end OOD framework for training, benchmarking, and evaluating a large set of methods

[PyTorch Out-of-Distribution Detection](https://github.com/kkirchheim/pytorch-ood): practical PyTorch library with detectors, losses, datasets, and evaluation utilities

[OODEEL](https://github.com/deel-ai/oodeel): compact post-hoc OOD toolkit for TensorFlow and PyTorch image classifiers

[TorchUncertainty](https://github.com/ENSTA-U2IS-AI/torch-uncertainty): broader uncertainty framework with strong support for OOD metrics, evaluation, and tutorials

[Alibi Detect](https://github.com/SeldonIO/alibi-detect): high-quality toolkit for outlier, adversarial, and drift detection across modalities


# Surveys

(TMLR 2025) [Generalized Out-of-Distribution Detection and Beyond in Vision Language Model Era: A Survey](https://openreview.net/forum?id=FO3IA4lUEY) by Miyai et al.

(2021) [Generalized Out-of-Distribution Detection: A Survey](https://arxiv.org/abs/2110.11334) by Yang et al.

(TMLR 2023) [A Survey on Out-of-distribution Detection in NLP](https://arxiv.org/abs/2305.03236) by Lang et al.

(ACM CSUR 2026) [Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances](https://doi.org/10.1145/3760390) by Lu et al.

(2025) [Out-of-Distribution Detection on Graphs: A Survey](https://arxiv.org/abs/2502.08105) by Cai et al.

(TPAMI 2023) [Domain Generalization: A Survey](https://doi.org/10.1109/TPAMI.2022.3195549) by Zhou et al.

(TPAMI 2025) [Out-of-Distribution Generalization on Graphs: A Survey](https://doi.org/10.1109/TPAMI.2025.3593897) by Li et al.

(IJCV 2025) [Dissecting Out-of-Distribution Detection and Open-Set Recognition: A Critical Analysis of Methods and Benchmarks](https://link.springer.com/article/10.1007/s11263-024-02222-4) by Wang et al.

# Theses

[(2021) Out of Distribution Generalization in Machine Learning](https://arxiv.org/abs/2103.02667) by Martin Arjovsky

[(2023) Robust Out-of-Distribution Detection in Deep Classifiers](https://publikationen.uni-tuebingen.de/xmlui/handle/10900/141438) by Alexander Meinke

[(2023) Detecting and Learning Out-of-Distribution Data in the Open-world: Algorithm and Theory](https://digital.library.wisc.edu/1711.dl/SA2T74ZEANBC39C) by Yiyou Sun

[(2024) Towards Reliable Foundation Models in the Open World](https://digital.library.wisc.edu/1711.dl/C4XVYFQBSF2HD85) by Yifei Ming

[(2025) Foundations of Unknown-aware Machine Learning](https://digital.library.wisc.edu/1711.dl/AZJFTB7NC4OXAF9A) by Xuefeng Du

[(2025) Learning to Generalize Across Distribution Shifts](https://publikationen.uni-tuebingen.de/xmlui/handle/10900/163498) by Frederik Joshua Träuble

[(2026) Out-of-Distribution Detection, Sharpness, and Unlearning: Advancing Robust and Trustworthy Deep Learning](https://publikationen.uni-tuebingen.de/xmlui/handle/10900/176056) by Maximilian Peter Müller

# Papers

> "Know thy literature"

## OOD Detection

(ICLR 2026) [Let OOD Feature Exploring Vast Predefined Classifiers](https://openreview.net/forum?id=DtkIH5TKe6) by Xia et al.

(ICLR 2026) [Towards a Certificate of Trust: Task-Aware OOD Detection for Scientific AI](https://openreview.net/forum?id=2RuSWLQK82) by Jovanovic et al.

(ICLR 2026) [Fisher-Rao Sensitivity for Out-of-Distribution Detection in Deep Neural Networks](https://openreview.net/forum?id=GEtOzC4MIi) by Nguyen et al.

(ICLR 2026) [EigenScore: OOD Detection using Posterior Covariance in Diffusion Models](https://openreview.net/forum?id=Dq64kthckN) by Shoushtari et al.

(ICLR 2026) [AP-OOD: Attention Pooling for Out-of-Distribution Detection](https://openreview.net/forum?id=fEYonozhKk) by Hofmann et al.

(ICLR 2026) [SCOPED: Score-Curvature Out-of-distribution Proximity Evaluator for Diffusion](https://openreview.net/forum?id=TMLiG9Rk2J) by Barkley et al.

(ICLR 2026) [GradPCA: Leveraging NTK Alignment for Reliable Out-of-Distribution Detection](https://openreview.net/forum?id=7rvMexIZA1) by Seleznova et al.

(TMLR 2026) [A Bayesian Nonparametric Perspective on Mahalanobis Distance for Out of Distribution Detection](https://openreview.net/forum?id=w3bMXPMDW1) [[Code]](https://github.com/rwl93/bnp4ood) by Linderman et al.

(ICCV 2025) [A Unified Interpretation of Training-Time Out-of-Distribution Detection](https://openaccess.thecvf.com/content/ICCV2025/html/Cheng_A_Unified_Interpretation_of_Training-Time_Out-of-Distribution_Detection_ICCV_2025_paper.html) by Cheng et al.

(TMLR 2025) [LEGO-Learn: Label-Efficient Graph Open-Set Learning](https://openreview.net/forum?id=J6oxTJPOyN) by Xu et al.


(AISTATS 2025) [Your Finetuned Large Language Model is Already a Powerful Out-of-distribution Detector](https://proceedings.mlr.press/v258/zhang25h.html) by Zhang et al.

(AAAI 2025) [HYBOOD: a hybrid generative model for out-of-distribution detection with corruption estimation](https://aaai.org/wp-content/uploads/2025/02/AAAI-25-Main-Track-Poster-Schedule.pdf) by Lee et al.

(AAAI 2025) [Diffusion-Based Semantic Outlier Generation via Nuisance Awareness for Out-of-Distribution Detection](https://aaai.org/wp-content/uploads/2025/01/AAAI-25-Poster-Schedule.pdf) by Yoon et al.

(AAAI 2025) [Filter or Compensate: Towards Invariant Representation from Distribution Shift for Anomaly Detection](https://aaai.org/wp-content/uploads/2025/02/AAAI-25-Main-Track-Poster-Schedule.pdf) by Chen et al.


(NeurIPS 2025) [ΔEnergy: Optimizing Energy Change During Vision-Language Alignment Improves both OOD Detection and OOD Generalization](https://neurips.cc/virtual/2025/poster/116579) [[Code]](https://github.com/LinLLLL/DeltaEnergy) by Zhu et al.

(NeurIPS 2025) [An Information-theoretical Framework for Understanding Out-of-distribution Detection with Pretrained Vision-Language Models](https://neurips.cc/virtual/2025/poster/116939) by Peng et al.

(NeurIPS 2025) [DIsoN: Decentralized Isolation Networks for Out-of-Distribution Detection in Medical Imaging](https://neurips.cc/virtual/2025/poster/115312) [[Code]](https://github.com/FelixWag/DIsoN) by Wagner et al.

(NeurIPS 2025) [DualCnst: Enhancing Zero-Shot Out-of-Distribution Detection via Text-Image Consistency in Vision-Language Models](https://neurips.cc/virtual/2025/poster/119757) by Le et al.

(NeurIPS 2025) [Geometric Logit Decoupling for Energy-Based Graph Out-of-distribution Detection](https://neurips.cc/virtual/2025/poster/116743) by Wang et al.

(NeurIPS 2025) [GOOD: Training-Free Guided Diffusion Sampling for Out-of-Distribution Detection](https://neurips.cc/virtual/2025/poster/116282) by Gao et al.

(NeurIPS 2025) [Harnessing Feature Resonance under Arbitrary Target Alignment for Out-of-Distribution Node Detection](https://neurips.cc/virtual/2025/poster/118716) by Yang et al.

(NeurIPS 2025) [Revisiting Logit Distributions for Reliable Out-of-Distribution Detection](https://neurips.cc/virtual/2025/poster/119045) by Liang et al.

(NeurIPS 2025) [Spurious-Aware Prototype Refinement for Reliable Out-of-Distribution Detection](https://neurips.cc/virtual/2025/poster/115233) [[Project Page]](https://sharif-ml-lab.github.io/SPROD/) by Zohrabi et al.

(NeurIPS 2025) [Redundancy-Aware Test-Time Graph Out-of-Distribution Detection](https://neurips.cc/virtual/2025/poster/119287) [[Code]](https://github.com/name-is-what/RedOUT) by Hou et al.

(NeurIPS 2025) [Double Descent Meets Out-of-Distribution Detection: Theoretical Insights and Empirical Analysis on the Role of Model Complexity](https://neurips.cc/virtual/2025/poster/117449) by Ben Ammar et al.

(ICCV 2025) [Activation Subspaces for Out-of-Distribution Detection](https://openaccess.thecvf.com/content/ICCV2025/html/Zongur_Activation_Subspaces_for_Out-of-Distribution_Detection_ICCV_2025_paper.html) by Zongur et al.

(ICCV 2025) [Gradient Short-Circuit: Efficient Out-of-Distribution Detection via Feature Intervention](https://openaccess.thecvf.com/content/ICCV2025/html/Gu_Gradient_Short-Circuit_Efficient_Out-of-Distribution_Detection_via_Feature_Intervention_ICCV_2025_paper.html) by Gu et al.

(ICCV 2025) [Synthesizing Near-Boundary OOD Samples for Out-of-Distribution Detection](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Synthesizing_Near-Boundary_OOD_Samples_for_Out-of-Distribution_Detection_ICCV_2025_paper.html) by Li et al.

(ICCV 2025) [Diagnosing Pretrained Models for Out-of-distribution Detection](https://openaccess.thecvf.com/content/ICCV2025/html/Xiong_Diagnosing_Pretrained_Models_for_Out-of-distribution_Detection_ICCV_2025_paper.html) by Xiong et al.

(ICCV 2025) [FA: Forced Prompt Learning of Vision-Language Models for Out-of-Distribution Detection](https://openaccess.thecvf.com/content/ICCV2025/html/Lu_FA_Forced_Prompt_Learning_of_Vision-Language_Models_for_Out-of-Distribution_Detection_ICCV_2025_paper.html) by Lu et al.

(ICCV 2025) [Adaptive Prompt Learning via Gaussian Outlier Synthesis for Out-of-distribution Detection](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Adaptive_Prompt_Learning_via_Gaussian_Outlier_Synthesis_for_Out-of-distribution_Detection_ICCV_2025_paper.html) by Zhang et al.

(ICCV 2025) [Secure On-Device Video OOD Detection Without Backpropagation](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Secure_On-Device_Video_OOD_Detection_Without_Backpropagation_ICCV_2025_paper.html) by Li et al.

(ICML 2025) [Splitting & Integrating: Out-of-Distribution Detection via Adversarial Gradient Attribution](https://proceedings.mlr.press/v267/zhang25by.html) by Zhang et al.

(ICML 2025) [BOOD: Boundary-based Out-Of-Distribution Data Generation](https://proceedings.mlr.press/v267/liao25g.html) by Liao et al.

(ICML 2025) [Improving Out-of-Distribution Detection via Dynamic Covariance Calibration](https://proceedings.mlr.press/v267/guo25m.html) by Guo et al.

(ICML 2025) [An Online Statistical Framework for Out-of-Distribution Detection](https://proceedings.mlr.press/v267/ma25y.html) by Ma et al.

(ICML 2025) [Position: Supervised Classifiers Answer the Wrong Questions for OOD Detection](https://proceedings.mlr.press/v267/li25ec.html) by Li et al.

(ICML 2025) [Adaptive Multi-prompt Contrastive Network for Few-shot Out-of-distribution Detection](https://proceedings.mlr.press/v267/fang25a.html) by Fang et al.

(ICML 2025) [Improving Out-of-Distribution Detection with Markov Logic Networks](https://proceedings.mlr.press/v267/kirchheim25a.html) by Kirchheim and Ortmeier

(ICML 2025) [A Variational Information Theoretic Approach to Out-of-Distribution Detection](https://proceedings.mlr.press/v267/mondal25a.html) by Mondal et al.

(ICLR 2025) [Advancing Out-of-Distribution Detection via Local Neuroplasticity](https://openreview.net/forum?id=1F8xTfv6ah) by Canevaro et al.

(ICLR 2025) [MetaOOD: Automatic Selection of OOD Detection Models](https://openreview.net/forum?id=9qpdDiDQ2H) by Qin et al.

(ICLR 2025) [Spreading Out-of-Distribution Detection on Graphs](https://openreview.net/forum?id=p1TBYyqy8v) by Um et al.

(ICLR 2025) [Outlier Synthesis via Hamiltonian Monte Carlo for Out-of-Distribution Detection](https://openreview.net/forum?id=N6ba2xsmds) by Li and Zhang

(ICLR 2025) [Local-Prompt: Extensible Local Prompts for Few-Shot Out-of-Distribution Detection](https://openreview.net/forum?id=Ew3VifXaxZ) by Zeng et al.

(ICLR 2025) [GOLD: Graph Out-of-Distribution Detection via Implicit Adversarial Latent Generation](https://openreview.net/forum?id=y5einmJ0Yx) by Wang et al.

(CVPR 2025) [Leveraging Perturbation Robustness to Enhance Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Leveraging_Perturbation_Robustness_to_Enhance_Out-of-Distribution_Detection_CVPR_2025_paper.html) [[Code]](https://github.com/wenxichen2746/Perturbation-Rectified-OOD-Detection) by Chen et al.

(CVPR 2025) [Detecting Out-of-Distribution Through the Lens of Neural Collapse](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Detecting_Out-of-Distribution_Through_the_Lens_of_Neural_Collapse_CVPR_2025_paper.html) [[Code]](https://github.com/litianliu/NCI-OOD) by Liu et al.

(CVPR 2025) [Dual Energy-Based Model with Open-World Uncertainty Estimation for Out-of-distribution Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Dual_Energy-Based_Model_with_Open-World_Uncertainty_Estimation_for_Out-of-distribution_Detection_CVPR_2025_paper.html) by Chen et al.

(CVPR 2025) [CADRef: Robust Out-of-Distribution Detection via Class-Aware Decoupled Relative Feature Leveraging](https://openaccess.thecvf.com/content/CVPR2025/html/Ling_CADRef_Robust_Out-of-Distribution_Detection_via_Class-Aware_Decoupled_Relative_Feature_Leveraging_CVPR_2025_paper.html) by Ling et al.

(CVPR 2025) [DPU: Dynamic Prototype Updating for Multimodal Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Li_DPU_Dynamic_Prototype_Updating_for_Multimodal_Out-of-Distribution_Detection_CVPR_2025_paper.html) by Li et al.

(CVPR 2025) [ProHOC: Probabilistic Hierarchical Out-of-Distribution Classification via Multi-Depth Networks](https://openaccess.thecvf.com/content/CVPR2025/html/Wallin_ProHOC_Probabilistic_Hierarchical_Out-of-Distribution_Classification_via_Multi-Depth_Networks_CVPR_2025_paper.html) [[Code]](https://github.com/walline/prohoc) by Wallin et al.

(CVPR 2025) [OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Gutbrod_OpenMIBOOD_Open_Medical_Imaging_Benchmarks_for_Out-Of-Distribution_Detection_CVPR_2025_paper.html) [[Code]](https://github.com/remic-othr/OpenMIBOOD) by Gutbrod et al.

(TMLR 2025) [GROOD: GRadient-Aware Out-of-Distribution Detection](https://openreview.net/forum?id=2V7itvvMVJ) [[Code]](https://github.com/mostafaelaraby/Gradient-Aware-OOD-Detection) by ElAraby et al.

(JMLR 2025) [Deep Out-of-Distribution Uncertainty Quantification via Weight Entropy Maximization](https://jmlr.org/papers/v26/23-1359.html) by de Mathelin et al.

(NeurIPS 2025) [Mysteries of the Deep: Role of Intermediate Representations in Out of Distribution Detection](https://neurips.cc/virtual/2025/poster/118891) by Meza de la Jara et al.

(NeurIPS 2025) [Extremely Simple Multimodal Outlier Synthesis for Out-of-Distribution Detection and Segmentation](https://neurips.cc/virtual/2025/poster/116022) by Liu et al.

(CVPR 2025) [Beyond Clean Training Data: A Versatile and Model-Agnostic Framework for Out-of-Distribution Detection with Contaminated Training Data](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Beyond_Clean_Training_Data_A_Versatile_and_Model-Agnostic_Framework_for_CVPR_2025_paper.html) by Li et al.

(CVPR 2025) [H2ST: Hierarchical Two-Sample Tests for Continual Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_H2ST_Hierarchical_Two-Sample_Tests_for_Continual_Out-of-Distribution_Detection_CVPR_2025_paper.html) by Liu et al.

(CVPR 2025) [Enhanced OoD Detection through Cross-Modal Alignment of Multi-Modal Representations](https://openaccess.thecvf.com/content/CVPR2025/html/Kim_Enhanced_OoD_Detection_through_Cross-Modal_Alignment_of_Multi-Modal_Representations_CVPR_2025_paper.html) by Kim et al.

(CVPR 2025) [Overcoming Shortcut Problem in VLM for Robust Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_Overcoming_Shortcut_Problem_in_VLM_for_Robust_Out-of-Distribution_Detection_CVPR_2025_paper.html) by Xu et al.

(CVPR 2025) [OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_OODD_Test-time_Out-of-Distribution_Detection_with_Dynamic_Dictionary_CVPR_2025_paper.html) by Yang et al.

(CVPR 2025) [Simplification Is All You Need against Out-of-Distribution Overconfidence](https://openaccess.thecvf.com/content/CVPR2025/html/Tang_Simplification_Is_All_You_Need_against_Out-of-Distribution_Overconfidence_CVPR_2025_paper.html) by Tang et al.

(ICML 2025) [A Closer Look at Generalized BH Algorithm for Out-of-Distribution Detection](https://proceedings.mlr.press/v267/ma25t.html) by Ma et al.

(ICML 2025) [Mahalanobis++: Improving OOD Detection via Feature Normalization](https://proceedings.mlr.press/v267/muller25a.html) by Müller and Hein

(ICML 2025) [Strengthen Out-of-Distribution Detection Capability with Progressive Self-Knowledge Distillation](https://proceedings.mlr.press/v267/yang25ap.html) by Yang and Xu

(UAI 2025) [Optimal Zero-shot Regret Minimization for Selective Classification with Out-of-Distribution Detection](https://proceedings.mlr.press/v286/dadalto-camara-gomes25a.html) by Gomes and Romanelli

(UAI 2024) [Latent Representation Entropy Density for Distribution Shift Detection](https://proceedings.mlr.press/v244/arnez24a.html) by Arnez et al.

(UAI 2024) [Mitigating Overconfidence in Out-of-Distribution Detection by Capturing Extreme Activations](https://proceedings.mlr.press/v244/azizmalayeri24a.html) by Azizmalayeri et al.

(ICML 2024) [RODEO: Robust Outlier Detection via Exposing Adaptive Out-of-Distribution Samples](https://proceedings.mlr.press/v235/mirzaei24a.html) by Mirzaei et al.

(JMLR 2024) [On the Learnability of Out-of-distribution Detection](https://jmlr.org/papers/v25/23-1257.html) by Fang et al.

(ECCV 2024) [Vision-Language Dual-Pattern Matching for Out-of-Distribution Detection](https://eccv2024.ecva.net/virtual/2024/poster/2615) by Zhang et al.

(ECCV 2024) [ProSub: Probabilistic Open-Set Semi-Supervised Learning with Subspace-Based Out-of-Distribution Detection](https://eccv2024.ecva.net/virtual/2024/poster/1192) by Li et al.

(ECCV 2024) [PixOOD: Pixel-Level Out-of-Distribution Detection](https://eccv2024.ecva.net/virtual/2024/poster/828) by Bansal et al.

(ECCV 2024) [Gradient-based Out-of-Distribution Detection](https://eccv2024.ecva.net/virtual/2024/poster/1930) by Yoo et al.

(ECCV 2024) [LAPT: Label-driven Automated Prompt Tuning for OOD Detection with Vision-Language Models](https://eccv2024.ecva.net/virtual/2024/poster/1567) by Wang et al.

(ECCV 2024) [TAG: Text Prompt Augmentation for Zero-Shot Out-of-Distribution Detection](https://eccv2024.ecva.net/virtual/2024/poster/772) by Chen et al.

(ECCV 2024) [Diffusion for Out-of-Distribution Detection on Road Scenes and Beyond](https://eccv2024.ecva.net/virtual/2024/poster/1152) by Xie et al.

(ECCV 2024) [Placing Objects in Context via Inpainting for Out-of-distribution Segmentation](https://eccv2024.ecva.net/virtual/2024/poster/1910) by Sakaridis et al.

(AAAI 2024) [Out-of-Distribution Detection in Long-Tailed Recognition with Calibrated Outlier Class Learning](https://aaai.org/wp-content/uploads/2024/01/AAAI_Main-Track_2024-01-04.pdf) by Miao et al.

(AAAI 2024) [TagFog: Textual Anchor Guidance and Fake Outlier Generation for Visual Out-of-Distribution Detection](https://aaai.org/wp-content/uploads/2024/02/AAAI-24-Saturday-Poster-Sessions-2.pdf) by Chen et al.

(AAAI 2024) [Exploiting Discrepancy in Feature Statistic for Out-of-Distribution Detection](https://aaai.org/wp-content/uploads/2024/02/AAAI-24-Saturday-Poster-Sessions-1.pdf) by Guan et al.

(NeurIPS 2024) [Learning Structured Representations with Hyperbolic Embeddings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a5d2da376bab7624b3caeb9f78fcaa2f-Abstract-Conference.html) [[Code]](https://github.com/uiuctml/HypStructure) by Sinha et al.

(NeurIPS 2024) [What If the Input is Expanded in OOD Detection?](https://proceedings.neurips.cc/paper_files/paper/2024/file/25cc3adf8c85f7c70989cb8a97a691a7-Paper-Conference.pdf) [[Code]](https://github.com/tmlr-group/CoVer) by Zhang et al.

(NeurIPS 2024) [AdaNeg: Adaptive Negative Proxy Guided OOD Detection with Vision-Language Models](https://openreview.net/pdf?id=vS5NC7jtCI) by Zhang et al.

(NeurIPS 2024) [Revisiting Score Propagation in Graph Out-of-Distribution Detection](https://neurips.cc/virtual/2024/poster/93946) by Ma et al.

(NeurIPS 2024) [AHA: Human-Assisted Out-of-Distribution Generalization and Detection](https://neurips.cc/virtual/2024/poster/96691) by Bai et al.

(NeurIPS 2024) [Long-Tailed Out-of-Distribution Detection via Normalized Outlier Distribution Adaptation](https://neurips.cc/virtual/2024/poster/94403) by Miao et al.

(NeurIPS 2024) [Embedding Trajectory for Out-of-Distribution Detection in Mathematical Reasoning](https://neurips.cc/virtual/2024/poster/94059) by Wang et al.

(NeurIPS 2024) [Rethinking Out-of-Distribution Detection on Imbalanced Data Distribution](https://neurips.cc/virtual/2024/poster/96028) by Liu et al.

(NeurIPS 2024) [Learning to Shape In-distribution Feature Space for Out-of-distribution Detection](https://neurips.cc/virtual/2024/poster/96879) by Zhang et al.

(NeurIPS 2024) [Self-Calibrated Tuning of Vision-Language Models for Out-of-Distribution Detection](https://neurips.cc/virtual/2024/poster/93172) by Yu et al.

(NeurIPS 2024) [FOOGD: Federated Collaboration for Both Out-of-distribution Generalization and Detection](https://neurips.cc/virtual/2024/poster/96103) by Liao et al.

(NeurIPS 2024) [Diffusion-based Layer-wise Semantic Reconstruction for Unsupervised Out-of-Distribution Detection](https://neurips.cc/virtual/2024/poster/96716) by Yang et al.

(NeurIPS 2024) [Out-of-Distribution Detection with a Single Unconditional Diffusion Model](https://neurips.cc/virtual/2024/poster/93332) by Heng et al.

(NeurIPS 2024) [Expecting The Unexpected: Towards Broad Out-Of-Distribution Detection](https://neurips.cc/virtual/2024/poster/97693) by Guille-Escuret et al.

(NeurIPS 2024) [Kernel PCA for Out-of-Distribution Detection](https://neurips.cc/virtual/2024/poster/96025) by Fang et al.

(NeurIPS 2024) [SeTAR: Out-of-Distribution Detection with Selective Low-Rank Approximation](https://neurips.cc/virtual/2024/poster/96549) by Li et al.

(NeurIPS 2024) [Rethinking the Evaluation of Out-of-Distribution Detection: A Sorites Paradox](https://neurips.cc/virtual/2024/poster/97509) by Long et al.

(NeurIPS 2024) [Energy-based Hopfield Boosting for Out-of-Distribution Detection](https://neurips.cc/virtual/2024/poster/94909) by Hofmann et al.

(NeurIPS 2024) [The Best of Both Worlds: On the Dilemma of Out-of-distribution Detection](https://neurips.cc/virtual/2024/poster/96211) by Zhang et al.

(NeurIPS 2024) [Out-Of-Distribution Detection with Diversification (Provably)](https://neurips.cc/virtual/2024/poster/96168) by Yao et al.

(NeurIPS 2024) [Hyper-opinion Evidential Deep Learning for Out-of-Distribution Detection](https://neurips.cc/virtual/2024/poster/95022) by Qu et al.

(NeurIPS 2024) [MultiOOD: Scaling Out-of-Distribution Detection for Multiple Modalities](https://arxiv.org/pdf/2405.17419) [Code](https://github.com/donghao51/MultiOOD) by Dong et al.

(ECCV 2024) [GalLoP: Learning Global and Local Prompts for Vision-Language Models](https://arxiv.org/pdf/2407.01400) by Lafon et al.

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

(ICML 2024) [Fast Decision Boundary based Out-of-Distribution Detector](https://proceedings.mlr.press/v235/xiong24b.html) [[Code]](https://github.com/litianliu/fDBD-OOD) by Liu et al.

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

(AAAI 2022) [iDECODe: In-Distribution Equivariance for Conformal Out-of-Distribution Detection](https://cdn.aaai.org/ojs/20670/20670-13-24683-1-2-20220628.pdf) by Kaur et al.

(AAAI 2022) [Provable Guarantees for Understanding Out-of-distribution Detection](https://arxiv.org/pdf/2112.00787.pdf) [[Code]](https://github.com/AlexMeinke/Provable-OOD-Detection) by Morteza and Li

(AAAI 2022) [Learning Modular Structures That Generalize Out-of-Distribution (Student Abstract)](https://cdn.aaai.org/ojs/21589/21589-13-25602-1-2-20220628.pdf) by Ashok et al.

(AAAI 2022) [Exploiting Mixed Unlabeled Data for Detecting Samples of Seen and Unseen Out-of-Distribution Classes](https://cdn.aaai.org/ojs/20814/20814-13-24827-1-2-20220628.pdf) by Sun and Wang

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

(ICLR 2026) [CoFact: Conformal Factuality Guarantees for Language Models under Covariate Shift](https://openreview.net/forum?id=eiBp7rsc3K) by Hu et al.

(ICCV 2025) [The Devil is in the Spurious Correlations: Boosting Moment Retrieval with Dynamic Learning](https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_The_Devil_is_in_the_Spurious_Correlations_Boosting_Moment_Retrieval_ICCV_2025_paper.html) by Zhou et al.

(ICCV 2025) [Bias in Gender Bias Benchmarks: How Spurious Features Distort Evaluation](https://openaccess.thecvf.com/content/ICCV2025/html/Hirota_Bias_in_Gender_Bias_Benchmarks_How_Spurious_Features_Distort_Evaluation_ICCV_2025_paper.html) by Hirota et al.

(AISTATS 2025) [Elastic Representation: Mitigating Spurious Correlations for Group Robustness](https://proceedings.mlr.press/v258/wen25a.html) by Wen et al.

(AISTATS 2025) [Out-of-distribution robustness for multivariate analysis via causal regularisation](https://proceedings.mlr.press/v258/durand25a.html) by Durand et al.

(AISTATS 2025) [Scalable Out-of-Distribution Robustness in the Presence of Unobserved Confounders](https://proceedings.mlr.press/v258/prashant25a.html) by Prashant et al.

(AISTATS 2025) [Mixed-Feature Logistic Regression Robust to Distribution Shifts](https://proceedings.mlr.press/v258/sun25e.html) by Sun et al.

(ICLR 2025) [Learning Neural Networks with Distribution Shift: Efficiently Certifiable Guarantees](https://openreview.net/forum?id=ed7zI29lRF) by Chandrasekaran et al.

(ICLR 2025) [Wasserstein-Regularized Conformal Prediction under General Distribution Shift](https://openreview.net/forum?id=aJ3tiX1Tu4) by Xu et al.

(TMLR 2025) [A Unified View of Double-Weighting for Marginal Distribution Shift](https://openreview.net/forum?id=aPyJilTiIb) by Segovia-Martin et al.

(TMLR 2025) [Distributionally Robust Coreset Selection under Covariate Shift](https://openreview.net/forum?id=Eu7XMLJqsC) by Tanaka et al.

(TMLR 2025) [An Analysis of Model Robustness across Concurrent Distribution Shifts](https://openreview.net/forum?id=nxQtoHHcj9) by Jeon et al.



(TMLR 2025) [Out of Spuriousity: Improving Robustness to Spurious Correlations without Group Annotations](https://openreview.net/forum?id=EEeVYfXor5) [[Code]](https://github.com/aix-group/prusc) by Le et al.

(TMLR 2025) [Making Self-supervised Learning Robust to Spurious Correlation via Learning-speed Aware Sampling](https://openreview.net/forum?id=8mgX3Uw2Ea) [[Code]](https://github.com/jackzhu727/LA-SSL) by Zhu et al.

(ICCV 2025) [DDB: Diffusion Driven Balancing to Address Spurious Correlations](https://openaccess.thecvf.com/content/ICCV2025/html/Parast_DDB_Diffusion_Driven_Balancing_to_Address_Spurious_Correlations_ICCV_2025_paper.html) by Parast et al.

(ICCV 2025) [Large Learning Rates Simultaneously Achieve Robustness to Spurious Correlations and Compressibility](https://openaccess.thecvf.com/content/ICCV2025/html/Barsbey_Large_Learning_Rates_Simultaneously_Achieve_Robustness_to_Spurious_Correlations_and_ICCV_2025_paper.html) by Barsbey et al.

(ICCV 2025) [CNS-Bench: Benchmarking Image Classifier Robustness Under Continuous Nuisance Shifts](https://openaccess.thecvf.com/content/ICCV2025/html/Dunkel_CNS-Bench_Benchmarking_Image_Classifier_Robustness_Under_Continuous_Nuisance_Shifts_ICCV_2025_paper.html) by Dunkel et al.

(ICML 2025) [FOCoOp: Enhancing Out-of-Distribution Robustness in Federated Prompt Learning for Vision-Language Models](https://proceedings.mlr.press/v267/liao25e.html) by Liao et al.

(ICLR 2025) [Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings](https://openreview.net/forum?id=GrDne4055L) by Mirzaei and Mathis

(AAAI 2024) [MaxEnt Loss: Constrained Maximum Entropy for Calibration under Out-of-Distribution Shift](https://aaai.org/wp-content/uploads/2024/01/AAAI_SRRAI_2024-01-04.pdf) by Ladner and Althoff

(AISTATS 2024) [Mind the GAP: Improving Robustness to Subpopulation Shifts with Group-Aware Priors](https://proceedings.mlr.press/v238/rudner24a.html) by Rudner et al.

(AISTATS 2024) [TransFusion: Covariate-Shift Robust Transfer Learning for High-Dimensional Regression](https://proceedings.mlr.press/v238/he24a.html) by He et al.

(AISTATS 2024) [Density-Regression: Efficient and Distance-aware Deep Regressor for Uncertainty Estimation under Distribution Shifts](https://proceedings.mlr.press/v238/manh-bui24a.html) by Bui and Liu

(ICLR 2024) [Adapting to Distribution Shift by Visual Domain Prompt Generation](https://openreview.net/forum?id=sSaN4gxuEf) by Chi et al.

(ICLR 2024) [Understanding the Robustness of Multi-modal Contrastive Learning to Distribution Shift](https://openreview.net/forum?id=rtl4XnJYBh) by Xue et al.

(ECCV 2024) [Robust Nearest Neighbors for Source-Free Domain Adaptation under Class Distribution Shift](https://eccv2024.ecva.net/virtual/2024/poster/833) by Tejero-de-Pablos et al.

(ECCV 2024) [TTT-MIM: Test-Time Training with Masked Image Modeling for Denoising Distribution Shifts](https://eccv2024.ecva.net/virtual/2024/poster/1788) by Mansour et al.

(ECCV 2024) [Human Motion Forecasting in Dynamic Domain Shifts: A Homeostatic Continual Test-time Adaptation Framework](https://eccv2024.ecva.net/virtual/2024/poster/2485) by Mansour et al.

(NeurIPS 2024) [Reconstruct and Match: Out-of-Distribution Robustness via Topological Homogeneity](https://neurips.cc/virtual/2024/poster/94189) by Chen et al.

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

(TMLR 2026) [Teaching Invariance Using Privileged Mediation Information](https://openreview.net/forum?id=8ZLhuo32Kz) [[Code]](https://github.com/DylanJamesZapzalka/tipmi-paper) by Zapzalka and Makar

(ICLR 2026) [Adaptive Mixture of Disentangled Experts for Dynamic Graph Out-of-Distribution Generalization](https://openreview.net/forum?id=q0O5LO7X4I) by Chen et al.

(ICLR 2026) [Out-of-Distribution Graph Models Merging](https://openreview.net/forum?id=93Y7jSUSpk) by Wang et al.

(ICLR 2026) [Diverse and Sparse Mixture-of-Experts for Causal Subgraph-Based Out-of-Distribution Graph Learning](https://openreview.net/forum?id=4XVczusV2K) by Sun et al.

(ICLR 2026) [Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning](https://openreview.net/forum?id=zrQkWSecMR) by Li et al.

(ICLR 2026) [Reasoning-Driven Multimodal LLM for Domain Generalization](https://openreview.net/forum?id=psJiUopUt7) by Xu et al.

(ICLR 2026) [Noise-Aware Generalization: Robustness to In-Domain Noise and Out-of-Domain Generalization](https://openreview.net/forum?id=wb83wO41QT) by Wang et al.

(ICLR 2026) [Does Weak-to-strong Generalization Happen under Spurious Correlations?](https://openreview.net/forum?id=5hfa2itwGz) by Liu et al.

(ICLR 2026) [Mitigating Spurious Correlation via Distributionally Robust Learning with Hierarchical Ambiguity Sets](https://openreview.net/forum?id=mruL1LDjzV) by Jo et al.

(ICLR 2026) [LABEL-FREE MITIGATION OF SPURIOUS CORRELATIONS IN VLMS USING SPARSE AUTOENCODERS](https://openreview.net/forum?id=NHOLsaHuFv) by Yalavarthi et al.

(ICLR 2026) [Bridging Explainability and Embeddings: BEE Aware of Spuriousness](https://openreview.net/forum?id=9jYpHmI8ot) by Paduraru et al.

(ICLR 2026) [Spurious Correlation-Aware Embedding Regularization for Worst-Group Robustness](https://openreview.net/forum?id=Grb5AOs7WC) by Park et al.

(ICLR 2026) [Federated Learning with Profile Mapping under Distribution Shifts and Drifts](https://openreview.net/forum?id=thoPskdIcE) by Li et al.

(TMLR 2026) [Diversity Sampling Regularization for Multi-Domain Generalization](https://openreview.net/forum?id=nXqMt7X2RX) by Tamang et al.

(TMLR 2025) [Disentangled Embedding through Style and Mutual Information for Domain Generalization](https://openreview.net/forum?id=552tedTByb) by Mehmood and Barner

(TMLR 2025) [Domain Generalization for Time Series: Enhancing Drilling Regression Models for Stick-Slip Index Prediction](https://openreview.net/forum?id=nNN1pPJRVL) by Yahia et al.

(TMLR 2025) [Latent Covariate Shift: Unlocking Partial Identifiability for Multi-Source Domain Adaptation](https://openreview.net/forum?id=9kFlOyLwyf) by Liu et al.

(TMLR 2025) [Prompt Tuning Vision Language Models with Margin Regularizer for Few-Shot Learning under Distribution Shifts](https://openreview.net/forum?id=ZnWqtPhHM7) by Brahma et al.

(TMLR 2024) [Domain-Generalizable Multiple-Domain Clustering](https://openreview.net/forum?id=O9RUANpPmb) by Rozner et al.

(TMLR 2024) [Weighted Risk Invariance: Domain Generalization under Invariant Feature Shift](https://openreview.net/forum?id=WyPKLWPYsr) by Wong et al.


(ICCV 2025) [ASGS: Single-Domain Generalizable Open-Set Object Detection via Adaptive Subgraph Searching](https://openaccess.thecvf.com/content/ICCV2025/html/Yuan_ASGS_Single-Domain_Generalizable_Open-Set_Object_Detection_via_Adaptive_Subgraph_Searching_ICCV_2025_paper.html) by Yuan et al.

(ICCV 2025) [Open-set Cross Modal Generalization via Multimodal Unified Representation](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_Open-set_Cross_Modal_Generalization_via_Multimodal_Unified_Representation_ICCV_2025_paper.html) by Huang et al.

(ICCV 2025) [Bridging Domain Generalization to Multimodal Domain Generalization via Unified Representations](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_Bridging_Domain_Generalization_to_Multimodal_Domain_Generalization_via_Unified_Representations_ICCV_2025_paper.html) by Huang et al.

(ICCV 2025) [Domain Generalizable Portrait Style Transfer](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Domain_Generalizable_Portrait_Style_Transfer_ICCV_2025_paper.html) by Wang et al.

(AISTATS 2025) [Accuracy on the wrong line: On the pitfalls of noisy data for out-of-distribution generalisation](https://proceedings.mlr.press/v258/sanyal25a.html) by Sanyal et al.

(AISTATS 2025) [Adapting to Online Distribution Shifts in Deep Learning: A Black-Box Approach](https://proceedings.mlr.press/v258/baby25a.html) by Baby et al.

(AISTATS 2025) [Harnessing the Power of Vicinity-Informed Analysis for Classification under Covariate Shift](https://proceedings.mlr.press/v258/fujikawa25a.html) by Fujikawa et al.

(AISTATS 2025) [Conformal Prediction Under Generalized Covariate Shift with Posterior Drift](https://proceedings.mlr.press/v258/wang25l.html) by Wang and Qiao

(AISTATS 2025) [Importance-weighted Positive-unlabeled Learning for Distribution Shift Adaptation](https://proceedings.mlr.press/v258/kumagai25a.html) by Kumagai et al.

(AISTATS 2025) [Invariant Link Selector for Spatial-Temporal Out-of-Distribution Problem](https://proceedings.mlr.press/v258/tieu25a.html) by Tieu et al.

(AISTATS 2025) [DeCaf: A Causal Decoupling Framework for OOD Generalization on Node Classification](https://proceedings.mlr.press/v258/han25b.html) by Han et al.

(ICLR 2025) [EvA: Erasing Spurious Correlations with Activations](https://openreview.net/forum?id=zKvrOOBouT) by He et al.

(ICLR 2025) [Minimax Optimal Two-Stage Algorithm For Moment Estimation Under Covariate Shift](https://openreview.net/forum?id=oc4yw7zX9T) by Zhang et al.

(ICLR 2025) [Group-robust Sample Reweighting for Subpopulation Shifts via Influence Functions](https://openreview.net/forum?id=aQj9Ifxrl6) by Qiao et al.

(ICLR 2025) [Multimodal Unsupervised Domain Generalization by Retrieving Across the Modality Gap](https://openreview.net/forum?id=bqoHdVMIbt) by Liao et al.

(ICLR 2025) [Federated Domain Generalization with Data-free On-server Matching Gradient](https://openreview.net/forum?id=8TERgu1Lb2) by Nguyen et al.

(AAAI 2025) [Partial Label Causal Representation Learning for Instance-Dependent Supervision and Domain Generalization](https://aaai.org/wp-content/uploads/2025/02/AAAI-25-Main-Track-Poster-Schedule.pdf) by Wang et al.

(AAAI 2025) [Federated Unsupervised Domain Generalization using Global and Local Alignment of Gradients](https://aaai.org/wp-content/uploads/2025/01/AAAI-25-Poster-Schedule.pdf) by Pourpanah et al.

(AAAI 2025) [TTA-FedDG: Leveraging Test-Time Adaptation to Address Federated Domain Generalization](https://aaai.org/wp-content/uploads/2025/01/AAAI-25-Poster-Schedule.pdf) by Liang et al.

(AAAI 2025) [PointDGMamba: Domain Generalization of Point Cloud Classification via Generalized State Space Model](https://aaai.org/wp-content/uploads/2025/02/AAAI-25-Main-Track-Poster-Schedule.pdf) by Yang et al.

(AAAI 2025) [DomCLP: Domain-wise Contrastive Learning with Prototype Mixup for Unsupervised Domain Generalization](https://aaai.org/wp-content/uploads/2025/01/AAAI-25-Poster-Schedule.pdf) by Lee et al.


(NeurIPS 2025) [Aggregation Hides Out-of-Distribution Generalization Failures from Spurious Correlations](https://neurips.cc/virtual/2025/poster/115351) by Salaudeen et al.

(NeurIPS 2025) [CCL: Causal-aware In-context Learning for Out-of-Distribution Generalization](https://neurips.cc/virtual/2025/poster/119001) by Byun et al.

(NeurIPS 2025) [Multi-Expert Distributionally Robust Optimization for Out-of-Distribution Generalization](https://neurips.cc/virtual/2025/poster/118509) by Jeong et al.

(NeurIPS 2025) [Pruning Spurious Subgraphs for Graph Out-of-Distribution Generalization](https://neurips.cc/virtual/2025/poster/116247) [[Code]](https://github.com/tianyao-aka/PrunE-GraphOOD) by Yao et al.

(NeurIPS 2025) [STRAP: Spatio-Temporal Pattern Retrieval for Out-of-Distribution Generalization](https://neurips.cc/virtual/2025/poster/117450) [[Code]](https://github.com/HoweyZ/STRAP) by Zhang et al.

(ICCV 2025) [Customizing Domain Adapters for Domain Generalization](https://openaccess.thecvf.com/content/ICCV2025/html/Ji_Customizing_Domain_Adapters_for_Domain_Generalization_ICCV_2025_paper.html) by Ji et al.

(ICCV 2025) [Federated Domain Generalization with Domain-specific Soft Prompts Generation](https://openaccess.thecvf.com/content/ICCV2025/html/Wu_Federated_Domain_Generalization_with_Domain-specific_Soft_Prompts_Generation_ICCV_2025_paper.html) by Wu et al.

(ICCV 2025) [ConstStyle: Robust Domain Generalization with Unified Style Transformation](https://openaccess.thecvf.com/content/ICCV2025/html/Tran_ConstStyle_Robust_Domain_Generalization_with_Unified_Style_Transformation_ICCV_2025_paper.html) by Tran et al.

(ICCV 2025) [Exploring Probabilistic Modeling Beyond Domain Generalization for Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Exploring_Probabilistic_Modeling_Beyond_Domain_Generalization_for_Semantic_Segmentation_ICCV_2025_paper.html) by Chen et al.

(ICML 2025) [Controlling Neural Collapse Enhances Out-of-Distribution Detection and Transfer Learning](https://proceedings.mlr.press/v267/harun25a.html) by Harun et al.

(ICML 2025) [On the Out-of-Distribution Generalization of Self-Supervised Learning](https://proceedings.mlr.press/v267/qiang25a.html) by Qiang et al.

(ICML 2025) [Enhancing Graph Invariant Learning from a Negative Inference Perspective](https://openreview.net/forum?id=0pUSBi1BjC) by Yang et al.

(ICML 2025) [One-Step Generalization Ratio Guided Optimization for Domain Generalization](https://proceedings.mlr.press/v267/cho25c.html) by Cho et al.

(ICLR 2025) [In Search of Forgotten Domain Generalization](https://openreview.net/forum?id=Fk3eod9aaD) by Mayilvahanan et al.

(ICLR 2025) [Intermediate Layer Classifiers for OOD generalization](https://openreview.net/forum?id=ByCV9xWfNK) by Uselis and Oh

(ICLR 2025) [BrainOOD: Out-of-distribution Generalizable Brain Network Analysis](https://openreview.net/forum?id=3xqqYOKILp) by Xu et al.

(ICLR 2025) [Is Large-scale Pretraining the Secret to Good Domain Generalization?](https://openreview.net/forum?id=wCOJpXm0Me) by Teterwak et al.

(ICLR 2025) [Regularizing Energy among Training Samples for Out-of-Distribution Generalization](https://openreview.net/forum?id=Lbx9zdURxe) by Chen et al.

(CVPR 2025) [On the Out-Of-Distribution Generalization of Large Multimodal Models](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_On_the_Out-Of-Distribution_Generalization_of_Large_Multimodal_Models_CVPR_2025_paper.html) by Zhang et al.

(CVPR 2025) [Sufficient Invariant Learning for Distribution Shift](https://openaccess.thecvf.com/content/CVPR2025/html/Kim_Sufficient_Invariant_Learning_for_Distribution_Shift_CVPR_2025_paper.html) by Kim et al.

(TMLR 2025) [An Empirical Study of Pre-trained Model Selection for Out-of-Distribution Generalization and Calibration](https://openreview.net/forum?id=tYjoHjShxF) [[Code]](https://github.com/Hiroki11x/Timm_OOD_Calibration) by Naganuma et al.

(TMLR 2025) [Generative Risk Minimization for Out-of-Distribution Generalization on Graphs](https://openreview.net/forum?id=EcMVskXo1n) [[Code]](https://github.com/SongW-SW/GRM) by Wang et al.

(TMLR 2025) [DivIL: Unveiling and Addressing Over-Invariance for Out-of-Distribution Generalization](https://openreview.net/forum?id=2Zan4ATYsh) [[Code]](https://github.com/kokolerk/DivIL) by Wang et al.

(TMLR 2025) [Are Domain Generalization Benchmarks with Accuracy on the Line Misspecified?](https://openreview.net/forum?id=fNywRyqPQo) [[Code]](https://github.com/olawalesalaudeen/misspecified_DG_benchmarks) by Salaudeen et al.

(JMLR 2026) [Boosted Control Functions: Distribution Generalization and Invariance in Confounded Models](https://jmlr.org/papers/v27/24-2207.html) by Gnecco et al.

(NeurIPS 2025) [Continuous Domain Generalization](https://neurips.cc/virtual/2025/poster/118589) by Cai et al.

(NeurIPS 2025) [How Many Domains Suffice for Domain Generalization? A Tight Characterization via the Domain Shattering Dimension](https://neurips.cc/virtual/2025/poster/116527) by Dwork et al.

(CVPR 2025) [Balanced Direction from Multifarious Choices: Arithmetic Meta-Learning for Domain Generalization](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Balanced_Direction_from_Multifarious_Choices_Arithmetic_Meta-Learning_for_Domain_Generalization_CVPR_2025_paper.html) by Wang et al.

(CVPR 2025) [Adversarial Domain Prompt Tuning and Generation for Single Domain Generalization](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_Adversarial_Domain_Prompt_Tuning_and_Generation_for_Single_Domain_Generalization_CVPR_2025_paper.html) by Xu et al.

(CVPR 2025) [Gradient-Guided Annealing for Domain Generalization](https://openaccess.thecvf.com/content/CVPR2025/html/Ballas_Gradient-Guided_Annealing_for_Domain_Generalization_CVPR_2025_paper.html) by Ballas et al.

(CVPR 2025) [Seeking Consistent Flat Minima for Better Domain Generalization via Refining Loss Landscapes](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Seeking_Consistent_Flat_Minima_for_Better_Domain_Generalization_via_Refining_CVPR_2025_paper.html) by Li et al.

(CVPR 2025) [OSLoPrompt: Bridging Low-Supervision Challenges and Open-Set Domain Generalization in CLIP](https://openaccess.thecvf.com/content/CVPR2025/html/C_OSLoPrompt_Bridging_Low-Supervision_Challenges_and_Open-Set_Domain_Generalization_in_CLIP_CVPR_2025_paper.html) by Mohamad Hassan N C et al.

(CVPR 2025) [Domain Generalization in CLIP via Learning with Diverse Text Prompts](https://openaccess.thecvf.com/content/CVPR2025/html/Wen_Domain_Generalization_in_CLIP_via_Learning_with_Diverse_Text_Prompts_CVPR_2025_paper.html) by Wen et al.

(CVPR 2025) [Unlocking the Potential of Unlabeled Data in Semi-Supervised Domain Generalization](https://openaccess.thecvf.com/content/CVPR2025/html/Lee_Unlocking_the_Potential_of_Unlabeled_Data_in_Semi-Supervised_Domain_Generalization_CVPR_2025_paper.html) by Lee et al.

(CVPR 2025) [Single Domain Generalization for Few-Shot Counting via Universal Representation Matching](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Single_Domain_Generalization_for_Few-Shot_Counting_via_Universal_Representation_Matching_CVPR_2025_paper.html) by Chen et al.

(CVPR 2025) [Test-Time Domain Generalization via Universe Learning: A Multi-Graph Matching Approach for Medical Image Segmentation](https://openaccess.thecvf.com/content/CVPR2025/html/Lv_Test-Time_Domain_Generalization_via_Universe_Learning_A_Multi-Graph_Matching_Approach_CVPR_2025_paper.html) by Lv et al.

(CVPR 2025) [TIDE: Training Locally Interpretable Domain Generalization Models Enables Test-time Correction](https://openaccess.thecvf.com/content/CVPR2025/html/Agarwal_TIDE_Training_Locally_Interpretable_Domain_Generalization_Models_Enables_Test-time_Correction_CVPR_2025_paper.html) by Agarwal et al.

(CVPR 2025) [When Domain Generalization meets Generalized Category Discovery: An Adaptive Task-Arithmetic Driven Approach](https://openaccess.thecvf.com/content/CVPR2025/html/Rathore_When_Domain_Generalization_meets_Generalized_Category_Discovery_An_Adaptive_Task-Arithmetic_CVPR_2025_paper.html) by Rathore et al.

(ICML 2025) [OOD-Chameleon: Is Algorithm Selection for OOD Generalization Learnable?](https://proceedings.mlr.press/v267/jiang25g.html) by Jiang and Teney

(ICML 2025) [LangDAug: Langevin Data Augmentation for Multi-Source Domain Generalization in Medical Image Segmentation](https://proceedings.mlr.press/v267/tiwary25a.html) by Tiwary et al.

(ICML 2025) [Set Valued Predictions For Robust Domain Generalization](https://proceedings.mlr.press/v267/tsibulsky25a.html) by Tsibulsky et al.

(ICML 2025) [Causality Inspired Federated Learning for OOD Generalization](https://proceedings.mlr.press/v267/zhang25ba.html) by Zhang et al.

(UAI 2025) [Moment Alignment: Unifying Gradient and Hessian Matching for Domain Generalization](https://proceedings.mlr.press/v286/chen25f.html) by Chen et al.

(UAI 2024) [Consistency Regularization for Domain Generalization with Logit Attribution Matching](https://proceedings.mlr.press/v244/gao24a.html) by Gao et al.

(UAI 2024) [Non-stationary Domain Generalization: Theory and Algorithm](https://proceedings.mlr.press/v244/pham24a.html) by Pham et al.


(ECCV 2024) [Disentangling Masked Autoencoders for Unsupervised Domain Generalization](https://eccv2024.ecva.net/virtual/2024/poster/836) by Zhao et al.

(ECCV 2024) [Improving Domain Generalization in Self-Supervised Monocular Depth Estimation via Stabilized Adversarial Training](https://eccv2024.ecva.net/virtual/2024/poster/625) by Kim et al.

(ECCV 2024) [Rethinking LiDAR Domain Generalization: Single Source as Multiple Density Domains](https://eccv2024.ecva.net/virtual/2024/poster/2073) by Li et al.

(ECCV 2024) [From Fake to Real: Pretraining on Balanced Synthetic Images to Prevent Spurious Correlations in Image Recognition](https://eccv2024.ecva.net/virtual/2024/poster/497) by Qraitem et al.

(ECCV 2024) [Constructing Concept-based Models to Mitigate Spurious Correlations with Minimal Human Effort](https://eccv2024.ecva.net/virtual/2024/poster/1563) by Mendez et al.

(ECCV 2024) [Benchmarking Spurious Bias in Few-Shot Image Classifiers](https://eccv2024.ecva.net/virtual/2024/poster/1190) by Zhang et al.

(ECCV 2024) [Feature Diversification and Adaptation for Federated Domain Generalization](https://eccv2024.ecva.net/virtual/2024/poster/1207) by Park et al.

(ECCV 2024) [Local and Global Flatness for Federated Domain Generalization](https://eccv2024.ecva.net/virtual/2024/poster/1581) by Zhang et al.

(ECCV 2024) [DG-PIC: Domain Generalized Point-In-Context Learning for Point Cloud Understanding](https://eccv2024.ecva.net/virtual/2024/poster/2434) by Jiang et al.

(ECCV 2024) [Face Reconstruction Transfer Attack as Out-of-Distribution Generalization](https://eccv2024.ecva.net/virtual/2024/poster/1810) by Jung et al.

(ECCV 2024) [SAFT: Towards Out-of-Distribution Generalization in Fine-Tuning](https://eccv2024.ecva.net/virtual/2024/poster/1561) by Nguyen et al.

(ECCV 2024) [Learn to Preserve and Diversify: Parameter-Efficient Group with Orthogonal Regularization for Domain Generalization](https://eccv2024.ecva.net/virtual/2024/poster/444) by Wang et al.

(ECCV 2024) [Domain Shifting: A Generalized Solution for Heterogeneous Cross-Modality Person Re-Identification](https://eccv2024.ecva.net/virtual/2024/poster/1482) by Zhang et al.

(ECCV 2024) [Towards Multimodal Open-Set Domain Generalization and Adaptation through Self-supervision](https://eccv2024.ecva.net/virtual/2024/poster/2313) by Liu et al.

(AISTATS 2024) [Learning Under Random Distributional Shifts](https://proceedings.mlr.press/v238/bansak24a.html) by Bansak et al.

(AISTATS 2024) [Approximate Bayesian Class-Conditional Models under Continuous Representation Shift](https://proceedings.mlr.press/v238/lee24c.html) by Lee and Storkey

(ICLR 2024) [Improving Domain Generalization with Domain Relations](https://openreview.net/forum?id=Dc4rXq3HIA) by Yao et al.

(ICLR 2024) [Understanding Domain Generalization: A Noise Robustness Perspective](https://openreview.net/forum?id=I2mIxuXA72) by Qiao and Low

(ICLR 2024) [Unknown Domain Inconsistency Minimization for Domain Generalization](https://openreview.net/forum?id=eNoiRal5xi) by Shin et al.

(ICLR 2024) [Continual Learning in the Presence of Spurious Correlations: Analyses and a Simple Baseline](https://openreview.net/forum?id=3Y7r6xueJJ) by Lee et al.

(ICLR 2024) [Out-of-Variable Generalisation for Discriminative Models](https://openreview.net/forum?id=zwMfg9PfPs) by Guo et al.

(ICLR 2024) [Domain-Inspired Sharpness-Aware Minimization Under Domain Shifts](https://openreview.net/forum?id=I4wB3HA3dJ) by Zhang et al.

(ICLR 2024) [Towards domain-invariant Self-Supervised Learning with Batch Styles Standardization](https://openreview.net/forum?id=qtE9K23ISq) by Scalbert et al.

(ICLR 2024) [Pooling Image Datasets with Multiple Covariate Shift and Imbalance](https://openreview.net/forum?id=2Mo7v69otj) by Chytas et al.

(AAAI 2024) [Sharpness-Aware Model-Agnostic Long-Tailed Domain Generalization](https://aaai.org/wp-content/uploads/2024/01/AAAI_Main-Track_2024-01-04.pdf) by Su et al.

(AAAI 2024) [Learning Content-Enhanced Mask Transformer for Domain Generalized Urban-Scene Segmentation](https://aaai.org/wp-content/uploads/2024/01/AAAI_Main-Track_2024-01-04.pdf) by Bi et al.

(AAAI 2024) [Object-Aware Domain Generalization for Object Detection](https://aaai.org/wp-content/uploads/2024/01/AAAI-24-Oral-Papers-Schedule-3.pdf) by Lee et al.

(AAAI 2024) [Enhancing Evolving Domain Generalization through Dynamic Latent Representations](https://aaai.org/wp-content/uploads/2024/01/AAAI-24-Oral-Papers-Schedule-3.pdf) by Xie et al.

(NeurIPS 2024) [Bridging Multicalibration and Out-of-distribution Generalization Beyond Covariate Shift](https://neurips.cc/virtual/2024/poster/94486) by Wu et al.

(NeurIPS 2024) [Advancing Open-Set Domain Generalization Using Evidential Bi-Level Hardest Domain Scheduler](https://neurips.cc/virtual/2024/poster/96166) by Peng et al.

(NeurIPS 2024) [AHA: Human-Assisted Out-of-Distribution Generalization and Detection](https://neurips.cc/virtual/2024/poster/96691) by Bai et al.

(NeurIPS 2024) [Vision Transformer Neural Architecture Search for Out-of-Distribution Generalization: Benchmark and Insights](https://neurips.cc/virtual/2024/poster/96829) by Ho et al.

(NeurIPS 2024) [What Variables Affect Out-Of-Distribution Generalization in Pretrained Models?](https://neurips.cc/virtual/2024/poster/93557) by Harun et al.

(NeurIPS 2024) [FOOGD: Federated Collaboration for Both Out-of-distribution Generalization and Detection](https://neurips.cc/virtual/2024/poster/96103) by Liao et al.

(NeurIPS 2024) [Benchmarking Out-of-Distribution Generalization Capabilities of DNN-based Encoding Models for the Ventral Visual Cortex](https://neurips.cc/virtual/2024/poster/97537) by Madan et al.

(NeurIPS 2024) [Neural Collapse Inspired Feature Alignment for Out-of-Distribution Generalization](https://neurips.cc/virtual/2024/poster/93156) by Chen et al.

(NeurIPS 2024) [EMGBench: Benchmarking Out-of-Distribution Generalization and Adaptation for Electromyography](https://neurips.cc/virtual/2024/poster/97478) by Yang et al.

(NeurIPS 2024) [WikiDO: A New Benchmark Evaluating Cross-Modal Retrieval for Vision-Language Models](https://neurips.cc/virtual/2024/poster/97785) by Tankala et al.

(Nature) [Out-of-distribution generalization for learning quantum dynamics](https://www.nature.com/articles/s41467-023-39381-w) by Caro et al.

(ICML 2024) [CRoFT: Robust Fine-Tuning with Concurrent Optimization for OOD Generalization and Open-Set OOD Detection](https://openreview.net/pdf?id=xFDJBzPhci) [[Code]](https://github.com/LinLLLL/CRoFT) by Zhu et al.

(ICML 2024) [Time-Series Forecasting for Out-of-Distribution Generalization Using Invariant Learning](https://openreview.net/pdf?id=SMUXPVKUBg) [[Code]](https://github.com/AdityaLab/FOIL) by Liu et al.

(CVPR 2024) [Improving Out-of-Distribution Generalization in Graphs via Hierarchical Semantic Environments](https://openaccess.thecvf.com/content/CVPR2024/papers/Piao_Improving_Out-of-Distribution_Generalization_in_Graphs_via_Hierarchical_Semantic_Environments_CVPR_2024_paper.pdf) by Piao et al.

(CVPR 2024) [PracticalDG: Perturbation Distillation on Vision-Language Models for Hybrid Domain Generalization](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_PracticalDG_Perturbation_Distillation_on_Vision-Language_Models_for_Hybrid_Domain_Generalization_CVPR_2024_paper.html) by Chen et al.

(CVPR 2024) [Rethinking the Evaluation Protocol of Domain Generalization](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Rethinking_the_Evaluation_Protocol_of_Domain_Generalization_CVPR_2024_paper.html) by Yu et al.

(CVPR 2024) [Single Domain Generalization for Crowd Counting](https://openaccess.thecvf.com/content/CVPR2024/html/Peng_Single_Domain_Generalization_for_Crowd_Counting_CVPR_2024_paper.html) by Peng et al.

(CVPR 2024) [DiPrompT: Disentangled Prompt Tuning for Multiple Latent Domain Generalization in Federated Learning](https://openaccess.thecvf.com/content/CVPR2024/html/Bai_DiPrompT_Disentangled_Prompt_Tuning_for_Multiple_Latent_Domain_Generalization_in_CVPR_2024_paper.html) by Bai et al.

(CVPR 2024) [A2XP: Towards Private Domain Generalization](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_A2XP_Towards_Private_Domain_Generalization_CVPR_2024_paper.html) by Yu et al.

(CVPR 2024) [Test-Time Domain Generalization for Face Anti-Spoofing](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_Test-Time_Domain_Generalization_for_Face_Anti-Spoofing_CVPR_2024_paper.html) by Zhou et al.

(CVPR 2024) [Prompt-Driven Dynamic Object-Centric Learning for Single Domain Generalization](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Prompt-Driven_Dynamic_Object-Centric_Learning_for_Single_Domain_Generalization_CVPR_2024_paper.html) by Li et al.

(CVPR 2024) [Efficiently Assemble Normalization Layers and Regularization for Federated Domain Generalization](https://openaccess.thecvf.com/content/CVPR2024/html/Le_Efficiently_Assemble_Normalization_Layers_and_Regularization_for_Federated_Domain_Generalization_CVPR_2024_paper.html) by Le et al.

(CVPR 2024) [Leveraging Vision-Language Models for Improving Domain Generalization in Image Classification](https://openaccess.thecvf.com/content/CVPR2024/html/Addepalli_Leveraging_Vision-Language_Models_for_Improving_Domain_Generalization_in_Image_Classification_CVPR_2024_paper.html) by Addepalli et al.

(CVPR 2024) [Rethinking Multi-domain Generalization with A General Learning Objective](https://openaccess.thecvf.com/content/CVPR2024/html/Tan_Rethinking_Multi-domain_Generalization_with_A_General_Learning_Objective_CVPR_2024_paper.html) by Tan et al.

(CVPR 2024) [Disentangled Prompt Representation for Domain Generalization](https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_Disentangled_Prompt_Representation_for_Domain_Generalization_CVPR_2024_paper.html) by Cheng et al.

(CVPR 2024) [A Dual-Augmentor Framework for Domain Generalization in 3D Human Pose Estimation](https://openaccess.thecvf.com/content/CVPR2024/html/Peng_A_Dual-Augmentor_Framework_for_Domain_Generalization_in_3D_Human_Pose_CVPR_2024_paper.html) by Peng et al.

(ICML 2024) [Out-of-Domain Generalization in Dynamical Systems Reconstruction](https://proceedings.mlr.press/v235/goring24a.html) by Goring et al.

(ICML 2024) [Graph Structure Extrapolation for Out-of-Distribution Generalization](https://proceedings.mlr.press/v235/li24y.html) by Li et al.

(ICML 2024) [Disentangled Graph Self-supervised Learning for Out-of-Distribution Generalization](https://proceedings.mlr.press/v235/li24br.html) by Li et al.

(ICML 2024) [Using Uncertainty Quantification to Characterize and Improve Out-of-Domain Learning for PDEs](https://proceedings.mlr.press/v235/mouli24a.html) by Mouli et al.

(ICML 2024) [Ensemble Pruning for Out-of-distribution Generalization](https://proceedings.mlr.press/v235/qiao24a.html) by Qiao and Peng

(ICML 2024) [LCA-on-the-Line: Benchmarking Out of Distribution Generalization with Class Taxonomies](https://proceedings.mlr.press/v235/shi24c.html) by Shi et al.


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

(AAAI 2022) [VITA: A Multi-Source Vicinal Transfer Augmentation Method for Out-of-Distribution Generalization](https://cdn.aaai.org/ojs/19908/19908-13-23921-1-2-20220628.pdf) by Chen et al.

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

(ICLR 2026) [Privacy-Protected Causal Survival Analysis Under Distribution Shift](https://openreview.net/forum?id=aTxnsFFO7t) by Liu et al.

(ICCV 2025) [MPBR: Multimodal Progressive Bidirectional Reasoning for Open-Set Fine-Grained Recognition](https://openaccess.thecvf.com/content/ICCV2025/html/Tan_MPBR_Multimodal_Progressive_Bidirectional_Reasoning_for_Open-Set_Fine-Grained_Recognition_ICCV_2025_paper.html) by Tan et al.

(TMLR 2026) [Training-Conditional Coverage Bounds under Covariate Shift](https://openreview.net/forum?id=F6hHT3qWxT) by Pournaderi and Xiang

(TMLR 2025) [Leveraging Gradients for Unsupervised Accuracy Estimation under Distribution Shift](https://openreview.net/forum?id=FIWHRSuoos) by Xie et al.

(TMLR 2025) [Importance Weighting for Aligning Language Models under Deployment Distribution Shift](https://openreview.net/forum?id=C7QWN4AXvp) by Lodkaew et al.

(TMLR 2024) [Selective Classification Under Distribution Shifts](https://openreview.net/forum?id=dmxMGW6J7N) by Liang et al.

(TMLR 2024) [Distributionally Robust Policy Evaluation under General Covariate Shift in Contextual Bandits](https://openreview.net/forum?id=R7PReNELww) by Guo et al.



(NeurIPS 2025) [Rethinking Out-of-Distribution Detection and Generalization with Collective Behavior Dynamics](https://neurips.cc/virtual/2025/poster/117703) by Wang et al.

(NeurIPS 2025) [Human Texts Are Outliers: Detecting LLM-generated Texts via Out-of-distribution Detection](https://neurips.cc/virtual/2025/poster/120309) [[Code]](https://github.com/cong-zeng/ood-llm-detect) by Zeng et al.

(NeurIPS 2025) [Reinforcement Learning for Out-of-Distribution Reasoning in LLMs: An Empirical Study on Diagnosis-Related Group Coding](https://neurips.cc/virtual/2025/poster/120291) [[Code]](https://github.com/hanyin88/DRG-Sapphire) by Wang et al.

(CVPR 2025) [COUNTS: Benchmarking Object Detectors and Multimodal Large Language Models under Distribution Shifts](https://openaccess.thecvf.com/content/CVPR2025/html/Li_COUNTS_Benchmarking_Object_Detectors_and_Multimodal_Large_Language_Models_under_CVPR_2025_paper.html) by Li et al.

(ICCV 2025) [ODP-Bench: Benchmarking Out-of-Distribution Performance Prediction](https://openaccess.thecvf.com/content/ICCV2025/html/Yu_ODP-Bench_Benchmarking_Out-of-Distribution_Performance_Prediction_ICCV_2025_paper.html) by Yu et al.

(TMLR 2025) [A Unified Approach Towards Active Learning and Out-of-Distribution Detection](https://openreview.net/forum?id=HL75La10FN) [[Code]](https://github.com/TUM-DAML/SISOM) by Schmidt et al.

(TMLR 2025) [Out-of-Distribution Learning with Human Feedback](https://openreview.net/forum?id=5qo8MF3QU1) [[Code]](https://github.com/HaoyueBaiZJU/ood-hf) by Bai et al.

(CVPR 2025) [Open Set Label Shift with Test Time Out-of-Distribution Reference](https://openaccess.thecvf.com/content/CVPR2025/html/Ye_Open_Set_Label_Shift_with_Test_Time_Out-of-Distribution_Reference_CVPR_2025_paper.html) by Ye et al.

(CVPR 2025) [Joint Out-of-Distribution Filtering and Data Discovery Active Learning](https://openaccess.thecvf.com/content/CVPR2025/html/Schmidt_Joint_Out-of-Distribution_Filtering_and_Data_Discovery_Active_Learning_CVPR_2025_paper.html) by Schmidt et al.

(CVPR 2025) [Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy](https://openaccess.thecvf.com/content/CVPR2025/html/Jeong_Playing_the_Fool_Jailbreaking_LLMs_and_Multimodal_LLMs_with_Out-of-Distribution_CVPR_2025_paper.html) by Jeong et al.

(ICML 2025) [LAION-C: An Out-of-Distribution Benchmark for Web-Scale Vision Models](https://proceedings.mlr.press/v267/li25aw.html) by Li et al.

(UAI 2025) [Out-of-distribution Robust Optimization](https://proceedings.mlr.press/v286/cai25c.html) by Cai et al.

(UAI 2025) [Adapting Prediction Sets to Distribution Shifts Without Labels](https://proceedings.mlr.press/v286/kasa25a.html) by Kasa et al.

(UAI 2025) [ODD: Overlap-aware Estimation of Model Performance under Distribution Shift](https://proceedings.mlr.press/v286/mishra25a.html) by Mishra and Liu

(UAI 2025) [Stochastic Embeddings : A Probabilistic and Geometric Analysis of Out-of-Distribution Behavior](https://proceedings.mlr.press/v286/nguyen25b.html) by Nguyen et al.


(ECCV 2024) [Bidirectional Uncertainty-Based Active Learning for Open-Set Annotation](https://eccv2024.ecva.net/virtual/2024/poster/458) by Zheng et al.

(AISTATS 2024) [Identifying Confounding from Causal Mechanism Shifts](https://proceedings.mlr.press/v238/mameche24a.html) by Mameche et al.

(NeurIPS 2024) [Adaptive Labeling for Efficient Out-of-distribution Model Evaluation](https://neurips.cc/virtual/2024/poster/93241) by Mittal et al.

(NeurIPS 2024) [TAIA: Large Language Models are Out-of-Distribution Data Learners](https://neurips.cc/virtual/2024/poster/94733) by Jiang et al.

(NeurIPS 2024) [PURE: Prompt Evolution with Graph ODE for Out-of-distribution Fluid Dynamics Modeling](https://neurips.cc/virtual/2024/poster/92971) by Wu et al.

(NeurIPS 2024) [Scanning Trojaned Models Using Out-of-Distribution Samples](https://neurips.cc/virtual/2024/poster/93781) by Mirzaei et al.

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
