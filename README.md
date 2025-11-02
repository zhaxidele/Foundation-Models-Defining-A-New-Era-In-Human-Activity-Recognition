# Foundation-Models-in-the-Area-of-Human-Activity-Recognition
## (continue updating ...)

> This repository aims to provide a curated, continuously updated index of foundation models (FMs) for human activity recognition (HAR). The organization follows our survey’s lifecycle-based taxonomy and major development directions.

## The brief history of sensor-based HAR
<p align="center">
    <img src="images/FM_HAR_TimeLine.png" alt="Description" width="500">
</p>

Historical development of sensor-based Human Activity Recognition (HAR) models. From classical machine learning with hand-crafted features and shallow classifiers to the rise of deep learning with CNNs and RNNs, the field progressed toward a phase focused on transfer and domain generalization (robustness across users, devices, and datasets). More recently, self-supervised learning (SSL) approaches have enabled pretraining on unlabeled sensor data using contrastive or masked objectives. Today, the field is moving toward foundation models, exemplified by large-scale sensor–language alignment, emphasizing scalability, generalization, and interpretability

## The Trends of FMs for HAR

Here are the growth trends of publications since 2022 and the model names cloud:

<p align="center">
  <img src="images/FM_HAR_Trend.png" width="400" />
  <img src="images/FM_HAR_Cloud.png" width="400" /> 
</p>


**Left**: HAR-FM papers showing a sharp acceleration with the vast majority of works emerging since 2024.

**Right**: Representative model name cloud.



## Table of Contents 

- [Base Architecture](#base-architecture)
    - [Encoder-only stacks](#encoder-only-stacks)
    - [Dual-encoder](#dual-encoder)
    - [Encoder-decoder stacks](#encoder-decoder-stacks)
    - [Language-model stacks](#language-model-stacks)

- [Major Directions](#major-directions)
    - [Developing HAR-Specific Foundation Models from Scratch](#developing-har-specific-foundation-models-from-scratch)
    - [Adapting General Time-Series and Multimodal Foundation Models to HAR](#adapting-general-time-series-and-multimodal-foundation-models-to-har)
    - [Leveraging Large Language Models for Human Activity Recognition](#leveraging-large-language-models-for-human-activity-recognition)
   

- [Datasets and Benchmarks](#datasets-and-benchmarks)
    - [IMU-Only Dataset](#imu-only-dataset)
    - [Multimodal Dataset](#multimodal-dataset)
    - [Tools and Benchmarks](#tools-and-benchmarks)



## Base Architecture

### Encoder-only stacks
1. **"A Novel Human Activity Recognition Framework Based on Pre‑Trained Foundation Model (Chronos HAR Adapters)"**. *Xiong et al..* IEEE BIBM 2024. [[Paper](https://doi.org/10.1109/BIBM57916.2024.10240605)]
2. **"Beyond Sensor Data: Foundation Models of Behavioral Data from Wearables Improve Health Predictions"**. *Erturk et al..* ICML 2025. [[Paper](https://arxiv.org/abs/2507.00191)]
3. **"Cosmo: Contrastive Fusion Learning with Small Data for Multimodal Human Activity Recognition"**. *Ouyang et al..* MobiCom 2022. [[Paper](https://dl.acm.org/doi/10.1145/3496968.3497015)]
4. **"CrossHAR: Generalizing Cross‑dataset Human Activity Recognition via Hierarchical Self‑Supervised Pretraining"**. *Hong et al..* IMWUT 2024. [[Paper](https://doi.org/10.1145/3631234)]
5. **"HAR‑DoReMi: Optimizing Data Mixture for Self‑Supervised Human Activity Recognition Across Heterogeneous IMU Datasets"**. *Ban et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2503.01234)]
6. **"Layout‑Agnostic Human Activity Recognition in Smart Homes through Textual Descriptions Of Sensor Triggers (TDOST)"**. *Thukral et al..* IMWUT 2025. [[Paper](https://dl.acm.org/doi/10.1145/3712278)]
7. **"LIMU‑BERT: Unleashing the Potential of Unlabeled Data for IMU Sensing Applications"**. *Xu et al..* SenSys 2021. [[Paper](https://dl.acm.org/doi/10.1145/3485730.3485783)]
8. **"MASTER: A Multi‑modal Foundation Model for Human Activity Recognition"**. *Zhu et al..* IMWUT 2025. [[Paper](http://dx.doi.org/10.1145/3749511)]
9. **"Pulse‑PPG: An Open‑Source Field‑Trained PPG Foundation Model for Wearable Applications across Lab and Field Settings"**. *Saha et al..* IMWUT 2025. [[Paper](https://doi.org/10.1145/3786408.3811654)]
10. **"RelCon: Relative Contrastive Learning for a Motion Foundation Model for Wearable Data"**. *Xu et al..* ICLR 2025. [[Paper](https://arxiv.org/abs/2411.18822v5)]
11. **"Self‑Supervised Learning for Human Activity Recognition Using 700,000 Person‑days of Wearable Data"**. *Yuan et al..* npj Digital Medicine 2024. [[Paper](https://doi.org/10.1038/s41746-024-01062-3)]
12. **"SelfPAB: Large‑Scale Pre‑training on Accelerometer Data for Human Activity Recognition"**. *Logacjov et al..* Applied Intelligence 2024. [[Paper](https://doi.org/10.1007/s10489-024-05322-3)]
13. **"Speech Foundation Models Generalize to Time Series Tasks from Wearable Sensor Data"**. *Narain et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2509.00221v1)]
14. **"Towards Customizable Foundation Models for Human Activity Recognition with Wearable Devices (HAR‑FM)"**. *Qiu et al..* IMWUT 2025. [[Paper](http://dx.doi.org/10.1145/3749479)]


### Dual-encoder
1. **"FM-Fi 2.0: Foundation Model for Cross-Modal Multi-Person Human Activity Recognition"**. *Weng et al..* IEEE TMC 2025. [[Paper](https://doi.org/10.1109/TMC.2025.3593406)]
2. **"IMU2CLIP: Multimodal Contrastive Learning for IMU Motion Sensors from Egocentric Videos and Text Narrations"**. *Moon et al..* EMNLP 2023. [[Paper](https://aclanthology.org/2023.findings-emnlp.883/)]
3. **"Inertial Signal Forecasting with Foundation Model Techniques (Dual-View FM)"**. *Wieland et al..* Sensors 2025. [[Paper](https://doi.org/10.3390/s25030834)]
4. **"Large Model for Small Data: Foundation Model for Cross-Modal RF Human Activity Recognition (FM-Fi)"**. *Weng et al..* SenSys 2024. [[Paper](https://doi.org/10.1145/3589132.3625578)]
5. **"Limitations in Employing Natural Language Supervision for Sensor-Based Human Activity Recognition – And Ways to Overcome Them"**. *Haresamudram et al..* AAAI 2025.
6. **"Multimodal Foundation Model for Cross-Modal Retrieval and Activity Recognition (AURA-MFM)"**. *Matsuishi et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2506.03174)]
7. **"PRimuS: Pretraining IMU Encoders with Multimodal Self-Supervision"**. *Das et al..* ICASSP 2025.
8. **"SensorLM: Learning the Language of Wearable Sensors"**. *Zhang et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2506.09108)]
9. **"SleepFM: Multi-modal Representation Learning for Sleep Across Brain Activity, ECG and Respiratory Signals"**. *Thapa et al..* arXiv 2024. [[Paper](https://arxiv.org/abs/2405.17766v1)]
10. **"Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals (NORMWEAR)"**. *Luo et al..* arXiv 2025.
11. **"UniMTS – Unified Pre-training for Motion Time-Series"**. *Zhang et al..* NeurIPS 2024. [[Paper](https://openreview.net/forum?id=NeurIPS2024-UniMTS)]
12. **"Weak-Annotation of HAR Datasets using Vision Foundation Models"**. *Bock et al..* ISWC 2024.
13. **"Gem: Empowering mllm for grounded ecg understanding with time series and images"**. *Xiang Lan et al.* Arxiv 2025. [[Paper](https://arxiv.org/abs/2503.06073)]


    
### Encoder-decoder stacks
1. **"LSM: Large-Scale Masked Modeling of Daily Summaries for Population-Level Behavior Modeling"**. *—.* arXiv 2025. [[Paper](https://arxiv.org/abs/2506.09010)]
2. **"NORMWEAR: Normalization-Enhanced Pretraining of Physiological Signals for Wearable Sensing"**. *Luo et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2505.02512)]
3. **"RobustHAR: Multi-scale Spatial-temporal Masked Self-supervised Pre-training for Robust Human Activity Recognition"**. *Liu et al..* IMWUT 2025.
4. **"Scaling Wearable Foundation Models"**. *Narayanswamy et al..* ARXIV 2024.
5. **"SelfPAB: Large-Scale Pre-training on Accelerometer Data for Human Activity Recognition"**. *Logacjov et al..* — 2024.
6. **"Sensor2Text: Enabling Natural Language Interactions for Daily Activity Tracking Using Wearable Sensors"**. *Chen et al..* IMWUT 2024. [[Paper](https://dl.acm.org/doi/10.1145/3699747)]
7. **"SensorLM: Learning the Language of Wearable Sensors"**. *Zhang et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2506.09108)]
8. **"Spatial-Temporal Masked Autoencoder for Multi-Device Wearable Human Activity Recognition (STMAE)"**. *Miao et al..* IMWUT 2023.
9. **"STMAE: Structured Temporal Masked Autoencoding for Sensor Reconstruction"**. *—.* arXiv 2024. [[Paper](https://arxiv.org/abs/2405.02541)]
10. **"Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals (NORMWEAR)"**. *Luo et al..* ARXIV 2025.

### Language-model stacks
1. **"Health‑LLM: Aligning Large Language Models with Wearable Sensor Health Data"**. *Liu et al..* Information Fusion 2025. [[Paper](https://doi.org/10.1016/j.inffus.2025.103403)]
2. **"HARGPT: A Language‑Conditioned Foundation Model for Human Activity Recognition"**. *Ji et al..* arXiv 2024. [[Paper](https://arxiv.org/abs/2403.19526)]
3. **"LanHAR: Language‑centered Human Activity Recognition"**. *Yan et al..* arXiv 2024. [[Paper](https://arxiv.org/abs/2410.00003)]
4. **"Sensor2Text: Enabling Natural Language Interactions for Daily Activity Tracking Using Wearable Sensors"**. *Chen et al..* IMWUT 2024. [[Paper](https://doi.org/10.1145/3699747)]
5. **"SensorGPT: Generative Pretraining for Wearable Sensing"**. *Sharma et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2504.06512)]
6. **"SensorLLM: Learning the Language of Wearable Sensors"**. *Zhang et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2506.09108)]
7. **"Time2Lang: Bridging Time‑Series Foundation Models and Natural Language"**. *Pillai et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2502.07608v3)]
8. **"DailyLLM: Large Language Models for Daily Behavior Understanding"**. *Kang et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2505.04563)]
9. **"ContextLLM: Multimodal Context Understanding from Wearable Devices"**. *Wang et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2506.07114)]
10. **"SensorCaps & LLaSA: A Large Multimodal Agent for Human Activity Analysis Through Wearable and Smartphone Sensors"**. *Imran et al..* arXiv 2024. [[Paper](https://arxiv.org/abs/2406.14498)]
11. **"StressLLM: Large Language Models for Wearable Stress Detection"**. *Thapa et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2506.04817)]
12. **"SensorLLM: Aligning Large Language Models with Motion Sensors for Human Activity Recognition"**. *Li et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2410.10624)]
13. **"Sensor‑Text: Bridging Sensing and Language for Multimodal Reasoning"**. *Pillai et al..* arXiv 2025.
14. **"HumanActivity‑GPT: A Generative Transformer for Multimodal HAR"**. *—.* arXiv 2025.



## Major Directions

### Developing HAR-Specific Foundation Models from Scratch
1. **"Towards Customizable Foundation Models for Human Activity Recognition with Wearable Devices"**. *Minghui Qiu et al.* IMWUT 2025. [[Paper](https://dl.acm.org/doi/10.1145/3749479)]

### Adapting General Time-Series and Multimodal Foundation Models to HAR
1. **"A Novel Human Activity Recognition Framework Based on Pre-Trained Foundation Model"**. *Fuhai Xiong et al.* BIBM 2024. [[Paper](https://ieeexplore.ieee.org/document/10822159)]

### Leveraging Large Language Models for Human Activity Recognition
1. **"SensorLM: Learning the Language of Wearable Sensors"**. *Yuwei Zhang et al.* Arxiv 2025. [[Paper](https://arxiv.org/abs/2506.09108)]



## Datasets and Benchmarks

### IMU-Only Dataset
### Multimodal Dataset
### Tools and Benchmarks

