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
1. **"A Personal Health Large Language Model for Sleep and Fitness Coaching (PH-LLM)"**. *Khasentino et al..* NATURE 2025.
2. **"A survey for wearable sensors empowering smart healthcare in the era of large language models"**. *Liu et al..* INFORMATION FUSION 2025.
3. **"Activity transitions for semi-supervised federated learning in sensor-based human activity recognition"**. *Bukit et al..* Applied Soft Computing 2025.
4. **"AgentSense: Virtual Sensor Data Generation Using LLM Agents in Simulated Home Environments"**. *Leng et al..* ARXIV 2025.
5. **"AI-Generated Fall Data: Assessing LLMs and Diffusion Model for Wearable Fall Detection"**. *Alamgeer et al..* SENSORS 2025.
6. **"ALPHA: Anomalous Physiological Health Assessment Using Large Language Models"**. *Tang et al..* ARXIV 2023.
7. **"BioSignal Copilot: Leveraging the power of LLMs in drafting reports for biomedical signals"**. *Liu et al..* medRxiv 2023.
8. **"ChatAnalysis revisited: can ChatGPT undermine privacy in smart homes with data analysis?"**. *J{\"u.* i-com 2025.
9. **"ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions"**. *Yang et al..* ARXIV 2025.
10. **"ContextLLM: Meaningful Context Reasoning from Multi-Sensor and Multi-Device Data Using LLMs"**. *Post et al..* Proceedings of the 26th International Workshop on Mobile Computing Systems and Applications 2025.
11. **"ContextLLM: Multimodal Context Understanding from Wearable Devices"**. *Wang et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2506.07114)]
12. **"DailyLLM: Context-Aware Activity Log Generation Using Multi-Modal Sensors and LLMs"**. *Tian et al..* arXiv preprint arXiv:2507.13737 2025.
13. **"DailyLLM: Large Language Models for Daily Behavior Understanding"**. *Kang et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2505.04563)]
14. **"Demo Abstract: An LLM-Powered Multimodal Mobile Sensing System for Personalized and Interactive Health Behavior Analysis (MobiBox)"**. *Zhang et al..* Proceedings of the 23rd ACM Conference on Embedded Networked Sensor Systems 2025.
15. **"Dynamic Uncertainty-aware Multimodal Fusion for Outdoor Health Monitoring"**. *Fang et al..* ARXIV 2025.
16. **"EEG-GPT: Exploring Capabilities of Large Language Models for EEG Classification and Interpretation"**. *Kim et al..* ARXIV 2024.
17. **"Enabling On-Device LLMs Personalization with Smartphone Sensing"**. *Zhang et al..* Companion of the 2024 on ACM International Joint Conference on Pervasive and Ubiquitous Computing 2024.
18. **"Enhancing Explainability of Deep Learning-Based ECG Diagnosis Using Large Language Models"**. *Wu et al..* Proceedings of the 2024 8th International Conference on Advances in Artificial Intelligence 2024.
19. **"Evaluating Large Language Models as Virtual Annotators for Time-series Physical Sensing Data (arXiv:2403.01133v2, Apr 2024)"**. *Hota et al..* ARXIV 2024.
20. **"Exploring the Capabilities of LLMs for IMU-based Fine-grained Human Activity Understanding"**. *Xu et al..* Proceedings of the 2nd International Workshop on Foundation Models for Cyber-Physical Systems \& Internet of Things 2025.
21. **"Few-Shot Human Activity Recognition Using Lightweight Language Models"**. *Cruciani et al..* ARXIV 2025.
22. **"Few-Shot Optimization for Sensor Data Using Large Language Models: A Case Study on Fatigue Detection"**. *Ronando et al..* SENSORS 2025.
23. **"FM-Fi 2.0: Foundation Model for Cross-Modal Multi-Person Human Activity Recognition"**. *Weng et al..* TMC 2025.
24. **"Game of LLMs: Discovering Structural Constructs in Activities using Large Language Models"**. *Hiremath et al..* Companion of the 2024 on ACM International Joint Conference on Pervasive and Ubiquitous Computing 2024.
25. **"GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images"**. *Lan et al..* ARXIV 2025.
26. **"Generalizable Seizure Prediction with LLMs: Converting EEG to Textual Representations"**. *Zhao et al..* ARXIV 2025.
27. **"GLOSS: Group of LLMs for Open-ended Sensemaking of Passive Sensing Data for Health and Wellbeing"**. *Choube et al..* IMWUT 2025.
28. **"GOAT: A Generalized Cross-Dataset Activity Recognition Framework with Natural Language Supervision"**. *Miao et al..* ARXIV 2025.
29. **"HARGPT: A Language-Conditioned Foundation Model for Human Activity Recognition"**. *Ji et al..* arXiv 2024. [[Paper](https://arxiv.org/abs/2403.19526)]
30. **"HARGPT: Are LLMs Zero-Shot Human Activity Recognizers?"**. *Ji et al..* ARXIV 2025.
31. **"Health-LLM: Aligning Large Language Models with Wearable Sensor Health Data"**. *Liu et al..* Information Fusion 2025. [[Paper](https://doi.org/10.1016/j.inffus.2025.103403)]
32. **"Health-LLM: Large Language Models for Health Prediction via Wearable Sensor Data"**. *Kim et al..* ARXIV 2024.
33. **"Improved Human Activity Recognition Through Controllable GAN-Generated Synthetic Data and Large Language Models for Classification"**. *Djemaa et al..* Cluster Computing 2024.
34. **"IMUGPT 2.0: Language-Based Cross Modality Transfer for Sensor-Based Human Activity Recognition"**. *Leng et al..* IMWUT 2024.
35. **"LanHAR: Language-centered Human Activity Recognition"**. *Yan et al..* ARXIV 2025.
36. **"LanHAR: Language-driven Human Activity Recognition"**. *Yan et al..* arXiv 2024. [[Paper](https://arxiv.org/abs/2408.12211)]
37. **"Large Language Models are Few-Shot Health Learners"**. *Liu et al..* ARXIV 2023.
38. **"Large Language Models Are Zero-Shot Recognizers for Activities of Daily Living"**. *Civitarese et al..* ACM Transactions on Intelligent Systems and Technology 2025.
39. **"Large Language Models for Cuffless Blood Pressure Measurement from Wearable Biosignals"**. *Liu et al..* Proceedings of the 15th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics 2024.
40. **"Large Language Models for Wearable Data Analysis and Interpretation"**. *Simon Boehi et al.* ICLR 2024. [[Paper](https://openreview.net/pdf?id=GoWD6logcd)]
41. **"Large Language Models for Wearable Sensor-Based Human Activity Recognition, Health Monitoring, and Behavioral Modeling: A Survey of Early Trends, Datasets, and Challenges"**. *Ferrara.* SENSORS 2024.
42. **"Large Model for Small Data: Foundation Model for Cross-Modal RF Human Activity Recognition (FM-Fi)"**. *Weng et al..* IEEE Transactions on Mobile Computing 2024. [[Paper](http://dx.doi.org/10.1109/tmc.2025.3593406)]
43. **"Layout-Agnostic Human Activity Recognition in Smart Homes through Textual Descriptions Of Sensor Triggers (TDOST)"**. *Thukral et al..* IMWUT 2025.
44. **"Leveraging Language Models for Human Activity Recognition for Intelligent Lighting"**. *Mahmoudi et al..* ARXIV 2025.
45. **"Leveraging Large Language Models for Activity Recognition in Smart Environments"**. *Cleland et al..* ARXIV 2025.
46. **"Leveraging Large Language Models for Digital Phenotyping: Detecting Depressive State Changes for Patients with Depressive Episodes"**. *Yuan et al..* medRxiv 2025.
47. **"Leveraging Large Language Models for Explainable Activity Recognition in Smart Homes: A Critical Evaluation"**. *Fiori et al..* ACM Transactions on Internet of Things 2025.
48. **"Leveraging LLMs to Predict Affective States via Smartphone Sensor Features"**. *Zhang et al..* Companion of the 2024 on ACM International Joint Conference on Pervasive and Ubiquitous Computing 2024.
49. **"LLaSA: Large Multimodal Agent for Human Activity Analysis Through Wearable Sensors"**. *Imran et al..* arXiv preprint arXiv:2406.14498 2024.
50. **"LLM-based Intermediate Interpretations for Predicting Nurse Stress and Its Causes from Step Count Data"**. *Fukazawa et al..* — 2025.
51. **"LLM4HAR: Generalizable On-device Human Activity Recognition with Pretrained LLMs"**. *Hong et al..* Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2 2025.
52. **"LLMSense: Harnessing LLMs for High-level Reasoning Over Spatiotemporal Sensor Traces"**. *Ouyang et al..* ARXIV 2024.
53. **"MASTER: A Multi-modal Foundation Model for Human Activity Recognition"**. *Zhu et al..* IMWUT 2025. [[Paper](http://dx.doi.org/10.1145/3749511)]
54. **"Mental-LLM: Leveraging Large Language Models for Mental Health Prediction via Online Text Data"**. *Xu et al..* IMWUT 2024.
55. **"MHARFedLLM: Multimodal Human Activity Recognition Using Federated Large Language Model"**. *Bandyopadhyay et al..* ARXIV 2025.
56. **"MultiAgentsCR: A Multi-Agent Collaborative Reasoning Framework Based on LLM for Human Activity Recognition"**. *Zhao et al..* 2025 10th International Conference on Information Science, Computer Technology and Transportation 2025.
57. **"Multimodal Large Language Models in Human-Centered Health: Practical Insights"**. *Dang et al..* ARXIV 2024.
58. **"On the Benefit of Generative Foundation Models for Human Activity Recognition"**. *Leng et al..* ARXIV 2023.
59. **"One Model to Fit Them All: Universal IMU-based Human Activity Recognition with LLM-assisted Cross-dataset Representation"**. *Wei et al..* IMWUT 2025. [[Paper](http://dx.doi.org/10.1145/3749509)]
60. **"PhysioLLM: Supporting Personalized Health Insights with Wearables and Large Language Models"**. *Fang et al..* 2024 IEEE EMBS International Conference on Biomedical and Health Informatics 2024.
61. **"PhysLLM: Harnessing Large Language Models for Cross-Modal Remote Physiological Sensing"**. *Xie et al..* ARXIV 2025.
62. **"Potential of Large Language Model for Activity Recognition in Activities of Daily Living: A Systematic Review"**. *Ugwu et al..* Authorea Preprints 2025.
63. **"Semantic Sensors: Multimodal Language Model Powered Sensors Capabilities"**. *Olwal et al..* ARXIV 2024.
64. **"Sensor2Text: Enabling Natural Language Interactions for Daily Activity Tracking Using Wearable Sensors"**. *Chen et al..* Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 2024.
65. **"SensorBench: Benchmarking LLMs in Coding-Based Sensor Processing"**. *Quan et al..* Proceedings of the 26th International Workshop on Mobile Computing Systems and Applications 2025.
66. **"SensorGPT: Generative Pretraining for Wearable Sensing"**. *Sharma et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2504.06512)]
67. **"SensorLLM: Aligning Large Language Models with Motion Sensors for Human Activity Recognition"**. *Li et al..* ARXIV 2025.
68. **"SensorLLM: Towards Large Language Models for Wearable Sensors"**. *Li et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2504.09012)]
69. **"StressLLM: Large Language Models for Stress Prediction via Wearable Sensor Data"**. *Thapa et al..* 2025 IEEE International Conference on Consumer Electronics 2025.
70. **"StressLLM: Large Language Models for Wearable Stress Detection"**. *Thapa et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2506.04817)]
71. **"The Strong Pull of Prior Knowledge in Large Language Models and Its Impact on Emotion Recognition"**. *Chochlakis et al..* 2024 12th International Conference on Affective Computing and Intelligent Interaction 2024.
72. **"Thou Shalt Not Prompt: Zero-Shot Human Activity Recognition in Smart Homes via Language Modeling of Sensor Data & Activities"**. *Dhekane et al..* arXiv preprint arXiv:2507.21964 2025.
73. **"Time2Lang: Bridging Multimodal Time-Series and Natural Language"**. *Pillai et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2503.08110)]
74. **"Time2Lang: Bridging Time-Series Foundation Models and Large Language Models for Health Sensing Beyond Prompting"**. *Pillai et al..* ARXIV 2025.
75. **"Toward Sensor-In-the-Loop LLM Agent: Benchmarks and Implications"**. *Ren et al..* Proceedings of the 23rd ACM Conference on Embedded Networked Sensor Systems 2025.
76. **"Towards LLM-Powered Ambient Sensor-Based Multi-Person Human Activity Recognition"**. *Chen et al..* 2024 IEEE 30th International Conference on Parallel and Distributed Systems 2024.
77. **"Unsupervised Human Activity Recognition Via Large Language Models and Iterative Evolution"**. *Gao et al..* ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing 2024.
78. **"Using LLMs for Late Multimodal Sensor Fusion for Activity Recognition"**. *Demirel et al..* ARXIV 2025.
79. **"Visible light human activity recognition driven by generative language model"**. *Yang et al..* INFORMATION FUSION 2025.
80. **"Vital Insight: Assisting Experts’ Context-Driven Sensemaking of Multi-modal Personal Tracking Data Using Visualization and Human-in-the-Loop LLM"**. *Li et al..* IMWUT 2025.
81. **"Wearable IoT, Edge AI and LLMs for Monitoring Health and Wellness: A Harmonization Framework for Responsible AI Interventions"**. *Bukhari et al..* — 2025.
82. **"ZARA: Zero-shot Motion Time-Series Analysis via Knowledge and Retrieval Driven LLM Agents"**. *Li et al..* ARXIV 2025.



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

