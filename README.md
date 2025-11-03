# Foundation-Models-in-the-Area-of-Human-Activity-Recognition
## (continue updating ...)
## (To include your related work in this repository, please create a pull request with the relevant details. ...)

This repository aims to provide a curated, continuously updated index of foundation models (FMs) in the human activity recognition (HAR) domain. The organization follows our survey’s lifecycle-based taxonomy and major development directions.
It serves as a living companion to the paper “**Foundation Models Defining a New Era in Human Activity Recognition: A Survey and Outlook**”, offering direct access to representative works, datasets, and model resources. Our goal is to foster transparency, reproducibility, and collaboration across the HAR community as the field transitions toward large-scale, multimodal, and language-grounded sensing models.

Contributions are welcome! Whether adding new papers, improving taxonomy coverage, linking open-source implementations, or updating existing groups! Please help the community build a shared, evolving reference for next-generation HAR foundation models.


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



- [Major Directions](#major-directions)
    - [Developing HAR-Specific Foundation Models from Scratch](#developing-har-specific-foundation-models-from-scratch)
    - [Adapting General Time-Series and Multimodal Foundation Models to HAR](#adapting-general-time-series-and-multimodal-foundation-models-to-har)
    - [Leveraging Large Language Models for Human Activity Recognition](#leveraging-large-language-models-for-human-activity-recognition)


- [Datasets Applied in FM Pretraining and Downstream Tasks](#datasets-applied-in-fm-pretraining-and-downstream-tasks)
    - [IMU-Only Datasets](#imu-only-datasets)
    - [Multimodal Datasets](#multimodal-datasets)
    - [Physiology Datasets](#physiology-datasets)
    - [Smart Home Datasets](#smart-home-datasets)
    - [RF Datasets](#rf-datasets)

- [Base Architecture](#base-architecture)
    - [Encoder-only stacks](#encoder-only-stacks)
    - [Dual-encoder](#dual-encoder)
    - [Encoder-decoder stacks](#encoder-decoder-stacks)
    - [Language-model stacks](#language-model-stacks)


- [Modality Scope](#modality-scope)
  - [Unimodal foundation models](#unimodal-foundation-models)
  - [Multimodal foundation models](#multimodal-foundation-models)
  - [Cross-modal foundation models](#cross-modal-foundation-models)

- [Data Landscape: Collected and Generated Corpora](#data-landscape-collected-and-generated-corpora)
  - [Corpora of sensor data collected in the wild](#corpora-of-sensor-data-collected-in-the-wild)
  - [Generated datasets and augmentation](#generated-datasets-and-augmentation)

- [Tokenization and Representation Strategies](#tokenization-and-representation-strategies)
  - [Window-based and patch-level segmentation](#window-based-and-patch-level-segmentation)
  - [Feature-based aggregation and statistical embeddings](#feature-based-aggregation-and-statistical-embeddings)
  - [Spectrogram and frequency-domain embeddings](#spectrogram-and-frequency-domain-embeddings)
  - [Discrete and quantized sensor tokens](#discrete-and-quantized-sensor-tokens)
  - [Multimodal alignment and positional encoding](#multimodal-alignment-and-positional-encoding)
  - [Token fusion and cross-modal projection](#token-fusion-and-cross-modal-projection)


- [Pretraining Paradigms](#pretraining-paradigms)
  - [Contrastive pretraining](#contrastive-pretraining)
  - [Generative pretraining](#generative-pretraining)
  - [Hybrid and self-supervised pretraining](#hybrid-and-self-supervised-pretraining)

- [Adaptation Strategies](#adaptation-strategies)
  - [Parameter-efficient fine-tuning (PEFT)](#parameter-efficient-fine-tuning-peft)
  - [Full or partial fine-tuning](#full-or-partial-fine-tuning)
  - [Instruction-tuning & alignment](#instruction-tuning--alignment)

- [Downstream Capabilities](#downstream-capabilities)
  - [Zero-/few-shot & open-set recognition](#zero-few-shot--open-set-recognition)
  - [Cross-dataset / cross-device / cross-user generalization](#cross-dataset--cross-device--cross-user-generalization)
  - [Cross-modal retrieval and search](#cross-modal-retrieval-and-search)
  - [Language-grounded captioning, Q&A, and reasoning](#language-grounded-captioning-qa-and-reasoning)
  - [Generative reconstruction, forecasting, and imputation](#generative-reconstruction-forecasting-and-imputation)
  - [On-device, federated, and online adaptation](#on-device-federated-and-online-adaptation)

- [Deployment Settings](#deployment-settings)
  - [Cloud-scale training and centralized evaluation](#cloud-scale-training-and-centralized-evaluation)
  - [On-device and mobile execution](#on-device-and-mobile-execution)

- [Application Domains](#application-domains)
  - [General-purpose HAR / ADL](#general-purpose-har--adl)
  - [Healthcare & wellbeing](#healthcare--wellbeing)
  - [Smart-home & context-aware environments](#smart-home--context-aware-environments)
  - [Interactive & agentic assistants](#interactive--agentic-assistants)




## Major Directions

### Developing HAR-Specific Foundation Models from Scratch
1. **"A Novel Human Activity Recognition Framework Based on Pre-Trained Foundation Model (Chronos HAR Adapters)"**. *Xiong et al..* IEEE BIBM 2024. [[Paper](https://doi.org/10.1109/BIBM57916.2024.10240605)]
2. **"Self-supervised Learning for Human Activity Recognition Using 700,000 Person-days of Wearable Data"**. *Yuan et al..* npj Digital Medicine 2024. [[Paper](https://doi.org/10.1038/s41746-024-01062-3)]
3. **"Scaling Wearable Foundation Models"**. *Narayanswamy et al..* arXiv 2024. [[Paper](https://arxiv.org/abs/2410.13638v1)]
4. **"RelCon: Relative Contrastive Learning for a Motion Foundation Model for Wearable Data"**. *Xu et al..* ICLR 2025. [[Paper](https://arxiv.org/abs/2411.18822v5)]
5. **"HAR-DoReMi: Optimizing Data Mixture for Self-Supervised Human Activity Recognition Across Heterogeneous IMU Datasets"**. *Ban et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2503.13542)]
6. **"Beyond Sensor Data: Foundation Models of Behavioral Data from Wearables Improve Health Predictions"**. *Erturk et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2507.00191)]
7. **"LIMU-BERT: Unleashing the Potential of Unlabeled Data for IMU Sensing Applications"**. *Xu et al..* SenSys 2021. [[Paper](https://dl.acm.org/doi/10.1145/3485730.3485937)]
8. **"Cosmo: Contrastive Fusion Learning with Small Data for Multimodal Human Activity Recognition"**. *Ouyang et al..* MobiCom 2022. [[Paper](https://doi.org/10.1145/3495243.3560519)]
9. **"Spatial-Temporal Masked Autoencoder for Multi-Device Wearable Human Activity Recognition (STMAE)"**. *Miao et al..* IMWUT 2024. [[Paper](https://doi.org/10.1145/3631415)]
10. **"CrossHAR: Generalizing Cross-dataset Human Activity Recognition via Hierarchical Self-Supervised Pretraining"**. *Hong et al..* IMWUT 2024. [[Paper](https://doi.org/10.1145/3659597)]
11. **"Pulse-PPG: An Open-Source Field-Trained PPG Foundation Model for Wearable Applications across Lab and Field Settings"**. *Saha et al..* IMWUT 2025. [[Paper](https://doi.org/10.1145/3749494)]
12. **"RobustHAR: Multi-scale Spatial-temporal Masked Self-supervised Pre-training for Robust Human Activity Recognition"**. *Liu et al..* IJCAI 2025. [[Paper](https://doi.org/10.24963/ijcai.2025/952)]
13. **"Towards Customizable Foundation Models for Human Activity Recognition with Wearable Devices"**. *Minghui Qiu et al.* IMWUT 2025. [[Paper](https://dl.acm.org/doi/10.1145/3749479)]

    
### Adapting General Time-Series and Multimodal Foundation Models to HAR
1. **"A Novel Human Activity Recognition Framework Based on Pre-Trained Foundation Model"**. *Fuhai Xiong et al.* BIBM 2024. [[Paper](https://ieeexplore.ieee.org/document/10822159)]
2. **"Inertial Signal Forecasting with Foundation Model Techniques (Dual-View FM)"**. *Wieland & Pankratius.* IEEE Sensors Journal 2025 (accepted). [[Paper](https://sites.google.com/view/victorpankratius/publications)]
3. **"IMU2CLIP: Multimodal Contrastive Learning for IMU Motion Sensors from Egocentric Videos and Text Narrations"**. *Moon et al..* EMNLP Findings 2023. [[Paper](https://aclanthology.org/2023.findings-emnlp.883/)]
4. **"Weak-Annotation of HAR Datasets using Vision Foundation Models"**. *Bock et al..* ISWC 2024. [[Paper](https://dl.acm.org/doi/10.1145/3675095.3676613)]
5. **"GOAT: A Generalized Cross-Dataset Activity Recognition Framework with Natural Language Supervision"**. *Miao & Chen.* IMWUT 2024. [[Paper](https://dl.acm.org/doi/10.1145/3699736)]
6. **"GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images"**. *Lan et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2503.06073)]
7. **"UniMTS: Unified Pre-training for Motion Time Series"**. *Zhang et al..* NeurIPS 2024. [[Paper](https://arxiv.org/abs/2410.19818)]
8. **"Time2Lang: Bridging Time‑Series Foundation Models and Large Language Models for Health Sensing"**. *Pillai et al..* PMLR 2025. [[Paper](https://proceedings.mlr.press/v287/pillai25a.html)]


### Leveraging Large Language Models for Human Activity Recognition
1. **"LanHAR: Language-centered Human Activity Recognition"**. *Yan et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2410.00003)]
2. **"SensorLLM: Aligning Large Language Models with Motion Sensors for Human Activity Recognition"**. *Li et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2410.10624)]
3. **"Time2Lang: Bridging Time-Series Foundation Models and Large Language Models for Health Sensing Beyond Prompting"**. *Pillai et al..* CHIL (PMLR) 2025. [[Paper](https://proceedings.mlr.press/v287/pillai25a.html)]
4. **"DailyLLM: Context-Aware Activity Log Generation Using Multi-Modal Sensors and LLMs"**. *Tian et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2507.13737)]
5. **"ContextLLM: Meaningful Context Reasoning from Multi-Sensor and Multi-Device Data Using LLMs"**. *Post et al..* HotMobile 2025. [[Paper](https://dl.acm.org/doi/10.1145/3708468.3711892)]
6. **"HARGPT: Are LLMs Zero-Shot Human Activity Recognizers?"**. *Ji et al..* arXiv 2024. [[Paper](https://arxiv.org/abs/2403.02727)]
7. **"LLaSA: Large Multimodal Agent for Human Activity Analysis Through Wearable Sensors"**. *Imran et al..* arXiv 2024. [[Paper](https://arxiv.org/abs/2406.14498)]
8. **"StressLLM: Large Language Models for Stress Prediction via Wearable Sensor Data"**. *Thapa et al..* IEEE ICCE 2025. [[Paper](https://doi.org/10.1109/ICCE63647.2025.10929774)]
9. **"SensorLM: Learning the Language of Wearable Sensors"**. *Zhang et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2506.09108)]
10. **"Sensor2Text: Enabling Natural Language Interactions for Daily Activity Tracking Using Wearable Sensors"**. *Chen et al..* IMWUT 2024. [[Paper](https://doi.org/10.1145/3699747)]




## Datasets Applied in FM Pretraining and Downstream Tasks

### IMU-only Datasets
| Dataset | Sensors | Datasize | Subjects | Activities |
|:--|:--|:--|:--|:--|
| [**CAPTURE-24**](https://www.nature.com/articles/s41597-024-03960-3) | acc | 3883 h | 151 | 200 unique labels |
| [**TNDA-HAR**](https://ieee-dataport.org/open-access/tnda-har-0) | acc, gyro | 5.7 h | 23 | 8 daily activities |
| [**HAR70+**](https://archive.ics.uci.edu/dataset/780/har70) | acc | 12.6 h | 18 | 8 daily activities |
| [**WISDM**](https://archive.ics.uci.edu/dataset/507/wisdm%2Bsmartphone%2Band%2Bsmartwatch%2Bactivity%2Band%2Bbiometrics%2Bdataset) | acc, gyro | 91.8 h | 51 | 18 daily activities |
| [**MotionSense**](https://github.com/mmalekzadeh/motion-sense) | acc, gyro | — | 24 | 6 daily activities |
| [**SHL Challenge**](http://www.shl-dataset.org/challenges/) | acc, gyro, mag | 2812 h | 3 | 8 transport modes |
| [**Shoaib**](https://www.utwente.nl/en/eemcs/ps/research/dataset/) | acc, gyro, mag | 6.5 h | 10 | 13 daily activities |
| [**HHAR**](https://archive.ics.uci.edu/ml/datasets/heterogeneity%2Bactivity%2Brecognition) | acc, gyro | — | 9 | 6 daily activities |
| [**WHARF**](https://github.com/fulviomas/WHARF) | acc | — | 16 | 8 motion primitives |
| [**DSADS**](https://archive.ics.uci.edu/dataset/256/daily%2Band%2Bsports%2Bactivities) | acc, gyro, mag | 12.7 h | 8 | 19 daily/sports activities |
| [**UCI-HAR**](https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones) | acc, gyro | — | 30 | 6 daily activities |
| [**USC-HAD**](https://sipi.usc.edu/had/) | acc, gyro, mag | — | 14 | 12 daily activities |
| [**Daphnet FoG**](https://archive.ics.uci.edu/dataset/245/daphnet%2Bfreezing%2Bof%2Bgait) | acc | 8.3 h | 10 | 3 walking activities |
| [**HAPT**](https://archive.ics.uci.edu/dataset/341/smartphone%2Bbased%2Brecognition%2Bof%2Bhuman%2Bactivities%2Band%2Bpostural%2Btransitions) | acc, gyro | — | 30 | 6 static, dynamic activities |
| [**REALDISP**](https://archive.ics.uci.edu/dataset/305/realdisp%2Bactivity%2Brecognition%2Bdataset) | acc, gyro, mag | — | 17 | 33 daily, fitness activities |
| [**UniMiB SHAR**](https://www.sal.disco.unimib.it/technologies/unimib-shar/) | acc | — | 30 | 17 daily, fall activities |
| [**UMAFall**](https://figshare.com/articles/dataset/UMA_ADL_FALL_Dataset_zip/4214283) | acc, gyro, mag | 2.2 h | 17 | 11 daily, fall activities |
| [**MobiAct**](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/) | acc, gyro | — | 57 | 13 daily, fall activities |
| [**Skoda Mini Checkpoint**](http://har-dataset.org/lib/exe/fetch.php?media=wiki:dataset:skodaminicp:skodaminicp_2015_08.zip) | acc (3D) | — | 1 | 10 assembly-line activities |



### IMU+ Multimodal Datasets
| Dataset | Sensors | Datasize | Subjects | Activities |
|:--|:--|:--|:--|:--|
| [**RecGym**](https://www.kaggle.com/datasets/zhaxidelebsz/10-gym-exercises-with-615-abstracted-features) | acc, gyro, human body capacitance | 50h | 10 | 12 fitness activities |
| [**WEAR**](https://mariusbock.github.io/wear/) | acc, video | 19 h | 22 | 18 sports activities |
| [**iSPL**](https://github.com/thunguyenth/HAR_IMU_Stretch) | acc, gyro, stretch | — | 1 | 9 daily activities |
| [**HARTH**](https://archive.ics.uci.edu/dataset/779/harth) | acc, video | 35.9 h | 22 | 12 daily activities |
| [**w-HAR**](https://github.com/gmbhat/human-activity-recognition) | acc, gyro, stretch | 3 h | 22 | 7 daily activities |
| [**RealLifeHAR**](https://lbd.udc.es/research/real-life-HAR-dataset/) | acc, gyro, mag, GPS | — | 19 | 4 daily activities |
| [**MMAct**](https://mmact19.github.io/2019/) | RGB, keypoints, acc, gyro, ori, Wi‑Fi, pressure | — | 40 | 37 activities |
| [**HuGaDB**](https://github.com/romanchereshnev/HuGaDB) | acc, gyro, EMG | 10 h | 18 | 12 activities |
| [**RealWorld HAR**](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/) | acc, gyro, mag, GPS, light, sound level | 124.3 h | 15 | 8 daily activities |
| [**ExtraSensory**](http://extrasensory.ucsd.edu/) | acc, gyro, mag, location, audio, additional | — | 60 | 51 activities |
| [**UTD-MHAD**](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html) | RGB, depth, skeleton, acc, gyro | — | 8 | 27 activities |
| [**MHEALTH**](https://archive.ics.uci.edu/dataset/319/mhealth+dataset) | acc, gyro, mag, ECG | — | 10 | 12 daily activities |
| [**Berkeley MHAD**](https://ieeexplore.ieee.org/document/6474999) | acc, optical capture, video, depth, audio | 1.37 h | 12 | 11 daily activities |
| [**PAMAP2**](https://archive.ics.uci.edu/dataset/231/pamap2%2Bphysical%2Bactivity%2Bmonitoring) | acc, gyro, mag, HR | 10 h | 9 | 18 daily activities |
| [**Opportunity**](https://archive.ics.uci.edu/dataset/226/opportunity%2Bactivity%2Brecognition) | acc, gyro, mag, ambient sensors | 25 h | 4 | 9 kitchen + 9 gestures |
| [**MRI**](https://github.com/sizhean/mri) | mmWave, RGB-D, IMU | 5.3 h | 20 | pose estimation | 
| [**NORMWEAR**](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear?tab=readme-ov-file)  | PPG, ECG, EEG, GSR, IMU | 14,943 h | 20 | pose estimation | 
| **SensorLM** [[Paper](https://arxiv.org/abs/2506.09108) ] | PPG, EDA, ACC, TEMP, ALT | 59,749 h | 103,731 | sensor-language study | 
| **Apple Study (AHMS; WBM)** [[Paper](https://arxiv.org/html/2507.00191v1)]  | HealthKit metrics (27) | > 2.5 B h | 162 K | 57 health tasks | 
| [**WESAD**](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)  | EDA/PPG/Temp + Acc | 45 h | 15 | stress detection | 
| [**PPG-Dalia**](https://archive.ics.uci.edu/dataset/495/ppg+dalia)  | ECG, PPG, IMU, GSR | 36 h | 15 | daily activities | 


### Physiology Datasets
| Dataset | Sensors | Datasize | Subjects | Activities |
|:--|:--|:--|:--|:--|
| [**Sleep-EDF**](https://www.physionet.org/content/sleep-edfx/1.0.0/) | EEG/EOG/EMG/ECG | 1,576 h | 197 | sleep stages |
| [**MIT-BIH Arrhythmia**](https://physionet.org/content/mitdb/1.0.0/) | 2‑lead ECG | 1128 h | 47 | ambulatory ECG |
| [**PTB-XL**](https://physionet.org/content/ptb-xl/1.0.3/) | 12‑lead ECG | 6.06 h | 18,885 | ECG status |
| [**TUH EEG**](https://isip.piconepress.com/projects/tuh_eeg/) | multi‑channel EEG | 1476 h | 675 | seizure activity |
| [**Cuff‑Less‑BP**](https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation) | ECG, PPG | 72 h | — | blood‑pressure estimation |
| [**Auditory‑EEG**](https://physionet.org/content/auditory-eeg/1.0.0/) | EEG | 23 h | — | auditory attention |
| [**PhyAAt**](https://phyaat.github.io/dataset)| EEG | 33 h | 25 | auditory attention |
| [**MAUS**](https://github.com/rickwu11/MAUS_dataset_baseline_system?tab=readme-ov-file) | ECG, PPG, GSR | 22 h | 22 | cognitive workload/stress |
| [**Mendeley‑YAAD**](https://data.mendeley.com/datasets/g2p7vwxyn2/4) | ECG, GSR | 5 h | — | affect/stress elicitation |
| [**Brain‑Cognitive**](https://www.physionet.org/content/brain-wearable-monitoring/1.0.0/) | EEG | 85 h | 20 | cognitive state regulation |
| [**EPHNOGRAM**](https://physionet.org/content/ephnogram/1.0.0/) | ECG, PCG | 61 h | 24 | cardiac auscultation |
| [**BIDMC**](https://physionet.org/content/bidmc/1.0.0/) | ECG, PPG | 14 h | 53 | clinical monitoring |
| **MOODS** [[Paper](https://arxiv.org/html/2502.01108v1)] | PPG | 54 K h | 122 | stress monitoring |
| [**SleepFM**](https://github.com/rthapa84/sleepfm-codebase?tab=readme-ov-file) | BAS, ECG, respiratory | 112,544 h | 14,068 | sleep quality monitoring |
| [**RecGym**](https://www.kaggle.com/datasets/zhaxidelebsz/10-gym-exercises-with-615-abstracted-features) | acc, gyro, human body capacitance | 50h | 10 | 12 fitness activities |
| [**PPG-Dalia**](https://archive.ics.uci.edu/dataset/495/ppg+dalia)  | ECG, PPG, IMU, GSR | 36 h | 15 | daily activities | 


### Smart Home Datasets
| Dataset | Sensors | Datasize | Subjects | Activities | 
|:--|:--|:--|:--|:--| 
| [**CASAS (Aruba/Milan/...)**](https://casas.wsu.edu/datasets/) | Ambient binary sensors (motion, doors) | — | — | smart home activities |

### RF Datasets
| Dataset | Sensors | Datasize | Subjects | Activities | 
|:--|:--|:--|:--|:--| 
| **mmWave (var.)**[[Paper](https://dl.acm.org/doi/abs/10.1145/3666025.3699349)] | Range–Doppler / RF point clouds | 5 h | 10 | daily activities (10 scenes) |
| [**MM-Fi**](https://github.com/ybhbingo/MMFi_dataset) | mmWave, LiDAR, Wi‑Fi, RGB‑D | 10.6 h | 40 | 27 daily activities |


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
1. **"SensorLM: Learning the Language of Wearable Sensors"**. *Zhang et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2506.09108)]
2. **"Time2Lang: Bridging Time‑Series Foundation Models and Natural Language"**. *Pillai et al..* PMLR 2025. [[Paper](https://proceedings.mlr.press/v287/pillai25a.html)]
3. **"SensorLLM: Aligning Large Language Models with Motion Sensors for Human Activity Recognition"**. *Li et al..* arXiv 2024. [[Paper](https://arxiv.org/abs/2410.10624)]
4. **"LLaSA: A Multimodal LLM for Human Activity Analysis Through Wearable and Smartphone Sensors"**. *Imran et al..* arXiv 2024. [[Paper](https://arxiv.org/abs/2406.14498)]
5. **"HARGPT: A Language‑Conditioned Foundation Model for Human Activity Recognition"**. *Ji et al..* arXiv 2024. [[Paper](https://arxiv.org/abs/2403.19526)]
6. **"LanHAR: Language‑Driven Human Activity Recognition"**. *Yan et al..* arXiv 2024. [[Paper](https://arxiv.org/abs/2410.00003)]
7. **"StressLLM: Large Language Models for Wearable Stress Detection"**. *Thapa et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2506.04817)]
8. **"DailyLLM: Large Language Models for Daily Behavior Understanding"**. *Kang et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2505.04563)]
9. **"ContextLLM: Multimodal Context Understanding from Wearable Devices"**. *Wang et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2506.07114)]
10. **"Health‑LLM: Aligning Large Language Models with Wearable Sensor Health Data"**. *Liu et al..* Information Fusion 2025. [[Paper](https://doi.org/10.1016/j.inffus.2025.103403)]
11. **"SensorGPT: Generative Pretraining for Wearable Sensing"**. *Sharma et al..* arXiv 2025. [[Paper](https://arxiv.org/abs/2504.06512)]
12. **"Sensor2Text: Enabling Natural Language Interactions for Daily Activity Tracking Using Wearable Sensors"**. *Chen et al..* IMWUT 2024. [[Paper](https://doi.org/10.1145/3699747)]






## Modality Scope

###  Unimodal foundation models
1. **"RelCon: Relative Contrastive Learning for a Motion Foundation Model for Wearable Data"**. *Xu et al.* ICLR 2025. [[Paper](https://arxiv.org/abs/2411.18822)]  
2. **"One Model to Fit Them All: Universal IMU-based Human Activity Recognition with LLM-assisted Cross-dataset Representation"**. *Wei et al.* IMWUT 2025. [[Paper](https://doi.org/10.1145/3749509)]  
3. **"SelfHAR-700k: Self-Supervised Learning for Human Activity Recognition Using 700,000 Person-Days of Wearable Data"**. *Yuan et al.* npj Digital Medicine 2024. [[Paper](https://www.nature.com/articles/s41746-024-01062-3)]  
4. **"Pulse-PPG: An Open-Source Field-Trained PPG Foundation Model for Wearable Applications"**. *Saha et al.* IMWUT 2025. [[Paper](https://doi.org/10.1145/3749494)]  

###  Multimodal foundation models
1. **"Towards Customizable Foundation Models for Human Activity Recognition with Wearable Devices"**. *Qiu et al.* arXiv 2025. [[Paper](https://dl.acm.org/doi/abs/10.1145/3749479)]  
2. **"MASTER: A Multi-modal Foundation Model for Human Activity Recognition"**. *Zhu et al.* IEEE TMC 2025. [[Paper](https://dl.acm.org/doi/10.1145/3749511)]  
3. **"NORMWEAR: Toward Foundation Models for Multivariate Wearable Sensing of Physiological Signals"**. *Luo et al.* arXiv 2024. [[Paper](https://arxiv.org/abs/2412.09758)]  
4. **"MuJo: Multimodal Joint Feature Space Learning for Human Activity Recognition"**. *Fritsch et al.* arXiv 2024. [[Paper](https://arxiv.org/abs/2406.03857)]  

###  Cross-modal foundation models
1. **"SensorLM: Learning the Language of Wearable Sensors"**. *Zhang et al.* arXiv 2025. [[Paper](https://arxiv.org/abs/2506.09108)]  
2. **"SensorLLM: Aligning Large Language Models with Motion Sensors for Human Activity Recognition"**. *Li et al.* arXiv 2024. [[Paper](https://arxiv.org/abs/2410.10624)]  
3. **"IMU2CLIP: Multimodal Contrastive Learning for IMU Motion Sensors from Egocentric Videos and Text"**. *Moon et al.* EMNLP Findings 2023. [[Paper](https://arxiv.org/abs/2210.14395)]  
4. **"FM-Fi 2.0: Foundation Model for Cross-Modal Multi-Person Human Activity Recognition"**. *Weng et al.* IEEE TMC 2025. [[Paper](https://doi.org/10.1109/TMC.2025.3593406)]  
5. **"Leveraging foundation models for zero-shot IoT sensing"**. *Xue et al.* arXiv 2024. [[Paper](https://arxiv.org/abs/2407.19893)]  






## Data Landscape: Collected and Generated Corpora
###  Corpora of sensor data collected in the wild
1. **"One Model to Fit Them All: Universal IMU-based Human Activity Recognition with LLM-assisted Cross-dataset Representation"**. *Wei et al.* IMWUT 2025. [[Paper](https://doi.org/10.1145/3749509)]

2. **"CrossHAR: Generalizing Cross-Dataset Human Activity Recognition via Hierarchical Self-Supervised Pretraining"**. *Hong et al.* IMWUT 2024. [[Paper](https://doi.org/10.1145/3659597)]

3. **"Self-Supervised Learning for Human Activity Recognition Using 700,000 Person-Days of Wearable Data"**. *Yuan et al.* npj Digital Medicine 2024. [[Paper](https://www.nature.com/articles/s41746-024-01062-3)]

4. **"Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals"**. *Luo et al.* arXiv 2024. [[Paper](https://arxiv.org/abs/2412.09758)]

5. **"SleepFM: Multi-Modal Representation Learning for Sleep Across Brain Activity, ECG and Respiratory Signals"**. *Thapa et al.* arXiv 2024. [[Paper](https://arxiv.org/abs/2405.17766)]

6. **"Scaling Wearable Foundation Models"**. *Narayanswamy et al.* arXiv 2024. [[Paper](https://arxiv.org/abs/2410.13638)]

7. **"DailyLLM: Context-Aware Activity Log Generation Using Multi-Modal Sensors and LLMs"**. *Tian et al.* arXiv 2025. [[Paper](https://arxiv.org/abs/2507.13737)]

8. **"IMU2CLIP: Language-Grounded Motion Sensor Translation with Multimodal Contrastive Learning"**. *Moon et al.* EMNLP Findings 2023. [[Paper](https://arxiv.org/abs/2210.14395)]

    
  ###  Generated datasets and augmentation

1. **"On the Benefit of Generative Foundation Models for Human Activity Recognition"**. *Leng et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2310.12085)]

2. **"FM-Fi 2.0: Foundation Model for Cross-Modal Multi-Person Human Activity Recognition"**. *Weng et al.* IEEE TMC 2025. [[Paper](https://doi.org/10.1109/TMC.2025.3593406)]

3. **"TxP: Reciprocal Generation of Ground Pressure Dynamics and Activity Descriptions for Improving Human Activity Recognition"**. *Lala et al.* IEEE TMC 2025. [[Paper](https://arxiv.org/abs/2505.02052)]

4. **"CrossHAR: Generalizing Cross-Dataset Human Activity Recognition via Hierarchical Self-Supervised Pretraining"**. *Hong et al.* IMWUT 2024. [[Paper](https://dl.acm.org/doi/10.1145/3659597)]





## Tokenization and Representation Strategies
###  Window-based and patch-level segmentation

1. **"RelCon: Relative Contrastive Learning for a Motion Foundation Model for Wearable Data"**. *Maxwell A. Xu et al.* arXiv 2024 (ICLR 2025). [[Paper](https://arxiv.org/abs/2411.18822)] 
2. **"Speech Foundation Models Generalize to Time Series Tasks from Wearable Sensor Data"**. *Jaya Narain et al.* arXiv 2025. [[Paper](https://arxiv.org/abs/2509.00221)] 

3. **"SleepFM: Multi-modal Representation Learning for Sleep Across Brain Activity, ECG and Respiratory Signals"**. *Rahul Thapa et al.* arXiv 2024. [[Paper](https://arxiv.org/abs/2405.17766)] 

4. **"Leveraging Foundation Models for Zero-Shot IoT Sensing"**. *Dinghao Xue et al.* arXiv 2024. [[Paper](https://arxiv.org/abs/2407.19893)]    

5. **"Inertial Signal Forecasting with Foundation Model Techniques (Dual-View FM)"**. *Christoph Wieland, Victor Pankratius.* IEEE Sensors Journal 2025. [[Paper](https://ieeexplore.ieee.org/document/11150553)]

6. **"A Novel Human Activity Recognition Framework Based on Pre-Trained Foundation Model (Chronos HAR Adapters)"**. *Xiong et al.* IEEE BIBM 2024. [[Paper](https://ieeexplore.ieee.org/document/10822159)]  
  

  ###  Feature-based aggregation and statistical embeddings

1. **"Beyond Sensor Data: Foundation Models of Behavioral Data from Wearables Improve Health Predictions"**. *Eray Erturk et al.* arXiv 2025. [[Paper](https://arxiv.org/abs/2507.00191)]

2. **"ContextLLM: Meaningful Context Reasoning from Multi-Sensor and Multi-Device Data Using LLMs"**. *Post et al.* arXiv 2025. [[Paper](https://dl.acm.org/doi/10.1145/3708468.3711892)]

3. **"SensorLLM: Aligning Large Language Models with Motion Sensors for Human Activity Recognition"**. *Li et al.* arXiv 2024. [[Paper](SensorLLM: Aligning Large Language Models with Motion Sensors for Human Activity Recognition)]

4. **"Scaling Wearable Foundation Models"**. *Narayanswamy et al.* arXiv 2024. [[Paper](https://arxiv.org/abs/2410.13638)]

5. **"A Personal Health Large Language Model for Sleep and Fitness Coaching (PH-LLM)"**. *Khasentino et al.* Nature Medicine 2025. [[Paper](https://www.nature.com/articles/s41591-025-03888-0)]

6. **"BioSignal Copilot: Leveraging the Power of LLMs in Drafting Reports for Biomedical Signals"**. *Liu et al.* arXiv 2023. [[Paper](https://www.medrxiv.org/content/10.1101/2023.06.28.23291916v1)]

  ###  Spectrogram and frequency-domain embeddings


1. **"Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals"**.  
   *Luo et al.* arXiv 2024. [[Paper](https://arxiv.org/abs/2412.09758)]

2. **"Self-Supervised Learning for Human Activity Recognition Using 700,000 Person-Days of Wearable Data"**.  
   *Yuan et al.* *npj Digital Medicine* 2024. [[Paper](https://www.nature.com/articles/s41746-024-01062-3)]

3. **"Sensor2Text: Enabling Natural Language Interactions for Daily Activity Tracking Using Wearable Sensors"**. *Chen et al..* IMWUT 2024. [[Paper](https://doi.org/10.1145/3699747)]

  ###  Discrete and quantized sensor tokens


1. **"Towards Learning Discrete Representations via Self-Supervision for Wearables-Based Human Activity Recognition"**.  
   *Harish Haresamudram et al.* Sensors 2024. [[Paper](https://www.mdpi.com/1424-8220/24/4/1238)]

2. **"Towards Customizable Foundation Models for Human Activity Recognition with Wearable Devices"**.  
   *Minghui Qiu et al.* ACM IMWUT 2025. [[Paper](https://dl.acm.org/doi/10.1145/3749479)]

3. **"Chronos: Learning the Language of Time Series"**.  
   *Ansari et al.* arXiv 2024. [[Paper](https://arxiv.org/abs/2403.07815)]

4. **"Time2Lang: Bridging Time-Series Foundation Models and Large Language Models for Health Sensing Beyond Prompting"**.  
   *Pillai et al.* arXiv 2025. [[Paper](https://arxiv.org/abs/2502.07608)]

  ###  Multimodal alignment and positional encoding


1. **"SensorLLM: Aligning Large Language Models with Motion Sensors for Human Activity Recognition"**.  
   *Li et al.* arXiv 2024. [[Paper](https://arxiv.org/abs/2410.10624)]

2. **"Visible Light Human Activity Recognition Driven by Generative Language Model"**.  
   *Yang et al.* Elsevier, Information Fusion 2025. [[Paper](https://www.sciencedirect.com/science/article/pii/S1566253525002325)]

3. **"LLMSense: Harnessing LLMs for High-Level Reasoning over Spatiotemporal Sensor Traces"**.  
   *Ouyang et al.* arXiv 2024. [[Paper](https://arxiv.org/abs/2403.19857)]


  ###  Token fusion and cross-modal projection

1. **"IMU2CLIP: Language-Grounded Motion Sensor Translation with Multimodal Contrastive Learning"**. *Moon et al.* EMNLP Findings 2023. [[Paper](https://arxiv.org/abs/2210.14395)]
2. **"FM-Fi 2.0: Foundation Model for Cross-Modal Multi-Person Human Activity Recognition"**. *Weng et al.* IEEE TMC 2025. [[Paper](https://doi.org/10.1109/TMC.2025.3593406)]

## Pretraining Paradigms
  ###  Contrastive pretraining

1. **"Large Model for Small Data: Foundation Model for Cross-Modal RF Human Activity Recognition (FM-Fi)"**.  
   *Weng et al.* ACM 2024. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3666025.3699349)]

2. **"FM-Fi 2.0: Foundation Model for Cross-Modal Multi-Person Human Activity Recognition"**.  
   *Weng et al.* IEEE TMC 2025. [[Paper](https://doi.org/10.1109/tmc.2025.3593406)]

3. **"RelCon: Relative Contrastive Learning for a Motion Foundation Model for Wearable Data"**.  
   *Xu et al.* arXiv 2024 (ICLR 2025). [[Paper](https://doi.org/10.48550/arXiv.2411.18822)]

4. **"IMU2CLIP: Multimodal Contrastive Learning for IMU Motion Sensors from Egocentric Videos and Text"**.  
   *Moon et al.* EMNLP Findings 2023. [[Paper](https://arxiv.org/pdf/2210.14395)]


  ###  Generative pretraining


1. **"Spatial-Temporal Masked Autoencoder for Multi-Sensor Human Activity Recognition (STMAE)"**.  
   *Miao et al.* ACM 2024. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3631415)]

2. **"Inertial Signal Forecasting with Foundation Model Techniques (Dual-View FM)"**.  
   *Wieland & Pankratius.* IEEE Sensors Journal 2025. [[Paper](https://doi.org/10.1109/JSEN.2025.3603456)]

3. **"LLaSA: Large Multimodal Agent for Human Activity Analysis Through Wearable Sensors"**.  
   *Imran et al.* arXiv 2024. [[Paper](https://www.semanticscholar.org/paper/91edd1d5ec1015952944c63fa43727e800699836)]

4. **"Scaling Wearable Foundation Models (LSM)"**.  
   *Narayanswamy et al.* arXiv 2024. [[Paper](https://arxiv.org/abs/2410.13638)]

5. **"SelfPAB: Large-Scale Pre-training on Accelerometer Recordings with Masked Spectrogram Reconstruction"**.  
   *Logacjov et al.* Springer 2024. [[Paper](https://link.springer.com/content/pdf/10.1007/s10489-024-05322-3.pdf)]

   
  ###  Hybrid and self-supervised pretraining


1. **"HAR-DoReMi: Optimizing Data Mixture for Self-Supervised Pretraining in Human Activity Recognition"**.  
   *Ban et al.* arXiv 2025. [[Paper](https://doi.org/10.48550/arXiv.2503.13542)]

2. **"CrossHAR: Generalizing Cross-Dataset Human Activity Recognition via Hierarchical Self-Supervised Pretraining"**.  
   *Hong et al.* IMWUT 2024. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3659597)]

3. **"A Personal Health Large Language Model for Sleep and Fitness Coaching (PH-LLM)"**.  
   *Khasentino et al.* *Nature Medicine* 2025. [[Paper](https://doi.org/10.1038/s41591-025-03888-0)]

4. **"Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals (NORMWEAR)"**.  
   *Luo et al.* arXiv 2024. [[Paper](https://doi.org/10.48550/arXiv.2412.09758)]

5. **"Scaling Wearable Foundation Models (LSM)"**.  
   *Narayanswamy et al.* arXiv 2024. [[Paper](https://arxiv.org/abs/2410.13638)]

6. **"SensorLM: Learning the Language of Wearable Sensors"**.  
   *Zhang et al.* arXiv 2025. [[Paper](https://doi.org/10.48550/arXiv.2506.09108)]

7. **"SensorLLM: Aligning Large Language Models with Motion Sensors for Human Activity Recognition"**.  
   *Li et al.* arXiv 2024. [[Paper](https://doi.org/10.48550/arXiv.2410.10624)]




## Adaptation Strategies
  ###  Parameter-efficient fine-tuning (PEFT)


1. **"LLaSA: Large Multimodal Agent for Human Activity Analysis Through Wearable Sensors"**.  
   *Imran et al.* arXiv 2024. [[Paper](https://www.semanticscholar.org/paper/91edd1d5ec1015952944c63fa43727e800699836)]

2. **"A Novel Human Activity Recognition Framework Based on Pre-Trained Foundation Model (Chronos HAR Adapters)"**.  
   *Xiong et al.* IEEE BIBM 2024. *(No verified link available in file.)*

3. **"Leveraging Large Language Models for Digital Phenotyping and Health Forecasting"**.  
   *Yuan et al.* bioRxiv 2025. [[Paper](https://doi.org/10.1101/2025.05.10.25327354)]

4. **"Large Language Models for Wearable Sensor-Based Activity Understanding"**.  
   *Liu et al.* Sensors 2024. [[Paper](https://doi.org/10.3390/s24155045)]

5. **"Time2Lang: Bridging Time-Series Foundation Models and Large Language Models for Health Sensing Beyond Prompting"**.  
   *Pillai et al.* PMLR 2025. [[Paper](https://arxiv.org/abs/2502.07608)]

6. **"PhysLLM: Harnessing Large Language Models for Physiological Understanding"**.  
   *Xie et al.* arXiv 2025. [[Paper](https://doi.org/10.48550/arXiv.2505.03621)]

7. **"MHARFedLLM: Multimodal Human Activity Recognition Using Federated Large Language Model"**.  
   *Bandyopadhyay et al.* arXiv 2025. [[Paper](https://doi.org/10.48550/arXiv.2508.01701)]

8. **"GOAT: A Generalized Cross-Dataset Activity Recognition Framework"**.  
   *Miao et al.* IMWUT 2024. [[Paper](https://doi.org/10.1145/3699736)]


  ###  Full or partial fine-tuning


1. **"LIMU-BERT: Unleashing the Potential of Unlabeled Data for IMU Sensing Applications"**.  
   *Xu et al.* SenSys 2021. [[Paper](https://doi.org/10.1145/3485730.3485937)]

2. **"Scaling Wearable Foundation Models (LSM)"**.  
   *Narayanswamy et al.* arXiv 2024. [[Paper](https://arxiv.org/abs/2410.13638)]

3. **"LLM4HAR: Generalizable On-device Human Activity Recognition with Large Language Models"**.  
   *Hong et al.* IMWUT 2025. [[Paper](https://doi.org/10.1145/3711896.3737226)]

4. **"RelCon: Relative Contrastive Learning for a Motion Foundation Model for Wearable Data"**.  
   *Xu et al.* ICLR 2025. [[Paper](https://doi.org/10.48550/arXiv.2411.18822)]

5. **"SelfPAB: large-scale pre-training on accelerometer data for human activity recognition"**.  
   *Logacjov et al.* ACM 2024. [[Paper](https://dl.acm.org/doi/abs/10.1007/s10489-024-05322-3)]

6. **"Spatial-Temporal Masked Autoencoder for Multi-Sensor Human Activity Recognition (STMAE)"**.  
   *Miao et al.* ACM 2024. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3631415)]

7. **"Leveraging Large Language Models for Digital Phenotyping and Health Forecasting"**.  
   *Yuan et al.* bioRxiv 2025. [[Paper](https://doi.org/10.1101/2025.05.10.25327354)]



  
  ###  Instruction-tuning & alignment


1. **"StressLLM: Large Language Models for Stress Prediction and Biomarker Reasoning"**.  
   *Thapa et al.* IEEE ICCE 2025. [[Paper](https://doi.org/10.1109/ICCE63647.2025.10929774)]

2. **"Leveraging Large Language Models for Digital Phenotyping: Detecting Depressive State Changes for Patients with Depressive Episodes"**.  
   *Yuan et al.* arXiv 2025. [[Paper](https://doi.org/10.1101/2025.05.10.25327354)]

3. **"LAHAR: Leveraging Language Models for Human Activity Recognition"**.  
   *Chen et al.* IEEE Access 2024. [[Paper](https://doi.org/10.1109/LS2463127.2024.10881660)]

4. **"DailyLLM: Context-Aware Activity Log Generation Using Multi-Modal Sensors and LLMs"**.  
   *Tian et al.* arXiv 2025. [[Paper](https://doi.org/10.48550/arXiv.2507.13737)]

5. **"Enabling On-Device LLMs Personalization with Sensor Prompts"**.  
   *Zhang et al.* arXiv 2024. [[Paper](https://arxiv.org/pdf/2407.04418)]

6. **"Sensor2Text: Enabling Natural Language Interactions for Daily Activity Tracking Using Wearable Sensors"**.  
   *Chen et al.* IMWUT 2024. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3699747)]

7. **"SensorLM: Learning the Language of Wearable Sensors"**.  
   *Zhang et al.* arXiv 2025. [[Paper](https://doi.org/10.48550/arXiv.2506.09108)]

8. **"A Personal Health Large Language Model for Sleep and Fitness Coaching (PH-LLM)"**.  
   *Khasentino et al.* Nature Medicine 2025. [[Paper](https://doi.org/10.1038/s41591-025-03888-0)]



## Downstream Capabilities
  ###  Zero-/few-shot & open-set recognition



1. **"Large Model for Small Data: Foundation Model for Cross-Modal RF Human Activity Recognition"**.  
   *Weng et al.* ACM 2024. [[Paper](https://dl.acm.org/doi/abs/10.1145/3666025.3699349)]

2. **"FM-Fi 2.0: Foundation Model for Cross-Modal Multi-Person Human Activity Recognition"**.  
   *Weng et al.* IEEE TMC 2025. [[Paper](https://doi.org/10.1109/tmc.2025.3593406)]

3. **"ZARA: Zero-Shot Motion Time-Series Analysis via LLM-Guided Knowledge Retrieval and Reasoning"**.  
   *Li et al.* arXiv 2025. [[Paper](https://doi.org/10.48550/arXiv.2508.04038)]

4. **"HARGPT: Are LLMs Zero-Shot Human Activity Recognizers?"**.  
   *Ji et al.* arXiv 2024. [[Paper](http://arxiv.org/pdf/2403.02727)]

5. **"EEG-GPT: Exploring Capabilities of Large Language Models for EEG-Based Abnormality Detection"**.  
   *Kim et al.* arXiv 2024. [[Paper](https://doi.org/10.48550/arXiv.2401.18006)]


  
  ###  Cross-dataset / cross-device / cross-user generalization

1. **"Towards Customizable Foundation Models for Human Activity Recognition with Wearable Devices (HAR-FM)"**.  
   *Qiu et al.* IMWUT 2025. [[Paper](https://doi.org/10.1145/3749479)]

2. **"One Model to Fit Them All: Universal IMU-based Human Activity Recognition with LLM-assisted Cross-dataset Representation"**.  
   *Wei et al.* ACM IMEUT 2025. [[Paper](https://dl.acm.org/doi/10.1145/3749509)]

3. **"MASTER: A Multi-Modal Foundation Model for Human Activity Recognition"**.  
   *Zhu et al.* IMWUT 2025. [[Paper](https://doi.org/10.1145/3749511)]

4. **"A Novel Human Activity Recognition Framework Based on Pre-Trained Foundation Model (Chronos HAR Adapters)"**.  
   *Xiong et al.* IEEE BIBM 2024. *(Link unavailable in verified file)*

5. **"Self-supervised learning for human activity recognition using 700,000 person-days of wearable data"**.  
   *Yuan et al.* *npj Digital Medicine* 2024. [[Paper](https://www.nature.com/articles/s41746-024-01062-3)]




  
  ###  Cross-modal retrieval and search

1. **"Multimodal Foundation Model for Cross-Modal Retrieval and Recognition (AURA-MFM)"**.  
   *Matsuishi et al.* arXiv 2025. [[Paper](https://doi.org/10.48550/arXiv.2506.03174)]

2. **"GLOSS: Group of LLMs for Open-ended Sensemaking of Passive Sensing Data for Health and Wellbeing"**.  
   *Choube et al.* IMWUT 2025. [[Paper](https://doi.org/10.1145/3749474)]

3. **"Vital Insight: Assisting Experts' Context-Driven Sensemaking of Multi-modal Personal Tracking Data Using Visualization and Human-in-the-Loop LLM"**.  
   *Li et al.* IMWUT 2025. [[Paper](https://doi.org/10.1145/3749508)]

4. **"PhysioLLM: Supporting Personalized Health Insights with Wearables and Large Language Models"**.  
   *Fang et al.* arXiv 2024. [[Paper](http://arxiv.org/pdf/2406.19283)]

5. **"SleepFM: Multi-modal Representation Learning for Sleep Across Brain Activity, ECG and Respiratory Signals"**.  
   *Thapa et al.* arXiv 2024. [[Paper](https://doi.org/10.48550/arXiv.2405.17766)]


  
  ###  Language-grounded captioning, Q&A, and reasoning

1. **"Sensor2Text: Enabling Natural Language Interactions for Daily Activity Tracking Using Wearable Sensors"**.  
   *Chen et al.* IMWUT 2024. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3699747)]

2. **"SensorLM: Learning the Language of Wearable Sensors"**.  
   *Zhang et al.* arXiv 2025. [[Paper](https://doi.org/10.48550/arXiv.2506.09108)]

3. **"SensorLLM: Aligning Large Language Models with Motion Sensors for Human Activity Recognition"**.  
   *Li et al.* arXiv 2024. [[Paper](https://doi.org/10.48550/arXiv.2410.10624)]

4. **"Time2Lang: Bridging Time-Series Foundation Models and Large Language Models for Health Sensing Beyond Prompting"**.  
   *Pillai et al.* PMLR 2025. [[Paper](https://arxiv.org/abs/2502.07608)]

5. **"Visible Light Human Activity Recognition Driven by Generative Language Model"**.  
   *Yang et al.* *Information Fusion* 2025. [[Paper](https://doi.org/10.1016/j.inffus.2025.103159)]

6. **"LLMSense: Harnessing LLMs for High-Level Reasoning over Spatiotemporal Sensor Traces"**.  
   *Ouyang et al.* SenSys-ML 2024. [[Paper](https://doi.org/10.1109/SenSys-ML62579.2024.00007)]

  
  ###  Generative reconstruction, forecasting, and imputation


1. **"Spatial-Temporal Masked Autoencoder for Multi-Sensor Human Activity Recognition (STMAE)"**.  
   *Miao et al.* ACM 2024. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3631415)]

2. **"CrossHAR: Generalizing Cross-Dataset Human Activity Recognition via Hierarchical Self-Supervised Pretraining"**.  
   *Hong et al.* IMWUT 2024. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3659597)]

3. **"Scaling Wearable Foundation Models (LSM)"**.  
   *Narayanswamy et al.* arXiv 2024. [[Paper](https://arxiv.org/abs/2410.13638)]

4. **"Inertial Signal Forecasting with Foundation Model Techniques (Dual-View FM)"**.  
   *Wieland & Pankratius.* IEEE Sensors Journal 2025. [[Paper](https://doi.org/10.1109/JSEN.2025.3603456)]

5. **"RobustHAR: Multi-Scale Spatial-Temporal Masked Autoencoder for Robust Human Activity Recognition"**.  
   *Liu et al.* IJCAI 2025. [[Paper](https://doi.org/10.24963/ijcai.2025/952)]

6. **"UniMTS – Unified Pre-training for Motion Time-Series Forecasting and Recognition"**.  
   *Zhang et al.* arXiv 2024. [[Paper](https://doi.org/10.48550/arXiv.2410.19818)]

   
  ###  On-device, federated, and online adaptation


1. **"LLM4HAR: Generalizable On-device Human Activity Recognition with Large Language Models"**.  
   *Hong et al.* KDD 2025. [[Paper](https://doi.org/10.1145/3711896.3737226)]

2. **"DailyLLM: Context-Aware Activity Log Generation Using Multi-Modal Sensors and LLMs"**.  
   *Tian et al.* arXiv 2025. [[Paper](https://doi.org/10.48550/arXiv.2507.13737)]

3. **"MHARFedLLM: Multimodal Human Activity Recognition Using Federated Large Language Model"**.  
   *Bandyopadhyay et al.* arXiv 2025. [[Paper](https://doi.org/10.48550/arXiv.2508.01701)]

4. **"ContextLLM: Meaningful Context Reasoning from Multi-Sensor and Multi-Device Data Using LLMs"**.  
   *Post et al.* ACM HotMobile 2025. [[Paper](https://doi.org/10.1145/3708468.3711892)]

5. **"Enabling On-Device LLMs Personalization with Smartphone Sensing"**.  
   *Zhang et al.* arXiv 2024. [[Paper](https://dl.acm.org/doi/10.1145/3675094.3677545)]

6. **"Activity transitions for semi-supervised federated learning in sensor-based human activity recognition"**.  
   *Bukit et al.* Elsevier, Applied Soft Computing 2025. [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S1568494625011068)]

  

## Deployment Settings
  ###  Cloud-scale training and centralized evaluation


1. **"Large Model for Small Data: Foundation Model for Cross-Modal RF Human Activity Recognition (FM-Fi)"**. *Weng et al..* SenSys 2024. [[Paper](https://doi.org/10.1145/3589132.3625578)]

2. **"FM-Fi 2.0: Foundation Model for Cross-Modal Multi-Person Human Activity Recognition"**.  
   *Weng et al.* IEEE TMC 2025. [[Paper](https://doi.org/10.1109/tmc.2025.3593406)]

3. **"Scaling Wearable Foundation Models (LSM)"**.  
   *Narayanswamy et al.* arXiv 2024. [[Paper](https://arxiv.org/abs/2410.13638)]
   
  ###  On-device and mobile execution

1. **"LLMSense: Harnessing LLMs for High-Level Reasoning over Spatiotemporal Sensor Traces"**.  
   *Ouyang et al.* SenSys-ML 2024. [[Paper](https://doi.org/10.1109/SenSys-ML62579.2024.00007)]

2. **"Leveraging Large Language Models for Digital Phenotyping: Detecting Depressive State Changes for Patients with Depressive Episodes"**.  
   *Yuan et al.* bioRxiv 2025. [[Paper](https://doi.org/10.1101/2025.05.10.25327354)]

3. **"LanHAR: Language-Centered Human Activity Recognition via Sensor–Text Alignment"**.  
   *Yan et al.* arXiv 2024. [[Paper](https://arxiv.org/pdf/2402.01049)]

4. **"Enabling On-Device LLMs Personalization with Sensor Prompts"**.  
   *Zhang et al.* arXiv 2024. [[Paper](https://arxiv.org/pdf/2407.04418)]

5. **"LLM4HAR: Generalizable On-Device Human Activity Recognition with Large Language Models"**.  
   *Hong et al.* IMWUT 2025. [[Paper](https://doi.org/10.1145/3711896.3737226)]

6. **"Few-Shot Human Activity Recognition Using Lightweight Language Models"**.  
   *Cruciani et al.* IEEE ICCCN 2025. [[Paper](https://doi.org/10.1109/ABC64332.2025.11118559)]

## Application Domains
  ###  General-purpose HAR / ADL


1. **"oneHAR: Universal Representation Learning for Human Activity Recognition via Inertial Signals"**.  
   *Wei et al.* IEEE JBHI 2025. [[Paper](https://doi.org/10.1109/JBHI.2025.3490181)]

2. **"CrossHAR: Generalizing Cross-Dataset Human Activity Recognition via Hierarchical Self-Supervised Pretraining"**.  
   *Hong et al.* IMWUT 2024. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3659597)]

3. **"Towards Customizable Foundation Models for Human Activity Recognition with Wearable Devices (HAR-FM)"**.  
   *Qiu et al.* IMWUT 2025. [[Paper](https://doi.org/10.1145/3749479)]
   
  ###  Healthcare & wellbeing

1. **"PhysioLLM: Supporting Personalized Health Insights from Multimodal Physiological Signals"**.  
   *Fang et al.* arXiv 2024. [[Paper](http://arxiv.org/pdf/2406.19283)]

2. **"Vital Insight: Assisting Experts’ Context-Driven Retrieval and Reasoning over Longitudinal Health Data"**.  
   *Li et al.* IMWUT 2025. [[Paper](https://doi.org/10.1145/3749508)]

3. **"Pulse-PPG: An Open-Source Field-Trained PPG Foundation Model for Continuous Cardiovascular Monitoring"**.  
   *Saha et al.* IMWUT 2025. [[Paper](https://doi.org/10.1145/3749494)]

4. **"SleepFM: Multi-Modal Representation Learning for Sleep Across Brain Activity, ECG, and Respiratory Signals"**.  
   *Thapa et al.* arXiv 2024. [[Paper](https://doi.org/10.48550/arXiv.2405.17766)]

5. **"Large Language Models for Wearable Data Analysis and Clinical Narratives"** *(bohi2024large – unverified)*.  
   *(No verified link found in uploaded file.)*

  
  ###  Smart-home & context-aware environments


1. **"Large Model for Small Data: Foundation Model for Cross-Modal RF Human Activity Recognition (FM-Fi)"**.  
   *Weng et al.* ACM 2024. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3666025.3697865)]

2. **"Visible Light Human Activity Recognition Driven by Generative Language Model"**.  
   *Yang et al.* *Information Fusion* 2025. [[Paper](https://doi.org/10.1016/j.inffus.2025.103159)]

3. **"ContextAgent: Context-Aware Proactive LLM Agent for Multimodal Human Activity Understanding"**.  
   *Yang et al.* arXiv 2025. [[Paper](https://doi.org/10.48550/arXiv.2505.14668)]

   
  ###  Interactive & agentic assistants


1. **"LLMSense: Harnessing LLMs for High-Level Reasoning over Spatiotemporal Sensor Traces"**.  
   *Ouyang et al.* SenSys-ML 2024. [[Paper](https://doi.org/10.1109/SenSys-ML62579.2024.00007)]

2. **"LanHAR: Language-Centered Human Activity Recognition via Sensor–Text Alignment"**.  
   *Yan et al.* arXiv 2024. [[Paper](https://arxiv.org/pdf/2402.01049)]

3. **"DailyLLM: Context-Aware Activity Log Generation Using Multi-Modal Sensors and LLMs"**.  
   *Tian et al.* arXiv 2025. [[Paper](https://doi.org/10.48550/arXiv.2507.13737)]




