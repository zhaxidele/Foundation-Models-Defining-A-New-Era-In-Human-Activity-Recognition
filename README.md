# Foundation-Models-in-the-Area-of-Human-Activity-Recognition
## (continue updating ...)

> This repository aims to provide a curated, continuously updated index of foundation models (FMs) for human activity recognition (HAR). The organization follows our survey’s lifecycle-based taxonomy and major development directions.

## The brief history of sensor-based HAR
<p align="center">
    <img src="images/FM_HAR_TimeLine.png" alt="Description" width="600">
</p>

Historical development of sensor-based Human Activity Recognition (HAR) models. From classical machine learning with hand-crafted features and shallow classifiers to the rise of deep learning with CNNs and RNNs, the field progressed toward a phase focused on transfer and domain generalization (robustness across users, devices, and datasets). More recently, self-supervised learning (SSL) approaches have enabled pretraining on unlabeled sensor data using contrastive or masked objectives. Today, the field is moving toward foundation models, exemplified by large-scale sensor–language alignment, emphasizing scalability, generalization, and interpretability

## The Trends of FMs for HAR

Here are the growth trends of publications since 2022 and the model names cloud:

<p align="center">
  <img src="images/FM_HAR_Trend.png" width="450" />
  <img src="images/FM_HAR_Cloud.png" width="450" /> 
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
1. **"Beyond Sensor Data: Foundation Models of Behavioral Data from Wearables Improve Health Predictions"**. *Eray Erturk et al.* Arxiv 2025. [[Paper](https://arxiv.org/abs/2507.00191)]

### Dual-encoder
1. **". Gem: Empowering mllm for grounded ecg understanding with time series and images"**. *Xiang Lan et al.* Arxiv 2025. [[Paper](https://arxiv.org/abs/2503.06073)]

### Encoder-decoder stacks
1. **"Sensor2text: Enabling natural language interactions for daily activity tracking using wearable sensors"**. *Wenqiang Chen et al.* IMWUT 2024. [[Paper](https://dl.acm.org/doi/10.1145/3699747)]

### Language-model stacks
1. **"Large language models for wearable data analysis and interpretation"**. *Simon Boehi et al.* ICLR 2024. [[Paper](https://openreview.net/pdf?id=GoWD6logcd)]




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

