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
  <img src="images/FM_HAR_Trend.png" width="400" />
  <img src="images/FM_HAR_Cloud.png" width="400" /> 
</p>


Left: HAR-FM papers showing a sharp acceleration with the vast majority of works emerging since 2024.

Right: Representative model name cloud.



## Table of Contents

- [The Trends of FMs for HAR](#the-trends-of-fms-for-har)
- [Generalization-Oriented Training Settings](#generalization-oriented-training-settings)
    - [Within-Group](#within-group)
    - [Cross-Person](#cross-person)
    - [Cross-Device](#cross-device)
    - [Cross-Position](#cross-position)
    - [Cross-Activity](#cross-activity)
    - [Cross-Dataset](#cross-dataset)
- [Datasets and Benchmarks](#datasets-and-benchmarks)
    - [IMU-Only Dataset](#imu-only-dataset)
    - [Multimodal Dataset](#multimodal-dataset)
    - [Dataset Library](#dataset-library)
    - [Tools and Benchmarks](#tools-and-benchmarks)
- [Model-Centric Methodology](#model-centric-methodology)
    - [Supervised Learning](#supervised-learning)
        - [Feature Invariance](#feature-invariance)
        - [Concept Invariance](#concept-invariance)
        - [Multi-Task Learning](#multi-task-learning)
        - [Federated Learning](#federated-learning)
    - [Weakly-Supervised Learning](#weakly-supervised-learning)
        - [Semi-Supervised Learning](#semi-supervised-learning)
        - [Active Learning](#active-learning)
        - [Inexact Supervised Learning](#inexact-supervised-learning)
    - [Unsupervised Learning](#unsupervised-learning)
        - [Clustering Analysis](#clustering-analysis)
        - [Association Analysis](#association-analysis)
    - [Self-Supervised Learning](#self-supervised-learning)
        - [Transformation Recognition](#transformation-recognition)
        - [Reconstruction](#reconstruction)
        - [Contrastive Learning](#contrastive-learning)
        - [Hybrid Approaches](#hybrid-approaches)
    - [LLM-based Learning](#llm-based-learning)
        - [LLM-Assisted Enhancer](#llm-assisted-enhancer)
        - [LLM-Centered Classifier](#llm-centered-classifier)
        - [LLM-Empowered Agent](#llm-empowered-agent)
- [Data-Centric Methodology](#data-centric-methodology)
    - [Multi-Modal Fusion](#multi-modal-fusion)
    - [Cross-Modal Learning](#cross-modal-learning)
        - [Cross-Modal Conversion](#cross-modal-conversion)
        - [Cross-Modal Contrastive](#cross-modal-contrastive)
    - [Data Augmentation](#data-augmentation)
- [Applications](#applications)
    - [Healthcare and Rehabilitation](#healthcare-and-rehabilitation)
    - [Sports and Fitness Monitoring](#sports-and-fitness-monitoring)
    - [Work Assessment](#work-assessment)
    - [Smart Home and Assisted Living](#smart-home-and-assisted-living)
    - [Transportation and Mobility](#transportation-and-mobility)
    - [Human-Robot Interaction](#human-robot-interaction)
    - [AR and VR Interaction](#ar-and-vr-interaction)
    - [Embodied Agents](#embodied-agents)
- [Citation](#citation)









