# ⚽ Fine-grained Soccer Action Recognition 
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![WandB](https://img.shields.io/badge/Weights_&_Biases-Tracked-yellow)](https://wandb.ai/)

## Project Overview
This is a lightweight and high-efficiency spatiotemporal video analysis system built upon the **PyTorch** framework. 

The core architecture utilizes **X3D**, an advanced 3D convolutional neural network optimized to capture critical **motion features** from consecutive video frames. By processing temporal sequences, the system accurately classifies fine-grained, complex player behaviors within highly dynamic soccer broadcast scenarios.

---
## ⚡ Key Accomplishments & Technical Highlights

* **雙路時序剪輯資料增強 (Dual-path Temporal Augmentation)：** 自定義 `split_temporal_segments` 機制，將 16 影格的影片動態裁剪為兩段不等長、不重疊的局部時序片段，強迫模型學習不同運動速度下的泛化特徵。
* **困難樣本聚焦優化 (Focal Loss Exploration)：** 引入 `FocalLoss` 結合類別不平衡處理（Class Balancing），有效解決運動影片中大量待機（Idle）等背景樣本引發的長尾效應（Long-tail distribution）。
* **滑動視窗即時推理 (Sliding-Window Inference)：** 實現 `Player` 物件追蹤暫存機制，利用時序滑動視窗技術，在不需要對整段影片重新編碼的情況下，對特定球員進行連續 8 影格的即時流式動作預測。
* **工業級實驗追蹤：** 整合 **Weights & Biases (WandB)** 與自動混淆矩陣（Confusion Matrix）視覺化，系統化追蹤超參數實驗與學習率（Cosine Annealing with Warmup）的收斂表現。

---

## Visualization Results 
The system utilizes distinct bounding box colors to real-time display different player actions:
* <font color="green">**Gree:**</font> Moving
* <font color="yellow">**Yello:**</font> Standing
* <font color="red">**Red :**</font> Kicking
* <font color="blue">**Blue :**</font> Falling



https://github.com/user-attachments/assets/8af9f32e-dbb8-419a-97a4-347560b81568





---

## 📁 Repository Structure & Implementation

* `main_proj.py`: 核心訓練管線。包含動態學習率排程、雙路時序增強與 Focal Loss 分類器優化。
* `vis_video.py`: 視覺化推論腳本。負責讀取球員標籤、局部畫面裁剪、流式時序預測，並使用 OpenCV 繪製動態 Bounding Box 與信心度。
* `model/X3D_proj.py`: 基於 3D 卷積（Spatiotemporal Convolution）的 X3D 模型骨幹網路實作。
* `data/dataset.py`: 影片資料集載入與動態平衡（Balanced Sampling）模組。
