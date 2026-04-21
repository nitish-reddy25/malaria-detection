# 🧬 Malaria Cell Detection — Hybrid Deep Learning Models

> **Final Year B.Tech Project · Computer Science & Engineering**  
> Madanapalle Institute of Technology & Science (MITS), 2024–2025

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Best%20Accuracy-96.20%25-brightgreen?style=flat-square)]()

---

## 📋 Overview

This project develops an automated malaria detection system using deep learning models trained on microscopic blood smear images. The system classifies red blood cell images as either **Parasitized** (malaria-infected) or **Uninfected** with high accuracy, supporting faster and more reliable diagnosis — especially in resource-constrained healthcare settings.

The core architecture benchmarks three models:
- A **custom CNN** optimized for this task
- **VGG19** with transfer learning
- **ResNet50** with transfer learning
- A **hybrid CNN-LSTM-BiLSTM** model combining spatial and sequential feature analysis

---

## 🏆 Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| **Custom CNN** | **96.84%** | 0.9785 | 0.9579 | 0.9681 | 0.9923 |
| VGG19 (Transfer Learning) | 92.71% | 0.9435 | 0.9086 | 0.9257 | 0.9782 |
| ResNet50 (Transfer Learning) | 69.34% | 0.6754 | 0.7446 | 0.7083 | 0.7610 |
| Hybrid CNN-BiLSTM | 96.41% | 0.9720 | 0.9557 | 0.9638 | 0.9916 |

> Dataset: NIH Malaria Cell Images Dataset — 27,558 images (Parasitized: 13,779 / Uninfected: 13,779)  
> Best model: **Custom CNN** at 96.84% test accuracy  
> Results generated on: NVIDIA T4 GPU (Google Colab)

---

## 📁 Project Structure

```
malaria-detection/
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA, class distribution, sample visualization
│   ├── 02_preprocessing.ipynb          # Preprocessing pipeline walkthrough
│   ├── 03_model_cnn.ipynb              # Custom CNN training & evaluation
│   ├── 04_model_vgg19.ipynb            # VGG19 transfer learning
│   ├── 05_model_resnet50.ipynb         # ResNet50 transfer learning
│   └── 06_model_hybrid.ipynb           # CNN-LSTM-BiLSTM hybrid model
├── src/
│   ├── preprocessing.py                # Image loading, augmentation, data splits
│   ├── models.py                       # All model architectures
│   ├── train.py                        # Training pipeline with callbacks
│   ├── evaluate.py                     # Metrics, confusion matrix, plots
│   └── predict.py                      # Single-image inference utility
├── app/
│   ├── app.py                          # Flask web application
│   ├── templates/
│   │   ├── index.html                  # Upload interface
│   │   └── result.html                 # Prediction result display
│   └── static/
│       └── style.css                   # App styling
├── results/
│   ├── training_curves/                # Loss & accuracy plots (PNG)
│   ├── confusion_matrices/             # Per-model confusion matrices
│   └── metrics_summary.csv            # All experiment results
├── tests/
│   └── test_preprocessing.py          # Unit tests
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/nitish-reddy25/malaria-detection.git
cd malaria-detection
pip install -r requirements.txt
```

### 2. Download Dataset

Download the [NIH Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) from Kaggle and extract to:

```
data/
├── Parasitized/
└── Uninfected/
```

### 3. Train a Model

```bash
# Train the custom CNN
python src/train.py --model cnn --epochs 50 --batch_size 32

# Train VGG19 with transfer learning
python src/train.py --model vgg19 --epochs 30 --batch_size 32

# Train the hybrid model
python src/train.py --model hybrid --epochs 50 --batch_size 32
```

### 4. Run the Web App

```bash
cd app
python app.py
# Open http://localhost:5000
```

---

## 🧠 Model Architectures

### Custom CNN
Six convolutional blocks with BatchNormalization and Dropout, designed from scratch for this binary classification task. Achieves high recall on parasitized cells with minimal compute.

### Transfer Learning (VGG19 / ResNet50)
ImageNet-pretrained backbones with frozen convolutional bases and custom classification heads (GlobalAveragePooling → Dense 256 → Dropout → Dense 1).

### Hybrid CNN-LSTM-BiLSTM
Spatial feature maps from CNN layers are reshaped into sequences and passed through BiLSTM layers, allowing the model to capture both local texture patterns and spatial dependencies across cell regions.

---

## 📊 Sample Results

| Input | Prediction | Confidence |
|---|---|---|
| Parasitized cell image | 🔴 Parasitized | 99.8% |
| Uninfected cell image | 🟢 Normal | 99.6% |

*(See `results/` folder for full confusion matrices and training curves)*

---

## ⚙️ Requirements

See `requirements.txt`. Key dependencies:
- TensorFlow 2.10+
- OpenCV-Python
- Flask 2.x
- scikit-learn
- NumPy, Pandas, Matplotlib, Seaborn

---

## 👥 Team

| Name | Roll Number |
|---|---|
| Y. Nitish Kumar Reddy | 21691A05C9 |
| B. Mahendra | 21691A0598 |
| M. Manjunath | 21691A05A5 |

**Guide:** G. Vasundhra Devi, Assistant Professor, Dept. of CSE, MITS  
**Institution:** Madanapalle Institute of Technology & Science (MITS), Andhra Pradesh

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- NIH / Kaggle for the [Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
- Dr. M. Sreedevi (HoD, CSE) and the faculty of MITS for their support
