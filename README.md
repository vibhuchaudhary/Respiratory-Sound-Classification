# Respiratory Disease Classification

A deep learning model for **automated respiratory disease classification** using audio recordings, with **selective augmentation** to address class imbalance.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Dpt0gq7pRSeaXmTOR_p4DEStpRIO5eye?usp=sharing)

---

## 📌 Overview

This project implements a **2D CNN** that classifies respiratory diseases from lung sound recordings.
The key innovation is **selective augmentation** – applying augmentation **only to minority classes** – which significantly improves balanced classification.

---

## 📂 Dataset

**Source**: [Kaggle – Respiratory Sound Database](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database/data)

**Characteristics**

* 920 audio recordings from 126 patients
* 6,898 respiratory cycles (~5.5 hours total)
* Duration: 10–90 seconds per recording
* Classes:
* The **original dataset** contained **8 diagnostic categories**:

  * Bronchiectasis
  * Bronchiolitis
  * COPD
  * Healthy
  * Pneumonia
  * URTI (Upper Respiratory Tract Infection)
  * Asthma
  * LRTI (Lower Respiratory Tract Infection)

**Challenge**: Severe class imbalance (8.5:1 ratio)

---

## ⚙️ Methodology

### 🔹 Pipeline

`Raw Audio → Preprocessing → Selective Augmentation → Mel-Spectrograms → 2D CNN`

### 1. Preprocessing

* Load audio with **librosa** (22 kHz, 20-second clips)
* Normalize and standardize duration

### 2. Selective Augmentation (Minority Classes Only)

| Technique       | Parameters               | Applied To       |
| --------------- | ------------------------ | ---------------- |
| Noise Addition  | 0.005, 0.01, 0.02        | Non-COPD classes |
| Time Stretching | 0.8×, 0.9×, 1.1×, 1.2×   | Non-COPD classes |
| Pitch Shifting  | -2, -1, +1, +2 semitones | Non-COPD classes |

✅ Up to **12× augmentation per minority sample**
✅ Balance improved from **8.5:1 → 2.1:1**

### 3. Feature Extraction

* **Mel-spectrograms**: 128 mel bins × 432 time frames
* Converted to **dB scale** and normalized to [0,1]
* Final input shape: `(128, 432, 1)`

### 4. Model Architecture (2D CNN)

```
Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25)
Conv2D(128) → BatchNorm → Conv2D(128) → MaxPool → Dropout(0.25)
Conv2D(256) → BatchNorm → Conv2D(256) → MaxPool → Dropout(0.25)
Conv2D(512) → BatchNorm → GlobalAvgPool
Dense(512) → Dropout(0.5) → Dense(256) → Dropout(0.4)
Dense(128) → Dropout(0.3) → Dense(#classes, softmax)
```

---

## 🚀 Quick Start

### 🔹 Run in Google Colab (Recommended)

1. Click the **“Open in Colab”** badge above
2. Run all cells – dataset downloads automatically via `kagglehub`

### 🔹 Local Setup

```bash
# Clone repository
git clone https://github.com/your-username/respiratory-disease-classification.git
cd respiratory-disease-classification

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook notebooks/Respiratory_Sound_Classification.ipynb
```

---

## 🎯 Use Pre-Trained Model

```python
import tensorflow as tf

model = tf.keras.models.load_model("models/trained_model.keras")
```

---

## 📁 Project Structure

```
Respiratory-Sound-Classification/
├── notebooks/
│   └── Respiratory_Sound_Classification.ipynb   # Main training notebook
├── models/
│   └── trained_model.keras                 # Pre-trained model (31.6 MB)
├── data/
│   └── README.md                           # Dataset download instructions
├── requirements.txt                        # Dependencies
├── README.md   
├── LICENSE                            # Project documentation
```

---

## 🌟 Key Features

* ✅ Automatic dataset download via **kagglehub**
* ✅ **Selective augmentation** for minority classes
* ✅ 2D CNN architecture preserving **time-frequency relationships**
* ✅ Balanced classification for medical audio datasets
* ✅ End-to-end pipeline: from raw audio → disease prediction

---

## 📖 Citation

```bibtex
@dataset{respiratory_sound_db_2019,
  title     = {Respiratory Sound Database},
  author    = {vbookshelf},
  year      = {2019},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database}
}
```
