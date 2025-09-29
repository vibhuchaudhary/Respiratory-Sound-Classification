# 📊 Dataset Information

This project uses the **Respiratory Sound Database** from Kaggle, downloaded automatically via [`kagglehub`](https://pypi.org/project/kagglehub/).

---

## 🔽 Automatic Download

The dataset is fetched programmatically when running the notebook:

```python
import kagglehub
path = kagglehub.dataset_download("vbookshelf/respiratory-sound-database")
```

✨ **No manual download needed!**
The dataset is cached locally after the first download for faster re-runs.

---

## 📁 Dataset Structure (After Download)

The dataset is stored under your cache directory:

```
~/.cache/kagglehub/datasets/vbookshelf/respiratory-sound-database/
├── Respiratory_Sound_Database/
│   ├── audio_and_txt_files/        # 920 .wav files + .txt annotations
│   │   ├── 101_1b1_Al_sc_Meditron.wav
│   │   ├── 101_1b1_Al_sc_Meditron.txt
│   │   └── ... (more files)
│   └── patient_diagnosis.csv       # Patient diagnosis labels
```

---

## 📌 Dataset Characteristics

* **Total Files**: 920 audio recordings + text annotations
* **Patients**: 126 individuals
* **Duration**: 10–90 seconds per recording
* **Format**: WAV audio files + TXT annotations + CSV labels
* **Size**: ~500 MB

### 🔹 Classes

* The **original dataset** contained **8 diagnostic categories**:

  * Bronchiectasis
  * Bronchiolitis
  * COPD
  * Healthy
  * Pneumonia
  * URTI (Upper Respiratory Tract Infection)
  * Asthma
  * LRTI (Lower Respiratory Tract Infection)

* For this project, **Asthma** and **LRTI** were removed due to **very few samples**, leaving **6 final classes** used for training:

---

## 💻 Local Usage

If running locally (outside Google Colab):

1. Install `kagglehub`:

   ```bash
   pip install kagglehub
   ```
2. Set up your Kaggle API credentials (`kaggle.json`).
3. Run the notebook — the dataset will be downloaded and cached automatically.

---

## ⚠️ Usage Note

* The dataset is **cached after first download** → subsequent runs are much faster.
* If cache is cleared, `kagglehub` will re-download the dataset.

---
