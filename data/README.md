# Data Directory

This folder is intentionally excluded from version control (see `.gitignore`).

## Setup Instructions

1. Download the **NIH Malaria Cell Images Dataset** from Kaggle:  
   👉 https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

2. Extract and place the folders as follows:

```
data/
├── Parasitized/     ← 13,779 infected cell images (.png)
└── Uninfected/      ← 13,779 healthy cell images (.png)
```

3. Run preprocessing:

```bash
python src/preprocessing.py
```

This will generate `X_train.npy`, `X_val.npy`, `X_test.npy`, and corresponding label files in this folder, which are used by `train.py` for fast data loading.
