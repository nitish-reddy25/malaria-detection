# Results

This folder contains outputs generated after training. Subfolders are created automatically.

```
results/
├── saved_models/
│   ├── cnn_best.h5              ← Best checkpoint for Custom CNN
│   ├── vgg19_best.h5            ← Best checkpoint for VGG19
│   ├── resnet50_best.h5         ← Best checkpoint for ResNet50
│   └── hybrid_best.h5           ← Best checkpoint for Hybrid CNN-BiLSTM
├── training_logs/
│   ├── cnn_training_log.csv
│   ├── vgg19_training_log.csv
│   ├── resnet50_training_log.csv
│   └── hybrid_training_log.csv
├── training_curves/
│   ├── cnn_training_curves.png
│   └── ...
├── confusion_matrices/
│   ├── cnn_confusion_matrix.png
│   └── ...
├── model_comparison.png         ← Bar chart comparing all models
└── metrics_summary.csv          ← All experiment results in one CSV
```

Model checkpoints (`.h5` files) are excluded from Git due to file size.  
Use `train.py` to generate them locally.
