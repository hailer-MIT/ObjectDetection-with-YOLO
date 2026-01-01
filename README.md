# YOLO Object Detection Training Project

Training results and code from Google Colab.

## 📁 Project Structure

```
.
├── YOLO_Training_Colab.ipynb    # Colab notebook for training
├── config/                       # Configuration files
│   ├── dataset.yaml
│   └── training_config.yaml
├── scripts/                      # Training scripts
│   └── train.py
├── models/                       # Trained models (excluded - too large)
├── results/                      # Training results
│   └── train/
└── README.md
```

## 🚀 Usage

1. Open `YOLO_Training_Colab.ipynb` in Google Colab
2. Upload your dataset (Step 4)
3. Configure classes in Step 5
4. Run all cells to train

## 📊 Results

Training results are in `results/train/` including:
- Training curves
- Confusion matrix
- mAP scores

## 📝 Notes

- Dataset and large model files not included (too large for GitHub)
- See `COLAB_SETUP.md` for detailed instructions
