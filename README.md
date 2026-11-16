# Pneumonia detection from chest X-rays with a CNN

A small convolutional neural network, trained from scratch on a CPU, that classifies a chest X-ray as **normal** or **pneumonia**.

## Dataset

[PneumoniaMNIST](https://medmnist.com/) (MedMNIST v3, 128×128 version): 5,856 pediatric chest X-rays from Kermany et al. (2018), CC BY 4.0.

| Split | Images | Normal | Pneumonia |
|---|---|---|---|
| Train | 4,708 | 1,214 | 3,494 |
| Validation | 524 | 135 | 389 |
| Test | 624 | 234 | 390 |

The data file (75 MB) is not in the repository. It is downloaded automatically and checked against its MD5 checksum.

## Model

```
Input 1x128x128
-> [Conv 3x3 -> BatchNorm -> ReLU -> MaxPool] x 4   (16, 32, 64, 128 filters)
-> Flatten 8192 -> Dropout -> Dense 128 -> Dropout -> Dense 2 -> Softmax
```

About 1.15M parameters. Training uses Adam, a class-weighted cross-entropy loss (to handle the 74% pneumonia imbalance) and light augmentation. It runs for 20 epochs, which takes about 8 minutes on a CPU.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python data.py
```

## Usage

Run everything in one place with **`pneumonia_cnn.ipynb`**, or use the scripts one step at a time:

| Step | Command | Output in `outputs/` |
|---|---|---|
| Train and save | `python train.py` | `model.pt`, `history.json`, `learning_curves.png` |
| Test | `python evaluate.py` | `test_metrics.txt`, `confusion_matrix.png` |
| Understand and explain | `python visualize.py` | `model_summary.txt`, `structure_diagram.png`, `feature_maps_*.png` |
| Predict | `python predict.py samples/*.png` | printed class and probabilities |

`samples/` contains test images exported as PNG files for the demo. `predict.py` accepts any PNG or JPG chest X-ray: it converts it to grayscale, center-crops it and resizes it to 128×128.

## Files

| File | Purpose |
|---|---|
| `data.py` | Download, loading, augmentation and preprocessing of new images |
| `model.py` | CNN definition, save and load |
| `train.py` | Training loop and learning curves |
| `evaluate.py` | Accuracy, precision, recall and confusion matrix on the test set |
| `visualize.py` | Layer summary, structure diagram and feature maps |
| `predict.py` | Classify new images |
| `pneumonia_cnn.ipynb` | All steps combined in one notebook |
