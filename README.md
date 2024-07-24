# Sentiment Classification on IMDb Dataset

This project involves sentiment classification on the IMDb dataset using two different models: an LSTM and a simple Transformer. The goal is to classify movie reviews as positive or negative.

## Models

### LSTM
- **Accuracy**: 90% after 10 epochs

### Transformer
- **Accuracy**: 80% after 20 epochs

## Results

| LSTM Confusion Matrix | Transformer Confusion Matrix |
|-----------------------|------------------------------|
| ![Confusion Matrix LSTM](figs/confusion_matrix_LSTM.png) | ![Confusion Matrix Transformer](figs/confusion_matrix_Transformer.png) |

| LSTM ROC Curve | Transformer ROC Curve |
|----------------|-----------------------|
| ![ROC Curve LSTM](figs/ROC_curve_LSTM.png) | ![ROC Curve Transformer](figs/ROC_curve_Transformer.png) |

## Dependencies

The project uses the following libraries:
- `torch`
- `torchvision`
- `torchtext`
- `numpy`
- `matplotlib`
- `sklearn`

## Installation

To install the required dependencies, run:
```sh
pip install -f requirements.txt
```
## Usage
1. To choose model, see the config
2. Train
```sh
python train.py
```
3. Evaluate/visualize
```sh
python evaluation.py
```