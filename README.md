# Chess Piece Classifier

A deep learning model for classifying chess pieces using computer vision. Built with FastAI and ConvNeXt Tiny architecture.

## Overview

This project trains a convolutional neural network to identify six different chess pieces: Bishop, King, Knight, Pawn, Queen, and Rook. The model achieves high accuracy through transfer learning and data augmentation techniques.

## Dataset

- **Source**: Chessman Image Dataset from Kaggle
- **Size**: 1,104 images total
- **Classes**: 6 chess piece types
- **Split**: 80% training, 20% validation

## Model Architecture

- **Base Model**: ConvNeXt Tiny (pretrained on ImageNet)
- **Framework**: FastAI v2
- **Input Size**: 320x320 pixels
- **Data Augmentation**: Random cropping, rotation (±10°), zoom (up to 10%), brightness/contrast adjustments
- **Batch Size**: 64
- **Training**: Fine-tuning for 5 epochs

## Features

- Automated data preprocessing and augmentation
- Real-time inference with confidence scores
- Interactive web interface using Gradio
- Model export for deployment
- Comprehensive evaluation metrics


## Training
Run the Jupyter notebook `chess_ai_notebook.ipynb` to:
1. Download and prepare the dataset
2. Train the ConvNeXt model
3. Evaluate performance
4. Export the trained model

### Inference
```python
from fastai.vision.all import *

# Load trained model
learn = load_learner('chessman_model.pkl')

# Predict on new image
pred, pred_idx, probs = learn.predict('path/to/chess_piece.jpg')
print(f"Prediction: {pred}")
```

## Files

- `chess_ai_notebook.ipynb` - Complete training and evaluation pipeline
