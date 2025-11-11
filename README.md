# Neural Language Model - Pride and Prejudice

A PyTorch implementation of neural language models trained on Jane Austen's "Pride and Prejudice" from scratch. This project demonstrates understanding of sequence models, model capacity, and generalization through three experimental scenarios: underfitting, overfitting, and best fit.

## 📋 Project Overview

This project implements LSTM-based language models trained to predict text at the character level. The implementation includes:

- **From-scratch PyTorch implementation** (no pre-trained models)
- **Three training scenarios** demonstrating different model behaviors
- **Comprehensive evaluation** with perplexity metrics
- **Text generation** capabilities with various sampling strategies
- **Complete reproducibility** with fixed random seeds

## 🎯 Objectives

1. ✅ Implement neural language models from scratch using PyTorch
2. ✅ Train and evaluate models on provided dataset
3. ✅ Demonstrate underfitting, overfitting, and best fit scenarios
4. ✅ Calculate and compare perplexity metrics
5. ✅ Generate comprehensive training visualizations

## 📊 Dataset

**Pride and Prejudice by Jane Austen**
- Source: Project Gutenberg
- Preprocessed text length: ~13,000 lines
- Tokenization: Character-level (vocabulary size: ~70 characters)
- Splits: 70% training / 15% validation / 15% test

## 🏗️ Architecture

**Model Type:** LSTM-based Language Model

### Three Experimental Scenarios:

#### 1. Underfitting Model
```
Embedding dim: 64
Hidden dim: 128
Layers: 1
Dropout: 0.0
Learning rate: 0.01 (too high)
Epochs: 5 (insufficient)
```
**Expected behavior:** High training AND validation loss

#### 2. Overfitting Model
```
Embedding dim: 512
Hidden dim: 1024
Layers: 4
Dropout: 0.0 (no regularization)
Learning rate: 0.001
Epochs: 50
Batch size: 32 (small)
```
**Expected behavior:** Low training loss, high validation loss (diverging)

#### 3. Best Fit Model
```
Embedding dim: 256
Hidden dim: 512
Layers: 2
Dropout: 0.3
Learning rate: 0.001
Weight decay: 1e-5
Epochs: 30
Early stopping: patience 5
```
**Expected behavior:** Converging training and validation loss

## 🚀 Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd Assignment_2

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Training

Train a specific scenario:
```bash
# Train underfit model
python src/train.py --scenario underfit

# Train overfit model
python src/train.py --scenario overfit

# Train best fit model
python src/train.py --scenario best_fit

# Train all scenarios
python src/train.py --scenario all
```

### Evaluation

Evaluate trained models:
```bash
# Evaluate specific scenario
python src/evaluate.py --scenario best_fit

# Evaluate all scenarios
python src/evaluate.py --scenario all
```

### Text Generation

Generate text with trained models:
```bash
# Generate with default settings
python src/generate.py --scenario best_fit

# Generate with custom parameters
python src/generate.py \
    --scenario best_fit \
    --seed "It is a truth universally acknowledged" \
    --length 500 \
    --temperature 0.8 \
    --top_k 50 \
    --num_samples 3
```

### Visualization

Create comparison plots:
```bash
python src/visualize.py
```

## 📈 Results

### Test Perplexity Comparison

| Scenario | Test Perplexity | Test Loss |
|----------|----------------|-----------|
| Underfit | TBD | TBD |
| Overfit | TBD | TBD |
| Best Fit | TBD | TBD |

*Note: Results will be updated after training*

### Training Curves

Training and validation loss plots for all three scenarios are available in the `plots/` directory:
- `underfit_training_curve.png`
- `overfit_training_curve.png`
- `best_fit_training_curve.png`
- `all_scenarios_comparison.png`
- `perplexity_comparison.png`

## 📁 Project Structure

```
Assignment_2/
├── src/
│   ├── config.py          # Hyperparameters and configurations
│   ├── data.py            # Data preprocessing and dataset classes
│   ├── model.py           # Neural language model architectures
│   ├── train.py           # Main training script
│   ├── evaluate.py        # Model evaluation
│   ├── generate.py        # Text generation
│   ├── utils.py           # Helper functions
│   └── visualize.py       # Plotting and visualization
├── dataset/
│   └── Pride_and_Prejudice-Jane_Austen.txt
├── models/                # Saved model checkpoints
│   ├── underfit/
│   ├── overfit/
│   └── best_fit/
├── plots/                 # Training curves and visualizations
├── logs/                  # Training logs (JSON)
├── requirements.txt
├── .gitignore
└── README.md
```

## 🔄 Reproducibility

All experiments use fixed random seeds for reproducibility:

```python
RANDOM_SEED = 42
```

Seeds are set for:
- Python's random module
- NumPy
- PyTorch (CPU and CUDA)

## 📥 Trained Models

Each scenario includes:
- `best_model.pt` - Model with best validation loss
- `final_model.pt` - Model after all training epochs
- Training logs (JSON)
- Loss curves (PNG)

### Loading a Trained Model

```python
import torch
from src.model import create_model

# Load checkpoint
checkpoint = torch.load('models/best_fit/best_model.pt')
config = checkpoint['config']

# Create and load model
model = create_model(
    model_type='LSTM',
    vocab_size=checkpoint['vocab_size'],
    embedding_dim=config['embedding_dim'],
    hidden_dim=config['hidden_dim'],
    num_layers=config['num_layers'],
    dropout=config['dropout']
)
model.load_state_dict(checkpoint['model_state_dict'])
```

## 🛠️ Technical Details

### Model Architecture
- **Embedding Layer:** Converts token indices to dense vectors
- **LSTM Layers:** Process sequences and capture long-term dependencies
- **Dropout:** Regularization to prevent overfitting
- **Output Layer:** Projects hidden states to vocabulary size
- **Loss Function:** Cross-entropy loss

### Training Procedure
- **Optimizer:** Adam with weight decay (L2 regularization)
- **Gradient Clipping:** Prevents exploding gradients (threshold: 5.0)
- **Early Stopping:** Stops training if validation loss doesn't improve (patience: 5)
- **Checkpointing:** Saves best model based on validation loss

### Evaluation Metrics
- **Loss:** Cross-entropy loss on test set
- **Perplexity:** exp(loss) - measures model uncertainty
  - Lower is better
  - Typical range: 50-200 for good models

## 🎓 Key Learnings

### Underfitting
- **Cause:** Insufficient model capacity or training
- **Symptoms:** High training and validation loss
- **Solution:** Increase model size or training duration

### Overfitting
- **Cause:** Model memorizes training data
- **Symptoms:** Low training loss, high validation loss (gap)
- **Solution:** Add regularization (dropout, weight decay, early stopping)

### Best Fit
- **Goal:** Balance between underfitting and overfitting
- **Indicators:** Small gap between training and validation loss
- **Techniques:** Proper regularization, appropriate model capacity

## 🚀 Future Improvements

- [ ] Implement Transformer architecture
- [ ] Add word-level tokenization option
- [ ] Implement beam search for generation
- [ ] Add attention visualization
- [ ] Deploy interactive web demo
- [ ] Experiment with different architectures (GRU, Bidirectional)

