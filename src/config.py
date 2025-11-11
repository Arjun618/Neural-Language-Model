"""
Configuration file for Neural Language Model
Contains hyperparameters for all three scenarios: underfit, overfit, and best_fit
"""

import torch

# Device configuration
# Prefer Apple Silicon GPU (MPS) if available, otherwise CUDA, else CPU
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# Random seed for reproducibility
RANDOM_SEED = 42

# Data paths
DATA_PATH = 'dataset/Pride_and_Prejudice-Jane_Austen.txt'
MODEL_SAVE_DIR = 'models'
PLOTS_DIR = 'plots'
LOGS_DIR = 'logs'

# Data preprocessing
TOKENIZATION = 'char'  # 'char' or 'word'
SEQUENCE_LENGTH = 100  # Length of input sequences
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Underfit configuration - deliberately weak model
UNDERFIT_CONFIG = {
    'name': 'underfit',
    'embedding_dim': 64,
    'hidden_dim': 128,
    'num_layers': 1,
    'dropout': 0.0,
    'learning_rate': 0.01,  # Too high
    'batch_size': 128,
    'epochs': 5,  # Too few
    'weight_decay': 0.0,
    'grad_clip': 5.0,
}

# Overfit configuration - large model without regularization
OVERFIT_CONFIG = {
    'name': 'overfit',
    # Keep model large enough to overfit, but lighter for faster epochs
    'embedding_dim': 384,
    'hidden_dim': 768,
    'num_layers': 3,
    'dropout': 0.0,  # No regularization
    # Slightly higher LR to reach overfitting faster
    'learning_rate': 0.0015,
    'batch_size': 32,  # Keep small batches to encourage overfitting
    'epochs': 20,  # Reduced for faster completion while still demonstrating overfit
    'weight_decay': 0.0,  # No regularization
    'grad_clip': 5.0,
    # Add early stopping so we don't waste time once divergence is clear
    'early_stopping_patience': 3,
}

# Best fit configuration - balanced model with regularization
BEST_FIT_CONFIG = {
    'name': 'best_fit',
    'embedding_dim': 256,
    'hidden_dim': 512,
    'num_layers': 2,
    'dropout': 0.3,  # Good regularization
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 30,
    'weight_decay': 1e-5,  # L2 regularization
    'grad_clip': 5.0,
    'early_stopping_patience': 5,
}

# Model architecture
MODEL_TYPE = 'LSTM'  # Options: 'RNN', 'LSTM', 'GRU', 'Transformer'

# Logging
LOG_INTERVAL = 10  # Print training stats every N batches
SAVE_INTERVAL = 1  # Save checkpoint every N epochs

def get_config(scenario='best_fit'):
    """
    Get configuration for a specific scenario
    
    Args:
        scenario: One of 'underfit', 'overfit', 'best_fit'
    
    Returns:
        Configuration dictionary
    """
    configs = {
        'underfit': UNDERFIT_CONFIG,
        'overfit': OVERFIT_CONFIG,
        'best_fit': BEST_FIT_CONFIG,
    }
    
    if scenario not in configs:
        raise ValueError(f"Unknown scenario: {scenario}. Choose from {list(configs.keys())}")
    
    return configs[scenario]
