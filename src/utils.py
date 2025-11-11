"""
Utility functions for training, evaluation, and visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import json
from typing import Dict, List, Tuple
import datetime


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_losses: List[float],
    val_losses: List[float],
    config: Dict,
    filepath: str,
    best_val_loss: float = None
):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        train_losses: List of training losses
        val_losses: List of validation losses
        config: Configuration dictionary
        filepath: Path to save checkpoint
        best_val_loss: Best validation loss so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config,
        'best_val_loss': best_val_loss,
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
    """
    Load model checkpoint
    
    Args:
        filepath: Path to checkpoint
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    return checkpoint


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from loss
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        Perplexity
    """
    return np.exp(loss)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    scenario: str,
    save_path: str = None
):
    """
    Plot training and validation loss curves
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        scenario: Scenario name (e.g., 'underfit', 'overfit', 'best_fit')
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training vs Validation Loss - {scenario.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()


def plot_all_scenarios_comparison(
    scenarios_data: Dict[str, Dict[str, List[float]]],
    save_path: str = None
):
    """
    Plot comparison of all three scenarios
    
    Args:
        scenarios_data: Dictionary with scenario names as keys and 
                       {'train_losses': [...], 'val_losses': [...]} as values
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = {'train': '#2E86AB', 'val': '#A23B72'}
    
    for idx, (scenario, data) in enumerate(scenarios_data.items()):
        ax = axes[idx]
        train_losses = data['train_losses']
        val_losses = data['val_losses']
        epochs = range(1, len(train_losses) + 1)
        
        ax.plot(epochs, train_losses, color=colors['train'], label='Training Loss', linewidth=2)
        ax.plot(epochs, val_losses, color=colors['val'], label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title(scenario.replace('_', ' ').title(), fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.close()


def plot_perplexity_comparison(
    scenarios_perplexities: Dict[str, float],
    save_path: str = None
):
    """
    Plot bar chart comparing perplexities
    
    Args:
        scenarios_perplexities: Dictionary with scenario names and their perplexities
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(10, 6))
    
    scenarios = list(scenarios_perplexities.keys())
    perplexities = list(scenarios_perplexities.values())
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = plt.bar(scenarios, perplexities, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, perp in zip(bars, perplexities):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{perp:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.xlabel('Scenario', fontsize=12)
    plt.ylabel('Perplexity', fontsize=12)
    plt.title('Test Perplexity Comparison Across Scenarios', fontsize=14, fontweight='bold')
    plt.xticks([s.replace('_', ' ').title() for s in scenarios])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Perplexity comparison plot saved to {save_path}")
    
    plt.close()


def save_training_log(
    scenario: str,
    config: Dict,
    train_losses: List[float],
    val_losses: List[float],
    test_loss: float,
    test_perplexity: float,
    save_dir: str
):
    """
    Save training log to JSON file
    
    Args:
        scenario: Scenario name
        config: Configuration dictionary
        train_losses: Training losses
        val_losses: Validation losses
        test_loss: Final test loss
        test_perplexity: Final test perplexity
        save_dir: Directory to save log
    """
    log = {
        'scenario': scenario,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': float(test_loss),
        'test_perplexity': float(test_perplexity),
        'best_val_loss': float(min(val_losses)),
        'best_val_perplexity': float(calculate_perplexity(min(val_losses))),
        'final_train_loss': float(train_losses[-1]),
        'final_val_loss': float(val_losses[-1]),
    }
    
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f'{scenario}_training_log.json')
    
    with open(log_path, 'w') as f:
        json.dump(log, indent=4, fp=f)
    
    print(f"Training log saved to {log_path}")


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, verbose: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_config(config: Dict):
    """
    Pretty print configuration
    
    Args:
        config: Configuration dictionary
    """
    print("\n" + "="*50)
    print("Configuration:")
    print("="*50)
    for key, value in config.items():
        print(f"  {key:20s}: {value}")
    print("="*50 + "\n")


if __name__ == "__main__":
    # Test utility functions
    set_seed(42)
    
    # Test plotting
    train_losses = [2.5, 2.2, 2.0, 1.8, 1.6, 1.5]
    val_losses = [2.6, 2.3, 2.1, 1.9, 1.8, 1.7]
    
    plot_training_curves(train_losses, val_losses, 'test', 'test_plot.png')
    
    # Test perplexity
    loss = 2.0
    perp = calculate_perplexity(loss)
    print(f"Loss: {loss:.2f}, Perplexity: {perp:.2f}")
