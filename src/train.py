"""
Main training script for Neural Language Model
Supports training for underfit, overfit, and best_fit scenarios
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time
from tqdm import tqdm
import contextlib

from config import *
from data import load_and_preprocess_data, create_dataloaders
from model import create_model
from utils import (
    set_seed, save_checkpoint, plot_training_curves,
    calculate_perplexity, save_training_log, EarlyStopping,
    format_time, print_config
)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip: float = 5.0
) -> float:
    """
    Train for one epoch
    
    Args:
        model: Language model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        grad_clip: Gradient clipping threshold
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    
    # Automatic Mixed Precision (AMP) setup for speed on GPU/MPS
    device_type = device.type
    use_amp = device_type in ('cuda', 'mps')
    autocast_ctx = (
        torch.autocast(device_type=device_type, dtype=torch.float16)
        if use_amp else contextlib.nullcontext()
    )
    scaler = torch.cuda.amp.GradScaler() if device_type == 'cuda' else None
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        # Move data to device
        inputs = inputs.to(device)  # (batch_size, seq_length)
        targets = targets.to(device)  # (batch_size, seq_length)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward + loss (with AMP if available)
        with autocast_ctx:
            outputs, _ = model(inputs)  # (batch_size, seq_length, vocab_size)
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)
            loss = criterion(outputs, targets)
        
        # Backward + step (with GradScaler on CUDA)
        if scaler is not None:
            scaler.scale(loss).backward()
            # Unscale before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        # Track loss
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Evaluate model on validation/test set
    
    Args:
        model: Language model
        dataloader: Validation/test dataloader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    # AMP context for evaluation
    device_type = device.type
    use_amp = device_type in ('cuda', 'mps')
    autocast_ctx = (
        torch.autocast(device_type=device_type, dtype=torch.float16)
        if use_amp else contextlib.nullcontext()
    )
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass (with AMP if available)
            with autocast_ctx:
                outputs, _ = model(inputs)
                outputs = outputs.reshape(-1, outputs.size(-1))
                targets = targets.reshape(-1)
                loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    return avg_loss


def train_model(
    scenario: str,
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    vocab_size: int,
    device: torch.device
):
    """
    Complete training procedure for a scenario
    
    Args:
        scenario: Scenario name ('underfit', 'overfit', 'best_fit')
        config: Configuration dictionary
        train_loader: Training dataloader
        val_loader: Validation dataloader
        test_loader: Test dataloader
        vocab_size: Vocabulary size
        device: Device to train on
    """
    print(f"\n{'='*60}")
    print(f"Training {scenario.upper()} model")
    print(f"{'='*60}")
    print_config(config)
    
    # Create model
    model = create_model(
        model_type=MODEL_TYPE,
        vocab_size=vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Early stopping (only for best_fit)
    early_stopping = None
    if 'early_stopping_patience' in config:
        early_stopping = EarlyStopping(
            patience=config['early_stopping_patience'],
            verbose=True
        )
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Model save directory
    model_dir = os.path.join(MODEL_SAVE_DIR, scenario)
    os.makedirs(model_dir, exist_ok=True)
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(1, config['epochs'] + 1):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, config['grad_clip']
        )
        
        # Validate
        val_loss = evaluate(model, val_loader, criterion, device)
        
        # Calculate perplexities
        train_perplexity = calculate_perplexity(train_loss)
        val_perplexity = calculate_perplexity(val_loss)
        
        # Save losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print progress
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch:3d}/{config['epochs']:3d} | "
              f"Time: {format_time(epoch_time)} | "
              f"Train Loss: {train_loss:.4f} (PPL: {train_perplexity:.2f}) | "
              f"Val Loss: {val_loss:.4f} (PPL: {val_perplexity:.2f})")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(model_dir, 'best_model.pt')
            save_checkpoint(
                model, optimizer, epoch, train_losses, val_losses,
                config, best_model_path, best_val_loss
            )
            print(f"  → New best model saved (val_loss: {val_loss:.4f})")
        
        # Save checkpoint every N epochs
        if epoch % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(model_dir, f'checkpoint_epoch_{epoch}.pt')
            save_checkpoint(
                model, optimizer, epoch, train_losses, val_losses,
                config, checkpoint_path, best_val_loss
            )
        
        # Early stopping check
        if early_stopping is not None:
            if early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\nTraining completed in {format_time(total_time)}")
    print(f"Best validation loss: {best_val_loss:.4f} (PPL: {calculate_perplexity(best_val_loss):.2f})")
    
    # Test evaluation
    print("\nEvaluating on test set...")
    test_loss = evaluate(model, test_loader, criterion, device)
    test_perplexity = calculate_perplexity(test_loss)
    print(f"Test Loss: {test_loss:.4f} | Test Perplexity: {test_perplexity:.2f}")
    
    # Save final model
    final_model_path = os.path.join(model_dir, 'final_model.pt')
    save_checkpoint(
        model, optimizer, config['epochs'], train_losses, val_losses,
        config, final_model_path, best_val_loss
    )
    
    # Plot training curves
    plot_path = os.path.join(PLOTS_DIR, f'{scenario}_training_curve.png')
    plot_training_curves(train_losses, val_losses, scenario, plot_path)
    
    # Save training log
    save_training_log(
        scenario, config, train_losses, val_losses,
        test_loss, test_perplexity, LOGS_DIR
    )
    
    print(f"\n{'='*60}")
    print(f"{scenario.upper()} training complete!")
    print(f"{'='*60}\n")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss,
        'test_perplexity': test_perplexity
    }


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Neural Language Model')
    parser.add_argument(
        '--scenario',
        type=str,
        default='best_fit',
        choices=['underfit', 'overfit', 'best_fit', 'all'],
        help='Training scenario'
    )
    args = parser.parse_args()
    
    # Set random seed
    set_seed(RANDOM_SEED)
    
    # Device
    print(f"Using device: {DEVICE}")
    
    # Load and preprocess data
    print("\n" + "="*60)
    print("Loading and preprocessing data...")
    print("="*60)
    
    preprocessor, train_dataset, val_dataset, test_dataset = load_and_preprocess_data(
        file_path=DATA_PATH,
        tokenization=TOKENIZATION,
        sequence_length=SEQUENCE_LENGTH,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT
    )
    
    vocab_size = preprocessor.vocab_size
    
    # Determine which scenarios to train
    if args.scenario == 'all':
        scenarios = ['underfit', 'overfit', 'best_fit']
    else:
        scenarios = [args.scenario]
    
    # Train each scenario
    results = {}
    
    for scenario in scenarios:
        config = get_config(scenario)
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=config['batch_size'],
            num_workers=2
        )
        
        # Train model
        result = train_model(
            scenario, config, train_loader, val_loader, test_loader,
            vocab_size, DEVICE
        )
        
        results[scenario] = result
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for scenario, result in results.items():
        print(f"\n{scenario.upper()}:")
        print(f"  Final Train Loss: {result['train_losses'][-1]:.4f}")
        print(f"  Final Val Loss:   {result['val_losses'][-1]:.4f}")
        print(f"  Test Loss:        {result['test_loss']:.4f}")
        print(f"  Test Perplexity:  {result['test_perplexity']:.2f}")
    
    print("\n" + "="*60)
    print("All training complete! Check 'plots/' and 'logs/' directories.")
    print("="*60)


if __name__ == "__main__":
    main()
