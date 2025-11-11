"""
Evaluation script for trained language models
"""

import torch
import torch.nn as nn
import argparse
import os
import json

from config import *
from data import load_and_preprocess_data, create_dataloaders
from model import create_model
from utils import load_checkpoint, calculate_perplexity, set_seed


def evaluate_model(
    model_path: str,
    test_loader: torch.utils.data.DataLoader,
    vocab_size: int,
    device: torch.device
):
    """
    Evaluate a trained model
    
    Args:
        model_path: Path to model checkpoint
        test_loader: Test dataloader
        vocab_size: Vocabulary size
        device: Device to evaluate on
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = create_model(
        model_type=MODEL_TYPE,
        vocab_size=vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate
    total_loss = 0.0
    num_batches = len(test_loader)
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs, _ = model(inputs)
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    perplexity = calculate_perplexity(avg_loss)
    
    results = {
        'test_loss': avg_loss,
        'test_perplexity': perplexity,
        'model_path': model_path,
        'config': config
    }
    
    return results


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate Neural Language Model')
    parser.add_argument(
        '--scenario',
        type=str,
        default='best_fit',
        choices=['underfit', 'overfit', 'best_fit', 'all'],
        help='Scenario to evaluate'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='best_model.pt',
        help='Checkpoint filename (default: best_model.pt)'
    )
    args = parser.parse_args()
    
    # Set seed
    set_seed(RANDOM_SEED)
    
    # Device
    print(f"Using device: {DEVICE}")
    
    # Load data
    print("\nLoading data...")
    preprocessor, train_dataset, val_dataset, test_dataset = load_and_preprocess_data(
        file_path=DATA_PATH,
        tokenization=TOKENIZATION,
        sequence_length=SEQUENCE_LENGTH,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT
    )
    
    vocab_size = preprocessor.vocab_size
    
    # Create test dataloader
    _, _, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=64
    )
    
    # Determine scenarios to evaluate
    if args.scenario == 'all':
        scenarios = ['underfit', 'overfit', 'best_fit']
    else:
        scenarios = [args.scenario]
    
    # Evaluate each scenario
    all_results = {}
    
    for scenario in scenarios:
        model_path = os.path.join(MODEL_SAVE_DIR, scenario, args.checkpoint)
        
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}, skipping...")
            continue
        
        results = evaluate_model(model_path, test_loader, vocab_size, DEVICE)
        all_results[scenario] = results
        
        print(f"\n{scenario.upper()} Results:")
        print(f"  Test Loss: {results['test_loss']:.4f}")
        print(f"  Test Perplexity: {results['test_perplexity']:.2f}")
    
    # Summary
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        for scenario, results in all_results.items():
            print(f"\n{scenario.upper()}:")
            print(f"  Test Perplexity: {results['test_perplexity']:.2f}")
        
        # Find best model
        best_scenario = min(all_results.items(), key=lambda x: x[1]['test_perplexity'])
        print(f"\nBest model: {best_scenario[0].upper()} "
              f"(Perplexity: {best_scenario[1]['test_perplexity']:.2f})")
    
    # Save evaluation results
    eval_results_path = os.path.join(LOGS_DIR, 'evaluation_results.json')
    with open(eval_results_path, 'w') as f:
        # Convert results to serializable format
        serializable_results = {}
        for scenario, results in all_results.items():
            serializable_results[scenario] = {
                'test_loss': float(results['test_loss']),
                'test_perplexity': float(results['test_perplexity']),
                'model_path': results['model_path']
            }
        json.dump(serializable_results, f, indent=4)
    
    print(f"\nEvaluation results saved to {eval_results_path}")


if __name__ == "__main__":
    main()
