"""
Text generation script using trained language models
"""

import torch
import torch.nn.functional as F
import argparse
import os

from config import *
from data import load_and_preprocess_data
from model import create_model
from utils import set_seed


def generate_text(
    model: torch.nn.Module,
    preprocessor,
    seed_text: str,
    max_length: int = 500,
    temperature: float = 1.0,
    top_k: int = 0,
    device: torch.device = torch.device('mps')
) -> str:
    """
    Generate text using the trained model
    
    Args:
        model: Trained language model
        preprocessor: Text preprocessor with encode/decode methods
        seed_text: Initial text to start generation
        max_length: Maximum length of generated text
        temperature: Sampling temperature (higher = more random)
        top_k: If > 0, only sample from top k tokens
        device: Device to generate on
        
    Returns:
        Generated text
    """
    model.eval()
    
    # Encode seed text
    current_text = seed_text
    encoded = preprocessor.encode(seed_text)
    
    # Generate
    with torch.no_grad():
        for _ in range(max_length):
            # Get input sequence (use last SEQUENCE_LENGTH tokens)
            if len(encoded) > SEQUENCE_LENGTH:
                input_seq = encoded[-SEQUENCE_LENGTH:]
            else:
                input_seq = encoded
            
            # Convert to tensor
            input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)
            
            # Forward pass
            output, _ = model(input_tensor)
            
            # Get logits for last token
            logits = output[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Add to sequence
            encoded.append(next_token)
            
            # Decode and check for stopping
            decoded = preprocessor.decode(encoded)
            if len(decoded) > len(current_text):
                new_char = decoded[len(current_text):]
                current_text += new_char
                
                # Stop if we've generated enough or hit end marker
                if len(current_text) >= len(seed_text) + max_length:
                    break
    
    return current_text


def main():
    """Main generation function"""
    parser = argparse.ArgumentParser(description='Generate text with Neural Language Model')
    parser.add_argument(
        '--scenario',
        type=str,
        default='best_fit',
        choices=['underfit', 'overfit', 'best_fit'],
        help='Model scenario to use'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='best_model.pt',
        help='Checkpoint filename'
    )
    parser.add_argument(
        '--seed',
        type=str,
        default='It is a truth universally acknowledged',
        help='Seed text to start generation'
    )
    parser.add_argument(
        '--length',
        type=int,
        default=500,
        help='Length of text to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (0.5-1.5)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=50,
        help='Top-k sampling (0 = disabled)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=3,
        help='Number of samples to generate'
    )
    args = parser.parse_args()
    
    # Set seed
    set_seed(RANDOM_SEED)
    
    # Device
    print(f"Using device: {DEVICE}")
    
    # Load preprocessor
    print("\nLoading preprocessor...")
    preprocessor, _, _, _ = load_and_preprocess_data(
        file_path=DATA_PATH,
        tokenization=TOKENIZATION,
        sequence_length=SEQUENCE_LENGTH,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT
    )
    
    vocab_size = preprocessor.vocab_size
    
    # Load model
    model_path = os.path.join(MODEL_SAVE_DIR, args.scenario, args.checkpoint)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    print(f"\nLoading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    config = checkpoint['config']
    
    # Create model
    model = create_model(
        model_type=MODEL_TYPE,
        vocab_size=vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=0.0  # No dropout during inference
    ).to(DEVICE)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\n" + "="*80)
    print(f"Generating text with {args.scenario.upper()} model")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}")
    print("="*80)
    
    # Generate multiple samples
    for i in range(args.num_samples):
        print(f"\n{'='*80}")
        print(f"Sample {i+1}")
        print(f"{'='*80}")
        
        generated = generate_text(
            model=model,
            preprocessor=preprocessor,
            seed_text=args.seed,
            max_length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            device=DEVICE
        )
        
        print(generated)
    
    print("\n" + "="*80)
    print("Generation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
