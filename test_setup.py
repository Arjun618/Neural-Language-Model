"""
Quick test script to verify setup and data loading
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import *
from src.data import load_and_preprocess_data, create_dataloaders
from src.model import create_model
from src.utils import set_seed, count_parameters

def test_setup():
    """Test basic setup and imports"""
    print("Testing basic setup...")
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ Device: {DEVICE}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    
def test_data_loading():
    """Test data loading and preprocessing"""
    print("\nTesting data loading...")
    
    try:
        preprocessor, train_dataset, val_dataset, test_dataset = load_and_preprocess_data(
            file_path=DATA_PATH,
            tokenization=TOKENIZATION,
            sequence_length=SEQUENCE_LENGTH,
            train_split=TRAIN_SPLIT,
            val_split=VAL_SPLIT,
            test_split=TEST_SPLIT
        )
        print(f"✓ Data loaded successfully")
        print(f"✓ Vocabulary size: {preprocessor.vocab_size}")
        print(f"✓ Train dataset size: {len(train_dataset)}")
        print(f"✓ Val dataset size: {len(val_dataset)}")
        print(f"✓ Test dataset size: {len(test_dataset)}")
        
        return preprocessor, train_dataset, val_dataset, test_dataset
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return None, None, None, None

def test_dataloader(train_dataset, val_dataset, test_dataset):
    """Test dataloader creation"""
    print("\nTesting dataloaders...")
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset, batch_size=32
        )
        
        # Test one batch
        for inputs, targets in train_loader:
            print(f"✓ Batch shape - Inputs: {inputs.shape}, Targets: {targets.shape}")
            break
        
        return train_loader, val_loader, test_loader
    except Exception as e:
        print(f"✗ Dataloader creation failed: {e}")
        return None, None, None

def test_model_creation(vocab_size):
    """Test model creation"""
    print("\nTesting model creation...")
    
    try:
        config = get_config('best_fit')
        model = create_model(
            model_type=MODEL_TYPE,
            vocab_size=vocab_size,
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
        
        total_params, trainable_params = count_parameters(model)
        print(f"✓ Model created successfully")
        print(f"✓ Total parameters: {total_params:,}")
        
        return model
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return None

def test_forward_pass(model, train_loader):
    """Test forward pass"""
    print("\nTesting forward pass...")
    
    try:
        model.eval()
        with torch.no_grad():
            for inputs, targets in train_loader:
                outputs, hidden = model(inputs)
                print(f"✓ Forward pass successful")
                print(f"✓ Output shape: {outputs.shape}")
                break
        
        print("\n✅ All tests passed!")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("NEURAL LANGUAGE MODEL - SETUP TEST")
    print("="*60)
    
    # Set seed
    set_seed(RANDOM_SEED)
    
    # Run tests
    test_setup()
    
    preprocessor, train_dataset, val_dataset, test_dataset = test_data_loading()
    if preprocessor is None:
        print("\n❌ Setup test failed at data loading")
        return
    
    train_loader, val_loader, test_loader = test_dataloader(
        train_dataset, val_dataset, test_dataset
    )
    if train_loader is None:
        print("\n❌ Setup test failed at dataloader creation")
        return
    
    model = test_model_creation(preprocessor.vocab_size)
    if model is None:
        print("\n❌ Setup test failed at model creation")
        return
    
    success = test_forward_pass(model, train_loader)
    
    if success:
        print("\n" + "="*60)
        print("✅ SETUP TEST COMPLETE - READY TO TRAIN!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Train underfit model:  python src/train.py --scenario underfit")
        print("  2. Train overfit model:   python src/train.py --scenario overfit")
        print("  3. Train best fit model:  python src/train.py --scenario best_fit")
        print("  4. Or train all:          python src/train.py --scenario all")
    else:
        print("\n❌ Setup test failed")

if __name__ == "__main__":
    main()
