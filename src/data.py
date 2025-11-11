"""
Data preprocessing and dataset creation for Neural Language Model
Handles text cleaning, tokenization, and PyTorch Dataset/DataLoader creation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from collections import Counter
from typing import Tuple, List, Dict


class TextPreprocessor:
    """Handles text cleaning and preprocessing"""
    
    def __init__(self, tokenization='char'):
        """
        Args:
            tokenization: 'char' for character-level or 'word' for word-level
        """
        self.tokenization = tokenization
        self.vocab = None
        self.char2idx = None
        self.idx2char = None
        self.vocab_size = 0
        
    def clean_text(self, text: str) -> str:
        """
        Remove Project Gutenberg headers/footers and clean text
        
        Args:
            text: Raw text from file
            
        Returns:
            Cleaned text
        """
        # Find start and end markers
        start_marker = "***START OF THE PROJECT GUTENBERG EBOOK"
        end_marker = "***END OF THE PROJECT GUTENBERG EBOOK"
        
        # Extract main content
        start_idx = text.find(start_marker)
        end_idx = text.find(end_marker)
        
        if start_idx != -1:
            # Find the end of the start marker line
            start_idx = text.find('\n', start_idx) + 1
        else:
            start_idx = 0
            
        if end_idx != -1:
            text = text[start_idx:end_idx]
        else:
            text = text[start_idx:]
        
        # Remove extra whitespace but keep paragraph structure
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def build_vocab(self, text: str):
        """
        Build vocabulary from text
        
        Args:
            text: Cleaned text
        """
        if self.tokenization == 'char':
            # Character-level vocabulary
            chars = sorted(set(text))
            self.vocab_size = len(chars)
            self.char2idx = {ch: i for i, ch in enumerate(chars)}
            self.idx2char = {i: ch for i, ch in enumerate(chars)}
            self.vocab = chars
            
        elif self.tokenization == 'word':
            # Word-level vocabulary
            words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
            word_counts = Counter(words)
            
            # Special tokens
            special_tokens = ['<PAD>', '<UNK>', '<EOS>']
            
            # Most common words (limit vocab size for efficiency)
            vocab_size = 10000
            common_words = [word for word, _ in word_counts.most_common(vocab_size - len(special_tokens))]
            
            self.vocab = special_tokens + common_words
            self.vocab_size = len(self.vocab)
            self.char2idx = {word: i for i, word in enumerate(self.vocab)}
            self.idx2char = {i: word for i, word in enumerate(self.vocab)}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Sample vocab: {self.vocab[:20]}")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to indices
        
        Args:
            text: Text to encode
            
        Returns:
            List of indices
        """
        if self.tokenization == 'char':
            return [self.char2idx.get(ch, 0) for ch in text]
        
        elif self.tokenization == 'word':
            words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
            unk_idx = self.char2idx['<UNK>']
            return [self.char2idx.get(word, unk_idx) for word in words]
    
    def decode(self, indices: List[int]) -> str:
        """
        Decode indices to text
        
        Args:
            indices: List of indices
            
        Returns:
            Decoded text
        """
        if self.tokenization == 'char':
            return ''.join([self.idx2char.get(idx, '') for idx in indices])
        
        elif self.tokenization == 'word':
            return ' '.join([self.idx2char.get(idx, '<UNK>') for idx in indices])


class LanguageModelDataset(Dataset):
    """PyTorch Dataset for language modeling"""
    
    def __init__(self, encoded_text: List[int], sequence_length: int):
        """
        Args:
            encoded_text: Text encoded as indices
            sequence_length: Length of input sequences
        """
        self.encoded_text = encoded_text
        self.sequence_length = sequence_length
        
        # Create sequences
        self.sequences = []
        for i in range(0, len(encoded_text) - sequence_length):
            seq = encoded_text[i:i + sequence_length]
            target = encoded_text[i + 1:i + sequence_length + 1]
            self.sequences.append((seq, target))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def load_and_preprocess_data(
    file_path: str,
    tokenization: str = 'char',
    sequence_length: int = 100,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15
) -> Tuple[TextPreprocessor, LanguageModelDataset, LanguageModelDataset, LanguageModelDataset]:
    """
    Load and preprocess text data
    
    Args:
        file_path: Path to text file
        tokenization: 'char' or 'word'
        sequence_length: Length of sequences
        train_split: Training set proportion
        val_split: Validation set proportion
        test_split: Test set proportion
        
    Returns:
        Tuple of (preprocessor, train_dataset, val_dataset, test_dataset)
    """
    print(f"Loading data from {file_path}...")
    
    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Original text length: {len(text)} characters")
    
    # Preprocess
    preprocessor = TextPreprocessor(tokenization=tokenization)
    cleaned_text = preprocessor.clean_text(text)
    print(f"Cleaned text length: {len(cleaned_text)} characters")
    
    # Build vocabulary
    preprocessor.build_vocab(cleaned_text)
    
    # Encode text
    encoded_text = preprocessor.encode(cleaned_text)
    print(f"Encoded text length: {len(encoded_text)} tokens")
    
    # Split data
    train_size = int(len(encoded_text) * train_split)
    val_size = int(len(encoded_text) * val_split)
    
    train_text = encoded_text[:train_size]
    val_text = encoded_text[train_size:train_size + val_size]
    test_text = encoded_text[train_size + val_size:]
    
    print(f"\nData splits:")
    print(f"  Training:   {len(train_text)} tokens")
    print(f"  Validation: {len(val_text)} tokens")
    print(f"  Test:       {len(test_text)} tokens")
    
    # Create datasets
    train_dataset = LanguageModelDataset(train_text, sequence_length)
    val_dataset = LanguageModelDataset(val_text, sequence_length)
    test_dataset = LanguageModelDataset(test_text, sequence_length)
    
    print(f"\nDataset sizes:")
    print(f"  Training:   {len(train_dataset)} sequences")
    print(f"  Validation: {len(val_dataset)} sequences")
    print(f"  Test:       {len(test_dataset)} sequences")
    
    return preprocessor, train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset: LanguageModelDataset,
    val_dataset: LanguageModelDataset,
    test_dataset: LanguageModelDataset,
    batch_size: int = 64,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data preprocessing
    from config import DATA_PATH, SEQUENCE_LENGTH, TOKENIZATION, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
    
    preprocessor, train_dataset, val_dataset, test_dataset = load_and_preprocess_data(
        file_path=DATA_PATH,
        tokenization=TOKENIZATION,
        sequence_length=SEQUENCE_LENGTH,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT
    )
    
    # Test dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=32
    )
    
    # Show sample batch
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"\nSample batch shape:")
        print(f"  Inputs:  {inputs.shape}")
        print(f"  Targets: {targets.shape}")
        print(f"\nFirst sequence (decoded):")
        print(preprocessor.decode(inputs[0].tolist()))
        break
