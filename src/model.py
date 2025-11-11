"""
Neural Language Model Architecture
Implements RNN, LSTM, GRU, and Transformer-based language models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LSTMLanguageModel(nn.Module):
    """LSTM-based Language Model"""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.5
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embedding vectors
            hidden_dim: Dimension of hidden states
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(LSTMLanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights with uniform distribution"""
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, x, hidden=None):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
            hidden: Hidden state tuple (h, c), optional
            
        Returns:
            output: Logits of shape (batch_size, seq_length, vocab_size)
            hidden: Updated hidden state
        """
        # Embed input
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        embedded = self.dropout_layer(embedded)
        
        # Pass through LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)  # (batch_size, seq_length, hidden_dim)
        lstm_out = self.dropout_layer(lstm_out)
        
        # Project to vocabulary size
        output = self.fc(lstm_out)  # (batch_size, seq_length, vocab_size)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """
        Initialize hidden state
        
        Args:
            batch_size: Batch size
            device: Device to create tensor on
            
        Returns:
            Tuple of (h0, c0)
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)


class GRULanguageModel(nn.Module):
    """GRU-based Language Model"""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.5
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embedding vectors
            hidden_dim: Dimension of hidden states
            num_layers: Number of GRU layers
            dropout: Dropout probability
        """
        super(GRULanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # GRU layers
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights with uniform distribution"""
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, x, hidden=None):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
            hidden: Hidden state, optional
            
        Returns:
            output: Logits of shape (batch_size, seq_length, vocab_size)
            hidden: Updated hidden state
        """
        # Embed input
        embedded = self.embedding(x)
        embedded = self.dropout_layer(embedded)
        
        # Pass through GRU
        gru_out, hidden = self.gru(embedded, hidden)
        gru_out = self.dropout_layer(gru_out)
        
        # Project to vocabulary size
        output = self.fc(gru_out)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """
        Initialize hidden state
        
        Args:
            batch_size: Batch size
            device: Device to create tensor on
            
        Returns:
            Hidden state tensor
        """
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)


class RNNLanguageModel(nn.Module):
    """Simple RNN-based Language Model"""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.5
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embedding vectors
            hidden_dim: Dimension of hidden states
            num_layers: Number of RNN layers
            dropout: Dropout probability
        """
        super(RNNLanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layers
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights with uniform distribution"""
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, x, hidden=None):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
            hidden: Hidden state, optional
            
        Returns:
            output: Logits of shape (batch_size, seq_length, vocab_size)
            hidden: Updated hidden state
        """
        # Embed input
        embedded = self.embedding(x)
        embedded = self.dropout_layer(embedded)
        
        # Pass through RNN
        rnn_out, hidden = self.rnn(embedded, hidden)
        rnn_out = self.dropout_layer(rnn_out)
        
        # Project to vocabulary size
        output = self.fc(rnn_out)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """
        Initialize hidden state
        
        Args:
            batch_size: Batch size
            device: Device to create tensor on
            
        Returns:
            Hidden state tensor
        """
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)


def create_model(
    model_type: str,
    vocab_size: int,
    embedding_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float = 0.5
) -> nn.Module:
    """
    Factory function to create a language model
    
    Args:
        model_type: Type of model ('RNN', 'LSTM', 'GRU')
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embedding vectors
        hidden_dim: Dimension of hidden states
        num_layers: Number of layers
        dropout: Dropout probability
        
    Returns:
        Language model
    """
    model_type = model_type.upper()
    
    if model_type == 'LSTM':
        model = LSTMLanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
    elif model_type == 'GRU':
        model = GRULanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
    elif model_type == 'RNN':
        model = RNNLanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from 'RNN', 'LSTM', 'GRU'")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{model_type} Language Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num layers: {num_layers}")
    print(f"  Dropout: {dropout}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    vocab_size = 100
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2
    dropout = 0.3
    
    # Test LSTM
    model = create_model('LSTM', vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
    
    # Test forward pass
    batch_size = 4
    seq_length = 50
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    output, hidden = model(x)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Hidden state shapes: {hidden[0].shape}, {hidden[1].shape}")
