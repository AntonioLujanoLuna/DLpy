from typing import Optional, Tuple
import numpy as np
from ..core import Module, Tensor
from ..nn.linear import Linear
from ..nn.layer_norm import LayerNorm
from ..nn.dropout import Dropout
from ..nn.sequential import Sequential
from ..nn.activations import ReLU

class MultiHeadAttention(Module):
    """
    Multi-head attention mechanism.
    
    This module splits the input into multiple heads, applies scaled dot-product
    attention independently on each head, and then concatenates the results.
    
    Args:
        embed_dim (int): Total dimension of the model
        num_heads (int): Number of parallel attention heads
        dropout (float): Dropout probability (optional)
        bias (bool): If True, use bias in linear layers
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 bias: bool = True):
        super().__init__()
        
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Embedding dimension {embed_dim} not divisible by num_heads {num_heads}"
            )
            
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        self.w_q = Linear(embed_dim, embed_dim, bias=bias)
        self.w_k = Linear(embed_dim, embed_dim, bias=bias)
        self.w_v = Linear(embed_dim, embed_dim, bias=bias)
        self.w_o = Linear(embed_dim, embed_dim, bias=bias)
        
    def _reshape_for_heads(self, x: Tensor) -> Tensor:
        """Reshapes input for parallel head processing."""
        batch_size, seq_len, _ = x.shape
        # First reshape to separate head dimensions
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        # Then transpose dimensions to (batch_size, num_heads, seq_len, head_dim)
        x = x.transpose(0, 2, 1, 3)  # Note: passing axes as a tuple
        return x
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, query_len, embed_dim)
            key: Key tensor of shape (batch_size, key_len, embed_dim)
            value: Value tensor of shape (batch_size, key_len, embed_dim)
            attention_mask: Optional mask tensor of shape (batch_size, num_heads, query_len, key_len)
            
        Returns:
            Tuple of:
                - Output tensor of shape (batch_size, query_len, embed_dim)
                - Attention weights of shape (batch_size, num_heads, query_len, key_len)
        """
        batch_size, query_len, _ = query.shape
        _, key_len, _ = key.shape

        # Linear projections
        q = self.w_q(query)  # (batch_size, query_len, embed_dim)
        k = self.w_k(key)    # (batch_size, key_len, embed_dim)
        v = self.w_v(value)  # (batch_size, key_len, embed_dim)

        # Reshape for multi-head attention
        q = self._reshape_for_heads(q)  # (batch_size, num_heads, query_len, head_dim)
        k = self._reshape_for_heads(k)
        v = self._reshape_for_heads(v)

        # Scale query
        q = q * self.scaling

        # Compute attention scores
        k_t = k.transpose(0, 1, 3, 2)  # (batch_size, num_heads, head_dim, key_len)
        attention_scores = q @ k_t      # (batch_size, num_heads, query_len, key_len)

        if attention_mask is not None:
            # Apply the mask first
            attention_scores = attention_scores + attention_mask

            # For numerical stability, subtract max after applying the mask
            attention_scores = attention_scores - attention_scores.max(axis=-1, keepdims=True)

            # Clip to avoid overflow (though unlikely after subtraction)
            attention_scores = attention_scores.clip(-1e30, 1e30)

        # Apply softmax to get attention weights
        attention_weights = attention_scores.softmax(dim=-1)

        if self.training and self.dropout > 0:
            attention_weights = Dropout(self.dropout)(attention_weights)

        # Apply attention to values
        output = attention_weights @ v  # (batch_size, num_heads, query_len, head_dim)

        # Reshape back to original dimensions
        output = output.transpose(0, 2, 1, 3)  # (batch_size, query_len, num_heads, head_dim)
        output = output.reshape(batch_size, query_len, self.embed_dim)

        # Final linear projection
        output = self.w_o(output)

        return output, attention_weights

class TransformerEncoderLayer(Module):
    """
    Transformer Encoder Layer.
    
    Implements a single layer of the transformer encoder, consisting of:
    1. Multi-head self-attention
    2. Add & Norm
    3. Feed-forward network
    4. Add & Norm
    
    Args:
        d_model (int): The dimension of the model
        nhead (int): Number of attention heads
        dim_feedforward (int): Dimension of feedforward network
        dropout (float): Dropout probability
        activation (str): Activation function to use
        layer_norm_eps (float): eps value in layer normalizations
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu",
                 layer_norm_eps: float = 1e-5):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        # Create ReLU activation as a Module instance
        self.activation = ReLU()
        
        # Fix Sequential layer construction
        self.ff = Sequential(
            Linear(d_model, dim_feedforward),
            self.activation,  # Use the Module instance
            Dropout(dropout),
            Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = LayerNorm([d_model], eps=layer_norm_eps)
        self.norm2 = LayerNorm([d_model], eps=layer_norm_eps)
        self.dropout = Dropout(dropout)
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

class PositionalEncoding(Module):
    """
    Positional Encoding module.
    
    Adds positional information to the input embeddings using sine and cosine
    functions of different frequencies.
    
    Args:
        d_model (int): Dimension of the model
        max_len (int): Maximum sequence length
        dropout (float): Dropout probability
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = Dropout(dropout)
        
        # Create positional encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        # Register buffer (not a parameter)
        self.register_buffer('pe', Tensor(pe[np.newaxis, :, :]))
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)

class TransformerEncoder(Module):
    """
    Transformer Encoder.
    
    A stack of N encoder layers.
    
    Args:
        encoder_layer: An instance of TransformerEncoderLayer
        num_layers (int): Number of encoder layers in the stack
        norm (Module, optional): Layer normalization component
    """
    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int,
                 norm: Optional[Module] = None):
        super().__init__()
        self.layers = Sequential(
            *[encoder_layer for _ in range(num_layers)]
        )
        self.norm = norm
        
    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of transformer encoder.
        
        Args:
            src: Source sequence of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        output = src
        
        for layer in self.layers:
            output = layer(output, mask)
            
        if self.norm is not None:
            output = self.norm(output)
            
        return output