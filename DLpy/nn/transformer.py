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

        self.attention_dropout = Dropout(self.dropout)
        
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
            # First stabilize the scores before applying mask
            attention_scores = attention_scores - attention_scores.max(axis=-1, keepdims=True)
            
            # Add the mask (broadcasting will handle shape differences)
            attention_scores = attention_scores + attention_mask
            
            # Clip values for numerical stability
            attention_scores = attention_scores.clip(-1e30, 1e30)
        else:
            # If no mask, still stabilize numerically
            attention_scores = attention_scores - attention_scores.max(axis=-1, keepdims=True)

        # Apply softmax to get attention weights
        attention_weights = attention_scores.softmax(dim=-1)

        if self.training:
            attention_weights = self.attention_dropout(attention_weights)

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
        # Create single instances of dropout layers
        self.attn_dropout = Dropout(dropout)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=0.0)  # Set to 0 as we handle dropout separately
        self.activation = ReLU()
        
        # Feed forward network with dropout
        self.ff = Sequential(
            Linear(d_model, dim_feedforward),
            self.activation,
            Dropout(dropout),  # This is fine as Sequential handles the instance properly
            Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = LayerNorm([d_model], eps=layer_norm_eps)
        self.norm2 = LayerNorm([d_model], eps=layer_norm_eps)
        
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Self attention block
        attn_output, _ = self.self_attn(x, x, x, mask)
        attn_output = self.attn_dropout(attn_output)  # Apply dropout to attention output
        x = x + self.dropout1(attn_output)  # Apply dropout to residual
        x = self.norm1(x)
        # Feedforward block
        ff_output = self.ff(x)
        x = x + self.dropout2(ff_output)  # Apply dropout to residual
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

class TransformerDecoderLayer(Module):
    """
    Transformer Decoder Layer.
    
    Implements a single layer of the transformer decoder, consisting of:
    1. Masked multi-head self-attention
    2. Add & Norm
    3. Multi-head attention over encoder output
    4. Add & Norm
    5. Feed-forward network
    6. Add & Norm
    
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
        
        # Self attention mechanism
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        # Cross attention mechanism
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        # Create ReLU activation
        self.activation = ReLU()
        
        # Initialize all required layers
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        
        # Layer normalization layers
        self.norm1 = LayerNorm([d_model], eps=layer_norm_eps)
        self.norm2 = LayerNorm([d_model], eps=layer_norm_eps)
        self.norm3 = LayerNorm([d_model], eps=layer_norm_eps)
        
        # Dropout layers
        self.dropout = Dropout(dropout)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        
        # Feed-forward network
        self.ff = Sequential(
            self.linear1,
            self.activation,
            self.dropout,
            self.linear2
        )
        
    def forward(self, x: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of decoder layer.
        
        Args:
            x: Input tensor (target sequence)
            memory: Output from encoder
            tgt_mask: Mask for target sequence
            memory_mask: Mask for source sequence
            
        Returns:
            Output tensor
        """
        # Self-attention block
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Cross-attention block
        attn_output, _ = self.multihead_attn(x, memory, memory, memory_mask)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)
        
        # Feedforward block
        ff_output = self.ff(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)
        
        return x
        
class TransformerDecoder(Module):
    """
    Transformer Decoder.
    
    A stack of N decoder layers with masking functionality.
    
    Args:
        decoder_layer: An instance of TransformerDecoderLayer
        num_layers (int): Number of decoder layers
        norm (Module, optional): Layer normalization component
    """
    
    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers: int,
                 norm: Optional[Module] = None):
        super().__init__()
        self.layers = Sequential(
            *[decoder_layer for _ in range(num_layers)]
        )
        self.norm = norm
        
    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of transformer decoder.
        
        Args:
            tgt: Target sequence
            memory: Output from encoder
            tgt_mask: Target sequence mask
            memory_mask: Source sequence mask
            
        Returns:
            Output tensor
        """
        output = tgt
        
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask)
            
        if self.norm is not None:
            output = self.norm(output)
            
        return output

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """
    Generate a square mask for the sequence where subsequent positions are masked.
    
    Args:
        sz: Size of square matrix
        
    Returns:
        Tensor of shape (sz, sz) containing mask where entries in upper triangle
        are -inf and lower triangle (including diagonal) are 0
    """
    mask = np.zeros((sz, sz))
    # Fill upper triangle with -inf (excluding diagonal)
    mask[np.triu(np.ones((sz, sz), dtype=bool), k=1)] = -np.inf
    return Tensor(mask)

class Transformer(Module):
    """
    A complete Transformer model.
    
    Combines encoder and decoder with all the necessary components.
    
    Args:
        d_model (int): Dimension of the model
        nhead (int): Number of attention heads
        num_encoder_layers (int): Number of encoder layers
        num_decoder_layers (int): Number of decoder layers
        dim_feedforward (int): Dimension of feedforward network
        dropout (float): Dropout probability
        activation (str): Activation function
        layer_norm_eps (float): Layer norm epsilon
    """
    
    def __init__(self, d_model: int, nhead: int, num_encoder_layers: int,
                 num_decoder_layers: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu",
                 layer_norm_eps: float = 1e-5):
        super().__init__()
        
        # Create encoder layer and full encoder
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation, layer_norm_eps
        )
        encoder_norm = LayerNorm([d_model], eps=layer_norm_eps)
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )
        
        # Create decoder layer and full decoder
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation, layer_norm_eps
        )
        decoder_norm = LayerNorm([d_model], eps=layer_norm_eps)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm
        )
        
        # Initialize parameters
        self._reset_parameters()
        
        self.d_model = d_model
        self.nhead = nhead
        
    def _reset_parameters(self):
        """
        Initialize parameters using Xavier uniform initialization.
        
        This is crucial for proper transformer operation - without good initialization,
        the network can collapse to outputting all zeros.
        """
        for p in self.parameters():
            if p.data.ndim > 1:
                # Xavier uniform initialization for weight matrices
                bound = np.sqrt(6.0 / sum(p.data.shape))
                p.data = np.random.uniform(-bound, bound, p.data.shape)
            else:
                # Initialize biases to small positive values to prevent dead neurons
                p.data.fill(0.01)
                
    def forward(self, src: Tensor, tgt: Tensor,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of transformer.
        
        Args:
            src: Source sequence
            tgt: Target sequence
            src_mask: Source sequence mask
            tgt_mask: Target sequence mask
            memory_mask: Memory mask
            
        Returns:
            Output tensor
        """
        # First run through encoder
        memory = self.encoder(src, src_mask)
        # Then through decoder
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        # Add a small epsilon to prevent exact zeros
        output = output + 1e-8
        return output