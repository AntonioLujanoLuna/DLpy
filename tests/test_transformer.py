import pytest
import numpy as np
from DLpy.core import Tensor
from DLpy.nn import (
    MultiHeadAttention,
    TransformerEncoderLayer,
    TransformerEncoder,
    PositionalEncoding,
    LayerNorm,
    TransformerDecoderLayer,
    TransformerDecoder,
    Transformer,
    generate_square_subsequent_mask
)

class TestMultiHeadAttention:
    """Tests for MultiHeadAttention module."""
    
    def test_initialization(self):
        """Test proper initialization of multi-head attention."""
        embed_dim = 512
        num_heads = 8
        
        mha = MultiHeadAttention(embed_dim, num_heads)
        
        # Test dimensions
        assert mha.embed_dim == embed_dim
        assert mha.num_heads == num_heads
        assert mha.head_dim == embed_dim // num_heads
        
        # Test scaling factor
        assert np.isclose(mha.scaling, (embed_dim // num_heads) ** -0.5)
        
        # Test invalid dimensions
        with pytest.raises(ValueError):
            MultiHeadAttention(512, 7)  # embed_dim not divisible by num_heads
            
    def test_attention_shape(self):
        """Test output shapes of attention computation."""
        batch_size = 32
        seq_len = 10
        embed_dim = 512
        num_heads = 8
        
        mha = MultiHeadAttention(embed_dim, num_heads)
        
        # Create input tensors
        query = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
        key = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
        value = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
        
        output, attention = mha(query, key, value)
        
        # Check output shapes
        assert output.shape == (batch_size, seq_len, embed_dim)
        assert attention.shape == (batch_size, num_heads, seq_len, seq_len)
        
    def test_attention_mask(self):
        """Test attention masking."""
        batch_size = 2
        seq_len = 4
        embed_dim = 8
        num_heads = 2

        mha = MultiHeadAttention(embed_dim, num_heads)

        # Create input tensors
        query = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
        key = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
        value = Tensor(np.random.randn(batch_size, seq_len, embed_dim))

        # Create attention mask (mask out upper triangular part)
        mask = np.triu(np.ones((batch_size, num_heads, seq_len, seq_len)), k=1) * -1e9
        mask = Tensor(mask)

        output, attention = mha(query, key, value, mask)

        # Check that masked positions have very low attention weights
        # For queries 0, 1, 2, the last key should be masked
        for q in range(seq_len - 1):
            assert np.all(attention.data[:, :, q, -1] < 0.1), f"Attention for query {q} to last key is not masked"

        # For the last query, the attention to the last key should not be masked
        assert np.all(attention.data[:, :, -1, -1] >= 0.1), "Attention for the last query to last key should not be masked"
        
    def test_attention_gradients(self):
        """Test gradient computation through attention."""
        batch_size = 2
        seq_len = 3
        embed_dim = 6
        num_heads = 2
        
        mha = MultiHeadAttention(embed_dim, num_heads)
        
        # Create input tensors with gradients
        query = Tensor(np.random.randn(batch_size, seq_len, embed_dim), requires_grad=True)
        key = Tensor(np.random.randn(batch_size, seq_len, embed_dim), requires_grad=True)
        value = Tensor(np.random.randn(batch_size, seq_len, embed_dim), requires_grad=True)
        
        output, _ = mha(query, key, value)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients were computed
        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None

class TestPositionalEncoding:
    """Tests for PositionalEncoding module."""
    
    def test_initialization(self):
        """Test proper initialization of positional encoding."""
        d_model = 512
        max_len = 100
        
        pe = PositionalEncoding(d_model, max_len)
        
        # Check buffer shape
        assert pe.pe.shape == (1, max_len, d_model)
        
        # Check alternating sine and cosine patterns
        even_indices = np.arange(0, d_model, 2)
        odd_indices = np.arange(1, d_model, 2)
        
        # First position should follow sin/cos pattern
        assert np.allclose(pe.pe.data[0, 0, even_indices], 
                         np.sin(np.zeros_like(even_indices)))
        assert np.allclose(pe.pe.data[0, 0, odd_indices], 
                         np.cos(np.zeros_like(odd_indices)))
        
    def test_forward(self):
        """Test forward pass of positional encoding."""
        batch_size = 32
        seq_len = 10
        d_model = 512
        
        pe = PositionalEncoding(d_model)
        x = Tensor(np.random.randn(batch_size, seq_len, d_model))
        
        output = pe(x)
        
        # Check output shape
        assert output.shape == x.shape
        
        # Check that positional encoding was added
        assert not np.array_equal(output.data, x.data)
        
    def test_different_sequence_lengths(self):
        """Test positional encoding with different sequence lengths."""
        d_model = 8
        max_len = 10
        pe = PositionalEncoding(d_model, max_len)
        
        # Test with different sequence lengths
        for seq_len in [1, 5, max_len]:
            x = Tensor(np.random.randn(2, seq_len, d_model))
            output = pe(x)
            assert output.shape == (2, seq_len, d_model)

class TestTransformerEncoderLayer:
    """Tests for TransformerEncoderLayer module."""
    
    def test_initialization(self):
        """Test proper initialization of encoder layer."""
        d_model = 512
        nhead = 8
        dim_feedforward = 2048
        
        layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        
        # Check component existence
        assert hasattr(layer, 'self_attn')
        assert hasattr(layer, 'ff')
        assert hasattr(layer, 'norm1')
        assert hasattr(layer, 'norm2')
        
    def test_forward(self):
        """Test forward pass of encoder layer."""
        batch_size = 32
        seq_len = 10
        d_model = 512
        nhead = 8
        
        layer = TransformerEncoderLayer(d_model, nhead)
        x = Tensor(np.random.randn(batch_size, seq_len, d_model))
        
        output = layer(x)
        
        # Check output shape
        assert output.shape == x.shape
        
    def test_with_mask(self):
        """Test encoder layer with attention mask."""
        batch_size = 2
        seq_len = 4
        d_model = 8
        nhead = 2
        
        layer = TransformerEncoderLayer(d_model, nhead)
        x = Tensor(np.random.randn(batch_size, seq_len, d_model))
        
        # Create attention mask
        mask = Tensor(np.triu(np.ones((batch_size, nhead, seq_len, seq_len)) * -1e9, k=1))
        
        output = layer(x, mask)
        assert output.shape == x.shape

class TestTransformerEncoder:
    """Tests for TransformerEncoder module."""
    
    def test_initialization(self):
        """Test proper initialization of transformer encoder."""
        d_model = 512
        nhead = 8
        num_layers = 6
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead)
        encoder = TransformerEncoder(encoder_layer, num_layers)
        
        # Check number of layers
        assert len(encoder.layers) == num_layers
        
    def test_forward(self):
        """Test forward pass of transformer encoder."""
        batch_size = 32
        seq_len = 10
        d_model = 512
        nhead = 8
        num_layers = 6
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead)
        encoder = TransformerEncoder(encoder_layer, num_layers)
        
        x = Tensor(np.random.randn(batch_size, seq_len, d_model))
        output = encoder(x)
        
        # Check output shape
        assert output.shape == x.shape
        
    def test_with_final_norm(self):
        """Test transformer encoder with final layer normalization."""
        d_model = 512
        nhead = 8
        num_layers = 6
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead)
        norm = LayerNorm([d_model])
        encoder = TransformerEncoder(encoder_layer, num_layers, norm=norm)
        
        x = Tensor(np.random.randn(2, 4, d_model))
        output = encoder(x)
        
        # Check output shape
        assert output.shape == x.shape
        
    def test_gradient_flow(self):
        """Test gradient computation through entire encoder."""
        batch_size = 2
        seq_len = 4
        d_model = 8
        nhead = 2
        num_layers = 2
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead)
        encoder = TransformerEncoder(encoder_layer, num_layers)
        
        x = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
        output = encoder(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients were computed
        assert x.grad is not None

class TestIntegration:
    """Integration tests for transformer components."""
    
    def test_full_transformer_stack(self):
        """Test integration of all transformer components."""
        batch_size = 2
        seq_len = 4
        d_model = 8
        nhead = 2
        num_layers = 2
        
        # Create components
        pe = PositionalEncoding(d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead)
        encoder = TransformerEncoder(encoder_layer, num_layers)
        
        # Input data
        x = Tensor(np.random.randn(batch_size, seq_len, d_model))
        
        # Forward pass through each component
        x = pe(x)
        output = encoder(x)
        
        # Check final output shape
        assert output.shape == (batch_size, seq_len, d_model)
        
    def test_training_mode(self):
        """Test components behavior in training vs eval mode."""
        d_model = 8
        nhead = 2
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dropout=0.5)
        x = Tensor(np.random.randn(2, 4, d_model))
        
        # Training mode
        encoder_layer.train()
        out1 = encoder_layer(x)
        out2 = encoder_layer(x)
        
        # Outputs should be different due to dropout
        assert not np.array_equal(out1.data, out2.data)
        
        # Eval mode
        encoder_layer.eval()
        out1 = encoder_layer(x)
        out2 = encoder_layer(x)
        
        # Outputs should be the same
        assert np.array_equal(out1.data, out2.data)

class TestTransformerDecoderLayer:
    """Tests for TransformerDecoderLayer module."""
    
    def test_initialization(self):
        """Test proper initialization of decoder layer."""
        d_model = 512
        nhead = 8
        dim_feedforward = 2048
        
        layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        
        # Check component existence
        assert hasattr(layer, 'self_attn')
        assert hasattr(layer, 'multihead_attn')
        assert hasattr(layer, 'ff')
        assert hasattr(layer, 'norm1')
        assert hasattr(layer, 'norm2')
        assert hasattr(layer, 'norm3')
        
    def test_forward(self):
        """Test forward pass of decoder layer."""
        batch_size = 32
        seq_len = 10
        d_model = 512
        nhead = 8
        
        layer = TransformerDecoderLayer(d_model, nhead)
        
        # Create input tensors
        tgt = Tensor(np.random.randn(batch_size, seq_len, d_model))
        memory = Tensor(np.random.randn(batch_size, seq_len, d_model))
        
        output = layer(tgt, memory)
        
        # Check output shape
        assert output.shape == tgt.shape
        
    def test_with_masks(self):
        """Test decoder layer with attention masks."""
        batch_size = 2
        seq_len = 4
        d_model = 8
        nhead = 2
        
        layer = TransformerDecoderLayer(d_model, nhead)
        
        # Create input tensors
        tgt = Tensor(np.random.randn(batch_size, seq_len, d_model))
        memory = Tensor(np.random.randn(batch_size, seq_len, d_model))
        
        # Create attention masks
        tgt_mask = Tensor(np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1))
        memory_mask = Tensor(np.ones((seq_len, seq_len)) * -1e9)
        
        output = layer(tgt, memory, tgt_mask, memory_mask)
        assert output.shape == tgt.shape
        
    def test_gradient_flow(self):
        """Test gradient computation through decoder layer."""
        batch_size = 2
        seq_len = 4
        d_model = 8
        nhead = 2
        
        layer = TransformerDecoderLayer(d_model, nhead)
        
        # Create input tensors with gradients
        tgt = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
        memory = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
        
        output = layer(tgt, memory)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients were computed
        assert tgt.grad is not None
        assert memory.grad is not None

class TestTransformerDecoder:
    """Tests for TransformerDecoder module."""
    
    def test_initialization(self):
        """Test proper initialization of transformer decoder."""
        d_model = 512
        nhead = 8
        num_layers = 6
        
        decoder_layer = TransformerDecoderLayer(d_model, nhead)
        decoder = TransformerDecoder(decoder_layer, num_layers)
        
        # Check number of layers
        assert len(decoder.layers) == num_layers
        
    def test_forward(self):
        """Test forward pass of transformer decoder."""
        batch_size = 32
        seq_len = 10
        d_model = 512
        nhead = 8
        num_layers = 6
        
        decoder_layer = TransformerDecoderLayer(d_model, nhead)
        decoder = TransformerDecoder(decoder_layer, num_layers)
        
        # Create input tensors
        tgt = Tensor(np.random.randn(batch_size, seq_len, d_model))
        memory = Tensor(np.random.randn(batch_size, seq_len, d_model))
        
        output = decoder(tgt, memory)
        
        # Check output shape
        assert output.shape == tgt.shape
        
    def test_with_final_norm(self):
        """Test transformer decoder with final layer normalization."""
        d_model = 512
        nhead = 8
        num_layers = 6
        
        decoder_layer = TransformerDecoderLayer(d_model, nhead)
        norm = LayerNorm([d_model])
        decoder = TransformerDecoder(decoder_layer, num_layers, norm=norm)
        
        tgt = Tensor(np.random.randn(2, 4, d_model))
        memory = Tensor(np.random.randn(2, 4, d_model))
        
        output = decoder(tgt, memory)
        
        # Check output shape
        assert output.shape == tgt.shape

class TestTransformer:
    """Tests for complete Transformer model."""
    
    def test_initialization(self):
        """Test proper initialization of transformer."""
        d_model = 512
        nhead = 8
        num_encoder_layers = 6
        num_decoder_layers = 6
        
        transformer = Transformer(
            d_model, nhead, num_encoder_layers, num_decoder_layers
        )
        
        # Check component existence
        assert hasattr(transformer, 'encoder')
        assert hasattr(transformer, 'decoder')
        assert transformer.d_model == d_model
        assert transformer.nhead == nhead
        
    def test_forward(self):
        """Test forward pass of transformer."""
        batch_size = 32
        seq_len = 10
        d_model = 512
        nhead = 8
        
        transformer = Transformer(
            d_model, nhead,
            num_encoder_layers=6,
            num_decoder_layers=6
        )
        
        # Create input tensors
        src = Tensor(np.random.randn(batch_size, seq_len, d_model))
        tgt = Tensor(np.random.randn(batch_size, seq_len, d_model))
        
        output = transformer(src, tgt)
        
        # Check output shape
        assert output.shape == tgt.shape
        
    def test_with_masks(self):
        """Test transformer with various masks."""
        batch_size = 2
        seq_len = 4
        d_model = 8
        nhead = 2
        
        transformer = Transformer(
            d_model, nhead,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        
        # Create input tensors
        src = Tensor(np.random.randn(batch_size, seq_len, d_model))
        tgt = Tensor(np.random.randn(batch_size, seq_len, d_model))
        
        # Create masks
        src_mask = Tensor(np.ones((seq_len, seq_len)) * -1e9)
        tgt_mask = generate_square_subsequent_mask(seq_len)
        memory_mask = Tensor(np.ones((seq_len, seq_len)) * -1e9)
        
        output = transformer(src, tgt, src_mask, tgt_mask, memory_mask)
        assert output.shape == tgt.shape
        
    def test_gradient_flow(self):
        """Test gradient computation through entire transformer."""
        batch_size = 2
        seq_len = 4
        d_model = 8
        nhead = 2
        
        transformer = Transformer(
            d_model, nhead,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        
        # Create input tensors with gradients
        src = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
        tgt = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
        
        output = transformer(src, tgt)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients were computed
        assert src.grad is not None
        assert tgt.grad is not None

class TestMaskGeneration:
    """Tests for mask generation utilities."""
    
    def test_square_subsequent_mask(self):
        """Test generation of square subsequent mask."""
        size = 5
        mask = generate_square_subsequent_mask(size)
        
        # Check shape
        assert mask.shape == (size, size)
        
        # Check mask values
        # Lower triangle should be 0, upper triangle should be -inf
        for i in range(size):
            for j in range(size):
                if i < j:  # Upper triangle
                    assert mask.data[i, j] == -np.inf
                else:  # Lower triangle and diagonal
                    assert mask.data[i, j] == 0
                    
    def test_mask_broadcasting(self):
        """Test that generated masks can be properly broadcast."""
        size = 4
        mask = generate_square_subsequent_mask(size)
        
        # Create dummy attention scores
        scores = Tensor(np.random.randn(2, 8, size, size))  # (batch, heads, seq, seq)
        
        # Add mask to scores
        masked_scores = scores + mask
        
        # Check that upper triangle was properly masked
        for i in range(size):
            for j in range(i + 1, size):
                assert np.all(masked_scores.data[..., i, j] < -1e30)

class TestAdvancedFeatures:
    """Tests for advanced transformer features and edge cases."""
    
    def test_variable_sequence_length(self):
        """Test transformer with different sequence lengths for source and target."""
        d_model = 8
        nhead = 2
        
        transformer = Transformer(
            d_model, nhead,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        
        # Create inputs with different sequence lengths
        src = Tensor(np.random.randn(2, 6, d_model))  # seq_len = 6
        tgt = Tensor(np.random.randn(2, 4, d_model))  # seq_len = 4
        
        output = transformer(src, tgt)
        assert output.shape == tgt.shape
        
    def test_training_vs_inference(self):
        """Test transformer behavior in training vs inference modes."""
        d_model = 8
        nhead = 2
        
        transformer = Transformer(
            d_model, nhead,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dropout=0.5
        )
        
        src = Tensor(np.random.randn(2, 4, d_model))
        tgt = Tensor(np.random.randn(2, 4, d_model))
        
        # Training mode
        transformer.train()
        out1 = transformer(src, tgt)
        out2 = transformer(src, tgt)
        
        # Outputs should be different due to dropout
        assert not np.array_equal(out1.data, out2.data)
        
        # Eval mode
        transformer.eval()
        out1 = transformer(src, tgt)
        out2 = transformer(src, tgt)
        
        # Outputs should be the same
        assert np.array_equal(out1.data, out2.data)
        
    def test_large_attention_scores(self):
        """Test numerical stability with large attention scores."""
        d_model = 8
        nhead = 2
        
        transformer = Transformer(
            d_model, nhead,
            num_encoder_layers=1,
            num_decoder_layers=1
        )
        
        # Create inputs with large values
        src = Tensor(np.random.randn(2, 4, d_model) * 100)
        tgt = Tensor(np.random.randn(2, 4, d_model) * 100)
        
        output = transformer(src, tgt)
        
        # Check that output values are reasonable
        assert not np.any(np.isnan(output.data))
        assert not np.any(np.isinf(output.data))