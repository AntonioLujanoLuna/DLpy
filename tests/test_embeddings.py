import pytest
import numpy as np
from DLpy.core import Tensor
from DLpy.nn.architectures.transformers.embedding import (
    PositionalEncoding,
    Embedding,
    LearnedPositionalEmbedding,
    RotaryPositionalEmbedding,
    ALiBiEmbedding,
)

# ---------------------------------------------------------------------------
# Tests for PositionalEncoding
# ---------------------------------------------------------------------------
class TestPositionalEncoding:
    def test_initialization(self):
        """Ensure the positional encoding buffer is created with the correct shape and pattern."""
        d_model = 16
        max_len = 50
        pe = PositionalEncoding(d_model, max_len, dropout=0.0)
        # Check that the registered buffer has shape (1, max_len, d_model)
        assert pe.pe.shape == (1, max_len, d_model)
        # For position 0, the sine should be sin(0)==0 and cosine should be cos(0)==1.
        even_idx = np.arange(0, d_model, 2)
        odd_idx = np.arange(1, d_model, 2)
        pos0 = pe.pe.data[0, 0]
        assert np.allclose(pos0[even_idx], np.sin(np.zeros_like(even_idx)), atol=1e-6)
        assert np.allclose(pos0[odd_idx], np.cos(np.zeros_like(odd_idx)), atol=1e-6)

    def test_forward(self):
        """Test that a forward pass adds the positional encoding to the input."""
        batch_size = 4
        seq_len = 10
        d_model = 16
        pe = PositionalEncoding(d_model, max_len=50, dropout=0.0)
        pe.eval()  # Disable dropout for deterministic output
        x = Tensor(np.zeros((batch_size, seq_len, d_model)))
        output = pe(x)
        # Output should have the same shape as input
        assert output.shape == (batch_size, seq_len, d_model)
        # Since input is zeros, output should equal the positional encoding slice
        expected = pe.pe.data[:, :seq_len, :]
        assert np.allclose(output.data, expected, atol=1e-6)

# ---------------------------------------------------------------------------
# Tests for Embedding (Token Embedding)
# ---------------------------------------------------------------------------
class TestEmbedding:
    def test_initialization(self):
        """Ensure that the embedding weight matrix is correctly initialized."""
        vocab_size = 100
        d_model = 16
        emb = Embedding(vocab_size, d_model)
        assert emb.weight.shape == (vocab_size, d_model)

    def test_forward(self):
        """Test that forward pass returns correct embedding vectors given token indices."""
        vocab_size = 50
        d_model = 16
        emb = Embedding(vocab_size, d_model)
        # Create a batch of token indices
        batch_size = 3
        seq_len = 7
        indices = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
        x = Tensor(indices)
        output = emb(x)
        # Expected output shape is (batch_size, seq_len, d_model)
        assert output.shape == (batch_size, seq_len, d_model)
        # Verify that the embeddings match the weight lookup
        expected = emb.weight.data[indices]
        assert np.allclose(output.data, expected, atol=1e-6)

# ---------------------------------------------------------------------------
# Tests for LearnedPositionalEmbedding
# ---------------------------------------------------------------------------
class TestLearnedPositionalEmbedding:
    def test_forward(self):
        """Test that learned positional embeddings add a learned offset to the input."""
        d_model = 16
        max_len = 30
        # Set dropout to zero for deterministic behavior.
        lpe = LearnedPositionalEmbedding(d_model, max_len, dropout=0.0)
        lpe.eval()  # Disable dropout
        batch_size = 4
        seq_len = 10
        x = Tensor(np.random.randn(batch_size, seq_len, d_model))
        output = lpe(x)
        # Check that output shape is preserved.
        assert output.shape == (batch_size, seq_len, d_model)
        # Since the operation is x + pe (with dropout off), the output should differ from x.
        assert not np.allclose(output.data, x.data, atol=1e-6)

# ---------------------------------------------------------------------------
# Tests for RotaryPositionalEmbedding
# ---------------------------------------------------------------------------
class TestRotaryPositionalEmbedding:
    def test_invalid_dimension(self):
        """Test that initializing with an odd d_model raises a ValueError."""
        with pytest.raises(ValueError):
            RotaryPositionalEmbedding(15)  # d_model must be even

    def test_forward_shape(self):
        """Test that the rotary embedding returns an output of the same shape as the input."""
        d_model = 16
        max_len = 20
        rope = RotaryPositionalEmbedding(d_model, max_len)
        batch_size = 3
        seq_len = 10
        x = Tensor(np.random.randn(batch_size, seq_len, d_model))
        output = rope(x)
        assert output.shape == (batch_size, seq_len, d_model)

    def test_forward_with_zeros(self):
        """Test that applying rotary embedding to an all-zero tensor returns zeros."""
        d_model = 16
        max_len = 20
        rope = RotaryPositionalEmbedding(d_model, max_len)
        batch_size = 2
        seq_len = 5
        x = Tensor(np.zeros((batch_size, seq_len, d_model)))
        output = rope(x)
        assert np.allclose(output.data, 0, atol=1e-6)

# ---------------------------------------------------------------------------
# Tests for ALiBiEmbedding
# ---------------------------------------------------------------------------
class TestALiBiEmbedding:
    def test_forward_shape(self):
        """Test that ALiBiEmbedding produces a bias matrix of the expected shape."""
        num_heads = 4
        alibi = ALiBiEmbedding(num_heads)
        seq_len = 10
        bias = alibi(seq_len)
        # Expected shape: (1, num_heads, seq_len, seq_len)
        assert bias.shape == (1, num_heads, seq_len, seq_len)

    def test_bias_values(self):
        """Test that the bias matrix is non-positive and diagonal entries are zero."""
        num_heads = 4
        alibi = ALiBiEmbedding(num_heads)
        seq_len = 8
        bias = alibi(seq_len).data  # shape (1, num_heads, seq_len, seq_len)
        # For each head and for each position, diagonal should be zero and upper-triangular entries should be negative.
        for h in range(num_heads):
            for i in range(seq_len):
                # Diagonal
                assert np.isclose(bias[0, h, i, i], 0, atol=1e-6)
                # For positions j > i, bias should be negative
                for j in range(i+1, seq_len):
                    assert bias[0, h, i, j] < 0
                # For positions j < i, mask was set to zero (since max(j-i,0)==0)
                for j in range(i):
                    assert np.isclose(bias[0, h, i, j], 0, atol=1e-6)
