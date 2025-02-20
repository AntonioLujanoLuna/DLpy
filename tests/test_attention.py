import pytest
import numpy as np
from DLpy.core import Tensor
from DLpy.nn.attention import (
    MultiHeadAttention,
    AdditiveAttention,
    LinearAttention,
    SparseAttention,
    get_angles,
)

# ---------------------------------------------------------------------------
# Tests for MultiHeadAttention (for completeness; similar tests already exist)
# ---------------------------------------------------------------------------
class TestMultiHeadAttention:
    def test_attention_shape(self):
        """Test that multi-head attention returns outputs and attention weights of correct shapes."""
        batch_size = 2
        seq_len = 4
        embed_dim = 16
        num_heads = 4

        mha = MultiHeadAttention(embed_dim, num_heads)
        query = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
        key = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
        value = Tensor(np.random.randn(batch_size, seq_len, embed_dim))

        output, attn, _ = mha(query, key, value)
        assert output.shape == (batch_size, seq_len, embed_dim)
        assert attn.shape == (batch_size, num_heads, seq_len, seq_len)

    def test_attention_mask(self):
        """Test that providing an attention mask results in masked attention weights."""
        batch_size = 1
        seq_len = 4
        embed_dim = 16
        num_heads = 4

        mha = MultiHeadAttention(embed_dim, num_heads)
        query = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
        key = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
        value = Tensor(np.random.randn(batch_size, seq_len, embed_dim))

        # Create an upper-triangular mask (with -inf) for each head
        mask_array = np.triu(np.ones((batch_size, num_heads, seq_len, seq_len)), k=1) * -1e9
        mask = Tensor(mask_array)
        output, attn, _ = mha(query, key, value, mask)
        # Check that for each query (except last) the final key gets negligible weight.
        for q in range(seq_len - 1):
            assert np.all(attn.data[:, :, q, -1] < 0.1)
        # The last query should not be masked against itself.
        assert np.all(attn.data[:, :, -1, -1] > 0.0)

    def test_gradient_flow(self):
        """Test that gradients propagate through multi-head attention."""
        batch_size = 2
        seq_len = 3
        embed_dim = 16
        num_heads = 4

        mha = MultiHeadAttention(embed_dim, num_heads)
        query = Tensor(np.random.randn(batch_size, seq_len, embed_dim), requires_grad=True)
        key = Tensor(np.random.randn(batch_size, seq_len, embed_dim), requires_grad=True)
        value = Tensor(np.random.randn(batch_size, seq_len, embed_dim), requires_grad=True)

        output, _, _ = mha(query, key, value)
        loss = output.sum()
        loss.backward()
        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None

# ---------------------------------------------------------------------------
# Tests for AdditiveAttention
# ---------------------------------------------------------------------------
class TestAdditiveAttention:
    def test_initialization(self):
        """Test proper initialization of the AdditiveAttention module."""
        query_dim = 16
        key_dim = 16
        hidden_dim = 32
        attn = AdditiveAttention(query_dim, key_dim, hidden_dim)
        assert hasattr(attn, "linear_query")
        assert hasattr(attn, "linear_key")
        assert hasattr(attn, "v")

    def test_forward_shape(self):
        """Test that forward pass returns context and attention weights of expected shapes."""
        batch_size = 2
        query_len = 3
        key_len = 4
        query_dim = 16
        key_dim = 16
        value_dim = 20
        hidden_dim = 32

        attn = AdditiveAttention(query_dim, key_dim, hidden_dim)
        query = Tensor(np.random.randn(batch_size, query_len, query_dim))
        keys = Tensor(np.random.randn(batch_size, key_len, key_dim))
        values = Tensor(np.random.randn(batch_size, key_len, value_dim))

        context, attn_weights = attn(query, keys, values)
        assert context.shape == (batch_size, query_len, value_dim)
        assert attn_weights.shape == (batch_size, query_len, key_len)

    def test_forward_with_mask(self):
        """Test that the attention mask properly suppresses masked positions."""
        batch_size = 1
        query_len = 3
        key_len = 4
        query_dim = 8
        key_dim = 8
        value_dim = 10
        hidden_dim = 16

        attn = AdditiveAttention(query_dim, key_dim, hidden_dim)
        query = Tensor(np.random.randn(batch_size, query_len, query_dim))
        keys = Tensor(np.random.randn(batch_size, key_len, key_dim))
        values = Tensor(np.random.randn(batch_size, key_len, value_dim))
        # Create a mask that sets the last key to a very low value
        mask_array = np.zeros((batch_size, query_len, key_len))
        mask_array[:, :, -1] = -1e9
        mask = Tensor(mask_array)

        context, attn_weights = attn(query, keys, values, mask=mask)
        # The attention weights for the last key should be near zero
        assert np.allclose(attn_weights.data[..., -1], 0, atol=1e-6)

    def test_gradient_flow(self):
        """Test that gradients propagate through additive attention."""
        batch_size = 1
        query_len = 3
        key_len = 4
        query_dim = 8
        key_dim = 8
        value_dim = 10
        hidden_dim = 16

        attn = AdditiveAttention(query_dim, key_dim, hidden_dim)
        query = Tensor(np.random.randn(batch_size, query_len, query_dim), requires_grad=True)
        keys = Tensor(np.random.randn(batch_size, key_len, key_dim), requires_grad=True)
        values = Tensor(np.random.randn(batch_size, key_len, value_dim), requires_grad=True)

        context, _ = attn(query, keys, values)
        loss = context.sum()
        loss.backward()
        assert query.grad is not None
        assert keys.grad is not None
        assert values.grad is not None

# ---------------------------------------------------------------------------
# Tests for LinearAttention
# ---------------------------------------------------------------------------
class TestLinearAttention:
    def test_initialization(self):
        """Test proper initialization of LinearAttention."""
        embed_dim = 16
        num_heads = 4
        attn = LinearAttention(embed_dim, num_heads)
        assert attn.embed_dim == embed_dim
        assert attn.num_heads == num_heads
        assert attn.head_dim == embed_dim // num_heads

    def test_forward_shape(self):
        """Test that LinearAttention returns output and dummy attention weights of correct shapes."""
        batch_size = 2
        seq_len = 5
        embed_dim = 16
        num_heads = 4

        attn = LinearAttention(embed_dim, num_heads)
        query = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
        key = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
        value = Tensor(np.random.randn(batch_size, seq_len, embed_dim))

        output, attn_weights = attn(query, key, value)
        assert output.shape == (batch_size, seq_len, embed_dim)
        assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
        # According to the implementation, attention weights are a dummy zero tensor.
        assert np.allclose(attn_weights.data, 0, atol=1e-6)

    def test_gradient_flow(self):
        """Test gradient propagation through LinearAttention."""
        batch_size = 2
        seq_len = 5
        embed_dim = 16
        num_heads = 4

        attn = LinearAttention(embed_dim, num_heads)
        query = Tensor(np.random.randn(batch_size, seq_len, embed_dim), requires_grad=True)
        key = Tensor(np.random.randn(batch_size, seq_len, embed_dim), requires_grad=True)
        value = Tensor(np.random.randn(batch_size, seq_len, embed_dim), requires_grad=True)

        output, _ = attn(query, key, value)
        loss = output.sum()
        loss.backward()
        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None

# ---------------------------------------------------------------------------
# Tests for SparseAttention
# ---------------------------------------------------------------------------
class TestSparseAttention:
    def test_initialization(self):
        """Test proper initialization of SparseAttention."""
        embed_dim = 16
        num_heads = 4
        block_size = 2
        attn = SparseAttention(embed_dim, num_heads, block_size)
        assert attn.embed_dim == embed_dim
        assert attn.num_heads == num_heads
        assert attn.head_dim == embed_dim // num_heads
        assert attn.block_size == block_size

    def test_forward_shape(self):
        """Test that SparseAttention returns outputs and attention weights of correct shapes."""
        batch_size = 2
        seq_len = 6
        embed_dim = 16
        num_heads = 4
        block_size = 2

        attn = SparseAttention(embed_dim, num_heads, block_size)
        query = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
        key = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
        value = Tensor(np.random.randn(batch_size, seq_len, embed_dim))

        output, attn_weights = attn(query, key, value)
        assert output.shape == (batch_size, seq_len, embed_dim)
        assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)

    def test_local_mask_effect(self):
        """
        Test that the block-sparse mask in SparseAttention zeroes out attention weights
        for positions outside the local window.
        """
        batch_size = 1
        seq_len = 6
        embed_dim = 16
        num_heads = 2
        block_size = 1  # Each token attends only to its immediate neighbors

        attn = SparseAttention(embed_dim, num_heads, block_size)
        query = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
        key = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
        value = Tensor(np.random.randn(batch_size, seq_len, embed_dim))

        _, attn_weights = attn(query, key, value)
        # For each query position i, allowed key indices are from max(0, i-block_size) to min(seq_len, i+block_size+1)
        for i in range(seq_len):
            allowed_start = max(0, i - block_size)
            allowed_end = min(seq_len, i + block_size + 1)
            for j in range(seq_len):
                if j < allowed_start or j >= allowed_end:
                    # The attention weight for positions outside the allowed window should be 0.
                    assert np.allclose(attn_weights.data[0, :, i, j], 0, atol=1e-6)

    def test_forward_with_mask(self):
        """Test that an external mask is correctly applied on top of the sparse mask."""
        batch_size = 1
        seq_len = 6
        embed_dim = 16
        num_heads = 2
        block_size = 2

        attn = SparseAttention(embed_dim, num_heads, block_size)
        query = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
        key = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
        value = Tensor(np.random.randn(batch_size, seq_len, embed_dim))
        # Create an external mask that forces the first key to be masked for all queries.
        ext_mask_array = np.zeros((batch_size, num_heads, seq_len, seq_len))
        ext_mask_array[:, :, :, 0] = -1e9
        ext_mask = Tensor(ext_mask_array)

        _, attn_weights = attn(query, key, value, mask=ext_mask)
        # Verify that for every query position the attention weight for key index 0 is near zero.
        assert np.allclose(attn_weights.data[..., 0], 0, atol=1e-6)

    def test_gradient_flow(self):
        """Test gradient propagation through SparseAttention."""
        batch_size = 2
        seq_len = 6
        embed_dim = 16
        num_heads = 2
        block_size = 2

        attn = SparseAttention(embed_dim, num_heads, block_size)
        query = Tensor(np.random.randn(batch_size, seq_len, embed_dim), requires_grad=True)
        key = Tensor(np.random.randn(batch_size, seq_len, embed_dim), requires_grad=True)
        value = Tensor(np.random.randn(batch_size, seq_len, embed_dim), requires_grad=True)

        output, _ = attn(query, key, value)
        loss = output.sum()
        loss.backward()
        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None

# ---------------------------------------------------------------------------
# Tests for Attention Utilities
# ---------------------------------------------------------------------------
class TestAttentionUtils:
    def test_get_angles(self):
        """Test that get_angles computes the correct shape and values for positional encoding."""
        pos = np.arange(5)
        i = np.arange(0, 16, 2)
        d_model = 16
        angles = get_angles(pos, i, d_model)
        # Expected shape: (5, len(i))
        assert angles.shape == (5, len(i))
        # For position 0, all angles should be 0
        assert np.allclose(angles[0], 0, atol=1e-6)
