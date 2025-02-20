import pytest
import numpy as np
from DLpy.core import Tensor
from DLpy.nn.architectures.transformers.advanced_llm import KVCacheTransformer

class TestKVCacheTransformerAdvancedLLM:
    def setup_method(self):
        # Define a small model instance for testing.
        self.d_model = 16
        self.nhead = 4
        self.num_encoder_layers = 2
        self.num_decoder_layers = 2
        self.vocab_size = 50
        
        # Initialize the advanced LLM (with KV caching for efficient decoding).
        self.model = KVCacheTransformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            vocab_size=self.vocab_size,
        )
        # Set model to evaluation mode to disable dropout for deterministic behavior.
        self.model.eval()

    def test_forward_output_shape(self):
        """
        Test that the forward pass returns an output tensor with shape:
        (batch_size, target_seq_length, vocab_size)
        """
        batch_size = 2
        src_seq_len = 6
        tgt_seq_len = 5
        src = Tensor(np.random.randn(batch_size, src_seq_len, self.d_model))
        tgt = Tensor(np.random.randn(batch_size, tgt_seq_len, self.d_model))
        output = self.model.forward(src, tgt)
        assert output.shape == (batch_size, tgt_seq_len, self.vocab_size)

    def test_generate(self):
        """
        Test standard autoregressive generation.
        Verifies that the generated sequence starts with the start-of-sequence token (assumed to be 1)
        and that its length does not exceed max_length+1 (to account for the initial token).
        """
        batch_size = 2
        src_seq_len = 6
        src = Tensor(np.random.randn(batch_size, src_seq_len, self.d_model))
        max_length = 10
        generated = self.model.generate(src, max_length)
        # Check that the first token is the start token (assumed to be 1)
        assert np.all(generated.data[:, 0] == 1)
        # Check batch size and that sequence length is within limits.
        assert generated.shape[0] == batch_size
        assert generated.shape[1] <= max_length + 1

    def test_generate_speculative(self):
        """
        Test speculative decoding.
        Verifies that the generated sequence starts with the start token and
        that the overall length is within the expected bounds.
        """
        batch_size = 2
        src_seq_len = 6
        src = Tensor(np.random.randn(batch_size, src_seq_len, self.d_model))
        max_length = 10
        draft_steps = 3  # Number of candidate tokens to draft per iteration.
        generated = self.model.generate_speculative(src, max_length, draft_steps=draft_steps)
        assert np.all(generated.data[:, 0] == 1)
        assert generated.shape[0] == batch_size
        assert generated.shape[1] <= max_length + 1

    def test_generate_beam(self):
        """
        Test beam search decoding.
        Verifies that the returned best sequence starts with the start token and has a valid shape.
        """
        batch_size = 2
        src_seq_len = 6
        src = Tensor(np.random.randn(batch_size, src_seq_len, self.d_model))
        max_length = 10
        beam_width = 3
        generated = self.model.generate_beam(src, max_length, beam_width=beam_width)
        assert np.all(generated.data[:, 0] == 1)
        assert generated.shape[0] == batch_size
        assert generated.shape[1] <= max_length + 1

    def test_gradients(self):
        """
        Test that gradients propagate correctly through the forward pass.
        """
        batch_size = 2
        src_seq_len = 6
        tgt_seq_len = 5
        src = Tensor(np.random.randn(batch_size, src_seq_len, self.d_model), requires_grad=True)
        tgt = Tensor(np.random.randn(batch_size, tgt_seq_len, self.d_model), requires_grad=True)
        output = self.model.forward(src, tgt)
        loss = output.sum()
        loss.backward()
        # Check that gradients are computed for both source and target inputs.
        assert src.grad is not None
        assert tgt.grad is not None

    def test_deterministic_generation(self):
        """
        Test that generation is deterministic in evaluation mode when the random seed is fixed.
        """
        batch_size = 2
        src_seq_len = 6
        src = Tensor(np.random.randn(batch_size, src_seq_len, self.d_model))
        max_length = 5
        self.model.eval()
        np.random.seed(42)
        gen1 = self.model.generate(src, max_length)
        np.random.seed(42)
        gen2 = self.model.generate(src, max_length)
        assert np.array_equal(gen1.data, gen2.data)
