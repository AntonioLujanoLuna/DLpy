# File: nn/architectures/transformers/advanced_llm.py
from typing import Any, List, Optional, Tuple, cast

import numpy as np

from ....core import Tensor
from ...base.linear import Linear
from .base import Transformer  # Base Transformer sets up the encoder, parameters, etc.
from .decoder import AdvancedTransformerDecoder, AdvancedTransformerDecoderLayer
from .embedding import Embedding
from .generation_utils import get_top_candidates  # New function for beam search.
from .generation_utils import (
    append_token,
    end_of_sequence,
    initialize_generation_token,
    process_output_with_sampling,
)


class KVCacheTransformer(Transformer):
    """
    Advanced Transformer model with KV caching for efficient autoregressive
    decoding and speculative decoding.

    This model reuses the base Transformer encoder but replaces the decoder with
    an advanced decoder module that supports KV caching. It also includes token
    embedding and an output projection for language modeling.

    Args:
        d_model (int): Model dimension.
        nhead (int): Number of attention heads.
        num_encoder_layers (int): Number of encoder layers.
        num_decoder_layers (int): Number of decoder layers.
        vocab_size (int): Vocabulary size.
        dim_feedforward (int): Feedforward network dimension.
        dropout (float): Dropout probability.
        activation (str): Activation function.
        layer_norm_eps (float): Epsilon for layer normalization.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        vocab_size: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
    ):
        # Initialize the base Transformer (which sets up the encoder)
        super().__init__(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
        )
        # Replace the decoder with the advanced decoder supporting KV caching.
        self.decoder = AdvancedTransformerDecoder(
            decoder_layer=AdvancedTransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps
            ),
            num_layers=num_decoder_layers,
            norm=None,
        )
        # Token embedding and output projection.
        self.token_embedding = Embedding(vocab_size, d_model)
        self.output_projection = Linear(d_model, vocab_size)
        self.vocab_size = vocab_size

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass (for training).
        """
        memory = self.encoder(src, src_mask)
        output, _ = self.decoder(tgt, memory, tgt_mask, memory_mask)
        # Add a small epsilon for numerical stability.
        output = output + 1e-8
        result = self.output_projection(output)
        return Tensor(result.data)

    def generate(self, src: Tensor, max_length: int, **generate_kwargs: Any) -> Tensor:
        """
        Standard autoregressive generation (greedy decoding) with KV caching.

        Args:
            src (Tensor): Source sequence.
            max_length (int): Maximum generated length.
            generate_kwargs: Additional parameters. Expected keys:
                - temperature: float, default 1.0
                - top_k: int, default None
                - nucleus_p: float, default None

        Returns:
            Tensor: Generated sequence of token ids.
        """
        temperature = generate_kwargs.get("temperature", 1.0)
        top_k = generate_kwargs.get("top_k", None)
        nucleus_p = generate_kwargs.get("nucleus_p", None)

        memory = self.encoder(src, None)  # Encode source once.
        batch_size = src.shape[0]
        # Initialize with the start-of-sequence token.
        generated = initialize_generation_token(
            batch_size, self.token_embedding, self.d_model
        )
        caches = [None] * self.decoder.num_layers

        for t in range(max_length):
            tgt_embed = self.token_embedding(generated)
            decoder_output, new_caches = self.decoder(
                tgt_embed, memory, None, None, caches
            )
            caches = new_caches
            logits = self.output_projection(decoder_output)
            next_token = process_output_with_sampling(
                logits, temperature, top_k, nucleus_p
            )
            generated = append_token(generated, next_token)
            if end_of_sequence(next_token):
                break
        return generated

    def generate_speculative(
        self, src: Tensor, max_length: int, draft_steps: int = 5, **generate_kwargs: Any
    ) -> Tensor:
        """
        Speculative decoding: use a draft phase to propose multiple tokens and then
        verify them with the full model. In this implementation, for each draft batch,
        we verify token-by-token using the full model’s prediction.

        Args:
            src (Tensor): Source sequence.
            max_length (int): Maximum generated length.
            draft_steps (int): Number of candidate tokens to draft.
            generate_kwargs: Additional parameters. Expected keys:
                - temperature: float, default 1.0
                - top_k: int, default None
                - nucleus_p: float, default None

        Returns:
            Tensor: Generated sequence of token ids.
        """
        temperature = generate_kwargs.get("temperature", 1.0)
        top_k = generate_kwargs.get("top_k", None)
        nucleus_p = generate_kwargs.get("nucleus_p", None)

        memory = self.encoder(src, None)
        batch_size = src.shape[0]
        generated = initialize_generation_token(
            batch_size, self.token_embedding, self.d_model
        )
        caches = [None] * self.decoder.num_layers
        t = 0

        while t < max_length:
            # Save current state for verification.
            prev_generated = generated
            prev_caches = caches

            # --- Draft Phase ---
            # Generate a batch of candidate tokens (draft_steps tokens).
            draft_tokens = []
            draft_caches = caches
            for _ in range(draft_steps):
                tgt_embed = self.token_embedding(generated)
                draft_decoder_output, draft_caches = self.decoder(
                    tgt_embed, memory, None, None, draft_caches
                )
                logits = self.output_projection(draft_decoder_output)
                candidate = process_output_with_sampling(
                    logits, temperature, top_k, nucleus_p
                )
                draft_tokens.append(candidate)
                generated = append_token(generated, candidate)
                t += 1
                if end_of_sequence(candidate):
                    break

            # --- Verification Phase ---
            # Re-run the full model (from the state before the draft) to verify candidates.
            accepted = 0
            verified_generated = prev_generated  # Roll back to state before draft.
            verified_caches = prev_caches

            for candidate in draft_tokens:
                tgt_embed = self.token_embedding(verified_generated)
                full_decoder_output, verified_caches = self.decoder(
                    tgt_embed, memory, None, None, verified_caches
                )
                logits = self.output_projection(full_decoder_output)
                full_token = process_output_with_sampling(
                    logits, temperature, top_k, nucleus_p
                )
                if full_token == candidate:
                    # Accept candidate token.
                    verified_generated = append_token(verified_generated, candidate)
                    accepted += 1
                else:
                    # Candidate did not match full model prediction; stop verification.
                    break

            if accepted < len(draft_tokens):
                # Roll back to verified sequence and take one full decoding step.
                generated = verified_generated
                tgt_embed = self.token_embedding(generated)
                full_decoder_output, caches = self.decoder(
                    tgt_embed, memory, None, None, verified_caches
                )
                logits = self.output_projection(full_decoder_output)
                next_token = process_output_with_sampling(
                    logits, temperature, top_k, nucleus_p
                )
                generated = append_token(generated, next_token)
                t += 1
                if end_of_sequence(next_token):
                    break
            else:
                # All draft tokens verified.
                caches = draft_caches
                if draft_tokens and end_of_sequence(draft_tokens[-1]):
                    break

        return generated

    def generate_beam(
        self, src: Tensor, max_length: int, beam_width: int = 3, **generate_kwargs: Any
    ) -> Tensor:
        """
        Generate sequence using beam search with KV caching and length normalization.

        Args:
            src (Tensor): Source sequence.
            max_length (int): Maximum generated length.
            beam_width (int): Beam width (number of candidate sequences maintained).
            generate_kwargs: Additional parameters. Expected keys:
                - temperature: float, default 1.0
                - top_k: int, default None
                - nucleus_p: float, default None
                - length_penalty: float, default 1.0 (no normalization if 1.0)

        Returns:
            Tensor: Best generated sequence of token ids.
        """
        temperature = generate_kwargs.get("temperature", 1.0)
        top_k = generate_kwargs.get("top_k", None)
        nucleus_p = generate_kwargs.get("nucleus_p", None)
        length_penalty = generate_kwargs.get("length_penalty", 1.0)

        memory = self.encoder(src, None)
        batch_size = src.shape[0]
        # Initialize with the start-of-sequence token.
        initial_seq = initialize_generation_token(
            batch_size, self.token_embedding, self.d_model
        )
        initial_caches = [None] * self.decoder.num_layers
        # Each beam is a tuple: (generated sequence, cumulative log probability, caches)
        beams = [(initial_seq, 0.0, initial_caches)]

        for t in range(max_length):
            new_beams = []
            complete_beams = []
            for seq, cum_log_prob, caches in beams:
                # If the last token is EOS, do not expand this beam.
                last_token = seq.data[0, -1]
                if end_of_sequence(last_token):
                    complete_beams.append((seq, cum_log_prob, caches))
                    continue

                tgt_embed = self.token_embedding(seq)
                decoder_output, new_caches = self.decoder(
                    tgt_embed, memory, None, None, caches
                )
                logits = self.output_projection(decoder_output)
                # Obtain candidate tokens and their log probabilities.
                candidate_tokens, candidate_log_probs = get_top_candidates(
                    logits, beam_width, temperature, top_k, nucleus_p
                )

                for token, token_log_prob in zip(candidate_tokens, candidate_log_probs):
                    new_seq = append_token(seq, int(token))
                    new_cum_log_prob = cum_log_prob + token_log_prob
                    new_beams.append((new_seq, new_cum_log_prob, new_caches))

            # Merge beams that have ended with those still in progress.
            beams = new_beams + complete_beams
            if not beams:
                break

            # Apply length normalization.
            # Compute normalized score = cumulative log probability / ((5 + length) / 6)^length_penalty.
            def normalized_score(beam: Tuple[Tensor, float, List[Any]]) -> float:
                seq, cum_log_prob, _ = beam
                seq_length = float(seq.data.shape[1])  # Explicitly convert to float
                norm = ((5.0 + seq_length) / 6.0) ** length_penalty
                return float(cum_log_prob / norm)  # Explicitly convert to float

            # Keep only the top beam_width beams based on normalized score.
            beams = sorted(beams, key=normalized_score, reverse=True)[:beam_width]
            # If all beams end with EOS, stop the search.
            if all(end_of_sequence(beam[0].data[0, -1]) for beam in beams):
                break

        # Select the beam with the highest normalized score.
        best_seq = max(beams, key=normalized_score)[0]
        return best_seq
