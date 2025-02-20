# File: nn/architectures/transformers/generation_utils.py
from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ....core import Tensor
from .embedding import Embedding


def initialize_generation_token(
    batch_size: int, embedding_layer: Embedding, d_model: int
) -> Tensor:
    """
    Initialize the generation with a start-of-sequence token.

    For simplicity, we assume the start token index is 1.

    Args:
        batch_size (int): Batch size.
        embedding_layer (Embedding): Token embedding layer.
        d_model (int): Model dimension.

    Returns:
        Tensor: Tensor of shape (batch_size, 1) with start token indices.
    """
    start_token = 1  # Assume start-of-sequence token id is 1.
    tokens = np.full((batch_size, 1), start_token, dtype=np.int32)
    return Tensor(tokens)


def process_output_with_sampling(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    nucleus_p: Optional[float] = None,
) -> int:
    """
    Process the model output logits to determine the next token using sampling strategies.

    Applies temperature scaling, top-k filtering, and nucleus (top-p) filtering to the logits,
    and then samples a token from the resulting probability distribution.

    Args:
        logits (Tensor): Logits of shape (batch_size, seq_len, vocab_size).
        temperature (float): Temperature for scaling logits.
        top_k (int, optional): Number of top tokens to consider.
        nucleus_p (float, optional): Cumulative probability threshold for nucleus sampling.

    Returns:
        int: Next token id.
    """
    # For simplicity, assume batch size is 1.
    last_logits = logits.data[0, -1, :]  # shape (vocab_size,)
    # Apply temperature scaling.
    scaled_logits = last_logits / temperature

    # Compute softmax probabilities.
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    probs = exp_logits / np.sum(exp_logits)

    # Apply top-k filtering if specified.
    if top_k is not None and top_k < len(probs):
        top_k_indices = np.argpartition(probs, -top_k)[-top_k:]
        top_k_mask = np.zeros_like(probs, dtype=bool)
        top_k_mask[top_k_indices] = True
        probs = np.where(top_k_mask, probs, 0)
        probs = probs / np.sum(probs)

    # Apply nucleus (top-p) filtering if specified.
    if nucleus_p is not None and 0 < nucleus_p < 1.0:
        sorted_indices = np.argsort(-probs)
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumulative_probs, nucleus_p, side="left") + 1
        nucleus_indices = sorted_indices[:cutoff]
        nucleus_mask = np.zeros_like(probs, dtype=bool)
        nucleus_mask[nucleus_indices] = True
        probs = np.where(nucleus_mask, probs, 0)
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
        else:
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
            probs = exp_logits / np.sum(exp_logits)

    # Sample token from the filtered probability distribution.
    next_token = int(np.random.choice(len(probs), p=probs))
    return next_token


def get_top_candidates(
    logits: Tensor,
    beam_width: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    nucleus_p: Optional[float] = None,
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """
    Process logits to return top candidate token ids and their log probabilities.

    Args:
        logits (Tensor): Logits of shape (batch_size, seq_len, vocab_size).
        beam_width (int): Maximum number of candidate tokens to consider.
        temperature (float): Temperature for scaling logits.
        top_k (int, optional): Top-k filtering.
        nucleus_p (float, optional): Nucleus (top-p) filtering.

    Returns:
        tuple:
            candidate_indices (np.ndarray): Candidate token ids.
            candidate_log_probs (np.ndarray): Corresponding log probabilities.
    """
    last_logits = logits.data[0, -1, :]  # shape (vocab_size,)
    scaled_logits = last_logits / temperature
    # Compute probabilities and log probabilities.
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    probs = exp_logits / np.sum(exp_logits)
    log_probs = np.log(probs + 1e-10)

    candidate_indices = np.arange(len(log_probs))
    candidate_log_probs = log_probs

    # Apply top-k filtering.
    if top_k is not None and top_k < len(candidate_indices):
        top_k_indices = np.argpartition(candidate_log_probs, -top_k)[-top_k:]
        candidate_indices = top_k_indices
        candidate_log_probs = candidate_log_probs[top_k_indices]

    # Apply nucleus filtering.
    if nucleus_p is not None and 0 < nucleus_p < 1.0:
        sorted_order = np.argsort(-candidate_log_probs)
        candidate_indices = candidate_indices[sorted_order]
        candidate_log_probs = candidate_log_probs[sorted_order]
        sorted_probs = np.exp(candidate_log_probs)
        cumulative_probs = np.cumsum(sorted_probs / np.sum(sorted_probs))
        cutoff = np.searchsorted(cumulative_probs, nucleus_p, side="left") + 1
        candidate_indices = candidate_indices[:cutoff]
        candidate_log_probs = candidate_log_probs[:cutoff]

    # Sort candidates by descending log probability.
    sorted_order = np.argsort(-candidate_log_probs)
    candidate_indices = candidate_indices[sorted_order]
    candidate_log_probs = candidate_log_probs[sorted_order]

    # Limit to beam_width candidates.
    if len(candidate_indices) > beam_width:
        candidate_indices = candidate_indices[:beam_width]
        candidate_log_probs = candidate_log_probs[:beam_width]

    return candidate_indices, candidate_log_probs


def append_token(generated: Tensor, next_token: int) -> Tensor:
    """
    Append a token to the generated sequence.

    Args:
        generated (Tensor): Tensor of shape (batch_size, seq_len) with token ids.
        next_token (int): Next token id to append.

    Returns:
        Tensor: Updated generated sequence.
    """
    current_tokens = generated.data  # shape (batch_size, seq_len)
    batch_size = current_tokens.shape[0]
    new_token = np.full((batch_size, 1), next_token, dtype=current_tokens.dtype)
    updated_tokens = np.concatenate([current_tokens, new_token], axis=1)
    return Tensor(updated_tokens)


def end_of_sequence(token: int) -> bool:
    """
    Check if the given token id is the end-of-sequence token.

    For simplicity, we assume the EOS token id is 2.

    Args:
        token (int): Token id.

    Returns:
        bool: True if token is EOS, else False.
    """
    eos_token = 2  # Assume end-of-sequence token id is 2.
    return token == eos_token
