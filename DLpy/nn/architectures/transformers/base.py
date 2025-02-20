from typing import Optional, Union

import numpy as np

from ....core import Module, Tensor
from ....utils import calculate_fan_in_fan_out
from ...norm.layer_norm import LayerNorm
from ..transformers.decoder import (
    AdvancedTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from ..transformers.encoder import TransformerEncoder, TransformerEncoderLayer


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

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()

        # Create encoder layer and full encoder
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps
        )
        encoder_norm = LayerNorm([d_model], eps=layer_norm_eps)
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        # Create decoder layer and full decoder
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps
        )
        decoder_norm = LayerNorm([d_model], eps=layer_norm_eps)
        self.decoder: Union[TransformerDecoder, AdvancedTransformerDecoder] = (
            TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        )

        # Initialize parameters
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self) -> None:
        """Initialize parameters using a layer-dependent scale."""
        for p in self.parameters():
            if isinstance(p, Tensor) and p.data.ndim > 1:
                # Calculate fan_in and fan_out
                fan_in, fan_out = calculate_fan_in_fan_out(p.data)
                # Layer-dependent scaling
                std = np.sqrt(2.0 / float(fan_in + fan_out))
                # Initialize using normal distribution
                p.data = np.random.normal(0.0, std, p.data.shape)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
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
        return Tensor(output)


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
