from typing import List, Optional, Tuple

from ....core import Module, Tensor
from ...attention import MultiHeadAttention
from ...base.activations import ReLU
from ...base.dropout import Dropout
from ...base.linear import Linear
from ...base.sequential import Sequential
from ...norm.layer_norm import LayerNorm


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

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
    ):
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
        self.ff = Sequential(self.linear1, self.activation, self.dropout, self.linear2)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
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
        attn_output, _, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Cross-attention block
        attn_output, _, _ = self.multihead_attn(x, memory, memory, memory_mask)
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

    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: Optional[Module] = None,
    ):
        super().__init__()
        self.layers = Sequential(*[decoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
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


class AdvancedTransformerDecoderLayer(Module):
    """
    Advanced transformer decoder layer supporting KV caching.

    Args:
        d_model (int): The dimension of the model.
        nhead (int): Number of attention heads.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout probability.
        activation (str): Activation function.
        layer_norm_eps (float): Epsilon for layer normalization.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.activation = ReLU()
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm([d_model], eps=layer_norm_eps)
        self.norm2 = LayerNorm([d_model], eps=layer_norm_eps)
        self.norm3 = LayerNorm([d_model], eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.ff = Sequential(
            self.linear1, self.activation, Dropout(dropout), self.linear2
        )

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        cache: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Forward pass with KV caching.

        Args:
            x (Tensor): Target sequence embeddings.
            memory (Tensor): Encoder output.
            tgt_mask (Optional[Tensor]): Target mask.
            memory_mask (Optional[Tensor]): Memory mask.
            cache (Optional[Tuple[Tensor, Tensor]]): Cached (key, value) pair.

        Returns:
            Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]: Output tensor and updated cache.
        """
        self_attn_output, _, updated_cache = self.self_attn(
            x,
            x,
            x,
            tgt_mask,
            past_key=cache[0] if cache is not None else None,
            past_value=cache[1] if cache is not None else None,
            use_cache=(cache is not None),
        )
        x = x + self.dropout1(self_attn_output)
        x = self.norm1(x)
        cross_attn_output, _, _ = self.multihead_attn(x, memory, memory, memory_mask)
        x = x + self.dropout2(cross_attn_output)
        x = self.norm2(x)
        ff_output = self.ff(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)
        return x, updated_cache


class AdvancedTransformerDecoder(Module):
    """
    Advanced transformer decoder composed of multiple AdvancedTransformerDecoderLayer.

    Args:
        decoder_layer: An instance of AdvancedTransformerDecoderLayer.
        num_layers (int): Number of decoder layers.
        norm (Optional[Module]): Optional normalization module.
    """

    def __init__(
        self,
        decoder_layer: AdvancedTransformerDecoderLayer,
        num_layers: int,
        norm: Optional[Module] = None,
    ):
        super().__init__()
        self.layers = Sequential(*[decoder_layer for _ in range(num_layers)])
        self.norm = norm
        self.num_layers = num_layers

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        caches: Optional[List[Optional[Tuple[Tensor, Tensor]]]] = None,
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        """
        Forward pass with per-layer KV caching.

        Args:
            tgt (Tensor): Target sequence embeddings.
            memory (Tensor): Encoder output.
            tgt_mask (Optional[Tensor]): Target mask.
            memory_mask (Optional[Tensor]): Memory mask.
            caches (Optional[List[Optional[Tuple[Tensor, Tensor]]]]): List of caches for each layer.

        Returns:
            Tuple[Tensor, List[Tuple[Tensor, Tensor]]]: Output tensor and list of updated caches.
        """
        if caches is None:
            caches = [None] * self.num_layers

        output = tgt
        new_caches = []
        for i, layer in enumerate(self.layers):
            output, layer_cache = layer(
                output, memory, tgt_mask, memory_mask, cache=caches[i]
            )
            new_caches.append(layer_cache)

        if self.norm is not None:
            output = self.norm(output)

        return output, new_caches
