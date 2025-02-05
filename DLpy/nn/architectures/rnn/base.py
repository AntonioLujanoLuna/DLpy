from typing import Optional, Tuple

import numpy as np

from ..core import Module, Tensor


class LSTM(Module):
    """
    Long Short-Term Memory (LSTM) layer.

    Applies a multi-layer LSTM to an input sequence.

    Args:
        input_size: Number of features in input
        hidden_size: Number of features in hidden state
        num_layers: Number of recurrent layers
        bias: If False, doesn't learn bias weights
        batch_first: If True, input shape is (batch, seq, feature)
        dropout: Dropout probability between layers (0 means no dropout)
        bidirectional: If True, becomes bidirectional LSTM

    Inputs: input, (h_0, c_0)
        input: tensor of shape (seq_len, batch, input_size)
            or (batch, seq_len, input_size) if batch_first=True
        h_0: initial hidden state (num_layers * num_directions, batch, hidden_size)
        c_0: initial cell state (num_layers * num_directions, batch, hidden_size)

    Outputs: output, (h_n, c_n)
        output: tensor of shape (seq_len, batch, num_directions * hidden_size)
        h_n: final hidden state
        c_n: final cell state
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        if dropout < 0 or dropout > 1:
            raise ValueError(
                f"Dropout probability has to be between 0 and 1, but got {dropout}"
            )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        # Create parameter tensors for each gate (input, forget, cell, output)
        # for each layer and direction
        self.cell_list = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = (
                    input_size if layer == 0 else hidden_size * num_directions
                )
                cell = LSTMCell(layer_input_size, hidden_size, bias)
                name = f"cell_{layer}_{direction}"
                setattr(self, name, cell)
                self.cell_list.append(cell)

    def forward(
        self, x: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass of LSTM."""
        # Handle batch_first
        if self.batch_first:
            x = Tensor(np.swapaxes(x.data, 0, 1))  # Changed to use swapaxes

        seq_len, batch_size, _ = x.shape
        num_directions = 2 if self.bidirectional else 1

        if hx is None:
            h_0 = Tensor(
                np.zeros(
                    (self.num_layers * num_directions, batch_size, self.hidden_size)
                )
            )
            c_0 = Tensor(
                np.zeros(
                    (self.num_layers * num_directions, batch_size, self.hidden_size)
                )
            )
            hx = (h_0, c_0)
        else:
            # Validate hidden state dimensions
            h_0, c_0 = hx
            expected_shape = (
                self.num_layers * num_directions,
                batch_size,
                self.hidden_size,
            )
            if h_0.shape != expected_shape or c_0.shape != expected_shape:
                raise ValueError(
                    f"Expected hidden size {expected_shape}, "
                    f"got {h_0.shape} and {c_0.shape}"
                )

        h_n, c_n = hx
        layer_output = x
        new_h = []
        new_c = []

        # Process each layer
        for layer in range(self.num_layers):
            layer_h_list = []
            h_forward = h_n[layer * num_directions]
            c_forward = c_n[layer * num_directions]

            # Forward direction
            for t in range(seq_len):
                h_forward, c_forward = self.cell_list[layer * num_directions](
                    layer_output[t], (h_forward, c_forward)
                )
                layer_h_list.append(h_forward)

            if self.bidirectional:
                # Backward direction
                h_backward = h_n[layer * num_directions + 1]
                c_backward = c_n[layer * num_directions + 1]
                layer_h_back = []

                for t in range(seq_len - 1, -1, -1):
                    h_backward, c_backward = self.cell_list[layer * num_directions + 1](
                        layer_output[t], (h_backward, c_backward)
                    )
                    layer_h_back.append(h_backward)

                # Combine forward and backward outputs
                layer_h_back.reverse()
                layer_forward = Tensor(np.stack([h.data for h in layer_h_list]))
                layer_backward = Tensor(np.stack([h.data for h in layer_h_back]))
                layer_output = Tensor(
                    np.concatenate([layer_forward.data, layer_backward.data], axis=-1)
                )
            else:
                layer_output = Tensor(np.stack([h.data for h in layer_h_list]))

            # Apply dropout except for last layer
            if layer < self.num_layers - 1 and self.training and self.dropout > 0:
                mask = (np.random.rand(*layer_output.shape) > self.dropout).astype(
                    np.float64
                )
                layer_output = Tensor(layer_output.data * mask / (1 - self.dropout))

            new_h.append(h_forward.data)
            if self.bidirectional:
                new_h.append(h_backward.data)
            new_c.append(c_forward.data)
            if self.bidirectional:
                new_c.append(c_backward.data)

        # Stack hidden states and cell states
        h_n = Tensor(np.stack(new_h))
        c_n = Tensor(np.stack(new_c))

        # Restore batch_first if needed
        if self.batch_first:
            layer_output = Tensor(np.swapaxes(layer_output.data, 0, 1))

        return layer_output, (h_n, c_n)


class GRU(Module):
    """
    Gated Recurrent Unit (GRU) layer.

    Applies a multi-layer GRU to an input sequence.

    Args:
        input_size: Number of features in input
        hidden_size: Number of features in hidden state
        num_layers: Number of recurrent layers
        bias: If False, doesn't learn bias weights
        batch_first: If True, input shape is (batch, seq, feature)
        dropout: Dropout probability between layers (0 means no dropout)
        bidirectional: If True, becomes bidirectional GRU

    Inputs: input, h_0
        input: tensor of shape (seq_len, batch, input_size)
            or (batch, seq_len, input_size) if batch_first=True
        h_0: initial hidden state (num_layers * num_directions, batch, hidden_size)

    Outputs: output, h_n
        output: tensor of shape (seq_len, batch, num_directions * hidden_size)
        h_n: final hidden state
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        if dropout < 0 or dropout > 1:
            raise ValueError(
                f"Dropout probability has to be between 0 and 1, but got {dropout}"
            )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        # Create parameter tensors for each gate (reset, update, new)
        # for each layer and direction
        self.cell_list = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = (
                    input_size if layer == 0 else hidden_size * num_directions
                )
                cell = GRUCell(layer_input_size, hidden_size, bias)
                name = f"cell_{layer}_{direction}"
                setattr(self, name, cell)
                self.cell_list.append(cell)

    def forward(self, x: Tensor, hx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Forward pass of GRU."""
        # Handle batch_first
        if self.batch_first:
            x = Tensor(np.swapaxes(x.data, 0, 1))

        seq_len, batch_size, _ = x.shape
        num_directions = 2 if self.bidirectional else 1

        if hx is None:
            hx = Tensor(
                np.zeros(
                    (self.num_layers * num_directions, batch_size, self.hidden_size)
                )
            )
        else:
            # Validate hidden state dimensions
            expected_shape = (
                self.num_layers * num_directions,
                batch_size,
                self.hidden_size,
            )
            if hx.shape != expected_shape:
                raise ValueError(
                    f"Expected hidden size {expected_shape}, got {hx.shape}"
                )

        layer_output = x
        new_h = []

        # Process each layer
        for layer in range(self.num_layers):
            layer_h_list = []
            h_forward = hx[layer * num_directions]

            # Forward direction
            for t in range(seq_len):
                h_forward = self.cell_list[layer * num_directions](
                    layer_output[t], h_forward
                )
                layer_h_list.append(h_forward)

            if self.bidirectional:
                # Backward direction
                h_backward = hx[layer * num_directions + 1]
                layer_h_back = []

                for t in range(seq_len - 1, -1, -1):
                    h_backward = self.cell_list[layer * num_directions + 1](
                        layer_output[t], h_backward
                    )
                    layer_h_back.append(h_backward)

                # Combine forward and backward outputs
                layer_h_back.reverse()
                layer_forward = Tensor(np.stack([h.data for h in layer_h_list]))
                layer_backward = Tensor(np.stack([h.data for h in layer_h_back]))
                layer_output = Tensor(
                    np.concatenate([layer_forward.data, layer_backward.data], axis=-1)
                )
            else:
                layer_output = Tensor(np.stack([h.data for h in layer_h_list]))

            # Apply dropout except for last layer
            if layer < self.num_layers - 1 and self.training and self.dropout > 0:
                mask = (np.random.rand(*layer_output.shape) > self.dropout).astype(
                    np.float64
                )
                layer_output = Tensor(layer_output.data * mask / (1 - self.dropout))

            new_h.append(h_forward.data)
            if self.bidirectional:
                new_h.append(h_backward.data)

        # Stack hidden states
        h_n = Tensor(np.stack(new_h))

        # Restore batch_first if needed
        if self.batch_first:
            layer_output = Tensor(np.swapaxes(layer_output.data, 0, 1))

        return layer_output, h_n


class LSTMCell(Module):
    """
    A single LSTM cell.

    Args:
        input_size: Number of features in input
        hidden_size: Number of features in hidden state
        bias: If False, doesn't learn bias weights
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Create weight matrices for the four gates
        self.weight_ih = Tensor(
            np.random.randn(4 * hidden_size, input_size) / np.sqrt(input_size),
            requires_grad=True,
        )
        self.weight_hh = Tensor(
            np.random.randn(4 * hidden_size, hidden_size) / np.sqrt(hidden_size),
            requires_grad=True,
        )

        if bias:
            self.bias_ih = Tensor(np.zeros(4 * hidden_size), requires_grad=True)
            self.bias_hh = Tensor(np.zeros(4 * hidden_size), requires_grad=True)
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

    def forward(
        self, x: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass of LSTM cell."""
        if hx is None:
            hx = (
                Tensor(np.zeros((x.shape[0], self.hidden_size))),
                Tensor(np.zeros((x.shape[0], self.hidden_size))),
            )

        h, c = hx

        # Calculate gates
        gates = x @ self.weight_ih.t()
        if self.bias_ih is not None:
            gates = gates + self.bias_ih

        gates = gates + (h @ self.weight_hh.t())
        if self.bias_hh is not None:
            gates = gates + self.bias_hh

        # Split into individual gates
        gates.shape[-1] // 4
        i, f, g, o = np.split(gates.data, 4, axis=-1)

        # Apply gate activations
        i = 1 / (1 + np.exp(-i))  # input gate
        f = 1 / (1 + np.exp(-f))  # forget gate
        g = np.tanh(g)  # cell gate
        o = 1 / (1 + np.exp(-o))  # output gate

        # Update cell state and hidden state
        c_next = f * c.data + i * g
        h_next = o * np.tanh(c_next)

        return Tensor(h_next), Tensor(c_next)


class GRUCell(Module):
    """
    A single GRU cell.

    Args:
        input_size: Number of features in input
        hidden_size: Number of features in hidden state
        bias: If False, doesn't learn bias weights
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Create weight matrices for the three gates (reset, update, new)
        self.weight_ih = Tensor(
            np.random.randn(3 * hidden_size, input_size) / np.sqrt(input_size),
            requires_grad=True,
        )
        self.weight_hh = Tensor(
            np.random.randn(3 * hidden_size, hidden_size) / np.sqrt(hidden_size),
            requires_grad=True,
        )

        if bias:
            self.bias_ih = Tensor(np.zeros(3 * hidden_size), requires_grad=True)
            self.bias_hh = Tensor(np.zeros(3 * hidden_size), requires_grad=True)
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

    def forward(self, x: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of GRU cell.

        Args:
            x: Input tensor of shape (batch, input_size)
            hx: Hidden state tensor of shape (batch, hidden_size)

        Returns:
            New hidden state tensor of shape (batch, hidden_size)
        """
        if hx is None:
            hx = Tensor(np.zeros((x.shape[0], self.hidden_size)))

        # Calculate gates
        gates_x = x @ self.weight_ih.t()
        if self.bias_ih is not None:
            gates_x = gates_x + self.bias_ih

        gates_h = hx @ self.weight_hh.t()
        if self.bias_hh is not None:
            gates_h = gates_h + self.bias_hh

        # Split into individual gates
        gates_x.shape[-1] // 3
        r_x, z_x, n_x = np.split(gates_x.data, 3, axis=-1)  # reset, update, new
        r_h, z_h, n_h = np.split(gates_h.data, 3, axis=-1)

        # Compute gate values
        r = 1 / (1 + np.exp(-(r_x + r_h)))  # reset gate
        z = 1 / (1 + np.exp(-(z_x + z_h)))  # update gate
        n = np.tanh(n_x + r * n_h)  # new gate

        # Compute new hidden state
        h_next = (1 - z) * hx.data + z * n

        return Tensor(h_next)

    def extra_repr(self) -> str:
        """Returns a string with extra representation information."""
        return f"input_size={self.input_size}, hidden_size={self.hidden_size}"
