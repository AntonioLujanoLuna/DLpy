import pytest
import numpy as np
from DLpy.core import Tensor
from DLpy.nn import LSTM, GRU, LSTMCell, GRUCell

class TestLSTMCell:
    """Tests for the LSTM Cell implementation."""
    
    def test_initialization(self):
        """Test LSTM cell initialization."""
        input_size = 10
        hidden_size = 20
        cell = LSTMCell(input_size, hidden_size)
        
        # Check parameter shapes
        assert cell.weight_ih.shape == (4 * hidden_size, input_size)
        assert cell.weight_hh.shape == (4 * hidden_size, hidden_size)
        assert cell.bias_ih.shape == (4 * hidden_size,)
        assert cell.bias_hh.shape == (4 * hidden_size,)
        
        # Test initialization without bias
        cell_no_bias = LSTMCell(input_size, hidden_size, has_bias=False)
        assert cell_no_bias.bias_ih is None
        assert cell_no_bias.bias_hh is None
        
    def test_forward(self):
        """Test forward pass of LSTM cell."""
        batch_size = 3
        input_size = 4
        hidden_size = 5
        cell = LSTMCell(input_size, hidden_size)
        
        # Test with provided hidden state
        x = Tensor(np.random.randn(batch_size, input_size))
        h = Tensor(np.random.randn(batch_size, hidden_size))
        c = Tensor(np.random.randn(batch_size, hidden_size))
        
        h_next, c_next = cell(x, (h, c))
        
        assert h_next.shape == (batch_size, hidden_size)
        assert c_next.shape == (batch_size, hidden_size)
        
        # Test with no hidden state
        h_next, c_next = cell(x)
        assert h_next.shape == (batch_size, hidden_size)
        assert c_next.shape == (batch_size, hidden_size)
        
    def test_gradient_flow(self):
        """Test gradient computation in LSTM cell."""
        cell = LSTMCell(3, 4)
        x = Tensor(np.random.randn(2, 3), requires_grad=True)
        h = Tensor(np.random.randn(2, 4), requires_grad=True)
        c = Tensor(np.random.randn(2, 4), requires_grad=True)
        
        h_next, c_next = cell(x, (h, c))
        loss = h_next.sum() + c_next.sum()
        loss.backward()
        
        assert x.grad is not None
        assert h.grad is not None
        assert c.grad is not None
        assert cell.weight_ih.grad is not None
        assert cell.weight_hh.grad is not None

class TestGRUCell:
    """Tests for the GRU Cell implementation."""
    
    def test_initialization(self):
        """Test GRU cell initialization."""
        input_size = 10
        hidden_size = 20
        cell = GRUCell(input_size, hidden_size)
        
        # Check parameter shapes
        assert cell.weight_ih.shape == (3 * hidden_size, input_size)
        assert cell.weight_hh.shape == (3 * hidden_size, hidden_size)
        assert cell.bias_ih.shape == (3 * hidden_size,)
        assert cell.bias_hh.shape == (3 * hidden_size,)
        
        # Test initialization without bias
        cell_no_bias = GRUCell(input_size, hidden_size, has_bias=False)
        assert cell_no_bias.bias_ih is None
        assert cell_no_bias.bias_hh is None
        
    def test_forward(self):
        """Test forward pass of GRU cell."""
        batch_size = 3
        input_size = 4
        hidden_size = 5
        cell = GRUCell(input_size, hidden_size)
        
        # Test with provided hidden state
        x = Tensor(np.random.randn(batch_size, input_size))
        h = Tensor(np.random.randn(batch_size, hidden_size))
        
        h_next = cell(x, h)
        assert h_next.shape == (batch_size, hidden_size)
        
        # Test with no hidden state
        h_next = cell(x)
        assert h_next.shape == (batch_size, hidden_size)
        
    def test_gradient_flow(self):
        """Test gradient computation in GRU cell."""
        cell = GRUCell(3, 4)
        x = Tensor(np.random.randn(2, 3), requires_grad=True)
        h = Tensor(np.random.randn(2, 4), requires_grad=True)
        
        h_next = cell(x, h)
        loss = h_next.sum()
        loss.backward()
        
        assert x.grad is not None
        assert h.grad is not None
        assert cell.weight_ih.grad is not None
        assert cell.weight_hh.grad is not None

class TestLSTM:
    """Tests for the LSTM layer."""
    
    def test_initialization(self):
        """Test LSTM layer initialization."""
        lstm = LSTM(10, 20, num_layers=2, bidirectional=True)
        assert len(lstm.cell_list) == 4  # 2 layers * 2 directions
        
        lstm_unidirectional = LSTM(10, 20, num_layers=2)
        assert len(lstm_unidirectional.cell_list) == 2  # 2 layers * 1 direction
        
    def test_forward_unidirectional(self):
        """Test forward pass of unidirectional LSTM."""
        batch_size = 3
        seq_length = 4
        input_size = 5
        hidden_size = 6
        
        lstm = LSTM(input_size, hidden_size, num_layers=2)
        x = Tensor(np.random.randn(seq_length, batch_size, input_size))
        
        output, (h_n, c_n) = lstm(x)
        
        assert output.shape == (seq_length, batch_size, hidden_size)
        assert h_n.shape == (2, batch_size, hidden_size)  # num_layers * batch * hidden
        assert c_n.shape == (2, batch_size, hidden_size)
        
    def test_forward_bidirectional(self):
        """Test forward pass of bidirectional LSTM."""
        batch_size = 3
        seq_length = 4
        input_size = 5
        hidden_size = 6
        
        lstm = LSTM(input_size, hidden_size, num_layers=2, bidirectional=True)
        x = Tensor(np.random.randn(seq_length, batch_size, input_size))
        
        output, (h_n, c_n) = lstm(x)
        
        assert output.shape == (seq_length, batch_size, 2 * hidden_size)
        assert h_n.shape == (4, batch_size, hidden_size)  # num_layers * 2 * batch * hidden
        assert c_n.shape == (4, batch_size, hidden_size)
        
    def test_batch_first(self):
        """Test batch_first option."""
        lstm = LSTM(5, 6, batch_first=True)
        x = Tensor(np.random.randn(3, 4, 5))  # batch * seq * input
        
        output, (h_n, c_n) = lstm(x)
        assert output.shape == (3, 4, 6)  # batch * seq * hidden
        
    def test_dropout(self):
        """Test dropout between layers."""
        lstm = LSTM(5, 6, num_layers=2, dropout=0.5)
        x = Tensor(np.random.randn(4, 3, 5))
        
        lstm.train()
        output1, _ = lstm(x)
        output2, _ = lstm(x)
        
        # Outputs should be different due to dropout
        assert not np.allclose(output1.data, output2.data)
        
        # In eval mode, outputs should be the same
        lstm.eval()
        output1, _ = lstm(x)
        output2, _ = lstm(x)
        assert np.allclose(output1.data, output2.data)

class TestGRU:
    """Tests for the GRU layer."""
    
    def test_initialization(self):
        """Test GRU layer initialization."""
        gru = GRU(10, 20, num_layers=2, bidirectional=True)
        assert len(gru.cell_list) == 4  # 2 layers * 2 directions
        
        gru_unidirectional = GRU(10, 20, num_layers=2)
        assert len(gru_unidirectional.cell_list) == 2  # 2 layers * 1 direction
        
    def test_forward_unidirectional(self):
        """Test forward pass of unidirectional GRU."""
        batch_size = 3
        seq_length = 4
        input_size = 5
        hidden_size = 6
        
        gru = GRU(input_size, hidden_size, num_layers=2)
        x = Tensor(np.random.randn(seq_length, batch_size, input_size))
        
        output, h_n = gru(x)
        
        assert output.shape == (seq_length, batch_size, hidden_size)
        assert h_n.shape == (2, batch_size, hidden_size)  # num_layers * batch * hidden
        
    def test_forward_bidirectional(self):
        """Test forward pass of bidirectional GRU."""
        batch_size = 3
        seq_length = 4
        input_size = 5
        hidden_size = 6
        
        gru = GRU(input_size, hidden_size, num_layers=2, bidirectional=True)
        x = Tensor(np.random.randn(seq_length, batch_size, input_size))
        
        output, h_n = gru(x)
        
        assert output.shape == (seq_length, batch_size, 2 * hidden_size)
        assert h_n.shape == (4, batch_size, hidden_size)  # num_layers * 2 * batch * hidden
        
    def test_batch_first(self):
        """Test batch_first option."""
        gru = GRU(5, 6, batch_first=True)
        x = Tensor(np.random.randn(3, 4, 5))  # batch * seq * input
        
        output, h_n = gru(x)
        assert output.shape == (3, 4, 6)  # batch * seq * hidden
        
    def test_dropout(self):
        """Test dropout between layers."""
        gru = GRU(5, 6, num_layers=2, dropout=0.5)
        x = Tensor(np.random.randn(4, 3, 5))
        
        gru.train()
        output1, _ = gru(x)
        output2, _ = gru(x)
        
        # Outputs should be different due to dropout
        assert not np.allclose(output1.data, output2.data)
        
        # In eval mode, outputs should be the same
        gru.eval()
        output1, _ = gru(x)
        output2, _ = gru(x)
        assert np.allclose(output1.data, output2.data)

class TestRNNEdgeCases:
    """Tests for edge cases in RNN layers."""
    
    def test_zero_length_sequence(self):
        """Test handling of zero-length sequences."""
        lstm = LSTM(5, 6)
        gru = GRU(5, 6)
        
        x = Tensor(np.random.randn(0, 3, 5))  # seq_len = 0
        
        with pytest.raises(ValueError):
            lstm(x)
        with pytest.raises(ValueError):
            gru(x)
            
    def test_invalid_hidden_state(self):
        """Test handling of invalid hidden state dimensions."""
        lstm = LSTM(5, 6, num_layers=2)
        gru = GRU(5, 6, num_layers=2)
        
        x = Tensor(np.random.randn(4, 3, 5))
        h_wrong = Tensor(np.random.randn(3, 3, 6))  # Wrong num_layers
        
        with pytest.raises(ValueError):
            lstm(x, (h_wrong, h_wrong))
        with pytest.raises(ValueError):
            gru(x, h_wrong)
            
    def test_invalid_dropout(self):
        """Test invalid dropout values."""
        with pytest.raises(ValueError):
            LSTM(5, 6, dropout=-0.1)
        with pytest.raises(ValueError):
            LSTM(5, 6, dropout=1.1)
        with pytest.raises(ValueError):
            GRU(5, 6, dropout=-0.1)
        with pytest.raises(ValueError):
            GRU(5, 6, dropout=1.1)