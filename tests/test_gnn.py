# tests_gnn.py
import pytest
import numpy as np

# Import our DLpy abstractions and the GNN modules.
from DLpy.core import Tensor
from DLpy.nn.architectures.gnn.message_passing import MessagePassing, DefaultUpdate
from DLpy.nn.architectures.gnn.convolution import GCNConv, GATConv, GraphSAGEConv, EdgeConv
from DLpy.nn.architectures.gnn.pooling import TopKPooling, SAGPooling, EdgePooling, DiffPooling

##############################
# Helper: Dummy MessagePassing
##############################
class DummyMP(MessagePassing):
    """A dummy message-passing layer that returns the neighbor features as messages."""
    def message(self, x_i: Tensor, x_j: Tensor, edge_attr=None) -> Tensor:
        return x_j  # trivial message function


class TestGNN:

    ##############################
    # Tests for MessagePassing
    ##############################
    def test_message_passing_propagate(self):
        # Create a simple graph with 4 nodes and 3 edges.
        # Let the node features be 2-dimensional.
        x_data = np.array([[1, 1],
                        [2, 2],
                        [3, 3],
                        [4, 4]], dtype=np.float64)
        x = Tensor(x_data)
        # Define edges: from node j to node i.
        # Let edge_index = ([source indices], [target indices])
        # For example, edge from 0->1, 1->2, 2->3.
        edge_index = ([0, 1, 2], [1, 2, 3])
        
        # Using the DummyMP without an update_module should return the aggregated messages.
        dummy_mp = DummyMP()  # update_module is None => update returns aggregated message.
        out = dummy_mp.propagate(edge_index, x)
        # Expected aggregation: node 0: no incoming message -> 0; 
        # node 1: receives from node 0: [1,1]; node 2: [2,2]; node 3: [3,3].
        expected = np.array([[0, 0],
                            [1, 1],
                            [2, 2],
                            [3, 3]], dtype=np.float64)
        np.testing.assert_allclose(out.data, expected)

    def test_message_passing_with_update(self):
        # Now use an update module that concatenates the original node feature with the aggregated message.
        # For simplicity, use DefaultUpdate with in_channels=2 and out_channels=2.
        update_module = DefaultUpdate(in_channels=2, out_channels=2)
        mp = DummyMP(update_module=update_module)
        x_data = np.array([[1, 2],
                        [3, 4],
                        [5, 6]], dtype=np.float64)
        x = Tensor(x_data)
        # Create edges: 0->1, 1->2.
        edge_index = ([0, 1], [1, 2])
        out = mp.propagate(edge_index, x)
        # The aggregated messages (with default sum aggregation) will be:
        # node 0: no incoming message, aggregated = 0.
        # node 1: aggregated = x[0] = [1,2].
        # node 2: aggregated = x[1] = [3,4].
        # Then update module concatenates original x with aggregated message:
        # For node 0: [1,2]||[0,0] -> Linear and ReLU (nonlinear, so we only test shape).
        # For node 1: [3,4]||[1,2] -> shape (4,) mapped to (2,).
        # For node 2: [5,6]||[3,4] -> shape (4,) mapped to (2,).
        # Check output shape is (3,2)
        assert out.data.shape == (3, 2)

    ##############################
    # Tests for Convolution Layers
    ##############################
    def test_gcnconv_forward(self):
        # Test the GCNConv layer.
        # Create a graph with 5 nodes and edges:
        # 0->1, 1->2, 2->3, 3->4.
        x_data = np.random.randn(5, 4)  # 5 nodes, 4 features each.
        x = Tensor(x_data)
        edge_index = ([0, 1, 2, 3], [1, 2, 3, 4])
        gcn = GCNConv(in_channels=4, out_channels=8)
        out = gcn.forward(x, edge_index)
        # Output should have shape (5, 8)
        assert out.data.shape == (5, 8)

    def test_gatconv_forward(self):
        # Test the GATConv layer.
        # Create a graph with 6 nodes.
        x_data = np.random.randn(6, 4)
        x = Tensor(x_data)
        # Create edges arbitrarily.
        edge_index = ([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
        # Instantiate GATConv with 2 heads, each producing 3 features.
        gat = GATConv(in_channels=4, out_channels=3, heads=2)
        out = gat.forward(x, edge_index)
        # Output shape should be (6, heads * out_channels) = (6, 6)
        assert out.data.shape == (6, 6)

    def test_graphsageconv_forward(self):
        # Test the GraphSAGEConv layer.
        x_data = np.random.randn(7, 5)
        x = Tensor(x_data)
        edge_index = ([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
        graphsage = GraphSAGEConv(in_channels=5, out_channels=5)
        out = graphsage.forward(x, edge_index)
        # Using the default update (concatenation then Linear+ReLU), output shape should be (7, 5)
        assert out.data.shape == (7, 5)

    def test_edgeconv_forward(self):
        # Test the EdgeConv layer.
        x_data = np.random.randn(8, 3)
        x = Tensor(x_data)
        # Create a chain graph: 0->1, 1->2, ..., 6->7.
        edge_index = (list(range(7)), list(range(1, 8)))
        edge_conv = EdgeConv(in_channels=3, out_channels=4)
        out = edge_conv.forward(x, edge_index)
        # Output should have shape (8, 4)
        assert out.data.shape == (8, 4)

    ##############################
    # Tests for Pooling Layers
    ##############################
    def test_topk_pooling(self):
        # Test TopKPooling.
        x_data = np.arange(20, dtype=np.float64).reshape(10, 2)  # 10 nodes, 2 features.
        x = Tensor(x_data)
        # Create a simple edge index (for instance, a circular graph).
        edge_index = (list(range(10)), list(range(1, 10)) + [0])
        # Instantiate TopKPooling with ratio 0.6 -> keep 6 nodes.
        topk = TopKPooling(in_channels=2, ratio=0.6)
        pooled, new_edge_index = topk.forward(x, edge_index)
        # Check pooled features shape: (6, 2)
        assert pooled.data.shape == (6, 2)
        # Check new_edge_index: both lists should have same length (could be zero or more).
        assert isinstance(new_edge_index, tuple)
        assert len(new_edge_index) == 2

    def test_sag_pooling(self):
        # Test SAGPooling.
        x_data = np.random.randn(12, 4)  # 12 nodes, 4 features.
        x = Tensor(x_data)
        # Create arbitrary edge index.
        edge_index = (list(np.random.randint(0, 12, size=20)), list(np.random.randint(0, 12, size=20)))
        sag = SAGPooling(in_channels=4, ratio=0.5)
        pooled, new_edge_index = sag.forward(x, edge_index)
        # With ratio 0.5 and 12 nodes, we expect 6 nodes.
        assert pooled.data.shape == (6, 4)
        assert isinstance(new_edge_index, tuple)
        assert len(new_edge_index) == 2

    def test_diff_pooling(self):
        # Test DiffPooling.
        x_data = np.random.randn(15, 6)  # 15 nodes, 6 features.
        x = Tensor(x_data)
        # Edge index: create a random set of edges.
        edge_index = (list(np.random.randint(0, 15, size=30)), list(np.random.randint(0, 15, size=30)))
        # DiffPooling: pool nodes into 4 clusters.
        diff_pool = DiffPooling(in_channels=6, assign_dim=4)
        pooled, new_edge_index = diff_pool.forward(x, edge_index)
        # Pooled features shape should be (assign_dim, in_channels) i.e., (4,6)
        assert pooled.data.shape == (4, 6)
        # New edge index for clusters: since we construct a complete graph among clusters,
        # new_edge_index should contain edges between clusters (except self-edges).
        new_src, new_tgt = new_edge_index
        # Check that all indices are in the range [0, 3].
        assert np.all(np.array(new_src) < 4)
        assert np.all(np.array(new_tgt) < 4)
    
    def test_edge_pooling(self):
        """Test EdgePooling layer."""
        # Create test data: 8 nodes with 3 features each
        x_data = np.random.randn(8, 3)
        x = Tensor(x_data)
        
        # Create a simple chain of edges: 0->1, 1->2, ..., 6->7
        edge_index = (list(range(7)), list(range(1, 8)))
        
        # Initialize EdgePooling
        edge_pool = EdgePooling(in_channels=3)
        
        # Apply pooling
        pooled_x, new_edge_index = edge_pool.forward(x, edge_index)
        
        # Verify output shapes
        assert isinstance(pooled_x, Tensor)
        assert pooled_x.data.shape[1] == 3  # Should preserve feature dimension
        assert pooled_x.data.shape[0] <= x.data.shape[0]  # Should have fewer or equal nodes
        
        # Verify edge index
        assert isinstance(new_edge_index, tuple)
        assert len(new_edge_index) == 2
        assert len(new_edge_index[0]) == len(new_edge_index[1])  # Source and target lists should match
        
        # Verify all indices in new edge_index are valid
        if len(new_edge_index[0]) > 0:  # If there are any edges
            assert max(new_edge_index[0]) < pooled_x.data.shape[0]
            assert max(new_edge_index[1]) < pooled_x.data.shape[0]
