import pytest
import numpy as np
from DLpy.core import (
    Tensor,
    get_autograd_engine
)
from DLpy.ops import Add, Multiply
from DLpy.core.autograd import Edge

class TestAutogradEngine:
    """Tests for the autograd engine's core functionality."""
    
    def setup_method(self):
        """Setup method run before each test."""
        self.engine = get_autograd_engine()
        self.engine.clear()

    def test_register_tensor(self):
        """Test registering a tensor with the autograd engine."""
        tensor = Tensor([1.0], requires_grad=True)
        self.engine.register_tensor(tensor)
        assert id(tensor) in self.engine._nodes

    def test_add_edge(self):
        """Test adding edges between tensors in the computational graph."""
        t1 = Tensor([1.0], requires_grad=True)
        t2 = Tensor([2.0], requires_grad=True)
        
        self.engine.register_tensor(t1)
        self.engine.register_tensor(t2)
        self.engine.add_edge(t1, t2)
        
        node1 = self.engine._nodes[id(t1)]
        node2 = self.engine._nodes[id(t2)]
        
        assert len(node1.out_edges) == 1
        assert len(node2.in_edges) == 1
        assert node1.out_edges[0].dst == node2

class TestGradientComputation:
    """Tests for gradient computation in different graph structures."""
    
    def setup_method(self):
        self.engine = get_autograd_engine()
        self.engine.clear()

    def test_linear_graph(self):
        """Test gradient computation in a linear graph."""
        # Create a simple linear computation: z = 2x + y
        x = Tensor([2.0], requires_grad=True)
        y = Tensor([3.0], requires_grad=True)
        z = Add.apply(x, y)
        self.engine.backward(z, np.array([1.0]))
        
        # Check gradients
        assert np.allclose(x.grad, [1.0])
        assert np.allclose(y.grad, [1.0])

    def test_branching_graph(self):
        """Test gradient computation in a graph with multiple paths."""

        # The test creates a computation graph shaped like:
        #     x
        #   /   \  
        #  y1   y2
        #   \   /
        #     z

        # This tests whether gradients properly flow and accumulate through 
        # multiple paths back to the same input.
        x = Tensor([2.0], requires_grad=True)
        y1 = Multiply.apply(x, Tensor([2.0]))  # y1 = 2x
        y2 = Multiply.apply(x, Tensor([3.0]))  # y2 = 3x
        z = Add.apply(y1, y2)  # z = y1 + y2 = 5x

        self.engine.backward(z, np.array([1.0]))
        # Gradient should be 5.0 (sum of both paths: 2 + 3)
        assert np.allclose(x.grad, [5.0])

    def test_diamond_graph(self):
        """Test gradient computation in a diamond-shaped graph."""
        # Create a diamond computation:
        #     x
        #    / \
        #   h1  h2
        #    \ /
        #     y
        x = Tensor([1.0], requires_grad=True)
        w1 = Tensor([2.0], requires_grad=True)
        w2 = Tensor([3.0], requires_grad=True)
        
        h1 = Multiply.apply(x, w1)
        h2 = Multiply.apply(x, w2)
        y = Add.apply(h1, h2)
        
        self.engine.backward(y, np.array([1.0]))
        
        # x's gradient should include effects from both paths
        assert np.allclose(x.grad, [5.0])  # 2 + 3
        assert np.allclose(w1.grad, [1.0])
        assert np.allclose(w2.grad, [1.0])

class TestGradientAccumulation:
    """Tests for correct gradient accumulation behavior."""

    def setup_method(self):
        self.engine = get_autograd_engine()
        self.engine.clear()

    def test_reused_variable(self):
        """Test gradient accumulation when a variable is used multiple times."""
        x = Tensor([2.0], requires_grad=True)
        
        # Use x in three separate computations
        y1 = Multiply.apply(x, Tensor([2.0]))
        y2 = Multiply.apply(x, Tensor([3.0]))
        y3 = Multiply.apply(x, Tensor([4.0]))
        
        # Backward on all three outputs
        self.engine.backward(y1, np.array([1.0]))
        self.engine.backward(y2, np.array([1.0]))
        self.engine.backward(y3, np.array([1.0]))
        
        # Gradient should accumulate: 2 + 3 + 4 = 9
        assert np.allclose(x.grad, [9.0])

    def test_shared_structure(self):
        """Test gradient computation with shared subgraphs."""
        # Create a computation where the same subgraph is used multiple times
        x = Tensor([1.0], requires_grad=True)
        y = Tensor([2.0], requires_grad=True)
        
        # Shared computation
        shared = Multiply.apply(x, y)
        
        # Use shared result multiple times
        out1 = Multiply.apply(shared, Tensor([2.0]))
        out2 = Multiply.apply(shared, Tensor([3.0]))
        
        # Final sum
        final = Add.apply(out1, out2)
        
        self.engine.backward(final, np.array([1.0]))
        
        # Verify gradients include effects from all paths
        assert x.grad is not None
        assert y.grad is not None

class TestAdvancedAutogradFeatures:
    """Tests for advanced AutogradEngine features and edge cases"""
    
    def setup_method(self):
        self.engine = get_autograd_engine()
        self.engine.clear()

    def test_validate_graph(self):
        """Test graph validation functionality"""
        # Create a disconnected subgraph
        x = Tensor([1.0], requires_grad=True)
        y = Tensor([2.0], requires_grad=True)
        _ = Add.apply(x, y)
        
        # Add an isolated node
        z = Tensor([3.0], requires_grad=True)
        self.engine.register_tensor(z)
        
        warnings = self.engine.validate_graph()
        assert len(warnings) > 0
        assert "isolated nodes" in warnings[0]  

    def test_nested_gradient_computation(self):
        """Test detection of nested gradient computations"""
        x = Tensor([1.0], requires_grad=True)
        y = Add.apply(x, Tensor([2.0]))
        
        # Simulate nested gradient computation
        self.engine._currently_computing_gradients = True
        with pytest.raises(RuntimeError, match="Nested gradient computation detected"):
            self.engine.backward(y)
        self.engine._currently_computing_gradients = False

    def test_cyclic_graph_detection(self):
        """Test detection of cycles in computational graph"""
        x = Tensor([1.0], requires_grad=True)
        y = Tensor([2.0], requires_grad=True)
        
        # Manually create a cycle in the graph
        node_x = self.engine._nodes[id(x)]
        node_y = self.engine._nodes[id(y)]
        
        edge1 = Edge(node_x, node_y)
        edge2 = Edge(node_y, node_x)
        
        node_x.out_edges.append(edge1)
        node_y.in_edges.append(edge1)
        node_y.out_edges.append(edge2)
        node_x.in_edges.append(edge2)
        
        with pytest.raises(RuntimeError, match="Cycle detected in computation graph"):
            self.engine.backward(x)

    def test_gradient_shape_mismatch(self):
        """Test detection of gradient shape mismatches"""
        x = Tensor([[1.0]], requires_grad=True)  # Shape (1, 1)
        y = Tensor([2.0], requires_grad=True)    # Shape (1,)
        
        # Create edge with obviously wrong shape
        node_x = self.engine._nodes[id(x)]
        node_y = self.engine._nodes[id(y)]
        
        edge = Edge(node_x, node_y)
        edge.grad = np.array([[1.0, 2.0]])  # Wrong shape (1, 2)
        
        # Add the edge to both nodes and the engine
        node_x.out_edges.append(edge)
        node_y.in_edges.append(edge)
        self.engine._edges.add(edge)
        
        warnings = self.engine.validate_graph()
        assert any("shape mismatch" in w for w in warnings), "Should detect shape mismatch"