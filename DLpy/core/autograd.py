from typing import Any, Dict, List, Optional, Set

import numpy as np
from numpy.typing import NDArray

from .tensor import Tensor


class Edge:
    """
    Represents a directed edge in the computational graph.

    Each edge connects a source node (input tensor) to a destination node
    (output tensor) and stores gradient information for that connection.
    """

    def __init__(self, src: "Node", dst: "Node"):
        self.src = src
        self.dst = dst
        self.grad: Optional[NDArray[Any]] = None


class Node:
    """
    Represents a node in the computational graph.

    Each node corresponds to an operation in the computation and maintains
    connections to its inputs and outputs through edges.
    """

    def __init__(self, tensor: "Tensor"):
        self.tensor = tensor
        self.in_edges: List[Edge] = []
        self.out_edges: List[Edge] = []
        self._backward_fn = tensor._backward_fn


class AutogradEngine:
    """
    Engine for managing automatic differentiation computations.

    This class handles the creation and execution of the computational graph,
    manages gradient computation and accumulation, and provides utilities for
    graph manipulation and visualization.
    """

    def __init__(self) -> None:
        self._nodes: Dict[int, Node] = {}
        self._edges: Set[Edge] = set()
        self._currently_computing_gradients = False

    def register_tensor(self, tensor: "Tensor") -> None:
        """
        Registers a tensor with the autograd engine.

        Args:
            tensor: Tensor to register
        """
        if id(tensor) not in self._nodes:
            self._nodes[id(tensor)] = Node(tensor)

    def add_edge(self, src: "Tensor", dst: "Tensor") -> None:
        """
        Adds a directed edge between two tensors in the computational graph.

        Args:
            src: Source tensor
            dst: Destination tensor
        """
        src_node = self._nodes[id(src)]
        dst_node = self._nodes[id(dst)]

        edge = Edge(src_node, dst_node)
        src_node.out_edges.append(edge)
        dst_node.in_edges.append(edge)
        self._edges.add(edge)

    def backward(
        self, tensor: "Tensor", gradient: Optional[NDArray[Any]] = None
    ) -> None:
        """Executes backward pass starting from the given tensor."""
        if self._currently_computing_gradients:
            raise RuntimeError("Nested gradient computation detected")

        self._currently_computing_gradients = True
        try:
            # Initialize grad_dict as a regular dictionary
            grad_dict: Dict[int, NDArray[Any]] = {}

            # If no gradient is provided, assume it's 1 (for scalar outputs)
            if gradient is None:
                if tensor.data.shape == ():
                    grad_dict[id(tensor)] = np.array(1.0)
                else:
                    grad_dict[id(tensor)] = np.ones_like(tensor.data)
            else:
                grad_dict[id(tensor)] = gradient

            # Perform topological sort
            sorted_nodes = self._topological_sort(tensor)

            # Traverse nodes in reverse topological order
            for node in reversed(sorted_nodes):
                node_id = id(node.tensor)
                if node_id not in grad_dict or not node.tensor.requires_grad:
                    continue  # No gradient to propagate

                current_grad = grad_dict[node_id]

                if node.tensor._backward_fn is not None:
                    node.tensor._backward_fn(current_grad, grad_dict)

                # Accumulate gradients for leaf nodes
                if not node.in_edges and node.tensor.requires_grad:
                    if node.tensor.grad is None:
                        node.tensor.grad = current_grad
                    else:
                        try:
                            node.tensor.grad += current_grad
                        except ValueError:
                            # If shapes don't match, reshape current_grad
                            node.tensor.grad += current_grad.reshape(
                                node.tensor.grad.shape
                            )
        finally:
            self._currently_computing_gradients = False

    def _topological_sort(self, start_tensor: "Tensor") -> List[Node]:
        """
        Performs topological sort on the computation graph.

        Args:
            start_tensor: Tensor to start the sort from

        Returns:
            List of nodes in topological order

        Raises:
            RuntimeError: If graph contains cycles
        """
        result: List[Node] = []
        visited: Set[Node] = set()
        temp_visited: Set[Node] = set()

        def visit(node: Node) -> None:
            if node in temp_visited:
                raise RuntimeError("Cycle detected in computation graph")

            if node not in visited:
                temp_visited.add(node)
                for edge in node.in_edges:
                    visit(edge.src)
                temp_visited.remove(node)
                visited.add(node)
                result.append(node)

        visit(self._nodes[id(start_tensor)])
        return result

    def clear(self) -> None:
        """Clears the computational graph."""
        self._nodes.clear()
        self._edges.clear()

    def validate_graph(self) -> List[str]:
        """
        Validates the computational graph structure.
        """
        warnings: List[str] = []

        # If no nodes in graph
        if not self._nodes:
            return warnings

        # Step 1: Find all nodes that are part of computations
        active_nodes = set()
        output_nodes = []
        for node in self._nodes.values():
            if not node.out_edges:  # Output node
                output_nodes.append(node)
            if node.in_edges or node.out_edges:  # Node is part of a computation
                active_nodes.add(node)

        # Step 2: Find all connected nodes starting from outputs
        connected_nodes = set()
        for output_node in output_nodes:
            stack = [output_node]
            while stack:
                curr = stack.pop()
                connected_nodes.add(curr)
                for edge in curr.in_edges:
                    if edge.src not in connected_nodes:
                        stack.append(edge.src)

        # Step 3: Find nodes not connected to outputs
        all_nodes = set(self._nodes.values())
        unconnected_nodes = all_nodes - connected_nodes

        # Step 4: Find completely isolated nodes
        isolated_nodes = all_nodes - active_nodes

        # Add appropriate warnings
        if unconnected_nodes:
            warnings.append(
                f"Found {len(unconnected_nodes)} nodes not connected to any output"
            )

        if isolated_nodes:
            warnings.append(f"Found {len(isolated_nodes)} isolated nodes")

        # Check gradient shapes
        for edge in self._edges:
            if edge.grad is not None:
                src_shape = edge.src.tensor.shape
                grad_shape = edge.grad.shape
                if src_shape != grad_shape:
                    warnings.append(
                        f"Gradient shape mismatch: "
                        f"grad shape {grad_shape} "
                        f"vs tensor shape {src_shape}"
                    )

        return warnings


# Global autograd engine instance
_autograd_engine = AutogradEngine()


def get_autograd_engine() -> AutogradEngine:
    """Returns the global autograd engine instance."""
    return _autograd_engine
