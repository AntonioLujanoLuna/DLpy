# tests/test_decomposition.py

import pytest
import numpy as np
from DLpy.core import Tensor
from DLpy.utils import TensorDecomposition

class TestTensorDecomposition:
    @pytest.fixture
    def simple_tensor(self):
        """Create a simple 2x3x4 tensor with known structure for testing."""
        data = np.arange(24).reshape(2, 3, 4)
        return Tensor(data)

    @pytest.fixture
    def rank_one_tensor(self):
        """Create a rank-1 tensor (outer product of vectors) for testing."""
        a = np.array([1, 2])
        b = np.array([3, 4, 5])
        c = np.array([6, 7, 8, 9])
        # Create rank-1 tensor through outer products
        tensor = np.outer(a, b).reshape(2, 3, 1) * c.reshape(1, 1, 4)
        return Tensor(tensor)

    def test_initialization(self, simple_tensor):
        """Test proper initialization of TensorDecomposition class."""
        decomp = TensorDecomposition(simple_tensor)
        assert decomp.shape == (2, 3, 4)
        assert isinstance(decomp.tensor, Tensor)

    def test_cp_decomposition_basic(self, simple_tensor):
        """Test basic functionality of CP decomposition."""
        decomp = TensorDecomposition(simple_tensor)
        factors = decomp.cp_decomposition(
                rank=2,
                max_iter=200,
                tol=1e-7,
                n_restarts=3,
                init='hosvd',
                stall_patience=5
            )
        
        # Check output structure
        assert len(factors) == 3  # Should have one factor per mode
        assert all(isinstance(f, Tensor) for f in factors)
        assert factors[0].shape == (2, 2)  # First mode factors
        assert factors[1].shape == (3, 2)  # Second mode factors
        assert factors[2].shape == (4, 2)  # Third mode factors

    def test_cp_decomposition_rank_one(self, rank_one_tensor):
        """Test CP decomposition on a known rank-1 tensor."""
        decomp = TensorDecomposition(rank_one_tensor)
        factors = decomp.cp_decomposition(
                rank=1,
                max_iter=200,
                tol=1e-7,
                n_restarts=3,
                init='hosvd',
                stall_patience=5
            )
        
        # Use the class's reconstruction method
        reconstructed = decomp.reconstruct_cp(factors)
        
        # Check reconstruction error
        relative_error = np.linalg.norm(reconstructed.numpy() - rank_one_tensor.numpy()) / \
                        np.linalg.norm(rank_one_tensor.numpy())
        assert relative_error < 1e-5

    def test_tucker_decomposition(self, simple_tensor):
        """Test basic functionality of Tucker decomposition."""
        decomp = TensorDecomposition(simple_tensor)
        ranks = [2, 2, 2]
        core, factors = decomp.tucker_decomposition(ranks)
        
        # Check output structure
        assert isinstance(core, Tensor)
        assert core.shape == tuple(ranks)
        assert len(factors) == 3
        assert all(isinstance(f, Tensor) for f in factors)
        assert factors[0].shape == (2, 2)
        assert factors[1].shape == (3, 2)
        assert factors[2].shape == (4, 2)

    def test_tensor_train(self, simple_tensor):
        """Test basic functionality of Tensor Train decomposition."""
        decomp = TensorDecomposition(simple_tensor)
        ranks = [1, 2, 2, 1]  # Ranks including boundary conditions
        cores = decomp.tensor_train(ranks)
        
        # Check output structure
        assert len(cores) == 3  # Number of cores should be number of modes
        assert all(isinstance(core, Tensor) for core in cores)
        # Check shapes of cores
        assert cores[0].shape == (1, 2, 2)  # First core
        assert cores[1].shape == (2, 3, 2)  # Middle core
        assert cores[2].shape == (2, 4, 1)  # Last core

    def test_unfold(self, simple_tensor):
        """Test tensor unfolding operation."""
        decomp = TensorDecomposition(simple_tensor)
        unfolded = decomp._unfold(simple_tensor, mode=0)
        
        # Check shape of unfolded tensor
        assert unfolded.shape == (2, 12)  # First mode unfolding
        
        # Check if unfolding preserves values
        original = simple_tensor.numpy()
        for i in range(2):
            assert np.array_equal(
                unfolded[i], 
                original[i].flatten()
            )

    def test_khatri_rao(self, simple_tensor):
        """Test Khatri-Rao product computation."""
        decomp = TensorDecomposition(simple_tensor)
        A = Tensor(np.random.randn(2, 3))
        B = Tensor(np.random.randn(3, 3))
        
        result = decomp._khatri_rao([A, B])
        assert result.shape == (6, 3)  # Shape should be (2*3, 3)

    def test_convergence_check(self, simple_tensor):
        """Test convergence checking mechanism."""
        decomp = TensorDecomposition(simple_tensor)
        current = [Tensor(np.ones((2, 2))), Tensor(np.ones((3, 2)))]
        previous = [Tensor(np.ones((2, 2))), Tensor(np.ones((3, 2)))]
        
        # Should converge when factors are identical
        assert decomp._converged(current, previous, tol=1e-6)
        
        # Should not converge when factors are different
        previous[0] = Tensor(np.zeros((2, 2)))
        assert not decomp._converged(current, previous, tol=1e-6)

class TestTensorDecompositionProperties:
    """Test mathematical properties and invariants of tensor decompositions."""
    
    @pytest.fixture
    def random_tensor(self):
        """Create a random 3x4x5 tensor for property testing."""
        return Tensor(np.random.randn(3, 4, 5))

    def test_cp_decomposition_rank_bound(self, random_tensor):
        """Test that CP decomposition respects rank bounds."""
        decomp = TensorDecomposition(random_tensor)
        max_rank = np.prod(random_tensor.shape) // max(random_tensor.shape)
        
        with pytest.raises(ValueError):
            decomp.cp_decomposition(
                rank=max_rank + 1,
                max_iter=200,
                tol=1e-7,
                n_restarts=3,
                init='hosvd',
                stall_patience=5
            )

    def test_tucker_decomposition_orthogonality(self, random_tensor):
        """Test that Tucker factor matrices are orthogonal."""
        decomp = TensorDecomposition(random_tensor)
        ranks = [2, 2, 2]
        _, factors = decomp.tucker_decomposition(ranks)
        
        for factor in factors:
            # Check if factor matrices have orthonormal columns
            factor_np = factor.numpy()
            gram_matrix = factor_np.T @ factor_np
            assert np.allclose(gram_matrix, np.eye(gram_matrix.shape[0]),
                             atol=1e-5)

    def test_tensor_train_ranks(self, random_tensor):
        """Test that Tensor Train decomposition preserves rank bounds."""
        decomp = TensorDecomposition(random_tensor)
        ranks = [1, 4, 4, 1]  # Including boundary ranks
        cores = decomp.tensor_train(ranks)
        
        # Check that ranks are preserved between cores
        for i in range(len(cores) - 1):
            assert cores[i].shape[-1] == cores[i + 1].shape[0]

    @pytest.mark.parametrize("rank", [1, 2, 3])
    def test_cp_decomposition_reconstruction(self, random_tensor, rank):
        """Test reconstruction error decreases with increasing rank."""
        decomp = TensorDecomposition(random_tensor)
        
        errors = []
        for r in range(1, rank + 1):
            factors = decomp.cp_decomposition(
                rank=r,
                max_iter=200,
                tol=1e-7,
                n_restarts=3,
                init='hosvd',
                stall_patience=5
            )
            reconstruction = decomp.reconstruct_cp(factors)
            error = np.linalg.norm(reconstruction.numpy() - random_tensor.numpy())
            errors.append(error)
        
        # Check that errors decrease with increasing rank
        assert all(errors[i] >= errors[i + 1] for i in range(len(errors) - 1))

    def test_decomposition_determinism(self, random_tensor):
        """Test that decompositions are deterministic with fixed random seed."""
        np.random.seed(42)
        decomp1 = TensorDecomposition(random_tensor)
        factors1 = decomp1.cp_decomposition(
                rank=2,
                max_iter=200,
                tol=1e-7,
                n_restarts=3,
                init='hosvd',
                stall_patience=5
            )
        
        np.random.seed(42)
        decomp2 = TensorDecomposition(random_tensor)
        factors2 = decomp2.cp_decomposition(
                rank=2,
                max_iter=200,
                tol=1e-7,
                n_restarts=3,
                init='hosvd',
                stall_patience=5
            )
        
        # Check that factors are identical
        for f1, f2 in zip(factors1, factors2):
            assert np.allclose(f1.numpy(), f2.numpy())