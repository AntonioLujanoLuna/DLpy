# DLpy/utils/decomposition.py

from typing import List, Tuple

import numpy as np

from ..core import Tensor


class TensorDecomposition:
    """
    Implementation of various tensor decomposition methods.

    This class provides implementations of three major tensor decomposition methods:
    1. CP (CANDECOMP/PARAFAC) Decomposition
    2. Tucker Decomposition
    3. Tensor Train Decomposition

    Each method decomposes a high-order tensor into simpler components while
    maintaining different properties and trade-offs.

    Attributes:
        tensor (Tensor): The input tensor to be decomposed
        shape (Tuple[int, ...]): Shape of the input tensor
    """

    def __init__(self, tensor: Tensor):
        """
        Initialize the decomposition object with an input tensor.

        Args:
            tensor: Input tensor to be decomposed
        """
        self.tensor = tensor
        self.shape = tensor.shape

    def cp_decomposition(
        self,
        rank: int,
        max_iter: int = 100,
        tol: float = 1e-6,
        n_restarts: int = 1,
        init: str = "svd",
        stall_patience: int = 3,
    ) -> List[Tensor]:
        """
        Perform CP decomposition via Alternating Least Squares (ALS) with
        improved initialization, multiple restarts, and an optional stall check.

        Args:
            rank (int): Desired rank for CP decomposition.
            max_iter (int): Maximum number of ALS iterations (per restart).
            tol (float): Convergence tolerance for relative error change.
            n_restarts (int): Number of independent runs with different inits.
                            The best (lowest-error) factor set is returned.
            init (str): Initialization method:
                        - "svd" uses truncated SVD on each mode-unfolding.
                        - "hosvd" uses multilinear SVD (Tucker-based) to build initial factors.
                        - "random" samples random initial factors.
            stall_patience (int): How many consecutive ALS iterations can stall
                                (no improvement) before we terminate early.
                                Set to None or negative to disable stall check.

        Returns:
            List[Tensor]: A list of factor matrices (one per mode),
                        each with shape (dim_mode, rank).

        Raises:
            ValueError: If rank is too large or invalid based on the tensor shape,
                        or if init string is not recognized.
        """
        # 1) Basic rank validation
        max_rank = np.prod(self.shape) // max(self.shape)
        if rank < 1:
            raise ValueError(f"Rank must be >= 1, got {rank}.")
        if rank > max_rank:
            raise ValueError(f"Requested rank={rank} exceeds maximum possible {max_rank}.")

        # 2) We'll do multiple restarts and track the best solution
        best_factors: List[Tensor] = []
        best_error = float("inf")

        # --- Helper: do a single ALS pass from an initial factor set ---
        def run_als(factors_init: List[Tensor]) -> Tuple[List[Tensor], float]:
            """
            Run ALS starting from 'factors_init', up to max_iter or stall limit.
            Returns the final factors and final reconstruction error.
            """
            factors = [f.copy() for f in factors_init]  # safe copy

            tensor_data = self.tensor.numpy()
            prev_error = None  # Initialize as None
            stall_counter = 0

            for iteration in range(max_iter):
                # Update each mode in turn
                for mode in range(len(self.shape)):
                    # Unfold the tensor along this mode
                    unfolded = self._unfold(self.tensor, mode)  # shape=(shape[mode], -1)
                    # Build Khatri-Rao product of all other factors
                    other = [factors[m] for m in range(len(self.shape)) if m != mode]
                    kr = self._khatri_rao(other)  # shape=(product_of_dims, rank)

                    # Solve for the current factor using least squares
                    pseudo_inv = np.linalg.pinv(kr.T @ kr)  # rank x rank
                    new_factor = unfolded @ kr @ pseudo_inv

                    factors[mode] = Tensor(new_factor)

                # Check reconstruction error
                reconstructed = self.reconstruct_cp(factors).numpy()
                error = np.linalg.norm(reconstructed - tensor_data)

                if prev_error is not None and np.isfinite(error) and np.isfinite(prev_error):
                    rel_change = abs(error - prev_error) / (max(abs(prev_error), 1e-12) + 1e-12)

                    if rel_change < tol:
                        # Converged by relative error threshold
                        break

                    if stall_patience is not None and stall_patience > 0:
                        if error >= prev_error:
                            stall_counter += 1
                        else:
                            stall_counter = 0
                        if stall_counter >= stall_patience:
                            # Stop early if no improvement for too many iterations
                            break
                # Else, it's the first iteration or non-finite values; skip rel_change

                prev_error = error

            return factors, error

        # --- Helper: factor initializers ---
        def init_factors(init_mode: str) -> List[Tensor]:
            """
            Return a list of Tensors that serve as the initial guess
            for the factor matrices in CP decomposition, each shaped (dim[mode], rank).
            """
            modes = len(self.shape)
            self.tensor.numpy()

            if init_mode == "svd":
                # Truncated SVD on each mode's unfolding
                inits = []
                for mode in range(modes):
                    mat = self._unfold(self.tensor, mode)  # shape= (shape[mode], -1)
                    U, S, Vt = np.linalg.svd(mat, full_matrices=False)
                    # Keep top 'rank' columns
                    U = U[:, :rank]
                    inits.append(Tensor(U))
                return inits

            elif init_mode == "hosvd":
                # Similar to Tucker: do multilinear SVD with rank-limited factor mats
                # Then discard the Tucker "core" and treat these as CP init.
                # We'll do a quick pass:
                factors_hosvd = []
                for mode in range(modes):
                    unfold_ = self._unfold(self.tensor, mode)
                    U, S, Vt = np.linalg.svd(unfold_, full_matrices=False)
                    U = U[:, :rank]
                    factors_hosvd.append(Tensor(U))
                return factors_hosvd

            elif init_mode == "random":
                # Just random normal or uniform
                inits = []
                for dim in self.shape:
                    mat = np.random.randn(dim, rank)
                    inits.append(Tensor(mat))
                return inits

            else:
                raise ValueError(
                    f"Unrecognized init method '{init_mode}'. "
                    "Use one of ['svd', 'hosvd', 'random']."
                )

        # 3) Do n_restarts with the chosen initialization
        for _attempt in range(n_restarts):
            # Create initial factor guess
            factors_init = init_factors(init)

            # Run ALS from that init
            final_factors, final_err = run_als(factors_init)

            # Track the best solution
            if final_err < best_error:
                best_error = final_err
                best_factors = final_factors

        # Return the best factors we found
        return best_factors

    def tensor_train(self, ranks: List[int]) -> List[Tensor]:
        """
        Implement Tensor Train decomposition with careful dimension tracking.

        This version maintains proper shape calculations throughout the process
        and ensures ranks are properly handled.
        """
        if len(ranks) != len(self.shape) + 1:
            raise ValueError("Number of ranks must be number of dimensions + 1")
        if ranks[0] != 1 or ranks[-1] != 1:
            raise ValueError("First and last ranks must be 1")

        # Convert tensor to numpy array and make a copy to avoid modifying original
        current = self.tensor.numpy().copy()
        n_dims = len(self.shape)
        cores = []

        # First core initialization
        n1 = self.shape[0]
        n2 = np.prod(self.shape[1:])
        matrix = current.reshape(n1, n2)

        # Process each dimension sequentially
        r_prev = 1
        for k in range(n_dims - 1):
            # Reshape for current dimension
            if k == 0:
                matrix = current.reshape(n1, -1)
            else:
                matrix = current.reshape(r_prev * self.shape[k], -1)

            # Compute SVD
            U, S, V = np.linalg.svd(matrix, full_matrices=False)

            # Determine rank (minimum of desired and available)
            r = min(ranks[k + 1], len(S))

            # Truncate SVD matrices
            U = U[:, :r]
            S = S[:r]
            V = V[:r, :]

            # Reshape U into core tensor
            if k == 0:
                core = U.reshape(1, self.shape[k], r)
            else:
                core = U.reshape(r_prev, self.shape[k], r)

            cores.append(Tensor(core))

            # Update for next iteration
            r_prev = r
            current = np.diag(S) @ V

        # Handle last core carefully
        last_core = current.reshape(r_prev, self.shape[-1], 1)
        cores.append(Tensor(last_core))

        return cores

    def _mode_n_product(self, tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
        """
        Perform the mode-n product of a tensor with a matrix.

        Args:
            tensor (np.ndarray): The input tensor.
            matrix (np.ndarray): The matrix to multiply with.
            mode (int): The mode along which to perform the multiplication.

        Returns:
            np.ndarray: The resulting tensor after the mode-n product.
        """
        # Move the specified mode to the first dimension
        tensor = np.moveaxis(tensor, mode, 0)
        # Reshape the tensor to 2D for matrix multiplication
        original_shape = tensor.shape
        tensor_reshaped = tensor.reshape(tensor.shape[0], -1)
        # Perform matrix multiplication
        result = matrix @ tensor_reshaped
        # Determine the new shape
        new_shape = (matrix.shape[0],) + original_shape[1:]
        # Reshape back to tensor form
        result = result.reshape(new_shape)
        # Move the axes back to their original positions
        result = np.moveaxis(result, 0, mode)
        return result

    def _tucker_core(self, tensor: Tensor, factors: List[Tensor]) -> Tensor:
        """
        Compute the core tensor for Tucker decomposition by sequentially applying
        mode-n products with the transposed factor matrices.

        Args:
            tensor (Tensor): The input tensor.
            factors (List[Tensor]): List of factor matrices.

        Returns:
            Tensor: The core tensor of the Tucker decomposition.
        """
        core = tensor.numpy()
        for mode, factor in enumerate(factors):
            # Multiply the core tensor with the transpose of the factor matrix along the current mode
            core = self._mode_n_product(core, factor.numpy().T, mode)
        return Tensor(core)

    def reconstruct_tucker(self, core: Tensor, factors: List[Tensor]) -> Tensor:
        """
        Reconstruct a tensor from its Tucker decomposition by sequentially applying
        mode-n products with the factor matrices.

        Args:
            core (Tensor): The core tensor from Tucker decomposition.
            factors (List[Tensor]): List of factor matrices.

        Returns:
            Tensor: The reconstructed tensor.
        """
        tensor = core.numpy()
        for mode, factor in enumerate(factors):
            tensor = self._mode_n_product(tensor, factor.numpy(), mode)
        return Tensor(tensor)

    def tucker_decomposition(self, ranks: List[int]) -> Tuple[Tensor, List[Tensor]]:
        """
        Implement Tucker decomposition using Higher-Order SVD (HOSVD).

        Args:
            ranks: List of ranks for each mode

        Returns:
            Tuple of (core tensor, list of factor matrices)

        Raises:
            ValueError: If requested ranks exceed tensor dimensions
        """
        # Validate ranks
        if any(r > s for r, s in zip(ranks, self.shape)):
            raise ValueError("Requested ranks exceed tensor dimensions")

        # Initialize factor matrices using truncated SVD
        factors = []
        for mode, rank in enumerate(ranks):
            # Unfold tensor along current mode
            unfolded = self._unfold(self.tensor, mode)

            # Compute truncated SVD
            U, S, Vt = np.linalg.svd(unfolded, full_matrices=False)
            U_truncated = U[:, :rank]
            factors.append(Tensor(U_truncated))

        # Compute core tensor by contracting with factor matrices
        core = self._tucker_core(self.tensor, factors)

        return core, factors

    def _unfold(self, tensor: Tensor, mode: int) -> np.ndarray:
        """
        Unfold/matricize a tensor along specified mode.

        This operation reshapes the tensor into a matrix where the specified mode
        becomes the first dimension and all other dimensions are combined.

        Args:
            tensor: Input tensor
            mode: Mode along which to unfold

        Returns:
            Unfolded tensor as a 2D array
        """
        arr = tensor.numpy()
        # Move specified mode to first dimension and flatten others
        return np.moveaxis(arr, mode, 0).reshape(arr.shape[mode], -1)

    def _khatri_rao(self, matrices: List[Tensor]) -> np.ndarray:
        """
        Compute Khatri-Rao product of a list of matrices.

        The Khatri-Rao product is a columnwise Kronecker product. For matrices
        with the same number of columns, it gives a matrix whose columns are
        Kronecker products of the corresponding columns.

        Args:
            matrices: List of matrices as Tensor objects

        Returns:
            numpy.ndarray: Khatri-Rao product
        """
        if not matrices:
            return None

        n_cols = matrices[0].shape[1]
        n_matrices = len(matrices)

        # Initialize result with the numpy array from first tensor
        result = matrices[0].numpy()

        # Compute product sequentially
        for i in range(1, n_matrices):
            n_rows = result.shape[0]
            next_matrix = matrices[i].numpy()

            # Compute Khatri-Rao product for current pair of matrices
            next_rows = next_matrix.shape[0]
            result = (result.reshape(-1, 1, n_cols) * next_matrix.reshape(1, -1, n_cols)).reshape(
                n_rows * next_rows, n_cols
            )

        return result

    def _converged(self, current: List[Tensor], previous: List[Tensor], tol: float) -> bool:
        """
        Check convergence of ALS iterations with improved metric.
        """
        # Compute relative change in factors
        changes = []
        for curr, prev in zip(current, previous):
            curr_norm = np.linalg.norm(curr.numpy())
            prev_norm = np.linalg.norm(prev.numpy())
            if prev_norm == 0:
                changes.append(curr_norm > tol)
            else:
                changes.append(np.linalg.norm(curr.numpy() - prev.numpy()) / prev_norm)

        return all(change < tol for change in changes)

    def reconstruct_cp(self, factors: List[Tensor]) -> Tensor:
        """
        Reconstruct tensor from CP decomposition factors.

        Uses an efficient implementation that properly handles the outer products
        for reconstruction.
        """
        # Get rank from factor shapes
        rank = factors[0].shape[1]
        result = np.zeros(self.shape)

        # Build rank-1 components and sum them
        for r in range(rank):
            # Extract rank-r vectors
            component = factors[0].numpy()[:, r : r + 1]

            # Build through successive outer products
            for factor in factors[1:]:
                component = component[..., np.newaxis] * factor.numpy()[:, r : r + 1]

            result += component.reshape(self.shape)

        return Tensor(result)

    def reconstruct_tt(self, cores: List[Tensor]) -> Tensor:
        """
        Reconstruct a tensor from its Tensor Train decomposition.

        This method rebuilds the tensor by sequentially contracting the cores
        along their shared edges.

        Args:
            cores: List of core tensors from TT decomposition

        Returns:
            Reconstructed tensor
        """
        # Start with first core
        result = cores[0].numpy()

        # Contract cores sequentially
        for core in cores[1:]:
            # Reshape result for contraction
            left_rank = result.shape[-1]
            result = result.reshape(-1, left_rank)

            # Reshape core for contraction
            core_data = core.numpy()
            right_shape = core_data.shape[1:]
            core_matrix = core_data.reshape(left_rank, -1)

            # Contract and reshape
            result = result @ core_matrix
            result = result.reshape(*result.shape[:-1], *right_shape)

        return Tensor(result)
