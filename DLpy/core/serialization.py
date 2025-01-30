# DLpy/core/serialization.py

import pickle
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from ..core import Module


class ModelSaver:
    """
    Handles saving and loading of models and their state dictionaries.

    Supports:
    - Complete model saving/loading (architecture + parameters)
    - State dict only saving/loading (just parameters)
    - Checkpointing (model state + optimizer state + training state)
    """

    @staticmethod
    def save_model(model: Module, path: Union[str, Path], optimize: bool = True) -> None:
        """
        Save complete model (architecture + parameters).

        Args:
            model: The model to save
            path: Path where to save the model
            optimize: If True, optimize the saved file size
        """
        path = Path(path)
        state = {
            "model_class": model.__class__.__name__,
            "model_dict": model.__dict__,
            "state_dict": ModelSaver.get_state_dict(model),
        }

        # Save module hierarchy for reconstruction
        if hasattr(model, "_modules"):
            state["modules"] = {
                name: module.__class__.__name__ for name, module in model._modules.items()
            }

        with open(path, "wb") as f:
            if optimize:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                pickle.dump(state, f)

    @staticmethod
    def load_model(path: Union[str, Path], custom_classes: Dict[str, type] = None) -> Module:
        """
        Load complete model (architecture + parameters).
        """
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)

        # Get model class - first try custom classes if provided
        model_class = None
        if custom_classes and state["model_class"] in custom_classes:
            model_class = custom_classes[state["model_class"]]

        if model_class is None:
            # Try modules in this order: nn module, main module, test module
            import DLpy.nn as nn

            model_class = getattr(nn, state["model_class"], None)

            if model_class is None:
                # Try test module
                import sys

                test_module_names = ["__main__", "tests.test_serialization"]
                for module_name in test_module_names:
                    if module_name in sys.modules:
                        model_class = getattr(sys.modules[module_name], state["model_class"], None)
                        if model_class is not None:
                            break

        if model_class is None:
            raise ValueError(f"Unknown model class: {state['model_class']}")

        # Create model instance
        model = model_class()  # Call __init__ to ensure proper initialization

        # Restore model state
        for key, value in state["model_dict"].items():
            if key in model.__dict__:
                model.__dict__[key] = value

        # Load state dict
        model.load_state_dict(state["state_dict"])

        return model

    @staticmethod
    def save_state_dict(model: Module, path: Union[str, Path]) -> None:
        """
        Save only the model's state dictionary (parameters).

        Args:
            model: The model whose parameters to save
            path: Path where to save the state dict
        """
        path = Path(path)
        state_dict = ModelSaver.get_state_dict(model)
        np.savez(path, **state_dict)

    @staticmethod
    def load_state_dict(model: Module, path: Union[str, Path]) -> None:
        """
        Load parameters into an existing model.

        Args:
            model: The model to load parameters into
            path: Path to the saved state dict
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")

        state_dict = dict(np.load(path))
        model.load_state_dict(state_dict)

    @staticmethod
    def save_checkpoint(
        path: Union[str, Path],
        model: Module,
        optimizer: Any = None,
        epoch: int = None,
        loss: float = None,
        additional_data: Dict[str, Any] = None,
    ) -> None:
        """
        Save a training checkpoint including model, optimizer and training state.

        Args:
            path: Path where to save the checkpoint
            model: The model to checkpoint
            optimizer: The optimizer to checkpoint (optional)
            epoch: Current epoch number (optional)
            loss: Current loss value (optional)
            additional_data: Additional data to save (optional)
        """
        path = Path(path)
        checkpoint = {
            "model_state_dict": ModelSaver.get_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "epoch": epoch,
            "loss": loss,
            "additional_data": additional_data or {},
        }

        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

    @staticmethod
    def load_checkpoint(
        path: Union[str, Path], model: Module, optimizer: Any = None
    ) -> Dict[str, Any]:
        """
        Load a training checkpoint.

        Args:
            path: Path to the checkpoint
            model: The model to load state into
            optimizer: The optimizer to load state into (optional)

        Returns:
            Dictionary containing the non-model/optimizer checkpoint data
        """
        path = Path(path)
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and checkpoint["optimizer_state_dict"]:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return {
            "epoch": checkpoint["epoch"],
            "loss": checkpoint["loss"],
            "additional_data": checkpoint["additional_data"],
        }

    @staticmethod
    def get_state_dict(model: Module) -> Dict[str, np.ndarray]:
        """Gets the state dictionary from a model."""
        state_dict = {}
        for name, param in model.named_parameters():
            if param is not None:
                state_dict[name] = param.data
        for name, buffer in model._buffers.items():
            if buffer is not None:
                state_dict[name] = buffer.data
        return state_dict
