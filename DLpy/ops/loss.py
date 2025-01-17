from typing import Dict, Optional
import numpy as np
from ..core import Function, Tensor

class MSELoss(Function):
    """
    Mean Squared Error Loss: L = 1/N * Σ(y - ŷ)²
    
    Args:
        reduction (str): Specifies the reduction to apply to the output:
            'mean' (default) | 'sum' | 'none'
    """
    
    @staticmethod
    def forward(ctx, predictions, targets, reduction='mean'):
        if not isinstance(predictions, Tensor):
            predictions = Tensor(predictions)
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)
            
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
            
        diff = predictions.data - targets.data
        squared_diff = diff * diff
        
        if reduction == 'none':
            result = squared_diff
        elif reduction == 'sum':
            result = np.sum(squared_diff)
        elif reduction == 'mean':
            result = np.mean(squared_diff)
        else:
            raise ValueError(f"Invalid reduction method: {reduction}")
            
        ctx.save_for_backward(predictions, targets)
        ctx.save_arguments(reduction=reduction)
        return Tensor(result)
        
    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        predictions, targets = ctx.saved_tensors
        reduction = ctx.saved_arguments['reduction']
        
        diff = predictions.data - targets.data
        
        if reduction == 'mean':
            grad = grad_output * 2 * diff / np.prod(diff.shape)
        elif reduction == 'sum':
            grad = grad_output * 2 * diff
        else:  # 'none'
            grad = grad_output * 2 * diff
            
        if predictions.requires_grad:
            grad_dict[id(predictions)] = grad

class CrossEntropyLoss(Function):
    """
    Cross Entropy Loss with built-in LogSoftmax: L = -Σ y_true * log(softmax(y_pred))
    
    Args:
        reduction (str): Specifies the reduction to apply to the output:
            'mean' (default) | 'sum' | 'none'
    """
    
    @staticmethod
    def _log_softmax(x):
        # Compute log(softmax(x)) in a numerically stable way
        max_x = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - max_x)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        return (x - max_x) - np.log(sum_exp_x)
        
    @staticmethod
    def forward(ctx, predictions, targets, reduction='mean'):
        if not isinstance(predictions, Tensor):
            predictions = Tensor(predictions)
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)
            
        log_softmax = CrossEntropyLoss._log_softmax(predictions.data)
        nll_loss = -np.sum(targets.data * log_softmax, axis=1)
        
        if reduction == 'none':
            result = nll_loss
        elif reduction == 'sum':
            result = np.sum(nll_loss)
        elif reduction == 'mean':
            result = np.mean(nll_loss)
        else:
            raise ValueError(f"Invalid reduction method: {reduction}")
            
        ctx.save_for_backward(predictions, targets)
        ctx.save_arguments(reduction=reduction, log_softmax=log_softmax)
        return Tensor(result)
        
    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        predictions, targets = ctx.saved_tensors
        reduction = ctx.saved_arguments['reduction']
        log_softmax = ctx.saved_arguments['log_softmax']
        
        grad_output = np.array(grad_output)
        if reduction == 'mean':
            grad_output = grad_output / len(targets.data)
        
        softmax = np.exp(log_softmax)
        grad = grad_output.reshape(-1, 1) * (softmax - targets.data)
            
        if predictions.requires_grad:
            grad_dict[id(predictions)] = grad

class BinaryCrossEntropyLoss(Function):
    """
    Binary Cross Entropy Loss: L = -Σ (y * log(p) + (1-y) * log(1-p))
    
    Args:
        reduction (str): Specifies the reduction to apply to the output:
            'mean' (default) | 'sum' | 'none'
        eps (float): Small value for numerical stability
    """
    
    @staticmethod
    def forward(ctx, predictions, targets, reduction='mean', eps=1e-7):
        if not isinstance(predictions, Tensor):
            predictions = Tensor(predictions)
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)

        # Check valid probability values
        if np.any(predictions.data < 0) or np.any(predictions.data > 1):
            raise ValueError("Predictions must be in range [0, 1]")
            
        # Clip predictions to prevent log(0)
        predictions_clipped = np.clip(predictions.data, eps, 1 - eps)
        
        loss = -(targets.data * np.log(predictions_clipped) + 
                (1 - targets.data) * np.log(1 - predictions_clipped))
                
        if reduction == 'none':
            result = loss
        elif reduction == 'sum':
            result = float(np.sum(loss))  # Convert to scalar
        elif reduction == 'mean':
            result = float(np.mean(loss))  # Convert to scalar
        else:
            raise ValueError(f"Invalid reduction method: {reduction}")
            
        ctx.save_for_backward(predictions, targets)
        ctx.save_arguments(reduction=reduction, eps=eps)
        return Tensor(result)
        
    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        predictions, targets = ctx.saved_tensors
        reduction = ctx.saved_arguments['reduction']
        eps = ctx.saved_arguments['eps']
        
        predictions_clipped = np.clip(predictions.data, eps, 1 - eps)
        
        grad = grad_output * (predictions_clipped - targets.data) / (
            predictions_clipped * (1 - predictions_clipped))
            
        if reduction == 'mean':
            grad = grad / np.prod(targets.shape)
            
        if predictions.requires_grad:
            grad_dict[id(predictions)] = grad

class L1Loss(Function):
    """
    L1 Loss (Mean Absolute Error): L = |y - ŷ|
    
    Args:
        reduction (str): Specifies the reduction to apply to the output:
            'mean' (default) | 'sum' | 'none'
    """
    
    @staticmethod
    def forward(ctx, predictions, targets, reduction='mean'):
        if not isinstance(predictions, Tensor):
            predictions = Tensor(predictions)
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)
            
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
            
        diff = predictions.data - targets.data
        abs_diff = np.abs(diff)
        
        if reduction == 'none':
            result = abs_diff
        elif reduction == 'sum':
            result = np.sum(abs_diff)
        elif reduction == 'mean':
            result = np.mean(abs_diff)
        else:
            raise ValueError(f"Invalid reduction method: {reduction}")
            
        ctx.save_for_backward(predictions, targets)
        ctx.save_arguments(reduction=reduction)
        return Tensor(result)
        
    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        predictions, targets = ctx.saved_tensors
        reduction = ctx.saved_arguments['reduction']
        
        diff = predictions.data - targets.data
        grad = np.sign(diff)
        
        if reduction == 'mean':
            grad = grad * grad_output / np.prod(diff.shape)
        else:  # 'sum' or 'none'
            grad = grad * grad_output
            
        if predictions.requires_grad:
            grad_dict[id(predictions)] = grad

class KLDivLoss(Function):
    """
    Kullback-Leibler Divergence Loss.
    KL divergence measures the relative entropy between two probability distributions.
    
    Args:
        reduction (str): Specifies the reduction to apply to the output:
            'mean' (default) | 'sum' | 'none'
        log_target (bool): If True, target is expected to be log-probabilities
    """
    
    @staticmethod
    def forward(ctx, predictions, targets, reduction='mean', log_target=False):
        if not isinstance(predictions, Tensor):
            predictions = Tensor(predictions)
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)
            
        if not log_target:
            targets_log = np.log(np.clip(targets.data, 1e-7, 1.0))
        else:
            targets_log = targets.data
            
        # KL divergence formula: KL(P||Q) = P * (log(P) - log(Q))
        loss = np.exp(targets_log) * (targets_log - predictions.data)
        loss = -loss  # Correct the sign to make it positive
        
        if reduction == 'none':
            result = loss
        elif reduction == 'sum':
            result = np.sum(loss)
        elif reduction == 'mean':
            result = np.mean(loss)
        else:
            raise ValueError(f"Invalid reduction method: {reduction}")
            
        ctx.save_for_backward(predictions, targets)
        ctx.save_arguments(reduction=reduction, log_target=log_target)
        return Tensor(result)
        
    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        predictions, targets = ctx.saved_tensors
        reduction = ctx.saved_arguments['reduction']
        log_target = ctx.saved_arguments['log_target']
        
        if not log_target:
            targets_log = np.log(np.clip(targets.data, 1e-7, 1.0))
        else:
            targets_log = targets.data
            
        grad = -np.exp(targets_log) * grad_output
        
        if reduction == 'mean':
            grad = grad / np.prod(predictions.shape)
            
        if predictions.requires_grad:
            grad_dict[id(predictions)] = grad

class CosineSimilarityLoss(Function):
    """
    Cosine Similarity Loss.
    Measures the cosine similarity between two vectors.
    
    Args:
        dim (int): Dimension along which cosine similarity is computed
        eps (float): Small value to avoid division by zero
        reduction (str): Specifies the reduction to apply to the output
    """
    
    @staticmethod
    def forward(ctx, x1, x2, dim=1, eps=1e-8, reduction='mean'):
        if not isinstance(x1, Tensor):
            x1 = Tensor(x1)
        if not isinstance(x2, Tensor):
            x2 = Tensor(x2)
            
        # Compute norms
        norm1 = np.sqrt(np.sum(x1.data * x1.data, axis=dim, keepdims=True))
        norm2 = np.sqrt(np.sum(x2.data * x2.data, axis=dim, keepdims=True))
        
        # Normalize vectors
        x1_normalized = x1.data / np.maximum(norm1, eps)
        x2_normalized = x2.data / np.maximum(norm2, eps)
        
        # Compute cosine similarity
        cos_sim = np.sum(x1_normalized * x2_normalized, axis=dim)
        
        # For orthogonal vectors, cos_sim = 0, we want loss = 1
        # For identical vectors, cos_sim = 1, we want loss = 0
        # Therefore, loss = 1 - cos_sim
        if reduction == 'none':
            result = 1 - cos_sim
        elif reduction == 'sum':
            result = float(np.sum(1 - cos_sim))
        elif reduction == 'mean':
            result = float(np.mean(1 - cos_sim))
        else:
            raise ValueError(f"Invalid reduction method: {reduction}")
            
        ctx.save_for_backward(x1, x2)
        ctx.save_arguments(dim=dim, eps=eps, reduction=reduction,
                         x1_normalized=x1_normalized,
                         x2_normalized=x2_normalized)
        return Tensor(result)
        
    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        x1, x2 = ctx.saved_tensors
        dim = ctx.saved_arguments['dim']
        eps = ctx.saved_arguments['eps']
        reduction = ctx.saved_arguments['reduction']
        x1_normalized = ctx.saved_arguments['x1_normalized']
        x2_normalized = ctx.saved_arguments['x2_normalized']
        
        if reduction == 'mean':
            grad_output = grad_output / x1.shape[0]
        
        # Gradient with respect to x1
        if x1.requires_grad:
            grad_x1 = -grad_output[..., None] * x2_normalized
            grad_dict[id(x1)] = grad_x1
            
        # Gradient with respect to x2
        if x2.requires_grad:
            grad_x2 = -grad_output[..., None] * x1_normalized
            grad_dict[id(x2)] = grad_x2

class HingeLoss(Function):
    """
    Hinge Loss (max-margin loss).
    Commonly used for SVM training.
    L = max(0, margin - y * f(x))
    
    Args:
        margin (float): Margin in the hinge loss
        reduction (str): Specifies the reduction to apply to the output
    """
    
    @staticmethod
    def forward(ctx, predictions, targets, margin=1.0, reduction='mean'):
        if not isinstance(predictions, Tensor):
            predictions = Tensor(predictions)
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)
            
        # Convert targets to ±1
        signed_targets = 2.0 * targets.data - 1.0
        
        # Compute raw hinge loss
        loss = np.maximum(0, margin - signed_targets * predictions.data)
        
        if reduction == 'none':
            result = loss
        elif reduction == 'sum':
            result = np.sum(loss)
        elif reduction == 'mean':
            result = np.mean(loss)
        else:
            raise ValueError(f"Invalid reduction method: {reduction}")
            
        ctx.save_for_backward(predictions, targets)
        ctx.save_arguments(margin=margin, reduction=reduction,
                         signed_targets=signed_targets)
        return Tensor(result)
        
    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        predictions, targets = ctx.saved_tensors
        margin = ctx.saved_arguments['margin']
        reduction = ctx.saved_arguments['reduction']
        signed_targets = ctx.saved_arguments['signed_targets']
        
        # Gradient is -y when margin - y*f(x) > 0, 0 otherwise
        mask = (margin - signed_targets * predictions.data) > 0
        grad = -signed_targets * mask * grad_output
        
        if reduction == 'mean':
            grad = grad / np.prod(predictions.shape)
            
        if predictions.requires_grad:
            grad_dict[id(predictions)] = grad

class FocalLoss(Function):
    """
    Focal Loss.
    Addresses class imbalance by down-weighting easily classified examples.
    FL(p) = -alpha * (1-p)^gamma * log(p)
    
    Args:
        alpha (float): Weighting factor for rare classes
        gamma (float): Focusing parameter
        reduction (str): Specifies the reduction to apply to the output
    """
    
    @staticmethod
    def forward(ctx, predictions, targets, alpha=0.25, gamma=2.0, reduction='mean'):
        if not isinstance(predictions, Tensor):
            predictions = Tensor(predictions)
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)
            
        # Clip predictions for numerical stability
        eps = 1e-7
        predictions_clipped = np.clip(predictions.data, eps, 1 - eps)
        
        # Compute pt (probability of target class)
        pt = predictions_clipped * targets.data + (1 - predictions_clipped) * (1 - targets.data)
        
        # Compute focal weight
        focal_weight = alpha * ((1 - pt) ** gamma)
        
        # Compute binary cross entropy
        bce = -(targets.data * np.log(predictions_clipped) + 
                (1 - targets.data) * np.log(1 - predictions_clipped))
        
        # Apply focal weight
        loss = focal_weight * bce
        
        if reduction == 'none':
            result = loss
        elif reduction == 'sum':
            result = np.sum(loss)
        elif reduction == 'mean':
            result = np.mean(loss)
        else:
            raise ValueError(f"Invalid reduction method: {reduction}")
            
        ctx.save_for_backward(predictions, targets)
        ctx.save_arguments(alpha=alpha, gamma=gamma, reduction=reduction,
                         pt=pt, focal_weight=focal_weight)
        return Tensor(result)
        
    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        predictions, targets = ctx.saved_tensors
        alpha = ctx.saved_arguments['alpha']
        gamma = ctx.saved_arguments['gamma']
        reduction = ctx.saved_arguments['reduction']
        pt = ctx.saved_arguments['pt']
        focal_weight = ctx.saved_arguments['focal_weight']
        
        # Compute gradient
        grad = grad_output * focal_weight * (
            gamma * pt * np.log(pt) + pt - targets.data
        )
        
        if reduction == 'mean':
            grad = grad / np.prod(predictions.shape)
            
        if predictions.requires_grad:
            grad_dict[id(predictions)] = grad

class HuberLoss(Function):
    """
    Huber Loss: L = 0.5 * (y - ŷ)² if |y - ŷ| <= delta else delta * |y - ŷ| - 0.5 * delta²
    
    This loss combines the best properties of MSE and L1 loss.
    For small errors it behaves like MSE, for large errors it behaves like L1.
    
    Args:
        delta (float): Threshold where loss transitions from squared to linear
        reduction (str): Specifies the reduction to apply to the output:
            'mean' (default) | 'sum' | 'none'
    """
    
    @staticmethod
    def forward(ctx, predictions, targets, delta=1.0, reduction='mean'):
        if not isinstance(predictions, Tensor):
            predictions = Tensor(predictions)
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)
            
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
            
        diff = predictions.data - targets.data
        abs_diff = np.abs(diff)
        quadratic = np.minimum(abs_diff, delta)
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic ** 2 + delta * linear
        
        if reduction == 'none':
            result = loss
        elif reduction == 'sum':
            result = np.sum(loss)
        elif reduction == 'mean':
            result = np.mean(loss)
        else:
            raise ValueError(f"Invalid reduction method: {reduction}")
            
        ctx.save_for_backward(predictions, targets)
        ctx.save_arguments(delta=delta, reduction=reduction)
        return Tensor(result)
        
    @staticmethod
    def backward(ctx, grad_output: np.ndarray, grad_dict: Dict[int, np.ndarray]) -> None:
        predictions, targets = ctx.saved_tensors
        delta = ctx.saved_arguments['delta']
        reduction = ctx.saved_arguments['reduction']
        
        diff = predictions.data - targets.data
        abs_diff = np.abs(diff)
        
        # Gradient is diff/|diff| * min(|diff|, delta)
        grad = np.sign(diff) * np.minimum(abs_diff, delta)
        
        if reduction == 'mean':
            grad = grad * grad_output / np.prod(diff.shape)
        else:  # 'sum' or 'none'
            grad = grad * grad_output
            
        if predictions.requires_grad:
            grad_dict[id(predictions)] = grad