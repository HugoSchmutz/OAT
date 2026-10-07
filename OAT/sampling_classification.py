import numpy as np
import numpy as np
import torch
from typing import Union, Callable

def _get_probabilities(model, X, device='cuda'):
    """Extract probabilities from any supported model type."""
    if isinstance(model, torch.nn.Module):
        if hasattr(model, 'predict_proba'):
            p_theta = model.predict_proba(X, device)
        else:
            model.eval()
            with torch.no_grad():
                X_tensor = torch.from_numpy(X).float().to(device) if isinstance(X, np.ndarray) else X.float().to(device)
                if X_tensor.ndim == 3:
                    X_tensor = X_tensor.unsqueeze(1)
                logits = model(X_tensor)
                p_theta = torch.softmax(logits, dim=1).cpu().numpy()
    else:
        X_flat = X.reshape(X.shape[0], -1)
        p_theta = model.predict_proba(X_flat)
    return np.clip(p_theta, 1e-15, 1 - 1e-15) # Numerical stability


def _get_surrogate_probabilities(model_aux, X, device='cuda'):
    """Extract probabilities from any supported surrogate type."""
    X_flat = X.reshape(X.shape[0], -1) if (isinstance(X, np.ndarray)) else X
    
    if hasattr(model_aux, 'predict_proba'):
        p_pi = model_aux.predict_proba(X_flat)
    
    elif hasattr(model_aux, 'predict_moments'):
        # Homemade model class.
        p_pi = model_aux.predict(X)
    else:
        if isinstance(model_aux, torch.nn.Module):
            model_aux.eval()
            with torch.no_grad():
                X_tensor = torch.from_numpy(X).float().to(device) if isinstance(X, np.ndarray) else X.float().to(device)
                if X_tensor.ndim == 3:
                    X_tensor = X_tensor.unsqueeze(1)
                logits_aux = model_aux(X_tensor)
                p_pi = torch.softmax(logits_aux, dim=1).cpu().numpy()
        else:
            raise TypeError("model_aux must provide probability distributions.")
    return np.clip(p_pi, 1e-15, 1 - 1e-15)# Numerical stability

def compute_generic_classification_metric(model, model_aux, X, loss_choice: str = 'CE', power=1, sqrt=False, device='cuda'):
    """Compute E[loss^power]"""
    p_theta = _get_probabilities(model, X, device=device)
    p_pi = _get_surrogate_probabilities(model_aux, X, device=device)
        
    if loss_choice == 'CE':
        losses = -np.log(p_theta)
        expected_val = np.sum(p_pi * (losses**power), axis=1)
        
    elif loss_choice == '01':
        y_pred = p_theta.argmax(axis=1)
        prob_of_pred = p_pi[np.arange(len(p_pi)), y_pred]
        expected_val = 1 - prob_of_pred   
    else:
        raise ValueError(f"loss_choice must be 'CE' or '01'. Got: {loss_choice}")

    if sqrt:
        return np.sqrt(np.maximum(expected_val, 0))
    return expected_val



def compute_expected_loss_classification_CE(model, model_aux, X, loss_choice='CE'):
    """Computes E_{y|x}[loss] for cross-entropy"""
    return compute_generic_classification_metric(model, model_aux, X, loss_choice, power=1, sqrt=False)

def compute_expected_squared_loss_classification_CE(model, model_aux, X, loss_choice='CE'):
    """Computes sqrt(E_{y|x}[loss^2]) for cross-entropy"""
    return compute_generic_classification_metric(model, model_aux, X, loss_choice, power=2, sqrt=True)

def compute_expected_loss_classification_01(model, model_aux, X, loss_choice='01'):
    """Computes E_{y|x}[loss] for 01-loss"""
    return compute_generic_classification_metric(model, model_aux, X, loss_choice, power=1, sqrt=False)

def compute_expected_squared_loss_classification_01(model, model_aux, X, loss_choice='01'):
    """Computes sqrt(E_{y|x}[loss^2]) for 01-loss"""
    return compute_generic_classification_metric(model, model_aux, X, loss_choice, power=2, sqrt=True)

