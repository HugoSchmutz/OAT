import numpy as np 

def per_sample_mse(model, X_test, y_test) -> np.ndarray:
    """Computes Mean Squared Error for each sample."""
    predictions = model.predict(X_test)
    return (predictions - y_test)**2

def per_sample_binary_log_loss(model, X_test, y_test) -> np.ndarray:
    """Computes Binary Log Loss for each sample with clipping for numerical stability."""
    predict_proba = model.predict_proba(X_test).clip(min = 1e-15, max = 1-1e-15) 
    pos_log_loss = -np.log(predict_proba[:, 1]) * y_test 
    neg_log_loss = -np.log(predict_proba[:, 0]) * (1 - y_test)
    return pos_log_loss + neg_log_loss

def per_sample_01_loss(model, X_test, y_test) -> np.ndarray:
    """Computes Log Loss for each sample."""
    y_pred = model.predict(X_test)
    return 1-(y_pred==y_test).astype(int)

def per_sample_log_loss(model, X_test, y_test) -> np.ndarray:
    """Computes Log Loss for each sample with clipping for numerical stability."""
    predict_proba = model.predict_proba(X_test).clip(min = 1e-15, max = 1-1e-15) 
    log_proba = np.log(predict_proba)
    return(-log_proba[np.arange(log_proba), y_test])

def per_sample_binary_entropy(model, X_test) -> np.ndarray:
    """Calculates the entropy of the model's predictions."""
    probs = model.predict_proba(X_test).clip(min = 1e-15, max = 1 - 1e-15)
    pos, neg = probs[:, 1], probs[:, 0]
    return pos * (-np.log(pos)) + neg * (-np.log(neg))


def per_sample_log_loss_with_device(model, X_test, y_test, device) -> np.ndarray:
    """Computes Log Loss for deep models for each sample with clipping for numerical stability."""
    predict_proba = model.predict_proba(X_test, device).clip(min = 1e-15, max = 1-1e-15) 
    log_proba = np.log(predict_proba)
    return(-log_proba[np.arange(log_proba), y_test])

def per_sample_01_loss_with_device(model, X_test, y_test, device) -> np.ndarray:
    """Computes Log Loss for deep models for each sample."""
    y_pred = model.predict(X_test, device)
    return 1-(y_pred==y_test).astype(int)