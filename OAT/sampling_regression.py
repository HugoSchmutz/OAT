import numpy as np


def compute_expected_loss_regression_MSE(model, model_aux, x_t):
    """1st moment of MSE"""
    try:
        mu, variance, _, _ = model_aux.predict_moments(x_t)
    except TypeError:
        raise TypeError("Unsupported model_aux type. model_aux must support predict_moment")
    
    f_theta_xt = np.squeeze(model.predict(x_t))
    bias = f_theta_xt - mu
    return (bias**2) + variance



def compute_expected_squared_loss_regression_MSE(model, model_aux, x_t):
    """2nd moment of MSE"""
    try:
        mu, variance, m3, m4 = model_aux.predict_moments(x_t)
    except TypeError:
        raise TypeError("Unsupported model_aux type. model_aux must support predict_moment")
    
    f_theta_xt = np.squeeze(model.predict(x_t))
    bias = f_theta_xt - mu
    
    s2 = (bias**4) + (6 * (bias**2) * variance) - (4 * bias * m3) + m4
    
    return np.sqrt(np.maximum(s2, 0))
