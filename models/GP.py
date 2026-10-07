import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

class MomentGPRegressor(GaussianProcessRegressor):
    """
    An extension of Sckitit-learn Gaussian Process that can return 
    the first four central moments of the predictive distribution.
    """
    def predict_moments(self, X, **kwargs):
        """
        Extended predict method.
        
        Args:
            X: Input samples.
            
        Returns:
            (mu, var, m3, m4) 
        """
        mu, std = super().predict(X, return_std=True, **kwargs)
        var = std**2
        m3 = np.zeros_like(var)# m3 is 0 for Gaussian
        m4 = 3 * (var**2)# m4 = 3 * sigma^4 for Gaussian
        return mu, var, m3, m4