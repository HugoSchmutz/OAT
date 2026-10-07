import numpy as np
from typing import Callable, Tuple, List, Dict
from .sampling_classification import *
from .sampling_regression import *


def oat_estimate(losses: np.ndarray, selection: np.ndarray, probabilities: np.ndarray, T: int) -> np.ndarray:
    """Computes the Online-Active-Testing (OAT) estimate of the risk for all 1<=t<=T."""
    if not (len(losses) == len(selection) == len(probabilities) == T):
        raise ValueError(f"Input lengths must all equal T ({T}).")
        
    weighted = np.zeros(T)
    mask = selection.astype(bool)
    #loss / probability of selection
    weighted[mask] = losses[mask] / probabilities[mask]
    return weighted.cumsum() / np.arange(1, T + 1)




def run_adaptive_selection(
    model_main, model_aux,
    X_train, y_train,
    X_test, y_test,
    permutation,
    T_stream,
    budget,
    heuristic,
    refit_milestones = None
):
    #If refit_milestones is not specified, refit every new sample selected
    if not refit_milestones: refit_milestones = np.arange(1,T_stream+1)
    
    # Initial fit
    model_aux.fit(X_train, y_train)

    n_selected = 0
    selection = np.zeros(T_stream, dtype=bool)
    beta_selection = np.zeros(T_stream)

    # Running mean of the weights for normalisation
    running_sum = 1e-15

    # Dataset of selected samples for adaptive training 
    X_buffer = [X_train]
    y_buffer = [y_train]

    for t in range(T_stream):

        x_t = X_test[permutation[t]]
        x_t = x_t[np.newaxis, :]
        y_t = y_test[permutation[t]]

        # Compute moment of the loss for the current sample
        if heuristic=='CE_1stmoment':
            weight = compute_expected_loss_classification_CE(model_main, model_aux, x_t)[0]
        elif heuristic=='CE_2ndmoment':
            weight = compute_expected_squared_loss_classification_CE(model_main, model_aux, x_t)[0]
        elif heuristic=='01_1stmoment':
            weight = compute_expected_loss_classification_01(model_main, model_aux, x_t)[0]
        elif heuristic=='01_2ndmoment':
            weight = compute_expected_squared_loss_classification_01(model_main, model_aux, x_t)[0]
        elif heuristic=='MSE_1stmoment':
            weight = compute_expected_loss_regression_MSE(model_main, model_aux, x_t)[0]
        elif heuristic =='MSE_2ndmoment':
            weight = compute_expected_squared_loss_regression_MSE(model_main, model_aux, x_t)[0]
            
        running_sum += weight
        norm_const = running_sum / (t + 1)

        pi = weight / norm_const * budget / T_stream
        pi = min(pi, 1.0) #pi<=1
        
        #To fit exact budget
        if T_stream - t == budget - n_selected:
            pi=1
        
        r = np.random.rand() < pi
        selection[t] = r
        beta_selection[t] = pi

        if r:
            n_selected+=1
            X_buffer.append(x_t.flatten().reshape(1, -1))
            y_buffer.append(np.array([y_t]))

            # Refit only when needed
            if n_selected in refit_milestones:
                model_aux.fit(
                    np.vstack(X_buffer),
                    np.concatenate(y_buffer)
                )

    return selection, beta_selection

