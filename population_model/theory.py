"""
theory.py
Theoretical conditions for replay propagation in population model.

This module implements analytical conditions from theory to predict whether replay
will successfully propagate in a network given: feedforward (F), recurrent (R),
number of assemblies (q), and number of time steps (t).
"""

import warnings
import numpy as np
import scipy.special


def cond_fin_q_fin_t(ff, rc, q, t):
    """
    Compute replay success condition for finite number of assemblies and 
    finite number of time steps. Corresponds to Eq (9) in the manuscript.
    
    Args:
        ff: Feedforward weight (F). Can be scalar or array.
        rc: Recurrent weight (R). Can be scalar or array.
        q: Number of assemblies (integer).
        t: Number of time steps (integer).
    
    Returns:
        Boolean array indicating success (True) or failure (False) for each (F,R) parameter combination.
    """

    # Compute all possible time steps where recurrent activation can occur
    warnings.filterwarnings("error")
    k_array = np.arange(0, t - q + 1)
    
    # Log of binomial coefficients: log(C(k+q-1, q-1))
    binom_coeff_log = np.log(scipy.special.comb(k_array + q - 1, q - 1))
    
    # Reshape for broadcasting
    rc_2d = rc[:, np.newaxis]
    
    # Log of power terms: k * log(R)
    power_terms_log = k_array * np.log(rc_2d)
    
    # Log of feedforward coefficient: (q-1) * log(F)
    ff_coeff_log = (q - 1) * np.log(ff)

    # Handle scalar vs array feedforward inputs for proper broadcasting
    if np.array(ff).size == 1:
        ff_coeff_log_term = ff_coeff_log
    else:
        ff_coeff_log_term = ff_coeff_log[:, None, None]

    # Combine all log terms for numerical stability
    sum_terms_log = ff_coeff_log_term + binom_coeff_log[None, None, :] + power_terms_log[None, :, :]

    # Compute the sum v = F^(q-1) * sum_k [ C(k+q-1, q-1) * R^k ]
    with np.errstate(over='ignore'):
        sum_terms = np.exp(sum_terms_log)
        v = np.sum(sum_terms, axis=2)

    # Replace NaN values with maximum float (indicating overflow/failure)
    v[np.argwhere(np.isnan(v))] = np.finfo(np.float64).max

    # Replay succeeds if v >= 1
    out = np.zeros_like(v, dtype=bool)
    out[v >= 1] = True

    return out


def cond_fin_q_inf_t(ff, rc, q):
    """
    Compute replay success condition for finite number of assemblies and 
    infinite number of time steps. Corresponds to Eq (9) when S → 0.
    
    Args:
        ff: Feedforward weight (F). Can be scalar or array.
        rc: Recurrent weight (R). Can be scalar or array.
        q: Number of assemblies (integer).
    
    Returns:
        Boolean array indicating success (True) or failure (False) for each (F,R) parameter combination.
    """
    # Reshape for broadcasting
    ff_2d = ff[:, None]
    rc_2d = rc[None, :]
    
    # Compute (F / (1 - R))^q / F, which is the limit for R < 1
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):  # Ignore division by zero
        v_small = ((ff_2d / (1 - rc_2d)) ** q) / ff_2d

    # When R >= 1, the series diverges; set v to maximum (guaranteed success)
    v = np.where(rc_2d >= 1, np.finfo(np.float64).max, v_small)

    # Replay succeeds if v >= 1
    out = np.zeros_like(v, dtype=bool)
    out[v >= 1] = True

    return out


def cond_inf_q(ff, rc, speed):
    """
    Compute replay success condition for infinite number of assemblies and 
    given speed S. Corresponds to Condition 3 in the manuscript.
    
    Args:
        ff: Feedforward weight (F). Can be scalar or array.
        rc: Recurrent weight (R). Can be scalar or array.
        speed: Propagation speed. Float in (0,1]
    
    Returns:
        Boolean array indicating success (True) or failure (False) for each (F,R) parameter combination.
    """
    # Reshape for broadcasting
    ff_2d = ff[:, None]
    rc_2d = rc[None, :]
    
    # Two regimes based on recurrent weight relative to speed
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):  # Ignore division by zero
        
        v_inf = 1 / (1 - rc_2d)
        
        v_small = (1 / speed) * ((rc_2d / (1 - speed)) ** (1 / speed - 1))

    # Select regime based on whether R < 1 - s
    v = ff_2d * np.where(rc_2d < 1 - speed, v_inf, v_small)
    
    # Replay succeeds if v >= 1
    out = np.zeros_like(v, dtype=bool)
    out[v >= 1] = True

    return out
