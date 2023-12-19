import numpy as np
from scipy.stats import multivariate_normal as sp_mvnorm
from scipy.stats import norm

from aaai_helpers.jax_spm import linear_sum
from aaai_helpers.jax_spm import sample_outcome
import jax.numpy as jnp
from jax import vmap


def evaluate_1d(outcomes, alloc, r):
    '''
    Compute average number of times when result of our allocation is inside success region.
    '''
    res = np.mean(np.sum(np.sum(outcomes * alloc, axis=1)) >= r)
    return res

def evaluate_1d_v2(alloc, mu, Sigma, r):
    '''
    Compute average number of times when result of our allocation is inside success region.
    '''
    mu_psi = np.sum(alloc * mu)
    Sigma_psi = np.sum(alloc * Sigma)
    
    cdf = norm.cdf(
        r, 
        mu_psi, 
        np.sqrt(Sigma_psi)
    )
    return 1. - cdf

def evaluate_2d(outcomes_v, outcomes_c, alloc, r_v, r_c):
    '''
    Compute average number of times when result of our allocation is inside success region.
    '''
    p_v = np.mean(np.sum(np.sum(outcomes_v * alloc, axis=1)) >= r_v)
    p_c = np.mean(np.sum(np.sum(outcomes_c * alloc, axis=1)) <= r_c)
    p_win = p_v * p_c
    return p_v, p_c, p_win


def evaluate_2d_v2(alloc, mu_0, cov_0, r_v, r_c):
    cdf_v_c = sp_mvnorm.cdf(
        np.array([r_v, r_c]), 
        linear_sum(alloc, mu_0), 
        linear_sum(alloc, cov_0).reshape(2,2)
    )
    
    cdf_c = norm.cdf(
        np.array(r_c), 
        linear_sum(alloc, mu_0)[1], 
        np.sqrt(linear_sum(alloc, cov_0).reshape(2,2)[1,1])
    )
    return cdf_c - cdf_v_c