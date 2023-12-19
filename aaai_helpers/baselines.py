import numpy as np
import scipy as sp
import itertools

import jax.numpy as jnp
from jax import vmap

from aaai_helpers.eval import evaluate_1d_v2, evaluate_2d_v2
from aaai_helpers.optim import *
from aaai_helpers.jax_spm import build_criteria_1d, build_criteria_my_v2

### 1D case

def hard_bruteforce_1d(y_bs, all_allocs, r):
    '''
    hard bruteforce 1d
    '''
    # candidates = [list(comb) for comb in list(itertools.product([0., 1.], repeat=npol)) if sum(comb) == 1]
    # all_allocs = list(itertools.product(candidates, repeat=nbucket))

    res_temp = []
    for alloc in all_allocs:
        res_temp.append([np.array(alloc), evaluate_1d(y_bs, alloc, r)])
    
    best_alloc, best_res = sorted(res_temp, key=lambda x:x[1])[-1]
    
    return best_alloc

def hard_bruteforce_1d_v2(all_allocs, mu, Sigma, r):
    '''
    hard bruteforce 1d
    '''
    # candidates = [list(comb) for comb in list(itertools.product([0., 1.], repeat=npol)) if sum(comb) == 1]
    # all_allocs = list(itertools.product(candidates, repeat=nbucket))

    res_temp = []
    for alloc in all_allocs:
        res_temp.append([np.array(alloc), evaluate_1d_v2(np.array(alloc), mu, Sigma, r)])
    
    best_alloc, best_res = sorted(res_temp, key=lambda x:x[1])[-1]
    
    return best_alloc

def hard_bruteforce_1d_v3(all_allocs, mu, Sigma, r):
    '''
    hard bruteforce 1d
    '''
    criteria = build_criteria_1d(Sigma, mu, r)
    res = vmap(criteria)(jnp.array(all_allocs))
    best_alloc = all_allocs[jnp.argmax(res)]
    
    return best_alloc


def greedy_1d(mu_0):
    '''
    greedy 1d
    '''
    res = (mu_0 == mu_0.max(axis=1)[:,None]).astype(float)
    return res
    



### 2D case
def hard_bruteforce_2d(all_allocs, mu, cov, r_v, r_c):
    '''
    hard bruteforce 2d
    '''
    # candidates = [list(comb) for comb in list(itertools.product([0., 1.], repeat=npol)) if sum(comb) == 1]
    # all_allocs = list(itertools.product(candidates, repeat=nbucket))

    res_temp = []
    for alloc in all_allocs:
        res_temp.append([np.array(alloc), evaluate_2d_v2(np.array(alloc), mu, cov, r_v, r_c)])
    
    best_alloc, best_res = sorted(res_temp, key=lambda x:x[1])[-1]
    
    return best_alloc

def hard_bruteforce_2d_v2(all_allocs, mu, cov, r_v, r_c):
    '''
    hard bruteforce 2d
    '''

    criteria = build_criteria_my_v2(cov, mu, r_v, r_c)
    res = vmap(criteria)(jnp.array(all_allocs))
    best_alloc = all_allocs[jnp.argmax(res)]
    
    return best_alloc


def greedy_2d(mu_v, mu_c, r_c, nbucket, npol, soft=1):
    '''
    greedy 2d
    '''
    if soft:
        res = solve_opt_pb(mu_v, mu_c, r_c, nbucket, npol)
    else:
        res = solve_opt_pb_mi(mu_v, mu_c, r_c, nbucket, npol)
    return res

