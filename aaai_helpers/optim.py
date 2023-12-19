import cvxpy as cvx
import numpy as np
import pandas as pd

def solve_opt_pb(mu_0_v, mu_0_c, r_cost, nbucket, npol):
    '''
    Linear programming solution (greedy, soft allocation) for maximizing value with cost constraint.
    '''
    
    list_pols = range(npol)

    P = {i:cvx.Variable(nbucket) for i in list_pols}

    obj_exp_value = cvx.sum([P[i] @ mu_0_v[:,i] for i in list_pols])
    cost_constraint = cvx.sum([P[i] @ mu_0_c[:,i] for i in list_pols]) <= r_cost

    obj = obj_exp_value
   
    constraints = [cost_constraint] + [P[i] >= 0 for i in list_pols] + [np.sum([P[i] for i in list_pols]) == 1]
    
    prob = cvx.Problem(cvx.Maximize(obj), constraints=constraints)
    prob.solve(verbose=False, solver=cvx.ECOS)
    
    # print([P[i].value for i in list_pols])
    
    opt_alloc = np.stack([P[i].value for i in list_pols], axis=1)
    return opt_alloc



def solve_opt_pb_mi(mu_0_v, mu_0_c, r_cost, nbucket, npol):
    '''
    Mixed-integer linear programming solution (hard allocation) for maximizing value with cost constraint.
    '''
    list_pols = range(npol)

    P = {i:cvx.Variable(nbucket, boolean=True) for i in list_pols}

    obj_exp_value = cvx.sum([P[i] @ mu_0_v[:,i] for i in list_pols])
    cost_constraint = cvx.sum([P[i] @ mu_0_c[:,i] for i in list_pols]) <= r_cost

    obj = obj_exp_value
    constraints = [cost_constraint] + [np.sum([P[i] for i in list_pols]) == 1]
    prob = cvx.Problem(cvx.Maximize(obj), constraints=constraints)
    prob.solve(verbose=False, solver=cvx.GLPK_MI)
    opt_alloc = np.stack([P[i].value for i in list_pols], axis=1)
    return opt_alloc