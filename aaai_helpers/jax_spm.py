import jax.numpy as jnp
from jax import grad, jit, lax, random, nn
from jax.scipy.stats import norm as jnorm
import numpy as np
from matplotlib import pyplot as plt

from jax import jacfwd, jacrev, jacobian, hessian, vmap, random, pmap
from jax.experimental import optimizers as jax_opt
from jax.scipy.special import erf
from jax.scipy.stats.norm import cdf as cdf1d

@jit
def _simplex_projection(x: jnp.ndarray) -> jnp.ndarray:
  """Projection onto the unit simplex."""
  s = 1.0
  n_features = x.shape[0]
  u = jnp.sort(x)[::-1]
  cumsum_u = jnp.cumsum(u,axis=0)
  ind = jnp.arange(n_features) + 1
  # print(ind.shape, u.shape, cumsum_u.shape)
  cond = s / ind + (u - cumsum_u / ind) > 0
  idx = jnp.count_nonzero(cond)
  res = nn.relu(s / idx + (x - cumsum_u[idx - 1] / idx))
  return res

@jit
def simplex_projection(x):
    proj = jnp.zeros_like(x)
    for i in range(x.shape[0]):
        proj = proj.at[i, :].set(_simplex_projection(x[i]))
    return proj



def build_criteria_1d(sigma, mu, r):
    def criteria(x):
        avg = jnp.sum(mu * x)
        var = jnp.sum(sigma * x)
        # return jnorm.sf(r, loc=avg, scale=jnp.sqrt(var))
        return 1. - jnorm.cdf(r, loc=avg, scale=jnp.sqrt(var))
    return jit(criteria)

@jit
def optimize_1d(sigma, mu, r, x_0, n_steps=100, eta = 0.1):
    criteria = build_criteria_1d(sigma, mu, r)
    def body_fun(i, x):
        g = grad(criteria)(x)
        return simplex_projection(x + eta * g/(criteria(x)+1e-7))
    return lax.fori_loop(0, n_steps, body_fun, x_0)



@jit
def optimize_1d_v2(sigma, mu, r, x_0s, n_steps=100, eta = 0.1):
    criteria = build_criteria_1d(sigma, mu, r)
    # @jit
    def body_fun(i,x):
        g = grad(criteria)(x)
        return simplex_projection(x + eta * g / (criteria(x) + 1e-7))
    # @jit
    # def gradient_loop(x_0):
    #     return lax.fori_loop(0, n_steps, body_fun, x_0)

#     optimized_x_0s = vmap(gradient_loop)(x_0s)
#     criterion_values = vmap(criteria)(optimized_x_0s)
#     best_index = jnp.argmax(criterion_values)

#     return optimized_x_0s[best_index]
    crit_values = vmap(criteria)(x_0s)
    best_index = jnp.argmax(crit_values)
    return lax.fori_loop(0, n_steps, body_fun, x_0s[best_index])


def optimize_1d_debug(sigma, mu, r, x_0, n_steps=200, eta = 0.1, verbose=1):
    x = x_0
    criteria = build_criteria_1d(sigma, mu, r)
    res = [x[0]]
    for i in range(n_steps):
        g = grad(criteria)(x)/(criteria(x)+1e-7)
        if verbose:
            print(i, 'crit:', criteria(x))
            # print('grad:', g)
            print('grad_norm:', jnp.linalg.norm(g))
        x = simplex_projection(x + eta * g)
        res.append(np.array(x[0]))
    return res#x









@jit
def linear_sum(x,y):
    return jnp.sum(jnp.expand_dims(x, axis=-1 ) * y, axis=(0,1))


def sample_outcome(x,avg,cov,n_samples=10_000):
    d = len(avg[0,0])
    agg_avg = linear_sum(x,avg)
    agg_cov = linear_sum(x,cov)
    agg_cov = agg_cov.reshape(d,d)
    samples = random.multivariate_normal(random.PRNGKey(0), agg_avg, agg_cov, shape=(n_samples,))
    return samples

@jit
def case1(p, q, rho, a, b):
    c1 = -1.0950081470333
    c2 = -0.75651138383854
    sqrt2 = jnp.sqrt(2)
    a2 = a * a
    b2 = b * b

    line11 = 0.5 * (erf(q / sqrt2) + erf(b / (sqrt2 * a)))
    aux1 = a2 * c1 * c1 - 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line12 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux1 / aux2)

    line21 = 1 - erf((sqrt2 * b - a2 * c1) / (2 * a * jnp.sqrt(1 - a2 * c2)))
    aux3 = a2 * c1 * c1 + 2 * sqrt2 * b * c1 + 2 * b2 * c2
    line22 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux3 / aux2)

    aux4 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 - a * c1
    aux5 = 2 * jnp.sqrt(1 - a2 * c2)
    line31 = erf(aux4 / aux5)
    line32 = erf((a2 * c1 + sqrt2 * b) / (2 * a * jnp.sqrt(1 - a2 * c2)))

    return line11 + (line12 * line21) - (line22 * (line31 + line32))


@jit
def case2(p, q):
    return cdf1d(p) * cdf1d(q)


@jit
def case3(p, q, rho, a, b):
    c1 = -1.0950081470333
    c2 = -0.75651138383854
    sqrt2 = jnp.sqrt(2)
    a2 = a * a
    b2 = b * b

    aux1 = a2 * c1 * c1 - 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line11 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux1 / aux2)

    aux3 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 + a * c1
    aux4 = 2 * jnp.sqrt(1 - a2 * c2)
    line12 = 1.0 + erf(aux3 / aux4)

    return line11 * line12


@jit
def case4(p, q, rho, a, b):
    c1 = -1.0950081470333
    c2 = -0.75651138383854
    sqrt2 = jnp.sqrt(2)
    a2 = a * a
    b2 = b * b

    line11 = 0.5 + 0.5 * erf(q / sqrt2)
    aux1 = a2 * c1 * c1 + 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line12 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux1 / aux2)

    aux3 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 - a * c1
    aux4 = 2 * jnp.sqrt(1 - a2 * c2)
    line2 = 1.0 + erf(aux3 / aux4)

    return line11 - (line12 * line2)


@jit
def case5(p, q, rho, a, b):
    c1 = -1.0950081470333
    c2 = -0.75651138383854
    sqrt2 = jnp.sqrt(2)
    a2 = a * a
    b2 = b * b

    line11 = 0.5 - 0.5 * erf(b / (sqrt2 * a))
    aux1 = a2 * c1 * c1 + 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line12 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux1 / aux2)

    line21 = 1 - erf((sqrt2 * b + a2 * c1) / (2 * a * jnp.sqrt(1 - a2 * c2)))
    aux3 = a2 * c1 * c1 - 2 * sqrt2 * b * c1 + 2 * b2 * c2
    line22 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux3 / aux2)

    aux4 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 + a * c1
    aux5 = 2 * jnp.sqrt(1 - a2 * c2)
    line31 = erf(aux4 / aux5)
    line32 = erf((-a2 * c1 + sqrt2 * b) / (2 * a * jnp.sqrt(1 - a2 * c2)))

    return line11 - (line12 * line21) + line22 * (line31 + line32)


def bivn_cdf_jax(x1, x2, mu1, mu2, sigma1, sigma2, rho):
    p = (x1 - mu1) / sigma1
    q = (x2 - mu2) / sigma2

    a = -rho / jnp.sqrt(1 - rho * rho)
    b = p / jnp.sqrt(1 - rho * rho)

    mask1 = jnp.where((a > 0) & (a * q + b >= 0), 1, 0)
    mask2 = jnp.where(a == 0, 1, 0)
    mask3 = jnp.where((a > 0) & (a * q + b < 0), 1, 0)
    mask4 = jnp.where((a < 0) & (a * q + b >= 0), 1, 0)
    mask5 = jnp.where((a < 0) & (a * q + b < 0), 1, 0)
    
    return mask1*case1(p, q, rho, a, b) + mask2*case2(p, q) + mask3*case3(p, q, rho, a, b) + mask4*case4(p, q, rho, a, b) + mask5*case5(p, q, rho, a, b)


def build_criteria_my_v2(cov, avg, r_v, r_c):
    def criteria(x):
        
        d = len(avg[0,0])
        agg_avg = linear_sum(x,avg)
        agg_cov = linear_sum(x,cov)
        agg_cov = agg_cov.reshape(d,d)
        
        sigma_v = jnp.sqrt(agg_cov[0,0])
        sigma_c = jnp.sqrt(agg_cov[1,1])
        
        rho = agg_cov[0,1] / (sigma_v * sigma_c)
        
        cdf_v_c = bivn_cdf_jax(r_v, r_c, agg_avg[0], agg_avg[1], sigma_v, sigma_c, rho+1e-7)### max(rho,1e-7)
        
        cdf_c = jnorm.cdf(r_c, loc=agg_avg[1], scale=sigma_c)
        res = cdf_c - cdf_v_c
        return res
    
    return jit(criteria)


@jit
def optimize_2d_my_v2(cov, avg, r_v, r_c, x_0, n_steps=200, eta = 0.1):
    criteria = build_criteria_my_v2(cov, avg, r_v, r_c)
    
    # crit = criteria(x_0)
    # def cond(t):
    #     return t[1] == 0.
    # def body(t):
    #     x_new = jnp.array(np.random.dirichlet(alpha=np.ones(x_0.shape[1]), size=x_0.shape[0]))
    #     crit = criteria(x_new)
    #     return (x_new, crit)
    # return lax.while_loop(cond, body, (x_0,crit))[0]
        
    def body_fun(i, x):
        g = grad(criteria)(x)
        return simplex_projection(x + eta * g/(criteria(x)+1e-7))
    return lax.fori_loop(0, n_steps, body_fun, x_0)


@jit
def optimize_2d_my_v4(cov, avg, r_v, r_c, x_0s, n_steps=200, eta=0.1):
    criteria = build_criteria_my_v2(cov, avg, r_v, r_c)
    # @jit
    def body_fun(i,x):
        g = grad(criteria)(x)
        return simplex_projection(x + eta * g / (criteria(x) + 1e-7))
    # @jit
    # def gradient_loop(x_0):
    #     return lax.fori_loop(0, n_steps, body_fun, x_0)
    
    crit_values = vmap(criteria)(x_0s)
    best_index = jnp.argmax(crit_values)
    return lax.fori_loop(0, n_steps, body_fun, x_0s[best_index])
    
    
def optimize_2d_my_v2_debug(cov, avg, r_v, r_c, x_0, n_steps=200, eta = 0.1, verbose=1):
    x = x_0
    criteria = build_criteria_my_v2(cov,avg,r_v,r_c)
    for _ in range(n_steps):
        g = grad(criteria)(x)/(criteria(x)+1e-7)
        if verbose:
            print('crit:', criteria(x))
            # print('grad:', g)
            print('grad_norm:', jnp.linalg.norm(g))
        x = simplex_projection(x + eta * g)
    return x
