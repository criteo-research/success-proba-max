import numpy as np
import jax.numpy as jnp

def create_bs(mu_0, Sigma_0, nbucket, npol, size=100000):
    '''
    For each pair of (bucket, policy) create normal distributions with corresponding mu and Sigma.
    '''
    y_bs = np.zeros_like(mu_0, dtype=object)
    
    for b in range(nbucket):
        for p in range(npol):
            y_bs[b][p] = np.random.normal(
                loc=mu_0[b][p], 
                scale=np.sqrt(Sigma_0[b][p]), 
                size=size
            )
    
    return y_bs

def get_synth_data_jax(mu_v, mu_c, Sigma_v, Sigma_c, rho):
    
    mu = jnp.stack((mu_v, mu_c)).transpose((1,2,0))
    Sigma = jnp.stack((Sigma_v, Sigma_c)).transpose((1,2,0))
    
    cov = jnp.zeros((mu_v.shape[0], mu_v.shape[1], 4))
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            cov = cov.at[i,j,0].set(Sigma[i,j,0])
            cov = cov.at[i,j,1].set(rho*jnp.sqrt(Sigma[i][j][0])*jnp.sqrt(Sigma[i][j][1]))
            cov = cov.at[i,j,2].set(rho*jnp.sqrt(Sigma[i][j][0])*jnp.sqrt(Sigma[i][j][1]))
            cov = cov.at[i,j,3].set(Sigma[i,j,1])
    
    return mu, cov


# def get_synth_data_numpy(mu_v, mu_c, Sigma_v, Sigma_c, rho):
    
#     mu = np.stack((mu_v, mu_c)).transpose((1,2,0))
#     Sigma = np.stack((Sigma_v, Sigma_c)).transpose((1,2,0))
    
#     cov = np.zeros((mu_v.shape[0], mu_v.shape[1], 4))
#     for i in range(cov.shape[0]):
#         for j in range(cov.shape[1]):
#             cov[i,j,0] = Sigma[i,j,0]
#             cov[i,j,1] = rho*jnp.sqrt(Sigma[i][j][0])*jnp.sqrt(Sigma[i][j][1])
#             cov[i,j,2] = rho*jnp.sqrt(Sigma[i][j][0])*jnp.sqrt(Sigma[i][j][1])
#             cov[i,j,3] = Sigma[i,j,1]
    
#     return mu, cov