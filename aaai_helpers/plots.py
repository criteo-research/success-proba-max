import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from aaai_helpers.jax_spm import linear_sum
    
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs
    )

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_data_2d(mu, cov, r_v, r_c, nbucket, npol):
    for b in range(nbucket):
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.gca().set_prop_cycle(plt.cycler("color", plt.cm.rainbow(np.linspace(0, npol, num=npol)/npol)))
        
        for p in range(npol):
            color = next(ax._get_lines.prop_cycler)["color"]
            y_bs = np.random.multivariate_normal(mean=mu[b][p], cov=cov[b][p].reshape(2,2), size=10000)
            # ax.scatter(y_bs[:,1], y_bs[:,0], label='p'+str(p), color=color, s=10, alpha=0.5)
            ax.scatter(np.mean(y_bs[:,1]), np.mean(y_bs[:,0]), marker="*", color=color, label='mean_p'+str(p), s=100)
            confidence_ellipse(y_bs[:,1], y_bs[:,0], ax=ax, n_std=3, edgecolor=color, alpha=0.5, label='p'+str(p)+'_3std')

        # plt.axhline(r_v, c='gray', label='r_v')
        # plt.axvline(r_c, c='gray', linestyle='--', label='r_c')
        
        plt.xlabel('r_c')
        plt.ylabel('r_v')
        plt.grid(alpha=0.2, ls="-.")
        plt.legend()
        plt.show()
        
        
def plot_eval_2d(mu, cov, allocs_dict, r_v, r_c, totals_ref=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    n_allocs = len(allocs_dict)
    plt.gca().set_prop_cycle(plt.cycler("color", plt.cm.rainbow(np.linspace(0, n_allocs, num=n_allocs)/n_allocs)))
    
    for alloc in allocs_dict.keys():
        color = next(ax._get_lines.prop_cycler)["color"]
        alloc_res_bs = np.random.multivariate_normal(mean=linear_sum(allocs_dict[alloc], mu), cov=linear_sum(allocs_dict[alloc], cov).reshape(2,2), size=10000)
        # ax.scatter(alloc_res_bs[:,1], alloc_res_bs[:,0], label=alloc, color=color, s=10, alpha=0.5)
        ax.scatter(np.mean(alloc_res_bs[:,1]), np.mean(alloc_res_bs[:,0]), marker="*", color=color, s=100)
        confidence_ellipse(alloc_res_bs[:,1], alloc_res_bs[:,0], ax=ax, n_std=3, edgecolor=color, alpha=0.5, label=alloc+'_3std')

    
    plt.axhline(r_v, c='gray', label='r_v')
    plt.axvline(r_c, c='gray', linestyle='--', label='r_c')
    
    if totals_ref is not None:
        plt.axhline(totals_ref[0], c='gray', label='v_ref', alpha=0.3)
        plt.axvline(totals_ref[1], c='gray', linestyle='--', label='c_ref', alpha=0.3)
        
    

    plt.legend()
    plt.grid(alpha=0.2, ls="-.")
    plt.show()