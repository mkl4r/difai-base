import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, random
from functools import partial
from jax.lax import fori_loop, squeeze
from jax.debug import print as jaxprint
import numpy as np
import yaml

EPSILON = 1e-12

### JAX
# Unscented Kalman Filter points
def sigma_points_jax(mean, cov, lam=3, alpha=1e-3, beta=2):
    """
    Compute the sigma points for the unscented transform for an *input*
    Gaussian with mean mu and covariance sigma.
    Returns a (D, 2D+1) array, where D is the dimensionality of the state space.    
    """
    n = len(mean)    
    ## EDIT Check diag is positive
    #cov = cov.at[jnp.diag_indices(n)].set(jnp.clip(jnp.diag(cov), min=EPSILON))
    ## 
    sqrt_sigma = jnp.linalg.cholesky((n + lam) * cov)   # needs to be Hermetian positive definite
    sigma_points = jnp.vstack([mean, mean + sqrt_sigma, mean - sqrt_sigma])
    return sigma_points.T

def unscented_weights_jax(n, lam=3, alpha=1e-3, beta=2):
    """
    Compute the weights for the unscented transform.
    Note: these depend on the dimensionality of the state space, but
    not on the function being applied or the current state,
    and so can be pre-computed if you want.
    """
    weights_mean = jnp.ones(2 * n + 1) / (2 * (n + lam))
    weights_cov = jnp.array(weights_mean)
    weights_mean = weights_mean.at[0].set(lam / (n + lam) ) #weights_mean[0] = lam / (n + lam) 
    weights_cov = weights_cov.at[0].set(weights_mean[0] + (1 - alpha**2 + beta))  #weights_cov[0] = weights_mean[0] + (1 - alpha**2 + beta)
    return weights_mean, weights_cov

def unscented_jax(mean, cov, fn, kappa=0, alpha=1e-3, beta=2):
    """
    Take a Gaussian, parameterised by mean and covariance, and apply a nonlinear function to it.
    Return the mean and covariance of the resulting Gaussian.
    fn(x) takes a (N,D) array of N samples of dimension D and returns a (N,D) array of N samples of dimension D.
    lam, alpha, beta are "fiddle factors" and don't usually need much tuning.
    """    
    n = len(mean)
    lam = alpha**2 * (n + kappa) - n
    # get the sigma points
    points = sigma_points_jax(mean, cov, lam, alpha, beta)
    # jaxprint("points {x}", x=points)
    # apply fn
    axis = 0
    #jaxprint(points)
    transformed = jnp.apply_along_axis(fn, 0, points)
    # get the weights
    weights_mean, weights_cov = unscented_weights_jax(n, lam, alpha, beta)
    # jaxprint("weights_mean {x}", x=weights_mean)
    # jaxprint("weights_cov {x}", x=weights_cov)
    # jaxprint("transformed {x}", x=transformed)

    # re-estimate the mean and covariance using the weighted samples
    mean_hat = jnp.dot(transformed, weights_mean)

    def body_fun(i, carry):
        carry += weights_cov[i] * \
                jnp.outer(transformed[:, i] - mean_hat,
                            transformed[:, i] - mean_hat)
        return carry
    cov_hat = fori_loop(0, 2 * n + 1, body_fun, jnp.zeros((n, n)))

    # jaxprint("mean_hat {x}", x=mean_hat)
    # jaxprint("cov_hat {x}", x=cov_hat)

    ## EDIT Check diag is positive
    cov_hat = cov_hat.at[jnp.diag_indices(n)].set(jnp.clip(jnp.diag(cov_hat), min=EPSILON))
    ## 

    

    return mean_hat, cov_hat

def unscented(mean, cov, fn, kappa=3e6, alpha=1e-3, beta=2):
    return unscented_jax(mean, cov, fn, kappa, alpha, beta)


# Jitting
@partial(jit, static_argnums=(2,3,4,5))
def unscented_jit(mean, cov, fn, kappa=3e6, alpha=1e-3, beta=2):
    return unscented_jax(mean, cov, fn, kappa, alpha, beta)


def softmax_jax(x):
    e = jnp.exp(x - x.max())
    return e / e.sum()

def kl_jax(m1, cov1, m2, cov2):
    m1 = jnp.atleast_1d(m1)
    cov1 = jnp.atleast_2d(cov1)
    m2 = jnp.atleast_1d(m2)
    cov2 = jnp.atleast_2d(cov2)
    if len(m1) == 1:
        return kl_normal_normal(m1, jnp.sqrt(cov1).flatten(), m2, jnp.sqrt(cov2).flatten())
    cov2inv = jnp.linalg.inv(cov2)
    n = m1.shape[0]
    kl = 0.5*(jnp.matmul(jnp.transpose(m2-m1),jnp.matmul(cov2inv,(m2-m1))) + 
            jnp.linalg.trace(jnp.matmul(cov2inv,cov1)) + 
            jnp.log(jnp.linalg.det(cov2)/jnp.linalg.det(cov1)) - 
            n)
    return jnp.squeeze(kl)

def kl_normal_normal(loc1, scale1, loc2, scale2):
    kl = (scale1**2 + (loc1-loc2)**2) / (2*scale2**2) + jnp.log(scale2/scale1) - 0.5 # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    return jnp.squeeze(kl)

# From Pytorch 
def kl_uniform_normal(p_low, p_high, loc, scale):
    common_term = p_high - p_low
    t1 = jnp.log((jnp.sqrt(jnp.pi * 2) * scale / common_term))
    t2 = jnp.pow(common_term, 2) / 12
    t3 = jnp.pow(((p_high + p_low - 2 * loc) / 2), 2)
    return t1 + 0.5 * (t2 + t3) / jnp.pow(scale, 2)

def kl_normal_to_sum_of_normals(a, b, mu_Q=0.5, sigma_Q=0.1, num_gaussians=100, sigma=0.01, num_samples=10000, key=None):
    """
    Computes the average KL divergence between a Gaussian distribution Q(x) and each Gaussian in a sum that approximates a uniform distribution.
    
    Parameters:
    - a, b: interval of the uniform distribution.
    - num_gaussians: number of Gaussians used in the sum to approximate the uniform distribution.
    - sigma: standard deviation of each Gaussian in the sum.
    - num_samples: number of samples to use for numerical integration.
    - mu_Q, sigma_Q: mean and standard deviation of the target Gaussian distribution Q(x).
    
    Returns:
    - avg_D_KL: Average KL divergence between Q(x) and each Gaussian P_i(x).
    """
    # Sample x values directly from the target Gaussian distribution Q(x)
    key, use_key = random.split(key)
    x_samples = mu_Q + sigma_Q * random.normal(key=use_key, shape=(num_samples,))
    
    # Define the target Gaussian distribution Q(x) evaluated at the sampled points
    Q_samples = jsp.stats.norm.pdf(x_samples, mu_Q, sigma_Q)
    
    # Compute individual KL divergences between Q(x) and each Gaussian P_i(x)
    means = jnp.linspace(a, b, num_gaussians)
    kl_divergences = jnp.zeros((num_gaussians,))
    
    for (i,mu) in enumerate(means):
        P_i_samples = jsp.stats.norm.pdf(x_samples, mu, sigma)
        epsilon = 1e-10  # small constant to avoid log(0)
        D_KL_i = jnp.mean(jnp.log((Q_samples + epsilon) / (P_i_samples + epsilon)))
        kl_divergences = kl_divergences.at[i].set(D_KL_i)
    
    # Calculate the average KL divergence
    avg_D_KL = jnp.mean(kl_divergences)
    
    return avg_D_KL

# From scipy, adapted
def _cdf_distance(p, u_values, v_values):
    r"""
    Compute, between two one-dimensional distributions :math:`u` and
    :math:`v`, whose respective CDFs are :math:`U` and :math:`V`, the
    statistical distance that is defined as:

    .. math::

        l_p(u, v) = \left( \int_{-\infty}^{+\infty} |U-V|^p \right)^{1/p}

    p is a positive parameter; p = 1 gives the Wasserstein distance, p = 2
    gives the energy distance.

    Parameters
    ----------
    u_values, v_values : array_like
        Values observed in the (empirical) distribution.
    u_weights, v_weights : array_like, optional
        Weight for each value. If unspecified, each value is assigned the same
        weight.
        `u_weights` (resp. `v_weights`) must have the same length as
        `u_values` (resp. `v_values`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.

    Returns
    -------
    distance : float
        The computed distance between the distributions.

    Notes
    -----
    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.

    References
    ----------
    .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
           Munos "The Cramer Distance as a Solution to Biased Wasserstein
           Gradients" (2017). :arXiv:`1705.10743`.

    """
    u_sorter = jnp.argsort(u_values)
    v_sorter = jnp.argsort(v_values)

    all_values = jnp.concatenate((u_values, v_values))
    all_values.sort()

    # Compute the differences between pairs of successive values of u and v.
    deltas = jnp.diff(all_values)

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
    v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

    # Calculate the CDFs of u and v using their weights, if specified.
    u_cdf = u_cdf_indices / u_values.size
    v_cdf = v_cdf_indices / v_values.size

    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
    # of about 15%.
    if p == 1:
        return jnp.sum(jnp.multiply(jnp.abs(u_cdf - v_cdf), deltas))
    if p == 2:
        return jnp.sqrt(jnp.sum(jnp.multiply(jnp.square(u_cdf - v_cdf), deltas)))
    return jnp.power(jnp.sum(jnp.multiply(jnp.power(jnp.abs(u_cdf - v_cdf), p),
                                       deltas)), 1/p)


def wasserstein_distance_uniform_normal(p_low, p_high, loc, scale, n_samples=1000, key=None):
    key, use_key = random.split(key)
    u_values = random.uniform(key=use_key, shape=(n_samples,), minval=p_low, maxval=p_high)
    key, use_key = random.split(key)
    v_values = loc + scale * random.normal(key=use_key, shape=(n_samples,)) 

    return _cdf_distance(1, u_values, v_values)

def wasserstein_distance_normal_normal(loc1, scale1, loc2, scale2, n_samples=1000, key=None):
    key, use_key = random.split(key)
    u_values = loc1 + scale1 * random.normal(key=use_key, shape=(n_samples,))
    key, use_key = random.split(key)
    v_values = loc2 + scale2 * random.normal(key=use_key, shape=(n_samples,)) 

    return _cdf_distance(1, u_values, v_values)

def wasserstein_distance(m1, cov1, m2, cov2):
    cov1sq = jnp.sqrt(cov1)
    w1 = jnp.linalg.norm(m1-m2)**2 
    w2 = jnp.linalg.trace(cov1 + cov2 - 2*jnp.sqrt(jnp.matmul(cov1sq, jnp.matmul(cov2, cov1sq)))) # https://danmackinlay.name/notebook/distance_between_gaussians
    return jnp.sqrt(w1 + w2)

def generate_n_bit_numbers(n):
    """Generate all possible bit numbers for an n-bit number."""
    for i in range(2**n):  # Iterate over all numbers from 0 to 2^n - 1
        yield format(i, f'0{n}b')  # Convert to binary string with leading zeros

def gen_fixed_plans(a_lims, horizon, dim_action):
    plans = []
    for bit_number in generate_n_bit_numbers(dim_action):
        action = jnp.zeros((dim_action,))
        for j in range(dim_action):
            if bit_number[j] == '0':
                action = action.at[j].set(a_lims[0][j])
            else:
                action = action.at[j].set(a_lims[1][j])
        plans.append(jnp.tile(action, (horizon,1)))
    return jnp.array(plans)


def generate_n_base_m_numbers(n, m):
    """Generate all possible numbers for an n-digit number in base m."""
    def to_base_m(num, base, length):
        """Convert a number to a base-m representation with leading zeros."""
        digits = []
        for _ in range(length):
            digits.append(str(num % base))
            num //= base
        return ''.join(reversed(digits))
    
    for i in range(m**n):  # Iterate over all numbers from 0 to m^n - 1
        yield to_base_m(i, m, n)

def mix_exp(x, min, max):
    res = x.copy()
    res = res.at[x>=0].set(jnp.exp(x[x>=0]*jnp.log(max+1))-1)
    res = res.at[x<0].set(-(jnp.exp(-x[x<0]*jnp.log(-min+1))-1))
    return res

def gen_fixed_plans(a_lims, horizon, dim_action, num_plans, uniform=True):
    plans = []

    num_actions_per_dim = int(num_plans**(1/dim_action))
    if num_actions_per_dim**dim_action != num_plans:
        print(f"Note on use_fixed_plans: Using {num_actions_per_dim**dim_action} different plans instead of {num_plans} for action selection (num actions per dim: {num_actions_per_dim}).")
    # base = num_plans_per_dim
    # if base >10:
    #     print(f"The number of plans are too large for the used technique. Using base 10 instead of {base}.")
    #     base = 10
    # plan_ids = generate_n_base_m_numbers(dim_action, base)
    print(f"dim_action: {dim_action}, num_actions_per_dim: {num_actions_per_dim}, num_plans: {num_plans}")
    plan_ids = create_plan_ids(dim_action, num_actions_per_dim)
    actions = create_fixed_actions(a_lims, dim_action, num_actions_per_dim, uniform)

    print(plan_ids)
    print(actions)

    for plan_id in plan_ids:
        action = jnp.zeros((dim_action,))
        for j in range(dim_action):
            action = action.at[j].set(actions[j][int(plan_id[j])])
        plans.append(jnp.tile(action, (horizon,1)))
    return jnp.array(plans)

def create_plan_ids(dim_action, num_actions_per_dim):
    return _create_plan_ids([[]], dim_action, num_actions_per_dim)

def _create_plan_ids(plans, dim_action, num_actions_per_dim):
    if dim_action > 0:
        new_plans = []
        for plan in plans:
            # print(plan)
            for i in range(num_actions_per_dim):
                plan_i = plan.copy()
                plan_i.append(i)
                # print(plan_i)
                new_plans.append(plan_i)
        return _create_plan_ids(new_plans, dim_action-1, num_actions_per_dim)
    else:
        return plans

def create_fixed_actions(a_lims, dim_action, num_actions_per_dim, uniform=True):
    actions = []
    if uniform:
        for j in range(dim_action):
            a = jnp.linspace(a_lims[0][j], a_lims[1][j], num_actions_per_dim)
            actions.append(a)
    else:
        for j in range(dim_action):
            uniform_distributed_values = jnp.linspace(-1, 1, num_actions_per_dim)
            a = mix_exp(uniform_distributed_values, a_lims[0][j], a_lims[1][j])
            actions.append(a)

    return actions

def refactor_noise_params(noise_params):

    noise_id = 0
    dim_noise = 0
    for (noise_name, noise_setting) in noise_params.items():
        assert 'id' in noise_setting.keys(), f"Noise parameter for '{noise_name}' does not contain an 'id' key defining the index/indices of the corresponding noise."
        try:
            num_noise_values = len(noise_setting['id'])
        except TypeError:
            num_noise_values = 1
        if num_noise_values > 1:
            noise_params[noise_name] = [np.array([index for index in range(noise_id, noise_id + num_noise_values)]), noise_setting['id']]
        else:
            noise_params[noise_name] = [jnp.atleast_1d(noise_id)] + [jnp.atleast_1d(noise_setting['id'])]

        if 'value' in noise_setting.keys():
            # Noise settings for generative process
            if num_noise_values > 1:
                noise_params[noise_name].append(noise_setting['value'])
            else:
                dim_noise += 1
                noise_params[noise_name].append(jnp.atleast_1d(noise_setting['value']))
        dim_noise += num_noise_values
        noise_id += num_noise_values
       
    return noise_params, dim_noise

 # Load data from a YAML file
def load_yaml_file(path):
    try:
        with open(path, 'r') as file:
            return yaml.safe_load(file) or {}
    except FileNotFoundError:
        print(f"Warning: File not found: {path}.")
        return {}
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file '{path}': {e}")