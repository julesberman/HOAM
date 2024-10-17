import random

import flax
import haiku as hk
import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev, jvp, vmap
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from einops import rearrange

def init_net(net, input_dim, key=None):
    if key is None:
        key = jax.random.PRNGKey(random.randint(0, 10_000))
    pt = jnp.zeros(input_dim)
    theta_init = net.init(key, pt)
    f = net.apply
    return theta_init, f


def split(theta_phi, filter_list):

    if not isinstance(filter_list, list):
        filter_list = [filter_list]

    def filter_rn(m, leaf_key, p):
        return leaf_key in filter_list

    _, theta_phi = flax.core.pop(theta_phi, 'params')
    phi, theta = hk.data_structures.partition(filter_rn, theta_phi)

    phi = {'params': phi}
    theta = {'params': theta}
    return phi, theta


def merge(phi, theta):
    theta_phi = hk.data_structures.merge(phi['params'], theta['params'])
    theta_phi = {'params': theta_phi}
    return theta_phi

def hess_trace_estimator(fn, argnum=0, diff='rev'):

    if diff == 'fwd':
        d_fn = jacfwd(fn, argnums=argnum)
    else:
        d_fn = jacrev(fn, argnums=argnum)

    def estimator(key, *args, **kwargs):
        args = list(args)
        primal = args[argnum]
        eps = jax.random.normal(key, shape=primal.shape)

        def s_dx_wrap(x):
            return d_fn(*args[:argnum], x, *args[argnum+1:], **kwargs)
        dx_val, jvp_val = jvp(s_dx_wrap, (primal,), (eps,))
        trace = jnp.dot(eps, jvp_val)
        return dx_val, trace

    return estimator



def meanvmap(f, mean_axes=(0,), in_axes=(0,)):
    return lambda *fargs, **fkwargs: jnp.mean(vmap(f, in_axes=in_axes)(*fargs, **fkwargs), axis=mean_axes)


def tracewrap(f, axis1=0, axis2=1):
    return lambda *fargs, **fkwargs: jnp.trace(f(*fargs, **fkwargs), axis1=axis1, axis2=axis2)

def get_rand_idx(key, N, bs):
    if bs > N:
        bs = N
    idx = jnp.arange(0, N)
    return jax.random.choice(key, idx, shape=(bs,), replace=False)

def pts_array_from_space(space):
    m_grids = jnp.meshgrid(*space,  indexing='ij')
    x_pts = jnp.asarray([m.flatten() for m in m_grids]).T
    return x_pts


def interplate_in_t(sols, true_t, interp_t):
    sols = np.asarray(sols)
    T, N, D = sols.shape

    data_spacing = [np.linspace(0.0, 1.0, n) for n in sols.shape[1:]]
    spacing = [np.squeeze(true_t), *data_spacing]

    gt_f = RegularGridInterpolator(
        spacing, sols, method='linear', bounds_error=True)

    interp_spacing = [np.squeeze(interp_t), *data_spacing]
    x_pts = pts_array_from_space(interp_spacing)
    interp_sols = gt_f(x_pts)

    interp_sols = rearrange(interp_sols, '(T N D) -> T N D', N=N, D=D)
    return interp_sols

def normalize_data(x, axis=None, method='std'):
    if method == '01':
        mm, mx = x.min(axis=axis, keepdims=True), x.max(
            axis=axis, keepdims=True)
        shift, scale = mm, (mx-mm)
    else:
        shift, scale = np.mean(x, axis=axis, keepdims=True),  np.std(
            x, axis=axis, keepdims=True)

    x = (x - shift) / scale
    
    def unnormalize_fn(data):
        return (data * scale) + shift
    
    return x, unnormalize_fn
    
    
def get_hist_single(frame, nx):
    frame = frame.T
    H, x, y = jnp.histogram2d(
        frame[0], frame[1], bins=nx, range=[[0, 1], [0, 1]])
    H = jnp.rot90(H)
    return H


def get_hist_over_time(frame, nx=100):
    return vmap(get_hist_single, (0, None))(frame, nx)
