
import jax
import jax.numpy as jnp
import random

import jax.numpy as jnp
import jax.random
from diffrax import (Euler, MultiTerm, ODETerm, SaveAt,
                     VirtualBrownianTree, ControlTerm,
                     diffeqsolve)
from jax import jit, vmap, jacrev
from einops import rearrange
import lineax

def solve_sde(drift, diffusion, t_eval, get_ic, n_samples, dt=1e-2, key=None):
    t_eval = jnp.asarray(t_eval)

    @jit
    def solve_single(key):
        ikey, skey = jax.random.split(key)
        y0 = get_ic(ikey)
        sol = solve_sde_ic(y0, skey, t_eval, dt, drift, diffusion)
        return sol

    if key is None:
        key = jax.random.PRNGKey(random.randint(0, 1e6))
    keys = jax.random.split(key, num=n_samples)
    solve_single = vmap(solve_single)
    sols = solve_single(keys)

    return sols


def solve_sde_ic(y0, key, t_eval, dt, drift, diffusion):
    t0, t1 = t_eval[0], t_eval[-1]
    brownian_motion = VirtualBrownianTree(
        t0, t1, tol=1e-3, shape=y0.shape, key=key)
    diag_diffusion = lambda *args: lineax.DiagonalLinearOperator(diffusion(*args))
    diffusion_term = ControlTerm(diag_diffusion, brownian_motion)
    terms = MultiTerm(ODETerm(drift), diffusion_term)
    solver = Euler()
    saveat = SaveAt(ts=t_eval)
    sol = diffeqsolve(terms, solver, t0, t1, dt0=dt,
                      y0=y0, saveat=saveat, max_steps=int(1e6))

    return sol.ys


def solve_test_sde(s_fn, params, ics, t_int, dt, epsilon, mu, key):
    s_dx = jacrev(s_fn, 1)

    def drift(t, y, *args):
        t = jnp.asarray([t])
        f = jnp.squeeze(s_dx(mu, y, t, params))
        return f

    def diffusion(t, y, *args):
        return epsilon * jnp.ones_like(y)

    keys = jax.random.split(key, num=len(ics))
    test_sol = vmap(solve_sde_ic, (0, 0, None, None, None, None))(
        ics, keys, t_int, dt, drift, diffusion)
    test_sol = rearrange(test_sol, 'N T D -> T N D')

    return test_sol