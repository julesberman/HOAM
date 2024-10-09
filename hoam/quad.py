
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_legendre

def get_simpson_quadrature(N):
    if (N - 1) % 2 != 0:
        raise ValueError("Number of intervals N - 1 must be even for Simpson's rule (N must be odd).")

    # Generate N equispaced points on [0,1]
    points = np.linspace(0, 1, N)
    h = 1 / (N - 1)  # Step size

    # Initialize weights
    weights = np.zeros(N)
    weights[0] = h / 3
    weights[-1] = h / 3

    # Apply Simpson's coefficients
    for i in range(1, N - 1):
        if i % 2 == 0:
            weights[i] = 2 * h / 3
        else:
            weights[i] = 4 * h / 3

    points = jnp.asarray(points)
    weights = jnp.asarray(weights)

    return points, weights

def get_gauss_quadrature(n, a=0, b=1):
    points, weights = roots_legendre(n)
    points = 0.5 * (points + 1) * (b - a) + a
    weights = weights * 0.5 * (b - a)
    points = jnp.asarray(points)
    weights = jnp.asarray(weights)
    return points, weights