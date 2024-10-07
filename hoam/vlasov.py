import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from einops import rearrange
from jax import random as jrandom
from scipy.sparse.linalg import spsolve

"""
Create Your Own Plasma PIC Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the 1D Two-Stream Instability
Code calculates the motions of electron under the Poisson-Maxwell equation
using the Particle-In-Cell (PIC) method

"""


def getAcc(pos, Nx, boxsize, n0, Gmtx, Lmtx):
    """
Calculate the acceleration on each particle due to electric field
    pos      is an Nx1 matrix of particle positions
    Nx       is the number of mesh cells
    boxsize  is the domain [0,boxsize]
    n0       is the electron number density
    Gmtx     is an Nx x Nx matrix for calculating the gradient on the grid
    Lmtx     is an Nx x Nx matrix for calculating the laplacian on the grid
    a        is an Nx1 matrix of accelerations
    """
    # Calculate Electron Number Density on the Mesh by
    # placing particles into the 2 nearest bins (j & j+1, with proper weights)
    # and normalizing
    N = pos.shape[0]
    dx = boxsize / Nx
    j = np.floor(pos/dx).astype(int)
    jp1 = j+1
    weight_j = (jp1*dx - pos)/dx
    weight_jp1 = (pos - j*dx)/dx
    j = np.mod(j, Nx)   # periodic BC
    jp1 = np.mod(jp1, Nx)   # periodic BC

    n = np.bincount(j[:, 0],   weights=weight_j[:, 0],   minlength=Nx)
    n += np.bincount(jp1[:, 0], weights=weight_jp1[:, 0], minlength=Nx)
    n *= n0 * boxsize / N / dx
    n_eff = n - n0
    n_eff[-1] = 0
    n_eff = n - n0
    n_eff[-1] = 0

    # Solve Poisson's Equation: laplacian(phi) = n-n0
    phi_grid = spsolve(Lmtx, n_eff, permc_spec="MMD_AT_PLUS_A")

    # Apply Derivative to get the Electric field
    E_grid = - Gmtx @ phi_grid

    # Interpolate grid value onto particle locations
    E = weight_j * E_grid[j] + weight_jp1 * E_grid[jp1]
    # interpolate grid value onto particle locations
    phi = weight_j * phi_grid[j] + weight_jp1 * phi_grid[jp1]

    a = -E

    return a, phi


def get_gradient_matrix(Nx, boxsize):
    dx = boxsize/Nx
    e = np.ones(Nx)
    diags = np.array([-1, 1])
    vals = np.vstack((-e, e))
    Gmtx = sp.spdiags(vals, diags, Nx, Nx)
    Gmtx = sp.lil_matrix(Gmtx)
    Gmtx[0, Nx-1] = -1
    Gmtx[Nx-1, 0] = 1
    Gmtx /= (2*dx)
    Gmtx = sp.csr_matrix(Gmtx)
    return Gmtx

    return Gmtx


def get_laplacian_matrix(Nx, boxsize):
    dx = boxsize/Nx
    e = np.ones(Nx)


def get_laplacian_matrix(Nx, boxsize):
    dx = boxsize/Nx
    e = np.ones(Nx)
    diags = np.array([-1, 0, 1])
    vals = np.vstack((e, -2*e, e))
    Lmtx = sp.spdiags(vals, diags, Nx, Nx)
    Lmtx = sp.lil_matrix(Lmtx)
    Lmtx[0, Nx-1] = 1
    Lmtx[Nx-1, 0] = 1
    Lmtx /= dx**2
    Lmtx[-1, :] = 1/Nx
    Lmtx[-1, :] = 1/Nx
    Lmtx = sp.csr_matrix(Lmtx)
    return Lmtx


def run_vlasov(n_samples, t_eval, mu=1.0, mode='two-stream', eta=0, seed=0):

    np.random.seed(seed=seed)

    dt = t_eval[1] - t_eval[0]
    Nt = len(t_eval)

    # Two-Stream Instability/bump-on-tail Parameters
    # two beams, at velocities v_1 and v_2 with widths vth_1 and vth_2
    # fractions of electrons in each beam are n1 and n2
    # spatial pertubation with amplitude eps, wavenumber kappa
    # collison frequency eta

    # Bump-on-tail:
    # v_1 = -0, v_2 = 4, vth_1 = 1, vth_2 = 0.5, n1 = 0.9, n2 = 0.1, eps = 0.05, mu 1 or larger
    # Two-stream:
    # v_1 = -3, v_2 = 3, vth_1 = 1, vth_2 = 1, n1 = 0.5, n2 = 0.5, eps = 0.05, mu between 0.5 and 1.5 (can be larger or smaller, too)

    Nx = n_samples // 8          # Number of mesh cells
    boxsize = 50                # periodic domain [0,boxsize]

    if mode == 'bump-on-tail':
        v_1 = 0
        v_2 = 4
        vth_1 = 1
        vth_2 = 0.5
        n_1 = 0.9
        n_2 = 0.1
    else:
        v_1 = -3
        v_2 = 3
        vth_1 = 1
        vth_2 = 1
        n_1 = 0.5
        n_2 = 0.5

    n0 = 1            # electron number density
    eps = 0.05
    lamb = mu/1
    v_th_coll = 1  # diffusivity is eta * v_th_coll**2

    n_1 = int(n_1 * n_samples)
    n_2 = n_samples - n_1

    # construct 2 Gaussian beams
    pos_1 = np.random.rand(n_1, 1) * boxsize  # uniform
    pos_2 = np.random.rand(n_2, 1) * boxsize  # uniform
    vel_1 = vth_1 * np.random.randn(n_1, 1) + v_1  # gaussian
    vel_2 = vth_2 * np.random.randn(n_2, 1) + v_2  # gaussian

    pos = np.vstack((pos_1, pos_2))
    vel = np.vstack((vel_1, vel_2))

    # add perturbation
    # pos *= (1 + eps*np.cos(2*np.pi*lamb*pos))
    # pos = np.mod(pos + boxsize, boxsize)
    vel *= (1 + eps*np.cos(2*np.pi*pos/boxsize*lamb))

    # Construct matrix G to compute gradient  (1st derivative)
    Gmtx = get_gradient_matrix(Nx, boxsize)

    # Construct matrix L to computer Laplacian (2nd derivative)
    Lmtx = get_laplacian_matrix(Nx, boxsize) * mu**2

    # calculate initial accelerations
    acc, phi = getAcc(pos, Nx, boxsize, n0, Gmtx, Lmtx)

    # number of timesteps
    sols = np.zeros((Nt, n_samples, 2))
    # Simulation Main Loop
    for i, t in enumerate(t_eval):

        sols[i, :, 0] = np.squeeze(pos)
        sols[i, :, 1] = np.squeeze(vel)
        # sols[i, :, 2] = np.squeeze(phi)
        # sols[i, :, 2] = np.squeeze(phi)
        # (1/2) kick
        vel += acc * dt/2.0

        # drift (and apply periodic boundary conditions)
        pos += vel * dt
        pos = np.mod(pos, boxsize)

        # update accelerations
        acc, phi = getAcc(pos, Nx, boxsize, n0, Gmtx, Lmtx)

        # (1/2) kick
        vel += acc * dt/2.0

        # collisions at the end
        if eta > 0:
            vel -= eta * dt * vel
            vel += eta * v_th_coll**2 * dt**0.5 * \
                np.random.rand(vel.shape[0], 1)

        # update time
        t += dt

    # add two quantties
    sols = rearrange(sols, 'T N D -> T N D')

    return sols
