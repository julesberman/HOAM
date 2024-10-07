from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable


def imshow_movie(sol, frames=50, t=None, interval=100, tight=False, title='', cmap='viridis', aspect='equal', live_cbar=False, save_to=None, show=True, fps=10):

    fig, ax = plt.subplots()
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')

    cv0 = sol[0]
    # Here make an AxesImage rather than contour
    im = ax.imshow(cv0, cmap=cmap, aspect=aspect)
    cb = fig.colorbar(im, cax=cax)
    tx = ax.set_title('Frame 0')
    vmax = np.max(sol)
    vmin = np.min(sol)
    ax.set_xticks([])
    ax.set_yticks([])
    if tight:
        plt.tight_layout()

    def animate(frame):
        arr, t = frame
        im.set_data(arr)
        if live_cbar:
            vmax = np.max(arr)
            vmin = np.min(arr)
            im.set_clim(vmin, vmax)
        tx.set_text(f'{title} t={t:.2f}')

    time, w, h = sol.shape
    if t is None:
        t = np.arange(time)
    inc = max(time//frames, 1)
    sol_frames = sol[::inc]
    t_frames = t[::inc]
    frames = list(zip(sol_frames, t_frames))
    ani = FuncAnimation(fig, animate,
                        frames=frames, interval=interval,)
    plt.close()

    if save_to is not None:
        p = Path(save_to).with_suffix('.gif')
        ani.save(p, writer='pillow', fps=fps)

    if show:
        return HTML(ani.to_jshtml())


def scatter_movie(pts, c='r', size=None, xlim=None, ylim=None, alpha=1, frames=60, t=None, title='', interval=100, save_to=None, show=True, fps=10):
    pts = rearrange(pts, 't n d -> t d n')
    fig, ax = plt.subplots()

    sct = ax.scatter(x=pts[0, 0], y=pts[0, 1], alpha=alpha, s=size, c=c)
    mm = pts.min(axis=(0, 2))
    mx = pts.max(axis=(0, 2))

    if xlim is None:
        xlim = [mm[0], mx[0]]
    if ylim is None:
        ylim = [mm[1], mx[1]]
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    tx = ax.set_title('Frame 0')

    def animate(frame):
        scatter, t = frame
        sct.set_offsets(scatter.T)
        tx.set_text(f'{title} t={t:.2f}')

    time = len(pts)
    if t is None:
        t = np.arange(time)
    inc = max(time//frames, 1)
    t_frames = t[::inc]
    pts = pts[::inc]

    frames = list(zip(pts, t_frames))
    ani = FuncAnimation(fig, animate,
                        frames=frames, interval=interval,)
    plt.close()

    if save_to is not None:
        p = Path(save_to).with_suffix('.gif')
        ani.save(p, writer='pillow', fps=fps)

    if show:
        return HTML(ani.to_jshtml())

