"""Plot helpping module.
"""
from matplotlib.animation import ArtistAnimation
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import tqdm

import diskflowsim2 as dfs2


def close():
    plt.close()


def plot_animation(xs, savename=None, show=False, verbose=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    frames = []
    vmin, vmax = np.unique(xs)[[1, -1]]  # Avoid array[0] == 0
    for i in tqdm.trange(xs.shape[0], disable=not(verbose)):
        frame = ax.imshow(xs[i], cmap="magma", vmax=vmax, vmin=vmin)
        frames.append([frame])
    ax.axis("off")
    fig.tight_layout()
    ani = ArtistAnimation(fig, frames, interval=100)
    if savename is not None:
        ani.save(savename)
    if show:
        plt.show()
    plt.close()


def plot_animation_multiple(xs, titles=None, interval=100) -> FuncAnimation:
    """Plot animation.

    Args:
        x: shape = (num_frames, num_targets, num_layers, num_segments)
        `num_targes` equals number of panels.

    Return:
        anim: FuncAnimation object.
    """
    xs = np.asarray(xs)
    num_frames, num_panels = xs.shape[:2]

    # Only show triagular areas.
    mask = np.ones(xs.shape[-2:])
    mask = dfs2.arrange_diskshape(mask)
    mask = np.where(mask == 0, np.nan, 1)
    xs = xs * mask

    fig, ax = plt.subplots(1, num_panels, figsize=(6*num_panels, 6))
    fig.tight_layout()

    # Compute maximum and minimum for each panel.
    vmaxs = np.nanmax(xs, axis=(0, 2, 3))
    vmins = np.nanmin(xs, axis=(0, 2, 3))

    def update(frame):
        for j in range(num_panels):
            ax[j].cla()
            plot_snapshot(xs[frame][j], ax[j], vmaxs[j], vmins[j],
                          verbose=False)
        if titles is not None:
            if len(titles) != num_panels:
                raise ValueError("'len(titles)' must match 'num_panels'")
            for j in range(num_panels):
                ax[j].set_title(titles[j])
        fig.suptitle(f"Frame {frame}")

    anim = FuncAnimation(fig, update, frames=num_frames,
                         interval=interval, repeat=False)

    return anim


def plot_snapshot(x, ax, vmax, vmin=0, verbose=False):
    ax.imshow(x, cmap="magma", vmax=vmax, vmin=vmin)
    ax.axis("off")
    return ax


def plot_snapshot_disk(x, ax, vmax, vmin=0, verbose=False):
    num_anulus, num_segments = x.shape
    r_out, r_in = num_segments, num_segments - num_anulus
    size = (r_out - r_in) / num_anulus

    cmap = plt.colormaps["magma"]

    x = (x - vmin) / (vmax - vmin)
    for i, anulus in enumerate(tqdm.tqdm(x, disable=not(verbose))):
        radius = int(r_out - i * size)
        anulus_true = anulus[:radius]
        colors = cmap(anulus_true)
        fractions = np.ones(anulus.shape)
        ax.pie(fractions, radius=radius, colors=colors,
               wedgeprops=dict(width=size))
    ax.set_xlim(-r_out, r_out)
    ax.set_ylim(-r_out, r_out)

    return ax


def plot_curves(times, radiations, num_slices, savename=None, show=False):
    num_layers = radiations.shape[1]

    fig, ax = plt.subplots(num_slices + 1, sharex="col")
    ax[0].plot(times, radiations.sum(axis=(1, 2)))
    ax[0].set_ylabel("total")

    jump = num_layers // (num_slices)
    for i in range(num_slices):
        curve = radiations[:, i*jump:(i+1)*jump].sum(axis=(1, 2))
        ax[i+1].plot(times, curve)
        ax[i+1].set_ylabel(f"layer {i}")
    ax[-1].set_xlabel("time")

    fig.align_ylabels()
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, dpi=150)
    if show:
        plt.show()


def save_animation(anim, savename):
    anim.save(savename)


def save_figure(fig, savename, dpi=150):
    fig.savefig(savename, dpi=dpi)


def show():
    plt.show()
