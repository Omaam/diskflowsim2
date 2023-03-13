"""Plot helpping module.
"""
from matplotlib.animation import ArtistAnimation
import matplotlib.pyplot as plt
import tqdm


def plot_animation(xs, savename=None, show=False, verbose=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    frames = []
    for i in tqdm.trange(xs.shape[0], disable=not(verbose)):
        frame = ax.imshow(xs[i], cmap="jet",
                          vmax=xs.max(), vmin=0)
        frames.append([frame])
    ax.axis("off")
    fig.tight_layout()
    ani = ArtistAnimation(fig, frames, interval=100)
    if savename is not None:
        ani.save(savename)
    if show:
        plt.show()
    plt.close()


def plot_snapshot(potential, ax, verbose=False):
    num_anulus, num_segments = potential.shape
    r_out, r_in = num_segments, num_segments - num_anulus
    size = (r_out - r_in) / num_anulus

    cmap = plt.colormaps["jet"]

    for i, anulus in enumerate(tqdm.tqdm(potential,
                                         disable=not(verbose))):
        radius = int(r_out - i * size)
        anulus_true = anulus[:radius]
        colors = cmap(anulus_true)
        fractions = (anulus_true > 0).astype(int)
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
