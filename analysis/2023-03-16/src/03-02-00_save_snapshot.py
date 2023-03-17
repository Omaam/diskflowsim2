"""Run simulation.
"""
from concurrent import futures
import os

from absl import app
from absl import flags
from absl import logging
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import numpy as np

import diskflowsim2 as dfs2
import plotting


FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", None, "")
logging.set_verbosity(logging.FATAL)


def save_potential_radiation(frame, p, r, savepath, vminmax_p, vminmax_r):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.tight_layout()
    fig.suptitle(f"Frame {frame}")
    plotting.plot_snapshot_disk(p, ax[0], vminmax_p[1], vminmax_p[0])
    plotting.plot_snapshot_disk(r, ax[1], vminmax_r[1], vminmax_r[0])
    ax[0].set_title("Potential")
    ax[1].set_title("Radiation")

    plt.savefig(savepath)
    plt.close()
    print(f"save {savepath}")


def save_anim_multiprocess(xs: ArrayLike, basename: str,
                           savedir: str, extension: str = "png"):

    xs = np.asarray(xs)
    xs = np.log10(1 + xs)

    vminmax_p = [np.nanmin(xs[:, 0]), np.nanmax(xs[:, 0])]
    vminmax_r = [np.nanmin(xs[:, 1]), np.nanmax(xs[:, 1])]

    with futures.ProcessPoolExecutor(10) as executor:
        for frame, x in enumerate(xs):
            p, r = x
            savename = f"{basename}_{frame:03d}." + extension
            savepath = os.path.join(savedir, savename)
            executor.submit(save_potential_radiation, frame, p, r,
                            savepath, vminmax_p, vminmax_r)


def do_simulation(p_init, ws, num_iter):
    potentials = np.zeros((num_iter, *p_init.shape))
    radiations = np.zeros((num_iter, *p_init.shape))
    p = p_init
    for i in range(num_iter):

        num_cells_r, num_cells_t = p.shape

        p[0, 0] = np.exp(0.01 * np.random.randn(1))

        r = np.arange(num_cells_t, num_cells_t - num_cells_r, -1)
        r = np.broadcast_to(r[:, None], p.shape)

        y = np.stack([p, r], axis=2)[..., None] * ws
        y = np.sum(y, axis=2)
        ratios = dfs2.softmax(y)

        dp = p[..., None] * ratios

        dp_s = dp[..., 0]

        # Radial direction.
        dp_r = dp[..., 1]
        dp_r = dfs2.propagate_conv(dp_r)
        dp_r[0] = 0

        # Transverse direction.
        dp_t = dp[..., 2]
        dp_t = np.roll(dp_t, shift=1, axis=1)
        # Insert diagonal conponent to head in layers.
        dp_t[:, 0] = np.diag(np.fliplr(dp_t))

        p = dp_s + dp_r + dp_t
        p = np.fliplr(np.triu(np.fliplr(p)))

        potentials[i] = p

        radiation = dfs2.arrange_diskshape(dp[..., 3])
        radiations[i] = radiation

    return potentials, radiations


def print_weights(names, values):
    max_len = max([len(n) for n in names])
    for n, v in zip(names, values):
        v = round(v, 3)
        print(f"{n:<{max_len + 1}}:", f"{v:>5}")


def main(argv):

    seed_str = "NSD"
    now_str = "2023-03-17_10-08-25"
    insertion = f"seed{seed_str}_{now_str}"
    potentials = np.load(f"../data/out/potentials_{insertion}.npy")
    radiations = np.load(f"../data/out/radiations_{insertion}.npy")

    xs = [[p, r] for p, r in zip(potentials, radiations)]

    titles = ["Potential", "Radiation"]
    anim = plotting.plot_animation_multiple(xs, titles)
    animname = f"../figs/anim/animation_{seed_str}_{now_str}.gif"
    plotting.save_animation(anim, animname)

    save_anim_multiprocess(xs, "animation", "anim", "png")

    # utils.save_info("run_info.txt", f"{seed_str:<10}", now_str, w)


if __name__ == "__main__":
    app.run(main)
