"""
"""
import os
import sys

import numpy as np

import diskflowsim2 as dfs2
import plotting
import utils

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def do_simulation(p_init, ws, num_iter):
    potentials = np.zeros((num_iter, *p_init.shape))
    radiations = np.zeros((num_iter, *p_init.shape))
    p = p_init
    for i in range(num_iter):

        num_cells_r, num_cells_t = p.shape

        p[0, 0] = np.exp(0.5 * np.random.randn(1))

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


def main():

    np.random.seed(0)

    p_init = np.zeros((97, 100))
    p_init = dfs2.arrange_diskshape(p_init)

    w_seed = 11
    if w_seed is None:
        # [dp_s, dp_r, dp_t, de]
        w = [[0.050, -0.20, 0.000, 0.400],  # p
             [0.100, 0.200, 0.200, 0.000]]  # r
        w_seed = "NSD"
    else:
        print(f"use w_seed = {w_seed}")
        with utils.change_seed_temp(seed=w_seed):
            w = np.abs(np.random.randn(2, 4))
            signs = np.array([[1, -1, 0, 1], [1, 1, 1, 0]])
            w = w * signs
    w /= np.sum(w)
    np.set_printoptions(3)
    print(w)

    num_iter = 1000
    potentials, radiations = do_simulation(p_init, w, num_iter)

    num_seqs = radiations.shape[0]
    times = np.arange(num_seqs)
    plotting.plot_curves(times, radiations, 3, show=True,
                         savename=f"figs/curve_seed{w_seed:0>3}.png")

    xs = [[p, r] for p, r in zip(potentials, radiations)]
    titles = ["Potential", "Radiation"]
    anim = plotting.plot_animation_multiple(xs, titles)
    plotting.save_animation(anim, f"figs/animation_seed{w_seed:0>3}.gif")


if __name__ == "__main__":
    main()
