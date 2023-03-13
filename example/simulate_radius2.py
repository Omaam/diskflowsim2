"""
"""
import numpy as np

import diskflowsim2 as dfs2
import utils
import plotting


def do_simulation(p_init, w, num_iter):
    potentials = np.zeros((num_iter, *p_init.shape))
    radiations = np.zeros((num_iter, *p_init.shape))
    p = p_init
    for i in range(num_iter):

        num_cells_r, num_cells_t = p.shape

        p[0, 0] = np.exp(0.01*np.random.randn(1))

        # The more a cell has potential, the more the cell gets photons.
        p = p + np.exp(0.001*np.random.randn(*p.shape)) * \
            utils.sigmoid(np.log10(p + 0.00001))

        r = np.arange(num_cells_t, num_cells_t - num_cells_r, -1)
        r2 = r**2

        y = p[..., None] * w
        y += r2[..., None, None] * 0.01 * w
        ratios = dfs2.softmax(y)

        dp = p[..., None] * ratios

        dp_s = dp[..., 0]

        # Radial propagation with potentials and radii.
        dp_r = dp[..., 1]
        dp_r = dfs2.propagate_conv(dp_r)
        dp_r[0] = 0

        # Transversal propagation with potentials.
        dp_t = dp[..., 2]
        dp_t = np.roll(dp_t, shift=1, axis=1)
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

    w_seed = 1
    with utils.change_seed_temp(seed=w_seed):
        # [dp_s, dp_r(p, r^2), dp_t(p, r^2), de]
        # w = [0.05, 0.20, 0.30, 0.10]
        w = np.abs(np.random.randn(4))
    w /= np.sum(w)
    weight_names = ["dp_s", "dp_r(p, r^2)", "dp_t(p, r^2)", "de"]
    print(f"random weight seed: {w_seed}")
    print_weights(weight_names, w)

    num_iter = 500
    potentials, radiations = do_simulation(p_init, w, num_iter)

    num_seqs = radiations.shape[0]
    times = np.arange(num_seqs)
    plotting.plot_curves(times, radiations, 3, show=True)

    xs = [[p, r] for p, r in zip(potentials, radiations)]
    titles = ["Potential", "Radiation"]
    anim = plotting.plot_animation_multiple(xs, titles)
    plotting.save_animation(
        anim, f"figs/animation_radius2_seed{w_seed:0>3}.gif")


if __name__ == "__main__":
    main()
