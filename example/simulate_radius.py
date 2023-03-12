"""
"""
import numpy as np

import diskflowsim2 as dfs2
import utils
import plot_helper


def do_simulation(p_init, w, num_iter):
    potentials = np.zeros((num_iter, *p_init.shape))
    radiations = np.zeros((num_iter, *p_init.shape))
    p = p_init
    for i in range(num_iter):

        num_cells_r, num_cells_t = p.shape

        p[0, 0] = np.exp(0.01*np.random.randn(1))

        # The more a cell has potential, the more the cell gets photons..
        p = p + np.exp(0.001*np.random.randn(*p.shape)) * \
            utils.sigmoid(np.log10(p + 0.00001))

        r = np.arange(num_cells_t, num_cells_r, -1)
        r2 = r**2
        ratios = dfs2.compute_propagation_ratio_with_radius(p, w, r2)

        dp = p[..., None] * ratios

        dp_s = dp[..., 0]

        # Radial propagation with potentials.
        dp_r_p = dp[..., 1]
        dp_r_p = dfs2.propagate_conv(dp_r_p)
        dp_r_p[0] = 0

        # Transversal propagation with potentials.
        dp_t_p = dp[..., 2]
        dp_t_p = np.roll(dp_t_p, shift=1, axis=1)
        dp_t_p[:, 0] = np.diag(np.fliplr(dp_t_p))

        # Radial propagation with radii.
        dp_r_r = dp[..., 3]
        dp_r_r = dfs2.propagate_conv(dp_r_r)
        dp_r_r[0] = 0

        # Transversal propagation with radii.
        dp_t_r = dp[..., 4]
        dp_t_r = np.roll(dp_t_r, shift=1, axis=1)
        dp_t_r[:, 0] = np.diag(np.fliplr(dp_t_r))

        p = dp_s + dp_r_p + dp_t_p + dp_r_r + dp_t_r
        p = np.fliplr(np.triu(np.fliplr(p)))

        potentials[i] = p

        radiation = dfs2.arrange_diskshape(dp[..., 5])
        radiations[i] = radiation

    return potentials, radiations


def print_weights(names, values):
    max_len = max([len(n) for n in names])
    for n, v in zip(names, values):
        v = round(v, 3)
        print(f"{n:<{max_len + 1}}:", f"{v:>5}")


def main():

    np.random.seed(0)

    p_init = np.exp(np.random.randn(50, 100))
    p_init = dfs2.arrange_diskshape(p_init)

    w_seed = 7
    with utils.change_seed_temp(seed=w_seed):
        # [dp_s, dp_r(p), dp_t(p), dp_r(r^2), dp_t(r^2), de]
        # w = [0.05, 0.20, 0.10, 0.01, 0.01, 0.10]
        w = np.abs(np.random.randn(6))
        w /= np.sum(w)
        weight_names = ["dp_s", "dp_r(p)", "dp_t(p)",
                        "dp_r(r^2)", "dp_t(r^2)", "de"]
        print(f"random weight seed: {w_seed}")
        print_weights(weight_names, w)

    num_iter = 500
    potentials, radiations = do_simulation(p_init, w, num_iter)

    plot_helper.plot_animation(
        potentials, f"figs/potentials_radius_seed{w_seed:0>3}.gif")
    plot_helper.plot_animation(
        radiations, f"figs/radiations_radius_seed{w_seed:0>3}.gif")


if __name__ == "__main__":
    main()
