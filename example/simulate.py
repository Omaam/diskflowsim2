"""
"""
import numpy as np

import diskflowsim2 as dfs2
import utils
import plot_helper


def do_simulation(init_p, w, num_iter):
    potentials = np.zeros((num_iter, *init_p.shape))
    radiations = np.zeros((num_iter, *init_p.shape))
    p = init_p
    for i in range(num_iter):

        num_cells_r, num_cells_t = p.shape

        p[0, 0] = np.exp(0.01*np.random.randn(1))

        # The more a cell has potential, the more the cell gets photons..
        p = p + np.exp(0.001*np.random.randn(*p.shape)) * \
            utils.sigmoid(np.log10(p + 0.00001))

        ratios = dfs2.compute_propagation_ratio(p, w)

        dp = p[..., None] * ratios

        dp_s = dp[..., 0]

        dp_r = dp[..., 1]
        dp_r = dfs2.propagate_conv(dp_r)
        dp_r[0] = 0

        dp_t = dp[..., 2]
        dp_t = np.roll(dp_t, shift=1, axis=1)
        dp_t[:, 0] = np.diag(np.fliplr(dp_t))

        p = dp_s + dp_r + dp_t
        p = np.fliplr(np.triu(np.fliplr(p)))

        potentials[i] = p

        radiations[i] = dp[..., 3]

    return potentials, radiations


def main():

    np.random.seed(0)

    init_p = np.exp(np.random.randn(50, 100))

    # [stay, prop_radial, prop_transverse, radiative_cooling]
    w = [0.05, 0.20, 0.30, 0.10]
    print(w)

    num_iter = 500
    potentials, radiations = do_simulation(init_p, w, num_iter)

    plot_helper.plot_animation(potentials, "figs/potentials.gif")
    plot_helper.plot_animation(radiations, "figs/radiations.gif")


if __name__ == "__main__":
    main()
