"""
"""
from matplotlib.animation import ArtistAnimation
import matplotlib.pyplot as plt
import numpy as np
import tqdm

import diskflowsim2 as dfs2
import utils


def propagate_conv(x):
    num_cells_r = x.shape[0]
    x = utils.im2col_array(x[None, None, :], kernel_size=(1, 2),
                           stride=1, pad=(0, 1))
    x = np.dot(x, [1., 1.])
    x = x.reshape((num_cells_r, -1))
    x = x[:, 1:]
    x = np.roll(x, shift=1, axis=0)
    return x


def do_simulation(init_p, w, num_iter):
    potentials = np.zeros((num_iter, *init_p.shape))
    radiations = np.zeros((num_iter, *init_p.shape))
    p = init_p
    for i in range(num_iter):

        num_cells_r, num_cells_t = p.shape

        p[0, 0] = np.exp(0.01*np.random.randn(1))

        p = p + np.exp(0.001*np.random.randn(*p.shape)) * \
            utils.sigmoid(np.log10(p + 0.00001))

        ratios = dfs2.compute_propagation_ratio(p, w)
        # r = np.arange(num_cells_t, num_cells_r, -1)
        # ratios = dfs2.compute_propagation_ratio_with_radius(p, w, r)

        dp = p[..., None] * ratios

        dp_s = dp[..., 0]

        dp_r = dp[..., 1]
        dp_r = propagate_conv(dp_r)
        dp_r[0] = 0

        dp_t = dp[..., 2]
        dp_t = np.roll(dp_t, shift=1, axis=1)
        dp_t[:, 0] = np.diag(np.fliplr(dp_t))

        p = dp_s + dp_r + dp_t
        p = np.fliplr(np.triu(np.fliplr(p)))

        potentials[i] = p

        radiations[i] = dp[..., 3]

    return potentials, radiations


def plot_animation(xs, to_file="animation.gif"):
    fig, ax = plt.subplots(figsize=(8, 5))
    frames = []
    for i in tqdm.trange(xs.shape[0]):
        frame = ax.imshow(xs[i], cmap="jet",
                          vmax=xs.max(), vmin=0)
        frames.append([frame])
    ax.axis("off")
    fig.tight_layout()
    ani = ArtistAnimation(fig, frames, interval=100)
    ani.save(to_file)
    # plt.show()
    plt.close()


def plot_snapshot(potential, ax, progress_bar=False):
    num_anulus, num_segments = potential.shape
    r_out, r_in = num_segments, num_segments - num_anulus
    size = (r_out - r_in) / num_anulus

    cmap = plt.colormaps["jet"]

    for i, anulus in enumerate(tqdm.tqdm(potential,
                                         disable=not(progress_bar))):
        radius = int(r_out - i * size)
        anulus_true = anulus[:radius]
        colors = cmap(anulus_true)
        fractions = (anulus_true > 0).astype(int)
        ax.pie(fractions, radius=radius, colors=colors,
               wedgeprops=dict(width=size))
    ax.set_xlim(-r_out, r_out)
    ax.set_ylim(-r_out, r_out)

    return ax


def main():

    np.random.seed(0)

    init_p = np.exp(np.random.randn(50, 500))

    # [stay, prop_radial, prop_transverse, radiative_cooling]
    w = [0.05, 0.20, 0.30, 0.10]
    print(w)

    num_iter = 400
    potentials, radiations = do_simulation(init_p, w, num_iter)

    # fig, ax = plt.subplots()
    # plot_snapshot(potentials[0], ax)
    # plt.show()

    plot_animation(potentials, "potentials.gif")
    plot_animation(radiations, "radiations.gif")


if __name__ == "__main__":
    main()
