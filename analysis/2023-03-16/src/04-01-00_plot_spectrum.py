"""Plot power spectrum.
"""
import glob

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


def main():

    radiation_path = sorted(glob.glob("../data/out/radiations_*.npy"))[-1]
    radiations = np.load(radiation_path)

    num_times, num_radius, num_trans = radiations.shape

    idx_stable = 500
    num_layers = 3

    radiations = radiations[idx_stable:]

    jump = num_radius // (num_layers)
    curve_list = []
    for i in range(num_layers):
        curve = radiations[:, i*jump:(i+1)*jump].sum(axis=(1, 2))
        curve_list.append(curve)

    freqs, powers_list = signal.welch(np.array(curve_list))

    times = np.arange(num_times-idx_stable)
    fig, ax = plt.subplots(num_layers, 2,
                           squeeze=False,
                           sharex="col",
                           width_ratios=(3, 1),
                           figsize=(12, 2*num_layers))
    for i in range(num_layers):
        ax[i, 0].plot(times, curve_list[i])
        ax[i, 1].plot(freqs, powers_list[i])

        ax[i, 0].set_ylabel(f"Layer {i}")
        ax[i, 1].set_xscale("log")
        ax[i, 1].set_yscale("log")

    ax[-1, 0].set_xlabel("Time (sample)")
    ax[-1, 1].set_xlabel("Frequency (Hz)")
    fig.align_ylabels()
    fig.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
