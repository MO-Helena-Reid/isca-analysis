import numpy as np
from matplotlib import pyplot as plt


def main():
    rng = np.random.default_rng(10)
    timestep = 0.2
    init = 0.0
    steps = int(1e6)
    mean_timescales = []
    sd_timescales = []
    phivals = np.concatenate((np.arange(0.1, 0.9, 0.1),
                              np.arange(0.9, 0.99, 0.01),
                              np.arange(0.99, 0.999, 0.001)))
    phivals = [get_phi_given_timescale(6, timestep)]
    print(phivals)
    for phi1 in phivals:
        seq = ar1(phi1, init, steps, rng)
        mean_steps_to_switch = np.mean(get_sequence_lengths(seq > 0))
        sd_steps_to_switch = np.std(get_sequence_lengths(seq > 0))
        mean_timescale = mean_steps_to_switch * timestep
        sd_timescale = sd_steps_to_switch * timestep
        mean_timescales.append(mean_timescale)
        sd_timescales.append(sd_timescale)
        print(f"PHI={phi1:.2f} "
              #     # f"min: {min(seq):.3f},\nmax: {max(seq):.3f},\nmean: {np.mean(seq):.5f},\nstd: {np.std(seq):.3f},\n"
              f"mean timescale: {mean_timescale:.1f}, sd timescale: {sd_timescale:.1f}")
        # print(f"--------------------------")
    mean_timescales = np.asarray(mean_timescales)
    sd_timescales = np.asarray(sd_timescales)
    expected_timescales = timestep * 2 * np.pi / (np.pi - 2 * np.arcsin(phivals))
    expected_phis = [get_phi_given_timescale(ts, timestep) for ts in mean_timescales]
    [print(f"phi:{phi:.3f},expected phi: {expected_phi:.3f}") for phi, expected_phi in zip(phivals, expected_phis)]
    [print(f"timescale:{mean_timescale:.3f},expected timescale: {expected_timescale:.3f}") for
     mean_timescale, expected_timescale in zip(mean_timescales, expected_timescales)]
    # plt.scatter(expected_timescales, mean_timescales)
    # plt.xlabel("expected timescales")
    # plt.ylabel("mean timescale (hours)")
    # plt.show()


def get_phi_given_timescale(timescale, timestep):
    if timescale < timestep:
        raise ValueError("cannot have AR(1) process where expected correlation timescale is shorter than the timestep!")
    return np.sin((1 / ((timescale / timestep) / (2 * np.pi)) - np.pi) / -2)


def ar1(phi, init, steps, rng):
    sequence = []
    for i in range(steps):
        # init = init * phi + rng.random() - 0.5
        init = init * phi + rng.normal()
        sequence.append(init)
    return np.asarray(sequence)


def get_sequence_lengths(condition):
    return np.diff(np.where(np.concatenate(([condition[0]], condition[:-1] != condition[1:], [True])))[0])[::2]


if __name__ == "__main__":
    main()
