"""Wall-clock alignment preserves the initial point and observed rebounds."""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from plot_results import _time_axis, time_profile


def test_time_profile_uses_saved_checkpoints_without_monotone_smoothing():
    checkpoints = pd.DataFrame([
        (0, 0, .03, .6), (0, 1, .10, .8),
        (1, 0, .05, .6), (1, 1, .20, .2),
    ], columns=["seed", "k", "elapsed", "residual"])
    times = np.array([0., .05, .10, .15, .20])
    low, median, high = time_profile(checkpoints, times, "residual")
    np.testing.assert_allclose(median, [.6, .6, .7, .7, .5])
    assert low[0] == high[0] == .6
    assert median[2] > median[1]  # An observed increase remains visible.


def test_wall_clock_axis_is_linear_seconds():
    fig, ax = plt.subplots()
    _time_axis(ax)
    assert ax.get_xlabel() == "Wall-clock time (s)"
    assert ax.get_xscale() == "linear"
    np.testing.assert_allclose(ax.get_xticks(), np.arange(0, 3.5, .5))
    plt.close(fig)


def test_bootstrap_intervals_resample_seeds_and_preserve_the_median():
    checkpoints = pd.DataFrame([
        (0, 0, 0., .6), (0, 1, .1, .8),
        (1, 0, 0., .6), (1, 1, .2, .2),
    ], columns=["seed", "k", "elapsed", "residual"])
    times = np.array([0., .1, .2])
    profile = time_profile(checkpoints, times, "residual", bootstrap_samples=2000)
    np.testing.assert_allclose(profile, [[.6, .6, .2], [.6, .7, .5], [.6, .8, .8]])
    np.testing.assert_array_equal(
        profile, time_profile(checkpoints, times, "residual", bootstrap_samples=2000))
