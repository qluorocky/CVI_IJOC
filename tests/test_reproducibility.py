"""Release integrity and protection against accidentally mixing experiment runs."""

from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiment_common import write_config
from cvx_benchmark import OE_SETTINGS
from plot_results import summarize


ROOT = Path(__file__).resolve().parents[1]


def test_released_checkpoints_and_summaries_are_consistent():
    grids = {
        "pfaos": {(n, 1) for n in [25, 50, 100, 200, 500, 1000]},
        "mc": ({(50, m) for m in [2, 5, 10, 20, 40, 100, 250, 500]}
               | {(n, 5) for n in [25, 50, 100]}),
    }
    for study, settings in grids.items():
        raw = pd.read_csv(ROOT / "results" / study / "checkpoints.csv")
        proposed = "PFAOS" if study == "pfaos" else "MC-PFAOS"
        expected = {(n, m, method, seed) for n, m in settings
                    for method in [proposed] + [name for _, name in OE_SETTINGS]
                    for seed in range(10)}
        assert set(raw.groupby(["n", "m", "method", "seed"]).groups) == expected
        assert not raw.duplicated(["n", "m", "method", "seed", "k"]).any()
        assert raw.gap_status.eq("optimal").all()
        assert np.isfinite(raw[["gap", "violation", "residual", "elapsed"]]).all().all()
        np.testing.assert_allclose(raw.residual, np.maximum(raw.gap, raw.violation))
        for factor, method in OE_SETTINGS:
            oe = raw[raw.method == method]
            np.testing.assert_allclose(oe.M0, factor * oe.M_true)
            assert oe.M0_factor.eq(factor).all()
            if factor >= 1:
                np.testing.assert_array_equal(oe.projection_calls, oe.k)
        for _, run in raw.groupby(["n", "m", "method", "seed"]):
            run = run.sort_values("k")
            assert run.k.iloc[0] == 0
            assert (np.diff(run[["elapsed", "F_calls", "grad_calls", "projection_calls"]], axis=0) >= 0).all()
        actual = summarize(raw).sort_values(["n", "m", "method", "seed"]).reset_index(drop=True)
        saved = pd.read_csv(ROOT / "results" / "figures" / f"{study}_runs.csv")
        saved = saved.sort_values(["n", "m", "method", "seed"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, saved, check_dtype=False)


def test_existing_output_is_preserved(tmp_path):
    checkpoints = tmp_path / "checkpoints.csv"
    checkpoints.write_text("existing experiment data\n")
    with pytest.raises(FileExistsError, match="choose a new --out"):
        write_config(tmp_path / "config.json", Namespace(seeds=1))
    assert checkpoints.read_text() == "existing experiment data\n"
    assert not (tmp_path / "config.json").exists()
