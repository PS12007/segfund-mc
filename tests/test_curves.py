import numpy as np
import pytest

from segfund.curves import BOC_N_TERMS, FlatCurve, InterpolatedZeroCurve, NelsonSiegelCurve, load_boc_zero_curve


def test_nelson_siegel_limits():
    c = NelsonSiegelCurve(0.035, -0.01, 0.005, 2.0)
    assert c.zero_rate(1e-12) == pytest.approx(0.025, abs=1e-9)  # short end = b0 + b1
    assert c.zero_rate(1e6) == pytest.approx(0.035, abs=1e-5)  # long end -> b0
    assert c.discount(0.0) == 1.0


def test_forward_rate_matches_flat():
    c = FlatCurve(0.04)
    assert np.allclose(c.forward_rate(np.array([0.0, 1.0, 10.0])), 0.04, atol=1e-9)


def test_forward_consistent_with_discount():
    c = NelsonSiegelCurve(0.035, -0.01, 0.005, 2.0)
    t = np.linspace(0, 10, 2001)
    integral = np.trapezoid(c.forward_rate(t), t)
    assert integral == pytest.approx(-np.log(c.discount(10.0)), rel=1e-6)


def test_interpolated_reproduces_knots_and_extrapolates_flat():
    t = np.array([0.5, 1.0, 5.0, 10.0])
    z = np.array([0.02, 0.025, 0.03, 0.032])
    c = InterpolatedZeroCurve(t, z)
    assert np.allclose(c.zero_rate(t), z)
    assert c.zero_rate(0.1) == pytest.approx(0.02)
    assert c.zero_rate(30.0) == pytest.approx(0.032)
    assert np.all(np.diff(c.discount(np.linspace(0.01, 30, 300))) < 0)


def _write_fixture(path, values, date="2026-10-29"):
    # FORMAT FIXTURE ONLY: these are made-up numbers in the layout described on the
    # Bank of Canada page (date, then 120 terms). They are not Bank of Canada data.
    path.write_text("Date," + ",".join(f"ZC{k:03d}" for k in range(1, BOC_N_TERMS + 1)) + "\n"
                    + date + "," + ",".join(values) + "\n")


def test_boc_loader_parses_and_drops_na(tmp_path):
    vals = ["na", "na"] + [f"{0.02 + 0.0001 * k:.4f}" for k in range(BOC_N_TERMS - 2)]
    p = tmp_path / "fixture.csv"
    _write_fixture(p, vals)
    c = load_boc_zero_curve(p, "2026-10-29")
    assert c.times[0] == pytest.approx(0.75)  # first two terms were 'na'
    assert c.times[-1] == pytest.approx(30.0)
    assert c.zero_rate(30.0) == pytest.approx(0.02 + 0.0001 * (BOC_N_TERMS - 3))


def test_boc_loader_rejects_percent_and_short_rows(tmp_path):
    p = tmp_path / "pct.csv"
    _write_fixture(p, ["3.5"] * BOC_N_TERMS)
    with pytest.raises(ValueError, match="percent"):
        load_boc_zero_curve(p)
    p2 = tmp_path / "short.csv"
    _write_fixture(p2, ["0.03"] * 10)
    with pytest.raises(ValueError, match="expected 120"):
        load_boc_zero_curve(p2)
    with pytest.raises(ValueError, match="no curve"):
        _write_fixture(p, ["0.03"] * BOC_N_TERMS)
        load_boc_zero_curve(p, "2020-01-01")
