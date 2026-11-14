"""The browser engine (web/engine.js) against the NumPy fp64 reference, on identical streams.

Runs only where Node.js is installed (skipped otherwise, e.g. in CI without node).
Both engines are fp64 and draw the same Philox numbers, so they should agree to
rounding: Philox words exactly, per-path values to ~1e-12 relative.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "web" / "engine.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")

DRIVER = r"""
const SF = require(process.argv[2]);
const req = JSON.parse(process.argv[3]);
const out = {};
const [k0, k1] = SF.seedKey(req.seed);
out.words = req.ctrs.map(c => Array.from(SF.philox(c, req.step, 0, 0, k0, k1)));
out.normals = req.ctrs.map(c => Array.from(SF.normals3(c, req.step, 0, k0, k1)));
const p = SF.BASE;
for (const [name, measure, anti] of [["rn", "risk_neutral", true], ["rw", "real_world", false]]) {
  const m = SF.buildTable(p, measure);
  const r = SF.simulate(m, req.n, { seed: req.seed, antiHalf: anti ? req.n / 2 : 0 });
  out[name] = { pvNet: Array.from(r.pvNet), FT: Array.from(r.FT), DT: Array.from(r.DT), lT: Array.from(r.lT) };
  if (anti) out.est = SF.estimate(r, p, "pvNet", true);
}
out.put = SF.hwGbmPut(p, 100000, 75000, 10);
out.erfc = req.erfc_x.map(SF.erfc);
// nested segment: steps [12, 120) from a given start state, inner stream
const m = SF.buildTable(p, "risk_neutral");
const r = SF.simulate(m, 64, { seed: req.seed, stream: 1, antiHalf: 32, kStart: 12, init: req.state });
out.seg = Array.from(r.pvNet);
console.log(JSON.stringify(out));
"""


def _run_node(req: dict) -> dict:
    drv = ROOT / "web" / ".test_driver.cjs"
    drv.write_text(DRIVER)
    try:
        r = subprocess.run(["node", str(drv), str(ENGINE), json.dumps(req)], capture_output=True, text=True, timeout=120)
    finally:
        drv.unlink()
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


@pytest.fixture(scope="module")
def js():
    req = dict(seed=20261113, step=37, ctrs=[0, 1, 17, 123456, 2**31 - 2], n=2000,
               erfc_x=[-3.0, -0.5, 0.0, 0.3, 1.7, 2.49, 2.51, 4.0, 7.5],
               state=dict(x=0.004, F=93000.0, GM=75000.0, GD=100000.0, l=0.91))
    return req, _run_node(req)


def _model(measure):
    from segfund.config import load_config, with_measure
    from segfund.validate import DEFAULT_CONFIG

    cfg, base = load_config(DEFAULT_CONFIG)
    return with_measure(cfg, measure, base)


def test_philox_and_normals(js):
    from segfund.rng import philox as P

    req, out = js
    ids = np.array(req["ctrs"], dtype=np.uint64)
    words = np.stack(P.philox4x32(ids, req["step"], 0, 0, req["seed"])).T.astype(np.int64)
    assert np.array_equal(words, np.array(out["words"], dtype=np.int64))
    z = P.normals3(ids, req["step"], req["seed"]).T
    np.testing.assert_allclose(np.array(out["normals"]), z, rtol=0, atol=1e-14)


@pytest.mark.parametrize("name,measure,anti", [("rn", "risk_neutral", True), ("rw", "real_world", False)])
def test_paths_match_reference(js, name, measure, anti):
    from segfund.reference.engine import simulate
    from segfund.rng.philox import PhiloxNormals

    req, out = js
    n = req["n"]
    ref = simulate(_model(measure), n, PhiloxNormals(req["seed"], antithetic_half=n // 2 if anti else 0))
    for f_js, f_py in [("FT", "fund_T"), ("DT", "discount_T"), ("lT", "inforce_T")]:
        np.testing.assert_allclose(np.array(out[name][f_js]), getattr(ref, f_py), rtol=1e-11)
    np.testing.assert_allclose(np.array(out[name]["pvNet"]), ref.pv_net, rtol=0, atol=1e-6)  # dollars


def test_estimator_and_closed_form(js):
    from segfund.estimators import estimate
    from segfund.reference import closed_form as cf
    from segfund.reference.engine import simulate
    from segfund.rng.philox import PhiloxNormals

    req, out = js
    m = _model("risk_neutral")
    put = float(cf.hw_gbm_put(m.hw, m.equity, 100000.0, 75000.0, 10.0, m.contract.mer))
    assert out["put"] == pytest.approx(put, rel=1e-12)
    import math

    for x, v in zip(req["erfc_x"], out["erfc"]):
        assert v == pytest.approx(math.erfc(x), rel=1e-13, abs=1e-300)
    n = req["n"]
    e = estimate(simulate(m, n, PhiloxNormals(req["seed"], antithetic_half=n // 2)), m)
    assert out["est"]["mean"] == pytest.approx(e.mean, rel=1e-10)
    assert out["est"]["se"] == pytest.approx(e.se, rel=1e-8)
    assert out["est"]["beta"] == pytest.approx(e.beta, rel=1e-8)


def test_nested_segment_matches_reference(js):
    from segfund.reference.engine import simulate
    from segfund.rng.philox import STREAM_INNER, PhiloxNormals

    req, out = js
    s = req["state"]
    init = {k: np.full(64, v) for k, v in s.items()}
    ref = simulate(_model("risk_neutral"), 64, PhiloxNormals(req["seed"], stream=STREAM_INNER, antithetic_half=32),
                   k_start=12, init_state=init)
    np.testing.assert_allclose(np.array(out["seg"]), ref.pv_net, rtol=0, atol=1e-6)
