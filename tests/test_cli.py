import json

from segfund.cli import main
from segfund.validate import DEFAULT_CONFIG


def test_cli_run_writes_json(tmp_path, capsys):
    out = tmp_path / "r.json"
    assert main(["run", "--config", str(DEFAULT_CONFIG), "--engine", "cpu", "--paths", "2000", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["n_paths"] == 2000 and data["engine"] == "cpu"
    assert {"pv", "cte", "delta", "rho"} <= set(data)
    assert data["cte"]["cte95"]["cte"] >= data["cte"]["cte70"]["cte"]
    assert data["delta"]["value"] < 0  # put-like liability falls when the fund rises


def test_cli_nested_cpu_small(tmp_path):
    out = tmp_path / "n.json"
    args = ["nested", "--engine", "cpu", "--n-outer", "200", "--n-inner-gt", "64", "--n-inner-cheap", "8",
            "--n-fit", "2000", "--out", str(out)]
    assert main(args) == 0
    rows = json.loads(out.read_text())["rows"]
    assert rows[0]["cte95_error"] == 0.0  # the reference compared with itself
    assert len(rows) == 3


def test_cli_report_builds_self_contained_html(tmp_path):
    out = tmp_path / "r.html"
    assert main(["report", "--out", str(out)]) == 0
    page = out.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and "Not actuarially approved" in page
    assert "<script" not in page and 'src="http' not in page  # no external resources


def test_cli_bench_without_gpu_says_so(capsys, monkeypatch):
    import segfund.cli as cli

    monkeypatch.setattr(cli, "_pick_engine", lambda name: "cpu")
    assert main(["bench"]) == 2
    assert "needs a CUDA GPU" in capsys.readouterr().out
