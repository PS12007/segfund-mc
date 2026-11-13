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


def test_cli_unbuilt_commands_say_so(capsys):
    assert main(["nested"]) == 2
    assert "phase 5" in capsys.readouterr().out
