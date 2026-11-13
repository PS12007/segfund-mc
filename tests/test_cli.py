import json

from segfund.cli import main
from segfund.validate import DEFAULT_CONFIG


def test_cli_run_writes_json(tmp_path, capsys):
    out = tmp_path / "r.json"
    assert main(["run", "--config", str(DEFAULT_CONFIG), "--paths", "500", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["n_paths"] == 500 and "pv_net" in data["results"]


def test_cli_unbuilt_commands_say_so(capsys):
    assert main(["nested"]) == 2
    assert "phase 5" in capsys.readouterr().out
