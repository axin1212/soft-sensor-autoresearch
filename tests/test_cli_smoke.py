from __future__ import annotations

from soft_sensor_autoresearch.cli import main


def test_cli_smoke(capsys):
    rc = main(["data.parquet", "target"])
    assert rc == 0
    assert "target=target" in capsys.readouterr().out
