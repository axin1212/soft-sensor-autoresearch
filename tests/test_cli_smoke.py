from __future__ import annotations

from pathlib import Path

from soft_sensor_autoresearch import cli


def test_cli_smoke(monkeypatch, capsys):
    def fake_run_autoresearch(**kwargs):
        assert kwargs["data_file"] == Path("data.parquet")
        assert kwargs["target_column"] == "target"
        assert kwargs["model_type"] == "tabpfn3"
        assert kwargs["forecast_horizons"] == (0,)
        return Path("/tmp/report.html")

    monkeypatch.setattr(cli, "run_autoresearch", fake_run_autoresearch)

    rc = cli.main(["data.parquet", "target"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "report.html: /tmp/report.html" in output
    assert "resource_usage.csv: /tmp/resource_usage.csv" in output


def test_zero_time_budget_stays_unlimited():
    assert cli._time_budget_seconds(0) == 0
    assert cli._time_budget_seconds(0.0) == 0
    assert cli._time_budget_seconds(0.5) == 30.0


def test_parse_forecast_horizons_accepts_ranges_and_lists():
    assert cli._parse_forecast_horizons("0") == (0,)
    assert cli._parse_forecast_horizons("0:3") == (0, 1, 2, 3)
    assert cli._parse_forecast_horizons("0,2,5") == (0, 2, 5)


def test_build_report_metadata_contains_target_and_training_parameters():
    metadata = cli._build_report_metadata(
        data_file=Path("data.parquet"),
        target_column="12PI-44026A",
        model_type="tabpfn3",
        resolved_window_minutes=10,
        num_train_samples=400,
        top_features_n=32,
        validation_fraction=0.30,
        forecast_horizons=(0, 10),
        include_frequency_candidate=False,
        tabpfn_device="auto",
        tabpfn_fit_mode="fit_preprocessors",
        tabpfn_n_estimators=1,
        tpt_device="mps",
        tpt_fit_mode="fit_preprocessors",
        tpt_n_estimators=1,
        resolved_fde=Path("/fde"),
    )

    assert metadata.target_column == "12PI-44026A"
    assert metadata.default_window_minutes == 10
    assert metadata.num_train_samples == 400
    assert metadata.forecast_horizons == (0, 10)
