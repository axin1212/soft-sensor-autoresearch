from __future__ import annotations

from soft_sensor_autoresearch.fde_bridge import _tpt_child_code


def test_tpt_child_code_patches_mps_version_check():
    code = _tpt_child_code()

    assert "_is_mps_supported" in code
    assert "torch.backends.mps.is_available()" in code


def test_tpt_child_code_logs_mps_resource_events():
    code = _tpt_child_code()

    assert "SOFT_SENSOR_RESOURCE_LOG_PATH" in code
    assert "mps_current_allocated_mb" in code
    assert "tpt_fit_start" in code
    assert "tpt_predict_end" in code
