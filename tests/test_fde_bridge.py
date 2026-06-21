from __future__ import annotations

from soft_sensor_autoresearch.fde_bridge import _tpt_child_code


def test_tpt_child_code_patches_mps_version_check():
    code = _tpt_child_code()

    assert "_is_mps_supported" in code
    assert "torch.backends.mps.is_available()" in code
