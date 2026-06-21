from __future__ import annotations

from pathlib import Path
import os
import sys

import pandas as pd

from soft_sensor_autoresearch.feature_pool import WindowFeatureRequest


def find_fde_root(start: Path, explicit: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    env = os.environ.get("FDE_SOURCE_PATH")
    if env:
        candidates.append(Path(env))
    for parent in [start, *start.parents]:
        candidates.extend([parent, parent / "FDE", parent / "benchmark"])
    candidates.append(start.parent / "FDE")

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except FileNotFoundError:
            resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _looks_like_fde_or_benchmark(resolved):
            return resolved
    return None


def add_fde_to_path(root: Path) -> None:
    paths = [
        root,
        root / "packages" / "kernels",
        root / "packages" / "icl_utils",
        root / "vendor" / "fde_packages" / "kernels",
        root / "vendor" / "fde_packages" / "icl_utils",
        root / "contestants" / "2_scoPe_regressor",
    ]
    for path in paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_tabpfn3_predictor_factory():
    try:
        from kernels.foundation.adapters.single_step import FoundationTabularPredictor
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"could not import FDE TabPFN3 predictor: {exc}") from exc

    def factory():
        return FoundationTabularPredictor(kernel_id="tabpfn3", device="auto")

    return factory


class FdeWindowFeatureBuilder:
    def build_feature_matrix(self, request: WindowFeatureRequest, extraction: str) -> pd.DataFrame:
        try:
            from pipeline.feature_engine import align_feature_matrix_to_samples, build_feature_matrix
            from pipeline.window_policy import WindowGateConfig, extract_windows
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"could not import FDE feature/window pipeline: {exc}") from exc

        data = request.data.rename(columns={request.time_column: "Timestamp"}).copy()
        data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
        data = data.sort_values("Timestamp")
        target_times = pd.to_datetime(request.target_times).to_numpy(dtype="datetime64[ns]")
        extracted = extract_windows(
            data,
            target_times,
            int(request.window_minutes),
            request.feature_columns,
            WindowGateConfig(),
        )
        if not extracted.window_dfs:
            return pd.DataFrame(index=range(len(target_times)))
        matrix = build_feature_matrix(
            extracted.window_dfs,
            request.feature_columns,
            resolved_extraction=extraction,
            target_points=10,
            raw_n_points=20,
            n_jobs=1,
        )
        return align_feature_matrix_to_samples(
            matrix,
            target_times,
            resolved_extraction=extraction,
        ).reset_index(drop=True)


def _looks_like_fde_or_benchmark(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return (
        (path / "packages" / "kernels").exists()
        or (path / "vendor" / "fde_packages" / "kernels").exists()
        or (path / "contestants" / "2_scoPe_regressor").exists()
    )
