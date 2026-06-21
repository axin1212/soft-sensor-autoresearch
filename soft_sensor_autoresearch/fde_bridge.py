from __future__ import annotations

from pathlib import Path
import os
import sys


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


def _looks_like_fde_or_benchmark(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return (
        (path / "packages" / "kernels").exists()
        or (path / "vendor" / "fde_packages" / "kernels").exists()
        or (path / "contestants" / "2_scoPe_regressor").exists()
    )
