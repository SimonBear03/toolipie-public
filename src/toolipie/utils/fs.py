from pathlib import Path
from typing import Iterable


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_files(root: Path, pattern: str) -> Iterable[Path]:
    if root.is_file():
        yield root
    elif root.exists():
        yield from root.glob(pattern)
