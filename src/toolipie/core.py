from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import os

import pypandoc
import yaml


@dataclass
class Context:
    task: str
    input_dir: Path
    output_dir: Path
    files: list[Path]
    config: dict
    overwrite: bool
    workers: int
    run_log: Path


def get_repo_root() -> Path:
    # Prefer current working directory if it contains config.yaml
    cwd = Path.cwd()
    if (cwd / "config.yaml").exists():
        return cwd
    # Fallback to package repo root: repo/ (config.yaml), src/toolipie/core.py
    return Path(__file__).resolve().parents[3]


def load_config() -> dict:
    cfg_path = get_repo_root() / "config.yaml"
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_context(
    task_name: str,
    input: str | None,
    output: str | None,
    glob: str | None,
    overwrite: bool | None,
    workers: int | None,
) -> Context:
    cfg = load_config()
    defaults = cfg.get("defaults") or {}
    paths = cfg.get("paths") or {}
    repo_root = get_repo_root()
    input_root = repo_root / Path(paths.get("input_root", "input"))
    output_root = repo_root / Path(paths.get("output_root", "output"))

    kebab_task = task_name
    default_input = input_root / kebab_task
    default_output = output_root / kebab_task

    input_dir = Path(input) if input else default_input
    output_dir = Path(output) if output else default_output
    ensure_dir(output_dir)

    use_glob = glob or defaults.get("glob", "*.[mM][dD]")
    # Accept a file path or directory.
    start = Path(input) if input else input_dir
    if start.is_file():
        files = [start]
    else:
        if start.exists():
            pattern = use_glob if use_glob.startswith("**/") else f"**/{use_glob}"
            files = sorted(start.glob(pattern))
        else:
            files = []

    use_overwrite = bool(defaults.get("overwrite", False)) if overwrite is None else bool(overwrite)
    use_workers = int(defaults.get("workers", 0)) if workers is None else int(workers)

    run_log = output_dir / "run.jsonl"
    # Ensure run log file exists for every run
    run_log.parent.mkdir(parents=True, exist_ok=True)
    run_log.touch(exist_ok=True)
    return Context(
        task=task_name,
        input_dir=start if start.is_dir() else start.parent,
        output_dir=output_dir,
        files=files,
        config=cfg,
        overwrite=use_overwrite,
        workers=use_workers,
        run_log=run_log,
    )


def append_run_log(run_log: Path, record: dict) -> None:
    run_log.parent.mkdir(parents=True, exist_ok=True)
    # Prefer repo root as the base for relative paths; run_log is output/<task>/run.jsonl
    base = run_log.parents[3] if len(run_log.parents) >= 4 else run_log.parent

    def to_rel(value: str | Path) -> str:
        p = Path(value)
        try:
            return str(p.relative_to(base))
        except Exception:
            # Best-effort relative path even if outside base
            try:
                return os.path.relpath(str(p), start=str(base))
            except Exception:
                return str(p)

    new_record = dict(record)
    if "input" in new_record:
        new_record["input"] = to_rel(new_record["input"])
    if "output" in new_record:
        new_record["output"] = to_rel(new_record["output"])

    with run_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(new_record, ensure_ascii=False) + "\n")


def ensure_pandoc() -> None:
    """Ensure a usable pandoc binary is available by downloading a private copy if needed."""
    try:
        pypandoc.get_pandoc_path()
    except OSError:
        pypandoc.download_pandoc()
