from __future__ import annotations

"""
Shared runner/registry for Toolipie.

This module centralizes tool discovery, option specification, effective defaults
calculation, and execution dispatch so that both the CLI and TUI (and future GUI)
can remain thin wrappers without duplicating business logic.

Public API:
- discover_tools() -> list[dict]
- get_tool_specs(tool_key: str) -> list[dict]
- get_effective_defaults(tool_key: str, cfg: dict | None = None) -> dict
- run_tool(tool_key: str, ctx, params: dict | None = None) -> None

Notes:
- This implementation mirrors existing behavior in cli.py to remain compatible.
- As we evolve toward per-tool specs (CLI_SPEC/tool.yaml), these functions can
  delegate to spec parsing without changing the callers.
"""

from pathlib import Path
from typing import Any, Optional
import inspect
import json
from datetime import datetime, timezone
import importlib
import importlib.util

from .core import get_repo_root, load_config
import yaml
import os
import sys


COMMON_SPECS = [
    {"name": "input", "label": "Input", "kind": "path"},
    {"name": "output", "label": "Output", "kind": "path"},
    {"name": "glob", "label": "Glob", "kind": "str"},
    {"name": "overwrite", "label": "Overwrite", "kind": "tri"},
    {"name": "workers", "label": "Workers (0=auto)", "kind": "int"},
]


def _snake_to_kebab(name: str) -> str:
    return name.replace("_", "-")


def _read_title_desc(readme_path: Path) -> tuple[str, str]:
    try:
        lines = readme_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return "", ""
    title = ""
    desc = ""
    for i, line in enumerate(lines):
        if line.strip().startswith("#"):
            title = line.lstrip("# ").strip()
            for j in range(i + 1, min(i + 10, len(lines))):
                if lines[j].strip():
                    desc = lines[j].strip()
                    break
            break
    return title, desc


def load_index() -> Optional[dict[str, Any]]:
    """Load the repository-level index file if it exists.

    Path: <repo>/.toolipie/index.json
    Returns a dict or None if missing/invalid.
    """
    root = get_repo_root()
    index_path = root / ".toolipie" / "index.json"
    try:
        if index_path.exists():
            return json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def get_repo_plugins_dir() -> Path:
    return Path(__file__).resolve().parent / "plugins"


def parse_tool_manifest(path: Path) -> Optional[dict[str, Any]]:
    """Parse a tool.yaml manifest and normalize it to an index entry.

    Returns a dict with at least: key, title, summary, default_glob, entry, path.
    Returns None if invalid or schema mismatch.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    if int(data.get("schema_version", 0)) != 1:
        return None
    name = str(data.get("name", "")).strip()
    if not name:
        # Derive from folder if omitted
        try:
            name = _snake_to_kebab(path.parent.name)
        except Exception:
            return None
    entry = data.get("entry")
    if not entry:
        # Default entry from folder name
        snake = path.parent.name
        entry = f"toolipie.tools.{snake}.run:run"
    title = data.get("title") or name.replace("-", " ").title()
    summary = data.get("summary") or ""
    default_glob = data.get("default_glob") or "*.md"
    options = data.get("options") or []
    requires = data.get("requires") or []
    rel_path = str(path.parent)
    return {
        "key": name,
        "title": title,
        "summary": summary,
        "default_glob": default_glob,
        "entry": entry,
        "options": options,
        "requires": requires,
        "path": rel_path,
    }


def discover_tools() -> list[dict[str, str]]:
    """Discover tools from the unified index (.toolipie/index.json).

    Returns list of {key, title, desc}. If no entries, prints
    a friendly hint to run `toolipie scan` or install plugins.
    """
    idx = load_index()
    tools: list[dict[str, str]] = []
    for t in idx.get("tools", []) if isinstance(idx, dict) else []:
        try:
            tools.append({
                "key": str(t["key"]),
                "title": str(t.get("title") or t["key"]).strip(),
                "desc": str(t.get("summary", "")),
            })
        except Exception:
            continue
    if not tools:
        print("No tools found. Run `toolipie scan` or install plugins with `toolipie install`.", file=sys.stderr)
    return tools


def scan_and_write_index() -> dict[str, Any]:
    """[Deprecated] Kept for backward-compat in imports. Use scan_all_and_write_index()."""
    return scan_all_and_write_index()


def _reflect_run_params(tool_key: str) -> list[dict[str, Any]]:
    """Reflect run(ctx, ...) to build per-tool option specs.

    Maps annotations to simple kinds: int | float | bool | str (default).
    """
    specs: list[dict[str, Any]] = []
    try:
        mod = __import__(f"toolipie.tools.{tool_key.replace('-', '_')}.run", fromlist=["run"])
        sig = inspect.signature(mod.run)
        for name, param in sig.parameters.items():
            if name == "ctx":
                continue
            ann = str(param.annotation)
            kind = "str"
            if "int" in ann:
                kind = "int"
            elif "float" in ann:
                kind = "float"
            elif "bool" in ann:
                kind = "tri"
            specs.append({"name": name, "label": name.replace("_", " ").title(), "kind": kind})
    except Exception:
        pass
    return specs


def _enrich_specs(tool_key: str, specs: list[dict[str, Any]]) -> None:
    """Apply per-tool enrichments (choices, path kinds, etc.)."""
    # md-to-pdf: preset choices from assets/presets/*.css, css treated as path
    if tool_key == "md-to-pdf":
        try:
            preset_dir = (
                Path(__file__).resolve().parent / "tools" / "md_to_pdf" / "assets" / "presets"
            )
            choices = [p.stem for p in preset_dir.glob("*.css")] if preset_dir.exists() else []
        except Exception:
            choices = []
        for s in specs:
            if s["name"] == "preset":
                s["kind"] = "choice"
                s["choices"] = choices
            if s["name"] == "css":
                s["kind"] = "path"
    # png-prep-ocr: fixed method choices
    if tool_key == "png-prep-ocr":
        for s in specs:
            if s["name"] == "method":
                s["kind"] = "choice"
                s["choices"] = ["auto", "hough", "minrect", "sweep"]


def get_tool_specs(tool_key: str) -> list[dict[str, Any]]:
    """Return UI specs strictly from the index/manifest.

    In strict mode, if the tool is missing or has no options, raise a RuntimeError
    with a clear message. Developers can enable a reflection fallback by setting
    TOOLIPIE_DEV_ALLOW_REFLECTION=1.
    """
    idx = load_index()
    tools = (idx or {}).get("tools", []) if isinstance(idx, dict) else []
    record = None
    for t in tools:
        if str(t.get("key")) == tool_key:
            record = t
            break
    if not record or not isinstance(record, dict):
        msg = (
            f"Tool '{tool_key}' is missing from the registry. "
            "Run `toolipie scan` or check tool.yaml."
        )
        # Dev escape hatch
        if os.environ.get("TOOLIPIE_DEV_ALLOW_REFLECTION") == "1":
            specs = list(COMMON_SPECS)
            specs += _reflect_run_params(tool_key)
            _enrich_specs(tool_key, specs)
            print(
                f"[dev] Reflection fallback used for '{tool_key}'. Add a tool.yaml and rescan.",
                file=sys.stderr,
            )
            return specs
        raise RuntimeError(msg)

    options = record.get("options")
    if not isinstance(options, list) or not options:
        msg = (
            f"Tool '{tool_key}' is missing a manifest or its options section. "
            "Run `toolipie scan` or check tool.yaml."
        )
        if os.environ.get("TOOLIPIE_DEV_ALLOW_REFLECTION") == "1":
            specs = list(COMMON_SPECS)
            specs += _reflect_run_params(tool_key)
            _enrich_specs(tool_key, specs)
            print(
                f"[dev] Reflection fallback used for '{tool_key}'. Define options in tool.yaml.",
                file=sys.stderr,
            )
            return specs
        raise RuntimeError(msg)

    # Map manifest options -> UI specs
    specs = list(COMMON_SPECS)
    for opt in options:
        try:
            name = str(opt["name"])  # required
            # Avoid duplicating common fields present in COMMON_SPECS
            if name in {s["name"] for s in COMMON_SPECS}:
                continue
            typ = str(opt.get("type", "str"))
            kind = {
                "bool": "tri",
                "int": "int",
                "float": "float",
                "str": "str",
                "path": "path",
            }.get(typ, "str")
            spec = {"name": name, "label": name.replace("_", " ").title(), "kind": kind}
            if "choices" in opt and isinstance(opt["choices"], list):
                spec["kind"] = "choice"
                spec["choices"] = [str(c) for c in opt["choices"]]
            specs.append(spec)
        except Exception:
            continue
    _enrich_specs(tool_key, specs)
    return specs


def validate_manifest_dict(data: dict[str, Any], path_hint: str = "tool.yaml") -> list[str]:
    """Validate a manifest dict. Returns a list of human-readable errors."""
    errors: list[str] = []
    if int(data.get("schema_version", 0)) != 1:
        errors.append(f"schema_version must be 1: {path_hint}")
    name = data.get("name")
    if not name or not isinstance(name, str):
        errors.append(f"name is required (kebab-case): {path_hint}")
    if not data.get("default_glob"):
        errors.append(f"default_glob is required: {path_hint}")
    options = data.get("options")
    if not isinstance(options, list) or len(options) == 0:
        errors.append(f"options must be a non-empty list: {path_hint}")
    else:
        for i, opt in enumerate(options):
            if not isinstance(opt, dict):
                errors.append(f"options[{i}] must be an object: {path_hint}")
                continue
            if not opt.get("name"):
                errors.append(f"options[{i}].name is required: {path_hint}")
            typ = opt.get("type")
            if typ not in {"str", "int", "float", "bool", "path"}:
                errors.append(f"options[{i}].type invalid: {typ} (path: {path_hint})")
            if "choices" in opt and not isinstance(opt["choices"], list):
                errors.append(f"options[{i}].choices must be a list: {path_hint}")
            # Disallow defaults for input/output (platform provides)
            if str(opt.get("name")) in {"input", "output"} and "default" in opt:
                errors.append(f"options[{i}]: do not set default for '{opt.get('name')}' (platform provides). {path_hint}")
    return errors


def validate_manifest_file(path: Path) -> list[str]:
    """Validate a single tool.yaml. Returns a list of human-readable errors."""
    if not path.exists():
        return [f"Manifest not found: {path}"]
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        return [f"YAML parse error: {path} — {e}"]
    return validate_manifest_dict(data, str(path))
    return errors


def validate_index_dict(idx: dict[str, Any]) -> list[str]:
    """Validate the structure of an index dict. Returns list of errors."""
    errors: list[str] = []
    if int(idx.get("version", 0)) != 1:
        errors.append("index.version must be 1")
    tools = idx.get("tools")
    if not isinstance(tools, list):
        errors.append("index.tools must be a list")
        return errors
    for i, t in enumerate(tools):
        if not isinstance(t, dict):
            errors.append(f"tools[{i}] must be an object")
            continue
        # Required fields
        for req in ("key", "title", "default_glob", "entry", "options", "source", "rel_path"):
            if req not in t:
                errors.append(f"tools[{i}].{req} is required")
        # Accept legacy 'path' for backward-compat, but prefer rel_path
        if not isinstance(t.get("options"), list) or len(t.get("options", [])) == 0:
            errors.append(f"tools[{i}].options must be a non-empty list")
        # Optional metadata
        if "source" in t and t["source"] not in {"core", "plugin"}:
            errors.append(f"tools[{i}].source must be 'core' or 'plugin' if present")
        if "installed_at" in t and not isinstance(t["installed_at"], str):
            errors.append(f"tools[{i}].installed_at must be an ISO datetime string if present")
    return errors


def get_effective_defaults(tool_key: str, cfg: Optional[dict[str, Any]] = None) -> dict:
    """Compute effective defaults strictly from config + index/manifest.

    - input/output from config roots + tool key
    - glob from manifest.default_glob (falls back to config default only if manifest missing)
    - overwrite/workers from config
    - tool option defaults from manifest.options[*].default (None if not provided)
    """
    if cfg is None:
        cfg = load_config()
    defaults = cfg.get("defaults") or {}
    paths = cfg.get("paths") or {}
    repo = get_repo_root()
    eff: dict[str, Any] = {}
    eff["input"] = str((repo / paths.get("input_root", "input") / tool_key).resolve())
    eff["output"] = str((repo / paths.get("output_root", "output") / tool_key).resolve())
    eff["overwrite"] = bool(defaults.get("overwrite", False))
    eff["workers"] = int(defaults.get("workers", 0))

    idx = load_index() or {}
    record = None
    for t in idx.get("tools", []) if isinstance(idx, dict) else []:
        if str(t.get("key")) == tool_key:
            record = t
            break
    if record:
        eff["glob"] = str(record.get("default_glob", defaults.get("glob", "*.md")))
        opts = record.get("options") or []
        if isinstance(opts, list):
            CORE_FIELDS = {"input", "output", "glob", "overwrite", "workers"}
            for opt in opts:
                try:
                    name = str(opt["name"])
                except Exception:
                    continue
                if name in CORE_FIELDS:
                    # Keep platform-driven defaults for common fields (esp. input/output)
                    continue
                if "default" in opt:
                    eff[name] = opt.get("default")
                else:
                    eff[name] = None
        return eff
    # If record missing and dev reflection allowed, keep previous behavior via reflection
    if os.environ.get("TOOLIPIE_DEV_ALLOW_REFLECTION") == "1":
        eff["glob"] = defaults.get("glob", "*.md")
        try:
            mod = __import__(f"toolipie.tools.{tool_key.replace('-', '_')}.run", fromlist=["run"])
            sig = inspect.signature(mod.run)
            for name, param in sig.parameters.items():
                if name in eff or name == "ctx":
                    continue
                d = param.default
                val = getattr(d, "default", d)
                eff[name] = None if val is inspect._empty else val
        except Exception:
            pass
        return eff
    # Strict default for glob if manifest missing
    eff["glob"] = defaults.get("glob", "*.md")
    return eff


def run_tool(tool_key: str, ctx: Any, params: Optional[dict[str, Any]] = None) -> None:
    """Dispatch to tool's run(ctx, ...) with filtered kwargs based on its signature.

    This avoids per-tool branching in callers. Callers pass only user-provided params;
    we remove common context fields and Nones, and forward what the tool accepts.
    """
    if params is None:
        params = {}
    # Resolve record from merged index
    idx = load_index()
    record = None
    for t in idx.get("tools", []) if isinstance(idx, dict) else []:
        if str(t.get("key")) == tool_key:
            record = t
            break
    if not record:
        raise RuntimeError(f"Tool '{tool_key}' is not registered. Run `toolipie scan` or install it.")

    entry = str(record.get("entry", "")).strip()
    # Resolve directory for file-based entries using source + rel_path
    src_kind = str(record.get("source", "")).strip()
    rel_path = str(record.get("rel_path", "")).strip()
    base_dir: Optional[Path] = None
    pkg_base = Path(__file__).resolve().parent
    if src_kind == "core":
        base_dir = pkg_base / "tools"
    elif src_kind == "plugin":
        base_dir = pkg_base / "plugins"
    run_func = None
    if ":" in entry:
        modspec, func_name = entry.split(":", 1)
        # File-based entry? treat as path (absolute or relative to record path)
        if modspec.endswith(".py") or (base_dir is not None and rel_path and (base_dir / rel_path / modspec).exists()):
            file_path = Path(modspec)
            if not file_path.is_absolute() and base_dir is not None:
                file_path = (base_dir / rel_path / modspec)
            if not file_path.exists():
                raise RuntimeError(f"Entry file not found for '{tool_key}': {file_path}")
            spec = importlib.util.spec_from_file_location(f"toolipie_plugin_{tool_key}", str(file_path))
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Failed to load entry module from {file_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            run_func = getattr(module, func_name, None)
        else:
            # Import via module path
            module = importlib.import_module(modspec)
            run_func = getattr(module, func_name, None)
    else:
        # Legacy core import as fallback (repo tools by module path)
        module = importlib.import_module(f"toolipie.tools.{tool_key.replace('-', '_')}.run")
        run_func = getattr(module, "run", None)
    if run_func is None:
        raise RuntimeError(f"Could not resolve run() for tool '{tool_key}' (entry: {entry}).")
    sig = inspect.signature(run_func)
    # Filter params: drop common fields and None values, keep only accepted kwargs
    core_fields = {"input", "output", "glob", "overwrite", "workers"}
    filtered: dict[str, Any] = {}
    for k, v in params.items():
        if k in core_fields:
            continue
        if v is None:
            continue
        if k in sig.parameters and k != "ctx":
            filtered[k] = v
    # Execute tool
    run_func(ctx, **filtered)
    # Post-run hint: if no input files were identified, inform the user succinctly.
    try:
        if not getattr(ctx, "files", []) or len(getattr(ctx, "files", [])) == 0:
            in_dir = None
            try:
                in_dir = getattr(ctx, "input_dir", None)
            except Exception:
                in_dir = None
            if in_dir is not None:
                print(f"0 input files identified in '{in_dir}'.")
            else:
                print("0 input files identified.")
    except Exception:
        # Best-effort hint; never fail the run due to messaging
        pass


def _write_index(index: dict[str, Any]) -> None:
    root = get_repo_root()
    out_dir = root / ".toolipie"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "index.json"
    out_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def scan_all_and_write_index() -> dict[str, Any]:
    """Scan repo tools and repo plugins; write unified .toolipie/index.json."""
    base_pkg = Path(__file__).resolve().parent
    tools_root = base_pkg / "tools"
    repo_plugins_root = base_pkg / "plugins"

    prev = load_index() or {}
    prev_map: dict[str, dict[str, Any]] = {}
    for t in prev.get("tools", []) if isinstance(prev, dict) else []:
        try:
            prev_map[str(t.get("key"))] = t
        except Exception:
            continue

    def scan(root: Path, source: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not root.exists():
            return out
        for child in sorted(p for p in root.iterdir() if p.is_dir()):
            manifest = child / "tool.yaml"
            if not manifest.exists():
                continue
            try:
                data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            if int(data.get("schema_version", 0)) != 1:
                continue
            name = str(data.get("name") or "").strip() or _snake_to_kebab(child.name)
            title = data.get("title") or name.replace("-", " ").title()
            summary = data.get("summary") or ""
            default_glob = data.get("default_glob") or "*"
            entry = str(data.get("entry") or ("run.py:run" if source == "plugin" else f"toolipie.tools.{child.name}.run:run"))
            options = data.get("options") or []
            requires = data.get("requires") or []
            rec: dict[str, Any] = {
                "key": name,
                "title": title,
                "summary": summary,
                "default_glob": default_glob,
                "entry": entry,
                "options": options,
                "requires": requires,
                "source": source,
                "rel_path": child.name,
            }
            prev_rec = prev_map.get(name)
            if prev_rec and isinstance(prev_rec.get("installed_at"), str):
                rec["installed_at"] = prev_rec["installed_at"]
            out.append(rec)
        return out

    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rec in scan(tools_root, "core") + scan(repo_plugins_root, "plugin"):
        k = str(rec.get("key"))
        if k in seen:
            continue
        seen.add(k)
        items.append(rec)

    idx: dict[str, Any] = {"version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "tools": items}
    _write_index(idx)
    return idx


__all__ = [
    "discover_tools",
    "get_tool_specs",
    "get_effective_defaults",
    "run_tool",
    "scan_all_and_write_index",
    "get_repo_plugins_dir",
]
