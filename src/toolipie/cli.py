from __future__ import annotations

from typing import Optional
from pathlib import Path
import sys
from typer import Context as TyperContext

import typer
from . import __version__

from .core import build_context, load_config, get_repo_root
# Lazy-import tool modules inside handlers to avoid loading heavy/native deps at startup

from .runner import (
    scan_all_and_write_index,
    validate_manifest_file,
    validate_index_dict,
    load_index,
    validate_manifest_dict,
    discover_tools as runner_discover_tools,
    get_tool_specs as runner_get_tool_specs,
    get_effective_defaults as runner_get_effective_defaults,
    run_tool as runner_run_tool,
)
import zipfile
import yaml
from datetime import datetime, timezone
import shutil
import os
import signal
import threading
from datetime import datetime

app = typer.Typer(
    help="Toolipie — personal CLI toolbox", no_args_is_help=False, add_completion=False
)


def _version_callback(value: bool) -> None:
    if value:
        print(__version__)
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: TyperContext,
    version: bool = typer.Option(
        False,
        "--version",
        help="Show Toolipie platform version and exit",
        is_eager=True,
        callback=_version_callback,
    )
):
    """Toolipie root command.

    If no subcommand is provided, launch the curses TUI.
    """
    # If a subcommand is being invoked, do nothing here
    if ctx.invoked_subcommand is not None:
        return
    # Only launch TUI in an interactive TTY
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        typer.echo(app.get_help(ctx))
        return
    # Discover tools via the runner/index (strict mode). Even if none found,
    # continue into the TUI so users can access System Commands (e.g., Scan).
    try:
        from .tui import launch_menu, launch_options_panel, launch_output_panel, launch_text_prompt
    except Exception as e:
        typer.secho(f"Curses UI not available: {e}", err=True, fg=typer.colors.RED)
        typer.echo(app.get_help(ctx))
        return
    while True:
        # Refresh tools each loop to reflect recent scans/installs
        tools = runner_discover_tools()
        # Compose top-level menu with a System Commands group at the top
        items = ([
            {
                "key": "::system",
                "title": "System Commands",
                "desc": "Scan registry, validate, help, and more",
            }
        ] + tools)

        sel = launch_menu(items, title="Toolipie — Select a Command")
        if not sel:
            return
        if sel == "::system":
            sys_items = [
                {"key": "scan", "title": "Scan Registry", "desc": "Rebuild .toolipie/index.json"},
                {"key": "validate", "title": "Validate Index & Manifests", "desc": "Check manifests and registry structure"},
                {"key": "list", "title": "List Tools", "desc": "Print registered tools"},
                {"key": "install", "title": "Install Plugin", "desc": "Enter a .zip path to install"},
                {"key": "uninstall", "title": "Uninstall Plugin", "desc": "Remove a plugin tool"},
                {"key": "package", "title": "Package Tool", "desc": "Create a distributable .zip from a tool"},
                {"key": "help", "title": "Help Overview", "desc": "Show CLI help"},
                {"key": "version", "title": "Version", "desc": "Show Toolipie version"},
            ]
            # Stay in System Commands menu until user backs out
            while True:
                sys_sel = launch_menu(sys_items, title="Toolipie — System Commands", allow_back=True)
                if not sys_sel:
                    break  # back to top-level menu
                try:
                    if sys_sel == "scan":
                        def do_scan() -> None:
                            idx_all = scan_all_and_write_index()
                            core = 0
                            plugins = 0
                            for t in idx_all.get("tools", []) if isinstance(idx_all, dict) else []:
                                src = str(t.get("source", ""))
                                if src == "core":
                                    core += 1
                                elif src == "plugin":
                                    plugins += 1
                            print(f"Indexed {core} core tool(s) and {plugins} plugin tool(s).")
                        launch_output_panel("Scan Registry", do_scan)
                    elif sys_sel == "validate":
                        def do_validate() -> None:
                            base_dir = Path(__file__).resolve().parent
                            tools_root = base_dir / "tools"
                            plugins_root = base_dir / "plugins"
                            errors: list[str] = []
                            checked = 0
                            if tools_root.exists():
                                for child in sorted(p for p in tools_root.iterdir() if p.is_dir()):
                                    manifest = child / "tool.yaml"
                                    if manifest.exists():
                                        checked += 1
                                        errs = validate_manifest_file(manifest)
                                        for e in errs:
                                            errors.append(f"manifest: {e}")
                            if plugins_root.exists():
                                for child in sorted(p for p in plugins_root.iterdir() if p.is_dir()):
                                    manifest = child / "tool.yaml"
                                    if manifest.exists():
                                        checked += 1
                                        errs = validate_manifest_file(manifest)
                                        for e in errs:
                                            errors.append(f"manifest: {e}")
                            idx_all = load_index()
                            if idx_all is not None:
                                errors.extend([f"index: {e}" for e in validate_index_dict(idx_all)])
                            print(f"Validated {checked} manifest(s).")
                            if idx_all is not None:
                                print("Validated index file .toolipie/index.json.")
                            if errors:
                                print("Issues found:")
                                for e in errors:
                                    print(f"- {e}")
                            else:
                                print("All good — no issues found.")
                        launch_output_panel("Validate Index & Manifests", do_validate)
                    elif sys_sel == "list":
                        def do_list() -> None:
                            lst = runner_discover_tools()
                            for t in lst:
                                title = f" — {t['title']}" if t.get("title") and t['title'] != t['key'].replace('-', ' ').title() else ""
                                desc = f": {t['desc']}" if t.get("desc") else ""
                                print(f"{t['key']}{title}{desc}")
                        launch_output_panel("List Tools", do_list)
                    elif sys_sel == "install":
                        zip_path = launch_text_prompt("Install Plugin", "Zip file path: ", "")
                        if not zip_path:
                            continue
                        # Sanitize common pasted path formats with surrounding quotes
                        if (zip_path.startswith('"') and zip_path.endswith('"')) or (zip_path.startswith("'") and zip_path.endswith("'")):
                            zip_path = zip_path[1:-1]
                        def do_install() -> None:
                            try:
                                install(zip_path, True)  # type: ignore[name-defined]
                            except SystemExit:
                                return
                            except Exception as e:
                                print(f"Install failed: {e}")
                                return
                            # After install, rescan unified index to include the new plugin
                            try:
                                scan_all_and_write_index()
                            except Exception as e:
                                print(f"Rescan failed after install: {e}")
                        launch_output_panel("Install Plugin", do_install)
                    elif sys_sel == "uninstall":
                        # Build a list of plugins from unified index
                        idx_all = load_index() or {}
                        plugins = []
                        for t in idx_all.get("tools", []) if isinstance(idx_all, dict) else []:
                            try:
                                if str(t.get("source")) != "plugin":
                                    continue
                                key = str(t.get("key"))
                                title = str(t.get("title") or key)
                                desc = str(t.get("summary") or "")
                                plugins.append({"key": key, "title": title, "desc": desc})
                            except Exception:
                                continue
                        if not plugins:
                            def no_plugins() -> None:
                                print("No plugins installed.")
                            launch_output_panel("Uninstall Plugin", no_plugins)
                        else:
                            sel_plugin = launch_menu(plugins, title="Select Plugin to Uninstall", allow_back=True)
                            if not sel_plugin:
                                continue
                            def do_uninstall() -> None:
                                try:
                                    # Force uninstall without interactive prompt inside TUI
                                    uninstall(sel_plugin, True)  # type: ignore[name-defined]
                                except SystemExit:
                                    # Typer may raise Exit/SystemExit; ignore here
                                    return
                                except Exception as e:
                                    print(f"Uninstall failed: {e}")
                                    return
                            launch_output_panel(f"Uninstall {sel_plugin}", do_uninstall)
                    elif sys_sel == "package":
                        # Select any tool (core or plugin)
                        all_tools = runner_discover_tools()
                        if not all_tools:
                            def no_tools() -> None:
                                print("No tools found. Run 'toolipie scan'.")
                            launch_output_panel("Package Tool", no_tools)
                        else:
                            sel_tool = launch_menu(all_tools, title="Select Tool to Package", allow_back=True)
                            if not sel_tool:
                                continue
                            def do_package_tui() -> None:
                                try:
                                    zpath = _package_tool(sel_tool, None, True)
                                    print(f"Packaged '{sel_tool}' → {zpath}")
                                except SystemExit:
                                    pass
                                except Exception as e:
                                    print(f"Packaging failed: {e}")
                            launch_output_panel(f"Packaging {sel_tool}", do_package_tui)
                    elif sys_sel == "help":
                        def do_help() -> None:
                            try:
                                from typer.main import get_command
                                import click  # type: ignore
                                cmd = get_command(app)
                                with click.Context(cmd) as c:
                                    print(cmd.get_help(c))
                            except Exception:
                                # Fallback minimal help
                                print("Toolipie — personal CLI toolbox")
                                print("Commands: list, run, scan, validate, install")
                                print("Tip: Run 'toolipie' to launch the TUI.")
                        launch_output_panel("Help Overview", do_help)
                    elif sys_sel == "version":
                        def do_version() -> None:
                            print(__version__)
                        launch_output_panel("Version", do_version)
                except Exception as e:
                    def show_err() -> None:
                        print(f"System command '{sys_sel}' failed: {e}")
                    launch_output_panel("System Command Error", show_err)
            # After exiting system menu, go back to top-level menu
            continue
        elif sel.startswith("::"):
            # Ignore any synthetic keys
            continue
        try:
            specs = runner_get_tool_specs(sel)
        except Exception as e:
            typer.secho(str(e), err=True, fg=typer.colors.RED)
            continue
        eff = runner_get_effective_defaults(sel)
        result = launch_options_panel(f"{sel} — Options", "Edit values or run", specs, eff)
        if result and result.get("action") == "exit":
            return
        if not result or result.get("action") == "back":
            # Go back to tool selection without exiting
            continue
        vals = result.get("values", {})
        if result.get("action") == "run_default":
            # Prefer running as a subprocess so we can cancel universally and preserve progress updates
            run_ctx = build_context(sel, None, None, eff.get("glob"), None, None)
            argv = [
                sys.executable,
                "-m",
                "toolipie.cli",
                "run",
                sel,
                "--input",
                str(run_ctx.input_dir),
                "--output",
                str(run_ctx.output_dir),
            ]
            if eff.get("glob"):
                argv += ["--glob", str(eff.get("glob"))]
            # workers/overwrite left to defaults unless user changes them in custom run
            try:
                from .tui import launch_process_panel
                launch_process_panel(f"Running {sel}", argv)
            except Exception:
                # Fallback to in-process if PTY or spawn fails
                import threading as _threading
                run_ctx.cancel_event = _threading.Event()
                def do_run() -> None:
                    runner_run_tool(sel, run_ctx, {})
                launch_output_panel(f"Running {sel}", do_run, cancel_event=run_ctx.cancel_event)
            continue
        g = vals.get("glob") if vals.get("glob") is not None else eff.get("glob")
        run_ctx = build_context(sel, vals.get("input"), vals.get("output"), g, vals.get("overwrite"), vals.get("workers"))
        # Build subprocess argv mirroring the selected options
        argv = [sys.executable, "-m", "toolipie.cli", "run", sel]
        def add_flag(name: str, value: object | None) -> None:
            if value is None:
                return
            argv.extend([f"--{name}", str(value)])
        add_flag("input", vals.get("input") or str(run_ctx.input_dir))
        add_flag("output", vals.get("output") or str(run_ctx.output_dir))
        add_flag("glob", vals.get("glob") if vals.get("glob") is not None else eff.get("glob"))
        if vals.get("workers") is not None:
            add_flag("workers", vals.get("workers"))
        if vals.get("overwrite") is not None:
            add_flag("overwrite", "true" if bool(vals.get("overwrite")) else "false")
        # tool-specific params: pass via --param key=value
        CORE = {"input", "output", "glob", "overwrite", "workers"}
        for k, v in (vals or {}).items():
            if k in CORE or v is None:
                continue
            argv.extend(["--param", f"{k}={v}"])
        try:
            from .tui import launch_process_panel
            launch_process_panel(f"Running {sel}", argv)
        except Exception:
            # Fallback to in-process
            import threading as _threading
            run_ctx.cancel_event = _threading.Event()
            def do_run_custom() -> None:
                runner_run_tool(sel, run_ctx, vals)
            launch_output_panel(f"Running {sel}", do_run_custom, cancel_event=run_ctx.cancel_event)
        continue

## menu command removed per request


def _run_selected(sel: str, ctx, vals: dict) -> None:
    try:
        runner_run_tool(sel, ctx, vals)
    except Exception as e:
        typer.secho(f"Failed to run {sel}: {e}", err=True, fg=typer.colors.RED)


def _package_tool(tool_key: str, dest_dir: str | None, timestamped: bool = True, override_key: str | None = None) -> Path:
    """Create a plugin zip for a tool (core or plugin) into output/packages by default.

    - Rewrites manifest entry to `run.py:run` for portability.
    - Packs tool folder contents under a top-level `<tool-key>/` directory in the zip.
    Returns the written zip path.
    """
    idx = load_index() or {}
    rec = None
    for t in idx.get("tools", []) if isinstance(idx, dict) else []:
        if str(t.get("key")) == tool_key:
            rec = t
            break
    if rec is None:
        raise typer.BadParameter(f"Tool '{tool_key}' is missing from the registry. Run `toolipie scan`.")
    src_kind = str(rec.get("source", ""))
    rel_path = str(rec.get("rel_path", ""))
    base_dir = Path(__file__).resolve().parent
    if src_kind == "core":
        tool_dir = base_dir / "tools" / rel_path
    elif src_kind == "plugin":
        tool_dir = base_dir / "plugins" / rel_path
    else:
        raise typer.BadParameter(f"Unknown source kind '{src_kind}' for tool '{tool_key}'.")
    if not tool_dir.exists():
        raise typer.BadParameter(f"Tool directory not found: {tool_dir}")

    # Output directory: <repo>/<output_root>/packages
    if dest_dir is None:
        cfg = load_config()
        repo = get_repo_root()
        out_root = Path(cfg.get("paths", {}).get("output_root", "output"))
        out_dir = (repo / out_root / "packages").resolve()
    else:
        out_dir = Path(dest_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S") if timestamped else None
    zip_name = f"{tool_key}-{ts}.zip" if ts else f"{tool_key}.zip"
    zip_path = out_dir / zip_name

    # Prepare manifest content (rewrite entry to run.py:run)
    manifest_path = tool_dir / "tool.yaml"
    if not manifest_path.exists():
        raise typer.BadParameter(f"Manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f.read()) or {}
    new_key = override_key or tool_key
    data["name"] = new_key
    data["schema_version"] = 1
    data["entry"] = "run.py:run"

    # Build file list to include
    ignore_names = {"__pycache__", ".DS_Store"}
    ignore_ext = {".pyc", ".pyo"}
    to_include: list[tuple[Path, str]] = []
    top = new_key
    # Walk tool_dir and include everything except tool.yaml (we write modified) and ignored files
    for root, dirs, files in os.walk(tool_dir):
        rpath = Path(root)
        # filter dirs in-place to skip ignored
        dirs[:] = [d for d in dirs if d not in ignore_names]
        for fn in files:
            if fn in ignore_names:
                continue
            ext = os.path.splitext(fn)[1]
            if ext in ignore_ext:
                continue
            p = rpath / fn
            # skip original manifest; we'll add the modified one below
            if p == manifest_path:
                continue
            rel = str(p.relative_to(tool_dir))
            arc = str(Path(top) / rel)
            to_include.append((p, arc))

    # Ensure run.py exists
    if not (tool_dir / "run.py").exists():
        raise typer.BadParameter(f"Entry file missing: {tool_dir / 'run.py'}")

    with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # write modified manifest at top/tool.yaml
        zf.writestr(str(Path(top) / "tool.yaml"), yaml.safe_dump(data, sort_keys=False).encode("utf-8"))
        # write other files
        for src, arc in to_include:
            zf.write(str(src), arcname=arc)
    return zip_path


@app.command("list")
def list_tools() -> None:
    """List available tools from the registry (.toolipie/index.json)."""
    tools = runner_discover_tools()
    if not tools:
        return
    for t in tools:
        title = f" — {t['title']}" if t.get("title") and t['title'] != t['key'].replace('-', ' ').title() else ""
        desc = f": {t['desc']}" if t.get("desc") else ""
        typer.echo(f"{t['key']}{title}{desc}")


@app.command("package")
def package(
    tool: str = typer.Argument(..., help="Tool key (kebab-case) to package"),
    dest: Optional[str] = typer.Option(None, "--dest", help="Destination output directory (defaults to output/packages)"),
    no_timestamp: bool = typer.Option(False, "--no-timestamp", help="Do not append a timestamp to the filename"),
    as_key: Optional[str] = typer.Option(None, "--as-key", help="Override the packaged tool key (useful to avoid collisions)"),
) -> None:
    """Package a tool (core or plugin) as a .zip consumable by `toolipie install`.

    The archive contains a top-level `<tool>/` folder with `tool.yaml`, `run.py`, and assets.
    The modified manifest uses `entry: run.py:run` for portability.
    """
    try:
        zpath = _package_tool(tool, dest, not no_timestamp, as_key)
    except typer.BadParameter as e:
        typer.secho(str(e), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.secho(f"Wrote {zpath}", fg=typer.colors.GREEN)


def _parse_params(param_items: list[str], tool_key: str) -> dict:
    """Parse --param key=value items, coerce types using manifest options, enforce choices."""
    # Use unified index
    idx = load_index() or {}
    record = None
    for t in idx.get("tools", []) if isinstance(idx, dict) else []:
        if str(t.get("key")) == tool_key:
            record = t
            break
    if not record:
        raise typer.BadParameter(
            f"Tool '{tool_key}' is missing from the registry. Run `toolipie scan`."
        )
    option_map = {}
    for opt in record.get("options", []) or []:
        name = str(opt.get("name"))
        option_map[name] = opt
    out: dict[str, object] = {}
    for item in param_items:
        if "=" not in item:
            raise typer.BadParameter(f"Invalid --param format: '{item}'. Use key=value.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in {"input", "output", "glob", "overwrite", "workers"}:
            # Common fields should be provided via dedicated flags
            raise typer.BadParameter(f"'{key}' must be provided via dedicated flags, not --param.")
        opt = option_map.get(key)
        if not opt:
            # unknown option; allow but pass as string
            out[key] = value
            continue
        typ = str(opt.get("type", "str"))
        if typ == "bool":
            val_lower = value.lower()
            if val_lower in {"1", "true", "yes", "y", "on"}:
                out[key] = True
            elif val_lower in {"0", "false", "no", "n", "off"}:
                out[key] = False
            else:
                raise typer.BadParameter(f"Invalid boolean for {key}: '{value}'. Use true/false.")
        elif typ == "int":
            try:
                out[key] = int(value)
            except Exception:
                raise typer.BadParameter(f"Invalid int for {key}: '{value}'.")
        elif typ == "float":
            try:
                out[key] = float(value)
            except Exception:
                raise typer.BadParameter(f"Invalid float for {key}: '{value}'.")
        else:
            out[key] = value
        if "choices" in opt and isinstance(opt["choices"], list):
            if str(out[key]) not in [str(c) for c in opt["choices"]]:
                raise typer.BadParameter(
                    f"Invalid value for {key}: '{out[key]}'. Choices: {opt['choices']}"
                )
    return out


@app.command("run")
def run_tool(
    tool: str = typer.Argument(..., help="Tool key (kebab-case) from the registry"),
    input: Optional[str] = typer.Option(None, help="Input directory or file"),
    output: Optional[str] = typer.Option(None, help="Output directory"),
    glob: Optional[str] = typer.Option(None, help="Glob for input files"),
    overwrite: Optional[bool] = typer.Option(None, help="Overwrite outputs if exist"),
    workers: Optional[int] = typer.Option(None, help="Number of workers (reserved)"),
    param: list[str] = typer.Option(
        [],
        "--param",
        help="Tool-specific parameter as key=value (repeatable)",
    ),
) -> None:
    """Run a tool by key using manifest-driven defaults and --param overrides.

    Examples:
      toolipie run md-to-pdf --input input/md --output output/pdf --param preset=a4_report
      toolipie run pdf-to-png --input input/pdf --output output/png --param dpi=200
    """
    # Build default set
    eff = runner_get_effective_defaults(tool)
    # Parse params and coerce types
    params = _parse_params(param or [], tool)
    # Enforce required options: any manifest option without a default must be provided
    idx = load_index() or {}
    record = None
    for t in idx.get("tools", []) if isinstance(idx, dict) else []:
        if str(t.get("key")) == tool:
            record = t
            break
    if not record:
        typer.secho(
            f"Tool '{tool}' is missing from the registry. Run `toolipie scan`.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    # Only enforce options explicitly marked as required=true in manifest.
    # Options without a default are treated as optional by default; tools may error if truly required.
    missing: list[str] = []
    for opt in record.get("options", []) or []:
        try:
            name = str(opt.get("name"))
        except Exception:
            continue
        if str(opt.get("required", "")).lower() in {"1", "true", "yes"}:
            if name not in params and "default" not in opt:
                missing.append(name)
    if missing:
        typer.secho(
            "Missing required option(s): " + ", ".join(f"'{m}'" for m in missing) +
            ". Provide them using --param name=value.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    # Common flags
    g = glob if glob is not None else eff.get("glob")
    ctx = build_context(tool, input, output, g, overwrite, workers)
    # Attach a cancel event for cooperative tools; set on first Ctrl+C
    ctx.cancel_event = threading.Event()

    # Informative tip for users running directly via CLI
    typer.secho("Tip: Press Ctrl+C to cancel; press again to force exit.", fg=typer.colors.CYAN)

    # SIGINT handling: first Ctrl+C requests cancel; second forces exit
    seen_sigint = {"count": 0}

    def _on_sigint(signum, frame):  # type: ignore[no-redef]
        try:
            seen_sigint["count"] += 1
            if seen_sigint["count"] == 1:
                try:
                    ctx.cancel_event.set()
                except Exception:
                    pass
                typer.secho(
                    "Cancel requested… finishing current work. Press Ctrl+C again to force exit.",
                    err=True,
                    fg=typer.colors.YELLOW,
                )
            else:
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            raise
        except Exception:
            # Best-effort: fall back to immediate exit on unexpected errors in handler
            raise KeyboardInterrupt

    prev = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _on_sigint)
    try:
        runner_run_tool(tool, ctx, params)
    except KeyboardInterrupt:
        raise typer.Exit(code=130)
    finally:
        # Restore previous handler
        try:
            signal.signal(signal.SIGINT, prev)  # type: ignore[arg-type]
        except Exception:
            pass


@app.command("scan")
def scan() -> None:
    """Scan tool folders for manifests and write .toolipie/index.json.

    This is a stub that builds a local index for fast UI startup. It only
    considers tools with a tool.yaml manifest (schema_version: 1).
    """
    # Rebuild unified index (core + plugins)
    try:
        idx = scan_all_and_write_index()
    except Exception as e:
        typer.secho(f"Scan failed: {e}", err=True, fg=typer.colors.RED)
        return
    core_count = 0
    plugin_count = 0
    for t in idx.get("tools", []) if isinstance(idx, dict) else []:
        src = str(t.get("source", ""))
        if src == "core":
            core_count += 1
        elif src == "plugin":
            plugin_count += 1
    typer.secho(
        f"Indexed {core_count} core tool(s) and {plugin_count} plugin tool(s).",
        fg=typer.colors.GREEN,
    )


@app.command("validate")
def validate() -> None:
    """Validate tool manifests and the tool registry index.

    - Validates all src/toolipie/tools/**/tool.yaml files
    - Validates .toolipie/index.json structure (if present)
    - Prints a human-friendly summary of issues
    """
    base_dir = Path(__file__).resolve().parent
    tools_root = base_dir / "tools"
    plugins_root = base_dir / "plugins"
    errors: list[str] = []
    checked = 0
    if tools_root.exists():
        for child in sorted(p for p in tools_root.iterdir() if p.is_dir()):
            manifest = child / "tool.yaml"
            if manifest.exists():
                checked += 1
                errs = validate_manifest_file(manifest)
                for e in errs:
                    errors.append(f"manifest: {e}")
    if plugins_root.exists():
        for child in sorted(p for p in plugins_root.iterdir() if p.is_dir()):
            manifest = child / "tool.yaml"
            if manifest.exists():
                checked += 1
                errs = validate_manifest_file(manifest)
                for e in errs:
                    errors.append(f"manifest: {e}")
    idx = load_index()
    if idx is not None:
        errors += [f"index: {e}" for e in validate_index_dict(idx)]
    typer.secho(f"Validated {checked} manifest(s).", fg=typer.colors.CYAN)
    if idx is not None:
        typer.secho("Validated index file .toolipie/index.json.", fg=typer.colors.CYAN)
    if errors:
        typer.secho("Issues found:", fg=typer.colors.RED)
        for e in errors:
            typer.echo(f"- {e}")
        raise typer.Exit(code=1)
    else:
        typer.secho("All good — no issues found.", fg=typer.colors.GREEN)


@app.command("install")
def install(
    package: str = typer.Argument(..., help="Path to a tool .zip package"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing plugin without prompting"),
) -> None:
    """Install a tool from a .zip package into src/toolipie/plugins and update the index.

    Package layout:
      - tool.yaml (schema_version: 1)
      - run.py
      - assets/ (optional)
      - README.md (optional)
    These may be at the root of the zip or within a single top-level folder.
    """
    zip_path = Path(package).expanduser().resolve()
    if not zip_path.exists() or not zipfile.is_zipfile(str(zip_path)):
        typer.secho(f"Not a valid zip file: {zip_path}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1)
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        names = zf.namelist()
        # Find manifest
        manifest_name = None
        base_prefix = ""
        candidates = [n for n in names if n.endswith("tool.yaml")]
        if not candidates:
            typer.secho("Package missing tool.yaml", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)
        # Prefer shortest path (root), else use first
        manifest_name = sorted(candidates, key=lambda s: len(s))[0]
        base_prefix = str(Path(manifest_name).parent)
        if base_prefix in (".", "/"):
            base_prefix = ""
        # Check run.py exists
        run_rel = str(Path(base_prefix) / "run.py") if base_prefix else "run.py"
        if run_rel not in names:
            typer.secho("Package missing run.py", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)
        # Load manifest content
        with zf.open(manifest_name) as f:
            try:
                data = yaml.safe_load(f.read().decode("utf-8")) or {}
            except Exception as e:
                typer.secho(f"Invalid YAML in manifest: {e}", err=True, fg=typer.colors.RED)
                raise typer.Exit(code=1)
        errs = validate_manifest_dict(data, path_hint=f"{zip_path}!/{manifest_name}")
        if errs:
            typer.secho("Manifest validation errors:", err=True, fg=typer.colors.RED)
            for e in errs:
                typer.echo(f"- {e}")
            raise typer.Exit(code=1)
        tool_key = str(data.get("name")).strip()
        # Destination (repo plugins folder)
        repo_plugins_dir = Path(__file__).resolve().parent / "plugins"
        dest = repo_plugins_dir / tool_key
        if dest.exists():
            if not force:
                overwrite = typer.confirm(
                    f"Plugin '{tool_key}' already exists at {dest}. Overwrite?",
                    default=False,
                )
                if not overwrite:
                    typer.secho("Aborted. Nothing installed.", fg=typer.colors.YELLOW)
                    raise typer.Exit(code=1)
            # remove existing
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        # Extract only members under base_prefix
        base_resolved = dest.resolve()
        for n in names:
            if base_prefix:
                if not n.startswith(base_prefix.rstrip("/") + "/") and n != base_prefix:
                    continue
                rel = n[len(base_prefix):].lstrip("/")
            else:
                rel = n
            if not rel:
                continue
            if n.endswith("/"):
                # directory
                target = (dest / rel).resolve()
                try:
                    target.relative_to(base_resolved)
                except Exception:
                    # Zip Slip attempt — skip
                    continue
                target.mkdir(parents=True, exist_ok=True)
            else:
                # file
                target = (dest / rel).resolve()
                try:
                    target.relative_to(base_resolved)
                except Exception:
                    # Zip Slip attempt — skip
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(n) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
        # Rebuild unified index to include the new plugin
        scan_all_and_write_index()
        typer.secho(f"Installed '{tool_key}' to {dest}", fg=typer.colors.GREEN)


@app.command("uninstall")
def uninstall(
    tool_key: str = typer.Argument(..., help="Tool key (kebab-case) to uninstall (plugins only)"),
    force: bool = typer.Option(False, "--force", help="Do not prompt before removal"),
) -> None:
    """Uninstall a plugin from src/toolipie/plugins. Does not remove core tools."""
    idx = load_index() or {"tools": []}
    rec = None
    for t in idx.get("tools", []) if isinstance(idx, dict) else []:
        if str(t.get("key")) == tool_key:
            rec = t
            break
    if rec is None:
        typer.secho(f"Plugin '{tool_key}' not found in index.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    if str(rec.get("source")) != "plugin":
        typer.secho(f"'{tool_key}' is a core tool; uninstall is only for plugins.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    rel_path = str(rec.get("rel_path", ""))
    repo_plugins_dir = Path(__file__).resolve().parent / "plugins"
    dest = repo_plugins_dir / rel_path if rel_path else None
    # Legacy fallback: accept absolute 'path' from older indices
    if dest is None or str(dest).endswith("plugins"):
        legacy_path = rec.get("path")
        if isinstance(legacy_path, str) and legacy_path:
            dest = Path(legacy_path).expanduser().resolve()
    if dest is None:
        typer.secho("Cannot determine plugin directory (missing rel_path/path). Rescan and try again.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    # Safety: only allow removal under the repo plugins directory
    try:
        dest_resolved = dest.resolve()
        repo_plugins_dir.resolve()
        if repo_plugins_dir not in dest_resolved.parents and dest_resolved != repo_plugins_dir:
            typer.secho(f"Refusing to remove non-repo path: {dest_resolved}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
    except Exception:
        pass
    if dest.exists() and not force:
        if not typer.confirm(f"Remove plugin '{tool_key}' at {dest}?", default=False):
            typer.secho("Aborted.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
    try:
        if dest.exists():
            shutil.rmtree(dest)
    except Exception as e:
        typer.secho(f"Failed to remove plugin folder: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    # Rebuild unified index after removal
    scan_all_and_write_index()
    typer.secho(f"Uninstalled '{tool_key}'.", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
