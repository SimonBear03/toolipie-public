 # Toolipie

Personal, extensible Python CLI platform to centralize your small “one‑off” scripts as reusable tools. Stop rewriting throwaway code — keep simple, personal utilities in one place with a consistent CLI.

Currently ships with four tools (more coming):
- [md-to-docx](src/toolipie/tools/md_to_docx/README.md)
- [md-to-pdf](src/toolipie/tools/md_to_pdf/README.md)
- [pdf-to-png](src/toolipie/tools/pdf_to_png/README.md)
- [png-prep-ocr](src/toolipie/tools/png_prep_ocr/README.md)

Build your own tool → see [Add your own tool](#add-your-own-tool-quick-guide).

**Tool Request** — Have a tool idea? Open an issue with the `tool request` label, and please include:
- Problem statement
- Example input/output
- Expected options
- Dependencies 

## Vision

- Build-your-own: add small focused tools under `src/toolipie/tools/<tool>/` and expose them via a strict manifest + index. More first‑party tools will be added over time.
- Plugin platform: manifests (`tool.yaml`) and a registry (`.toolipie/index.json`) are in place; future auto‑registration and installers will build on this. See `planning/PLUGIN_PLATFORM_PLAN.md`.
- AI assistance (future): an AI layer to guide tool creation (scaffold code, configure options, and generate docs).

## Install

Requirements: Python 3.10+ (matches `pyproject.toml`).

Step 1 (recommended): Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

Step 2: Install Toolipie (editable) and set up Git hooks
```bash
pip install -e .
pre-commit install
toolipie --help
```

## Usage

Run from the repository root so defaults resolve via `config.yaml`. Otherwise, pass explicit `--input` and `--output` paths.

General pattern (strict manifest + index):

```bash
# 0) Build the tool registry (run after adding/editing a tool or installing/removing plugins)
toolipie scan  # refreshes core tools and repo plugins in the index

# 1) Interactive TUI (arrow keys + Enter; reads the index)
#    Includes a “System Commands” group and an in‑TUI output console for runs
toolipie

# 2) List tools from the registry
toolipie list

# 3) Run a tool by key (use --param for tool options)
toolipie run md-to-pdf \
  --input input/md-to-pdf \
  --output output/md-to-pdf \
  --param preset=a4_report

toolipie run pdf-to-png \
  --input input/pdf-to-png \
  --output output/pdf-to-png \
  --param dpi=300

# Help overview
toolipie --help
```

## Core vs Plugin Tools

Toolipie treats tools in two categories, both managed inside the repo:

- Core tools: live under `src/toolipie/tools/<tool>/` and are maintained with the repository.
- Plugin tools: live under `src/toolipie/plugins/<tool>/` and are installed from a `.zip` via `toolipie install`.

Discovery is index‑driven and unified using a single index file at `.toolipie/index.json`. The `scan` command indexes both core tools and repo plugins. The tool manifest (`tool.yaml`) is identical for both; “core vs plugin” is tracked only in index metadata (`source: core|plugin`, `rel_path`). No `~/.toolipie` is used.

Common workflows:

- Add/remove a core tool:
  - Create/remove `src/toolipie/tools/<tool>/` with `run.py` and `tool.yaml`.
  - Run `toolipie scan` to refresh `.toolipie/index.json`.

- Install a plugin tool (into repo):
  - `toolipie install /path/to/tool.zip`
  - Then `toolipie list` or launch the TUI to see it.

- Uninstall a plugin tool (from repo):
  - `toolipie uninstall <tool-key>` removes `src/toolipie/plugins/<tool-key>/` and refreshes the index.

Security and safety notes:
- The installer validates manifests, prevents Zip Slip, and does not import plugin code during discovery. Only the index is used at startup; code is loaded lazily at run time.

Example plugin for local testing:
- A sample plugin that mirrors `pdf-to-png` is provided at `src/toolipie/plugins/pdf_to_png_plugin/`.
- From the repo root, create a zip and install it:
  ```bash
  zip -r pdf-to-png-plugin.zip src/toolipie/plugins/pdf_to_png_plugin
  toolipie install pdf-to-png-plugin.zip
  toolipie list
  ```


TUI basics:
- System Commands: scan registry, validate, list, install, uninstall, package, help, version.
  - Stays inside the System Commands submenu after each action until you back out.
  - Install Plugin: enter a .zip path; installs into `src/toolipie/plugins/` and rescans.
  - Uninstall Plugin: pick a plugin to remove from `src/toolipie/plugins/`.
  - Package Tool: create a distributable `.zip` for any core or plugin tool into `output/packages/`.
- Two‑pane selector: ↑/↓ navigate tools; right pane shows title + description.
- Options panel: Enter opens; q backs out; Ctrl+C twice exits the program.
- Text prompts: Enter confirms; Esc cancels; ‘q’ is treated as a normal character.
- Runs: execute in an in‑TUI console under a subprocess/PTY for universal behavior.
  - Progress bars update in place; scroll with ↑/↓.
  - q requests cancel (SIGTERM). If the process doesn’t exit within a short grace period, it escalates to SIGKILL and returns automatically.
  - If no input files are identified for a run (e.g., the default input folder does not exist or contains no files matching the glob), the console prints: "0 input files identified in '<path>'."
- Options are generated from each tool’s manifest (`tool.yaml`) plus common fields (`input`, `output`, `glob`, `overwrite`, `workers`).

Pandoc is only required for the DOCX tool (`md-to-docx`) and is auto-handled by the tool. See the `md-to-docx` tool docs.

Note: PDF generation uses a pure-Python pipeline (MarkdownIt → WeasyPrint). Styling is controlled via `--param css=/path/to.css` or `--param preset=a4_report` (presets live under `src/toolipie/tools/md_to_pdf/assets/presets/<name>.css`). The `pdf-to-png` tool uses `pypdfium2` and outputs PNGs into per-PDF folders.

## How Toolipie works

- Tools live under `src/toolipie/tools/<tool>/` (snake_case) and expose `run(ctx, ...)`.
- Each tool declares a manifest `tool.yaml` (schema_version: 1) with its key (kebab-case), title, summary, default_glob, options, and optional `requires`.
- `toolipie scan` reads tool manifests and writes a registry at `.toolipie/index.json`.
- The TUI and CLI read only the registry for discovery and options (fast startup). No runtime scanning/imports.
- For each run, Toolipie builds a Context with resolved input/output paths, file list (via glob), defaults, and writes `run.jsonl` per tool run.
- TUI runs tools in a subprocess under a PTY: consistent progress rendering and universal cancel/kill.
- Input/output defaults: `<repo>/<paths.input_root>/<tool-key>` and `<repo>/<paths.output_root>/<tool-key>`; override via `--input`/`--output` or from GUI pickers.
- Default input behavior:
  - If you pass a file path for `--input`, the platform treats its parent directory as `ctx.input_dir` and processes only that file.
  - If you pass a directory (or rely on the default), the platform uses that directory as-is, even if it does not exist. There is no fallback to the parent folder. If no files are found, the run processes zero files and prints: "0 input files identified in '<path>'."

Manifest notes:
- Options are optional by default. If an option must be provided, add `required: true` (or supply a `default`).
- Example option: `{ name: dpi, type: int, default: 300, help: "Render DPI" }`.

Cancellation model:
- TUI runs: subprocess with q → SIGTERM and auto‑escalation to SIGKILL after ~2s if needed. A short status is shown, then the panel closes automatically.
- CLI runs: normal process behavior (Ctrl+C to interrupt). Tools may opt‑in to cooperative cancellation via `ctx.cancel_event`, but this is not required for platform‑level cancel.

Folder conventions:

- `tools/md_to_docx` → `input/md-to-docx/` and `output/md-to-docx/`
- `tools/md_to_pdf`  → `input/md-to-pdf/` and `output/md-to-pdf/`
- `tools/pdf_to_png` → `input/pdf-to-png/` and `output/pdf-to-png/<pdf-name>/`
- Presets live per tool under `src/toolipie/tools/<tool>/assets/presets/` (e.g., CSS or templates).

## Quickstart — add your own tool

1) Create a folder: `src/toolipie/tools/my_tool/`

2) Add `run.py` exposing `run(ctx, ...)` (iterate `ctx.files`, write to `ctx.output_dir`, log each file to `ctx.run_log`).

3) Add a manifest `tool.yaml` with: `schema_version: 1`, `name` (kebab-case), `title`, `summary`, `default_glob`, and `options` (list of {name, type, help, default?, choices?}).

4) Run `toolipie scan` to refresh `.toolipie/index.json`.

5) Run your tool:
```bash
toolipie list
toolipie run my-tool --input input/my-tool --output output/my-tool --param option=value
```

See `CONTRIBUTING.md` for a fuller quickstart.

## Tool documentation

- Each tool ships a focused README under `src/toolipie/tools/<tool>/README.md`.
- See `CONTRIBUTING.md` for adding new tools.

## Versioning

- Platform (core) uses Semantic Versioning (MAJOR.MINOR.PATCH). Only core/platform changes bump the version and get a `vX.Y.Z` git tag and GitHub Release. See `CHANGELOG.md`.
- Tools can evolve independently without bumping the platform version. Tool-only updates may be mentioned under “Unreleased” but do not create a new platform release.
- Programmatic version: `toolipie --version` prints the platform version. Internal `PLATFORM_API_VERSION` signals plugin interface compatibility.
