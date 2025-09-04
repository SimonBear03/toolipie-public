 # Toolipie

Personal, extensible Python CLI platform to centralize your small “one‑off” scripts as reusable tools. Stop rewriting throwaway code — keep simple, personal utilities in one place with a consistent CLI.

Currently ships with four tools (more coming):
- [md-to-docx](src/toolipie/tools/md_to_docx/README.md)
- [md-to-pdf](src/toolipie/tools/md_to_pdf/README.md)
- [pdf-to-png](src/toolipie/tools/pdf_to_png/README.md)
- [png-prep-ocr](src/toolipie/tools/png_prep_ocr/README.md)

Build your own tool → see [Add your own tool](#add-your-own-tool-quick-guide).

**Tool Request** — Have a tool idea? Open an issue with the `tool request` label, and plz include:
- Problem statement
- Example input/output
- Expected options
- Dependencies 

## Vision

- Build-your-own: add small focused tools under `src/toolipie/tools/<tool>/` and expose them via the CLI. More first‑party tools will be added over time.
- Plugin platform: planned auto‑discovery and per‑tool specs for zero‑boilerplate command registration. See [`PLUGIN_PLATFORM_PLAN.md`](PLUGIN_PLATFORM_PLAN.md).
- AI assistance (future): an AI layer to guide tool creation (scaffold code, configure options, and generate docs) similar to how Cursor assists this repo today.

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

Run commands from the repository root so defaults resolve via `config.yaml`. Otherwise, pass explicit `--input` and `--output` paths.

General pattern:

```bash
# Syntax
toolipie <command> [options]

# Quick start (defaults)
toolipie md-to-pdf

# Override paths (optional)
toolipie md-to-pdf --input input/md-to-pdf --output output/md-to-pdf

# Discover commands and options
toolipie --help
toolipie <command> --help
```

Pandoc is only required for the DOCX tool (`md-to-docx`) and is auto-handled by the tool. See the `md-to-docx` tool docs.

Note: PDF generation uses a pure-Python pipeline (MarkdownIt → WeasyPrint) and supports CSS styling via `--css` or `--preset <name>` with files under `src/toolipie/tools/md_to_pdf/assets/presets/<name>.css`. The `pdf-to-png` tool uses `pypdfium2` (no system dependency) and outputs PNGs into per-PDF folders.

## How Toolipie works

- Each command (kebab-case) maps to a Python module (snake_case) under `src/toolipie/tools/<tool>/` that exposes a `run(ctx, ...)` function.
- Toolipie builds a Context for each run with resolved input/output paths, file list (via glob), defaults, and a `run.jsonl` log in the output folder.
- Input/output folders default to the command name (kebab-case) under `input/` and `output/`, and can be overridden via `--input` and `--output`.

Folder conventions:

- `tools/md_to_docx` → `input/md-to-docx/` and `output/md-to-docx/`
- `tools/md_to_pdf`  → `input/md-to-pdf/` and `output/md-to-pdf/`
- `tools/pdf_to_png` → `input/pdf-to-png/` and `output/pdf-to-png/<pdf-name>/`
- Presets live per tool under `src/toolipie/tools/<tool>/assets/presets/` (e.g., CSS or templates).

## Add your own tool (quick guide)

1) Create a folder: `src/toolipie/tools/my_tool/`

2) Add `run.py` exposing a `run(ctx, ...)` function to process files and write outputs.

3) Add an entry to the CLI in `src/toolipie/cli.py` with a command name (kebab-case) and map options to your `run` function.

4) (Optional) Add `assets/` for presets, and a `README.md` for tool-specific notes.

See `CONTRIBUTING.md` for a fuller quickstart.

## Tool documentation

- Each tool ships a focused README under `src/toolipie/tools/<tool>/README.md`.
- See `CONTRIBUTING.md` for adding new tools.

## Versioning

- Platform (core) uses Semantic Versioning (MAJOR.MINOR.PATCH). Only core/platform changes bump the version and get a `vX.Y.Z` git tag and GitHub Release. See `CHANGELOG.md`.
- Tools can evolve independently without bumping the platform version. Tool-only updates may be mentioned under “Unreleased” but do not create a new platform release.
- Programmatic version: `toolipie --version` prints the platform version. Internal `PLATFORM_API_VERSION` signals plugin interface compatibility.
