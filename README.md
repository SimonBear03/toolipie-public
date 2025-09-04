 # Toolipie

Personal CLI toolbox — supports Markdown to DOCX/PDF and PDF-to-PNG rasterization.

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

## Available tools

- md-to-docx: `src/toolipie/tools/md_to_docx/README.md`
- md-to-pdf:  `src/toolipie/tools/md_to_pdf/README.md`
- pdf-to-png: `src/toolipie/tools/pdf_to_png/README.md`
- png-prep-ocr: `src/toolipie/tools/png_prep_ocr/README.md`
