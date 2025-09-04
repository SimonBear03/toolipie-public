# Contributing to Toolipie

This guide is optimized for both humans and AI assistants to add new tools to Toolipie.

## Quickstart: add a tool in 60 seconds

1. Create the folder:
   - `src/toolipie/tools/<snake_case>/` with `__init__.py`, `run.py`, `README.md`, and `assets/presets/` (optional)
2. Wire the CLI in `src/toolipie/cli.py`:
   - `@app.command("<kebab-case>")` → build context → import `toolipie.tools.<snake_case>.run:run` → call `run(ctx, ...)`
3. Implement `run(ctx, ...)`:
   - Iterate files in `ctx.files`, write outputs under `ctx.output_dir`, append `{input, output, status, time}` to `ctx.run_log`
   - Show per-file Rich progress bars
4. Test:
   - `toolipie <kebab-case>` should process `input/<kebab-case>/` to `output/<kebab-case>/`

## Naming conventions
- CLI command: kebab-case (e.g., `md-to-pdf`).
- Python module and folder: snake_case (e.g., `md_to_pdf`).
- There is a 1:1 mapping: `md-to-pdf` ↔ `md_to_pdf`.

## Minimum structure for a new tool
```
src/toolipie/tools/<snake_case>/
  ├─ __init__.py               # usually empty
  ├─ run.py                    # entry module exposing run(ctx, ...)
  ├─ README.md                 # short usage notes, options, caveats
  └─ assets/
     └─ presets/               # optional (e.g., CSS for PDF, .docx template)
```

- Default I/O folders: `input/<kebab-case>/` → `output/<kebab-case>/`.
- Default glob: `*.md` (override if your tool processes other types).
- Logs: write one JSON line per processed file to `output/<kebab-case>/run.jsonl` with `{input, output, status, time}`.

## CLI wiring (manual registration)
Edit `src/toolipie/cli.py` and add a new command:
```python
@app.command("<kebab-case>")

def <snake_case>(
    input: Optional[str] = typer.Option(None),
    output: Optional[str] = typer.Option(None),
    glob: Optional[str] = typer.Option(None),
    overwrite: Optional[bool] = typer.Option(None),
    workers: Optional[int] = typer.Option(None),
    # add tool-specific flags below
):
    ctx = build_context("<kebab-case>", input, output, glob, overwrite, workers)
    from toolipie.tools.<snake_case>.run import run
    run(ctx, ...)
```

## Context and config utilities
Use core helpers from `src/toolipie/core.py`:
- `build_context(task_name, input, output, glob, overwrite, workers)` to construct a Context with resolved defaults and file list.
- `append_run_log(run_log, record)` to append JSON lines.
- `ensure_pandoc()` if your tool calls Pandoc.

## Progress display
Use Rich to show per-file progress bars:
```python
with Progress(TextColumn("{task.description}"), BarColumn(), TimeElapsedColumn()) as progress:
    task_ids = []
    for idx, file in enumerate(ctx.files, start=1):
        task_ids.append(progress.add_task(f"{ctx.task} {idx}/{len(ctx.files)} {file.name}", total=1))
    for idx, file in enumerate(ctx.files):
        # process file → output
        progress.update(task_ids[idx], advance=1)
```

### Progress patterns

- For multi-folder or multi-page tasks, consider a bottom "TOTAL X/N" bar with per-scope bars above (see `pdf-to-png` and `png-prep-ocr`).
- Keep task descriptions short and high-signal; include counts and, when helpful, the filename or folder.

## Presets and assets
- Prefer per-tool presets under `assets/presets/` within the tool folder.
- Expose a `--preset NAME` flag and resolve `assets/presets/NAME.*`.
- Allow explicit file flags (e.g., `--css`, `--template`) to override presets.

## I/O rules
- Accept either a single file path or a directory via `--input`.
- Filter with `--glob`. The default comes from `config.yaml` (currently `*.md`). Tools can override (e.g., `pdf-to-png` defaults to `*.pdf`).
- Skip existing outputs unless `--overwrite` is set.

## Quality checklist
- Manual CLI registration done in `cli.py`.
- Per-file run logging includes `{input, output, status, time}`.
- Defaults respect `config.yaml`; flags override.
- Progress bars render per file.
- Tool README created with examples.
- If an aggregate progress makes sense, prefer a bottom bar labeled `TOTAL X/N`.

## Example input/output defaults
```
input/<kebab-case>/
  └─ example.ext
output/<kebab-case>/
  ├─ run.jsonl
  └─ example.out
```
