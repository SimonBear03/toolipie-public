# Contributing to Toolipie

This guide is optimized for both humans and AI assistants to add new tools to Toolipie.

## Quickstart: add a tool in 60 seconds

1. Create the folder:
   - `src/toolipie/tools/<snake_case>/` with `__init__.py`, `run.py`, `README.md`, and `assets/presets/` (optional)
2. Implement `run(ctx, ...)`:
   - Iterate files in `ctx.files`, write outputs under `ctx.output_dir`, append `{input, output, status, time}` to `ctx.run_log`
   - Show per-file Rich progress bars (TUI will capture and display progress in-place)
3. Add a manifest `tool.yaml` (schema_version: 1) under your tool folder (see below)
4. Run `toolipie scan` to refresh the tool registry (.toolipie/index.json)
5. Test:
   - `toolipie run <kebab-case> --input input/<kebab-case> --output output/<kebab-case>`
   - Add `--param name=value` for tool-specific options as needed

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
  ├─ tool.yaml                 # manifest (schema_version: 1)
  └─ assets/
     └─ presets/               # optional (e.g., CSS for PDF, .docx template)
```

- Default I/O folders: `input/<kebab-case>/` → `output/<kebab-case>/`.
- Default glob: `*.md` (override if your tool processes other types).
- Logs: write one JSON line per processed file to `output/<kebab-case>/run.jsonl` with `{input, output, status, time}`.

## Manifests and the registry index

Toolipie uses a strict manifest + index workflow for fast, deterministic discovery:

- Each tool includes `tool.yaml` following `tool.schema.yaml` (schema_version: 1).
- Run `toolipie scan` to build `.toolipie/index.json` with all tools’ metadata.
- The TUI/CLI read only the index for discovery and options; no runtime scanning.
- If a tool is missing from the index or lacks `options`, the UI will prompt you to run `toolipie scan` or fix the manifest.

Manifest essentials:
- `name` (kebab), `title`, `summary`, `default_glob`, `entry` (module:function)
- `options`: list of {name, type [str|int|float|bool|path], help, default?, choices?}
- Options are optional by default. If an option must be supplied by the user, set `required: true` (or provide a `default`).
- `requires`: optional list of Python packages

See `tool.schema.yaml` for a commented template.

### Input/Output handling

- Tools may declare `input` and `output` in `options` with `type: path` so UIs can render fields.
- Defaults for these are provided by the platform, not the manifest:
  - `input`  defaults to `<repo>/<paths.input_root>/<tool-key>`
  - `output` defaults to `<repo>/<paths.output_root>/<tool-key>`
- CLI users override via `--input` and `--output` flags; GUIs can pass absolute paths directly to the runner.
- Do not set manifest defaults for `input` or `output` — the platform pre-fills them.

Behavior details:
- If a single file path is provided for `--input`, that file is processed and `ctx.input_dir` is the file's parent directory.
- If a directory is provided (or defaulted), the platform uses that directory as-is, even if it does not exist. There is no fallback to the parent directory.
- If no input files are identified (empty/missing folder or no files match the glob), the tool should do nothing; after the run, the runner prints a short hint: `0 input files identified in '<path>'.`

Validation:
- `toolipie validate` checks all manifests and the index file and prints any issues.

## Plugin packaging & install

You can distribute tools as a simple `.zip` package. Minimum contents:

- `tool.yaml` (schema_version: 1; includes `name`, `default_glob`, `options`, and optional `entry`/`requires`)
- `run.py` (exports `run(ctx, ...)`)
- `assets/` and `README.md` are optional but encouraged

Layout options:
- Files may be at the zip root, or within a single top-level folder (both are supported by the installer).
- If your tool is a plugin (not shipped in the repo), set `entry: "run.py:run"` in `tool.yaml` so the runner loads the file directly from the plugin folder.

Install a plugin (into the repo):
```bash
toolipie install /path/to/my-tool.zip
toolipie list    # unified view of core + repo plugins
toolipie run my-tool --input input/my-tool --output output/my-tool --param option=value
```

Where plugins live:
- Installed plugins are extracted to `src/toolipie/plugins/<tool-key>/` within the repo.
- A single unified index lives at `.toolipie/index.json` and includes both core tools and repo plugins. No `~/.toolipie` is used.

Uninstall:
- `toolipie uninstall <tool-key>` removes the plugin folder from `src/toolipie/plugins/` and rebuilds the index.

## Choosing Core vs Plugin for new tools

- Prefer starting as a plugin under `src/toolipie/plugins/<tool>/` when:
  - The tool is optional, domain‑specific, or experimental
  - It has heavy or rapidly changing dependencies
  - You want easy distribution as a `.zip` and independent iteration

- Consider core under `src/toolipie/tools/<tool>/` when:
  - The tool is broadly useful and stable
  - Dependencies are lightweight
  - It meaningfully showcases the platform for most users

- Promotion flow:
  - Move plugin → core folder; keep the `name` in `tool.yaml`, update `entry` if needed
  - Run `toolipie scan`, update README/tool docs
  - Do not bump platform version for tool‑only changes; platform releases track core changes

## Context and config utilities
Use core helpers from `src/toolipie/core.py`:
- `build_context(task_name, input, output, glob, overwrite, workers)` to construct a Context with resolved defaults and file list.
- `append_run_log(run_log, record)` to append JSON lines.
- `ensure_pandoc()` if your tool calls Pandoc.

Cancellation:
- The TUI runs tools in a subprocess with a PTY; pressing `q` sends SIGTERM and auto‑escalates to SIGKILL after a short grace period. No tool changes are required for platform‑level cancel.
- If you want faster cooperative cancel during long operations, optionally check `ctx.cancel_event` inside your tool's `run()` loops and return early when set.

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

TUI console notes:
- Runs are executed under a subprocess/PTY. A minimal ANSI/CSI handler is implemented so common Rich progress updates render in place.
- Avoid relying on full terminal control sequences beyond standard cursor moves and erase‑in‑line if possible.

## Presets and assets
- Prefer per-tool presets under `assets/presets/` within the tool folder.
- Surface preset choices via the manifest `options` (e.g., `choices: [a4_report]`).
- Users pass tool-specific options via `--param name=value` when running `toolipie run`.

## I/O rules
- Accept either a single file path or a directory via `--input`.
- Filter with `--glob`. The default comes from the tool manifest (`default_glob`).
- Skip existing outputs unless `--overwrite` is set.
- Handle zero-file runs gracefully (no errors, no outputs) — the runner will inform users post‑run.

## Quality checklist
- Per-file run logging includes `{input, output, status, time}`.
- Defaults respect `config.yaml`; flags override.
- Progress bars render per file.
- Tool README created with examples.
- If an aggregate progress makes sense, prefer a bottom bar labeled `TOTAL X/N`.
 - Zero-file runs behave sensibly (do nothing); the runner prints `0 input files identified …` after completion.

## Example input/output defaults
```
input/<kebab-case>/
  └─ example.ext
output/<kebab-case>/
  ├─ run.jsonl
  └─ example.out
```
