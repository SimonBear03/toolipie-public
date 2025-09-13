# Repository Guidelines

## Project Structure & Module Organization
- `src/toolipie/cli.py`: Typer CLI entrypoint; provides `list`, `run`, `scan`, `validate`, `install`, `uninstall`, `package`, and the TUI launcher. The TUI includes a “System Commands” submenu and runs tools in a subprocess/PTY with an in‑TUI console; delegates business logic to the runner.
- `src/toolipie/runner.py`: shared discovery/registry, specs, defaults, and run dispatcher (strict, manifest/index-driven).
- `src/toolipie/core.py`: context, config, logging, and Pandoc helpers.
- `src/toolipie/utils/`: small utilities (`fs.py`, `timeit.py`).
- `src/toolipie/tools/<snake_case>/`: each tool exposes `run(ctx, ...)`, plus `tool.yaml`, `README.md`, and optional `assets/presets/`.
- `input/` and `output/`: default I/O roots for runs (per command folder). Configure via `config.yaml`.

## Build, Test, and Development Commands
```bash
# Setup (venv + editable install + hooks)
python -m venv .venv && source .venv/bin/activate && pip install -e . && pre-commit install

# Discover commands and options
toolipie --help
toolipie list

# Common examples (manifest/index-driven)
toolipie scan
toolipie run md-to-pdf --input examples/md --output output/pdf --param preset=a4_report
toolipie run pdf-to-png --input input/pdf-to-png --output output/pdf-to-png --param dpi=300

# Justfile shortcuts (optional)
just setup
just pdf
just docx
```

## Coding Style & Naming Conventions
- Python 3.10; Ruff enforces lint/format (`E`, `F`, `I`; line-length 100). Run via pre-commit.
- Indentation 4 spaces; prefer type hints; keep functions small and focused.
- Tool keys kebab-case (e.g., `md-to-pdf`); Python packages/folders snake_case (e.g., `md_to_pdf`). Keep a 1:1 mapping.
- Place reusable assets under `src/toolipie/tools/<tool>/assets/presets/`.

## Testing Guidelines
- No formal test suite yet; add smoke tests by running tools against small inputs under `input/<kebab-case>/` and verifying outputs and `run.jsonl` entries.
- When testing via the TUI:
  - Confirm System Commands (scan/validate/list/install/uninstall/package/help/version) stream into the in‑TUI console.
  - Confirm tools render progress in-place (no duplicated rows), that q requests cancel and returns automatically with a clear status, and that double Ctrl+C exits the program from any panel.
- For new tools, include a minimal example in the tool README and document expected output paths.
- Avoid committing large binaries; prefer tiny fixtures or generate outputs during local runs.

## Commit & Pull Request Guidelines
- Commits: use clear, imperative messages. Conventional prefixes encouraged: `feat:`, `fix:`, `chore:`, `docs:`, `refactor:` (e.g., `feat(md-to-pdf): add a4_report preset`).
- PRs: include a concise description, linked issues, CLI examples (command + sample paths), and screenshots only if UI-like output is relevant (e.g., rendered PDF preview).
- Ensure `ruff` passes and tools run locally before requesting review.

## Security & Configuration Tips
- Configure defaults in `config.yaml`; override via CLI flags. Never commit secrets; use `.env` locally (see `.env.example`).
- `md-to-docx` auto-manages Pandoc; no system install required. Other tools use pure-Python deps.

## Platform Behaviors (Universal)
- Index-driven discovery; strict manifest format enforced by `validate`.
- TUI runs tools in a subprocess with a PTY for consistent progress rendering and universal cancel (SIGTERM then SIGKILL).
- CLI enforces types/choices and only options marked `required: true`.
- Per-tool cooperative cancel via `ctx.cancel_event` is optional; platform cancel is still effective due to subprocess isolation.
 - Double Ctrl+C consistently exits the program. In text prompts, Esc cancels input; ‘q’ is just a character.
 - Default input/output resolution:
   - Defaults always target `input/<tool-key>` and `output/<tool-key>` under the repo roots from `config.yaml`.
   - If a file path is provided as `--input`, `ctx.input_dir` is the file's parent; the file itself is processed directly.
   - If a directory is provided (or defaulted), it is used as-is even if it does not exist. There is no fallback to the parent directory. Runs with zero inputs do nothing.
   - After any run with zero inputs, the runner prints a short hint: `0 input files identified in '<path>'.`

## Docs To Update On Platform Changes
When platform behavior changes, review and update these records in the repo:
- `README.md` — user-facing usage, TUI behavior, cancellation semantics
- `AGENTS.md` — agent guidance, platform behaviors, testing guidelines
- `CHANGELOG.md` — summarize notable platform changes under Unreleased
- `CONTRIBUTING.md` — tool authoring rules, manifests, progress/cancel notes
- `planning/tui-menu-roadmap.md` — TUI roadmap and current state
- `planning/todo.md` — updated tasks list
- `PLUGIN_PLATFORM_PLAN.md` — if plugin/discovery model changes
- Tool docs under `src/toolipie/tools/*/README.md` — if tool UX/options changed

## Plugins Folder (Repo)
- The in-repo folder `src/toolipie/plugins/` holds plugin-style tools, whether shipped or installed.
- `toolipie scan` indexes both `src/toolipie/tools/` and `src/toolipie/plugins/` into a single repo index at `.toolipie/index.json`. Entries carry `source: core|plugin` and a `rel_path` relative to their root.
- `toolipie install <zip>` extracts into `src/toolipie/plugins/<tool>/` and updates `.toolipie/index.json`. `toolipie uninstall <tool>` removes the folder and rescans. No `~/.toolipie` is used.
