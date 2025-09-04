# Toolipie Plugin Platform Plan

## Vision

Make Toolipie a small, reliable "platform" where tools are drop‑in plugins:
- Copy a tool folder into `src/toolipie/tools/<tool>/` → it auto‑appears in the CLI
- Remove the folder → it disappears from the CLI
- Each tool declares its own options and (optional) dependencies
- Core stays small, tools remain modular and easy to share

## Goals (high‑level)
- Auto‑discover tools on startup (no manual `cli.py` edits)
- Auto‑register Typer commands from a per‑tool spec
- Provide clear dependency checks and one‑shot install helpers (opt‑in)
- Keep pure‑CLI UX simple; no hidden auto‑installs during normal runs

## User stories
- As a user, I can drop a tool folder in `tools/` and immediately run `toolipie <tool>`
- As a developer, I can define simple metadata for my tool (help text, options, defaults)
- As a maintainer, I can see which tools are missing deps and install them quickly

## Architecture overview
- Discovery: scan `src/toolipie/tools/*/` for a `run.py` exposing `run(ctx, ...)`
- Tool spec: each tool optionally publishes either of:
  - a module constant `CLI_SPEC` in `run.py`, or
  - a `tool.yaml` file in the tool folder
- Registration: build Typer commands from the spec; fallback to a generic wrapper that reflects `run()` parameters if no spec is present
- Dependencies: each tool can declare optional deps; core exposes helper commands to check and (optionally) install them

### Proposed tool spec (Python)
```python
# src/toolipie/tools/my_tool/run.py
CLI_SPEC = {
  "name": "my-tool",               # kebab-case command name
  "summary": "One-line description",
  "default_glob": "*.ext",        # overrides global default
  "options": [                      # translated to Typer options
    {"name": "threshold", "type": "int", "default": 3, "help": "…"},
    {"name": "mode", "type": "str", "choices": ["fast", "best"], "default": "fast"},
    {"name": "dry_run", "type": "bool", "default": False, "help": "…"},
  ],
  "requires": ["somepkg>=1.2"],    # optional: used by plugins installer
}

def run(ctx, threshold: int = 3, mode: str = "fast", dry_run: bool = False):
    ...
```

### Alternative spec (YAML)
```yaml
name: my-tool
summary: One-line description
default_glob: "*.ext"
options:
  - name: threshold
    type: int
    default: 3
    help: …
  - name: mode
    type: str
    choices: [fast, best]
    default: fast
  - name: dry_run
    type: bool
    default: false
requires:
  - somepkg>=1.2
```

### Registration flow (runtime)
1) Scan tool folders, load `CLI_SPEC` or `tool.yaml` if present
2) Convert spec → Typer command (plus common flags: `--input`, `--output`, `--glob`, `--overwrite`, `--workers`)
3) Fallback: if no spec, reflect `run()` signature for basic options
4) Register command under the kebab‑case tool name

### Dependencies strategy
- Preferred: declare per‑tool optional dependencies in the tool spec
- Core provides helper commands:
  - `toolipie plugins list` (show all tools, spec status, dep status)
  - `toolipie plugins check` (report missing deps per tool)
  - `toolipie plugins install <tool>` (pip install the tool’s `requires` list)
- Avoid auto‑install on normal runs; prompt with a clear error and remedy

## Phased plan and tasks

### Phase 1: Discovery & basic auto‑registration
- [ ] Add discovery in `cli.py` to scan `src/toolipie/tools/*/`
- [ ] Support Python `CLI_SPEC` in `run.py`
- [ ] Support fallback to `run()` signature if no spec
- [ ] Map snake_case folder → kebab‑case command name
- [ ] Helpful errors when `run()` not found or signature invalid

### Phase 2: Options & validation
- [ ] Translate spec options to proper Typer types (bool/int/float/str, choices)
- [ ] Apply defaults from spec (or `run()` signature if absent)
- [ ] Validate `default_glob` and merge with global defaults
- [ ] Ensure common flags (`--input`, `--output`, `--glob`, `--overwrite`, `--workers`) remain available

### Phase 3: Dependencies & tooling
- [ ] Parse `requires` from spec
- [ ] Implement `toolipie plugins list`
- [ ] Implement `toolipie plugins check`
- [ ] Implement `toolipie plugins install <tool>` (opt‑in installer)
- [ ] Document dependency workflow

### Phase 4: UX polish
- [ ] Uniform command help from `summary` and option help
- [ ] Consistent "TOTAL X/N" aggregate progress guidance
- [ ] Rich errors when a tool fails to load or deps are missing
- [ ] Cache discovery results between runs (optional)

### Phase 5: QA & docs
- [ ] Unit tests for discovery, spec parsing, and registration
- [ ] Integration tests covering 1–2 sample tools
- [ ] Update `README.md` with plugin model overview
- [ ] Update `CONTRIBUTING.md` with spec examples and workflow

## Acceptance criteria
- Dropping a valid tool folder into `tools/` creates a working `toolipie <tool>` command on next run
- Removing the folder removes the command
- Tools with specs get correct options and help
- Missing deps are clearly reported with a one‑line remedy

## Open questions
- Prefer Python spec (`CLI_SPEC`) or YAML by default?
- Support third‑party tool namespaces (outside the repo) later?
- Handle versioned specs / breaking spec changes how?

## Out of scope (for now)
- Auto‑installing dependencies on normal command runs (not sure if this is a good idea yet...)
- Non‑Python plugin languages
- Networked/plugin registries

## Rough timeline
- P1–P2: 1–2 days
- P3: 0.5–1 day
- P4–P5: 1 day

---

Use this file as a living to‑do and design reference while implementing the plugin platform.
