# Changelog

All notable platform changes to this project will be documented in this file.

The platform is versioned with Semantic Versioning (MAJOR.MINOR.PATCH). Tool-only changes do not bump the platform version.

## [Unreleased]

- No changes yet.

## [0.2.0] - 2025-09-13

### Added
- Expose platform version via `toolipie --version` and define `PLATFORM_API_VERSION = "1"`.
- TUI (curses) interface auto‑launched with `toolipie` (TTY only): two‑pane selector, options panel, and System Commands submenu (scan, validate, list, install, uninstall, package, help, version).
- In‑TUI output console: runs tools in a subprocess attached to a PTY; minimal ANSI/CSI handling so Rich progress renders in place; scroll with ↑/↓.
- Universal cancel in the TUI run panel: `q` sends SIGTERM, auto‑escalates to SIGKILL after a short grace period; panel auto‑closes when stopped.
- Plugin lifecycle inside the repo: `install`/`uninstall` plugins under `src/toolipie/plugins/<tool>/` with schema validation and safe extraction; `package` creates distributable `.zip` files to `output/packages/`.
- Unified discovery index at `.toolipie/index.json` (single source of truth) covering both core tools and repo plugins; used by both CLI and TUI for fast startup.
- Strict manifest schema (`tool.yaml`, schema_version: 1) and index validation helpers.
- Runner: post‑run hint when no inputs — prints `0 input files identified in '<path>'.`

### Changed
- Discovery and options are strictly manifest/index‑driven; no direct per‑tool Typer auto‑registration.
- CLI `run`: only enforces options marked `required: true`; options without defaults remain optional by default.
- Import and run dispatch supports both module entries and file‑based plugin entries (`run.py:run`) resolved from index metadata.
- Help rendering is robust when used inside the TUI; double Ctrl+C consistently exits from menus, options, prompts, and consoles; text prompts treat ‘q’ as a character and Esc as cancel.
- Terminology: use "identified" (not "matched") when referring to discovered input files.

### Fixed
- `python -m toolipie.cli scan` and other subcommands: ensure availability by moving app() call after command definitions.
- Input default handling: `build_context` preserves the intended `input/<tool-key>` and no longer falls back to the parent directory when the folder is missing (zero‑file runs are valid).

## [0.1.0] - 2025-09-04
- Initial public release of the platform and first tools

[Unreleased]: https://github.com/SimonBear03/toolipie-public/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/SimonBear03/toolipie-public/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/SimonBear03/toolipie-public/releases/tag/v0.1.0
