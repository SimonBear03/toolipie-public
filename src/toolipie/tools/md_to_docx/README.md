# md-to-docx

Convert Markdown to DOCX using Pandoc via the `pypandoc` Python wrapper. A private
Pandoc binary is auto-downloaded on first use if none is found on your system.

## Command

```bash
# Build/refresh the registry after changes
toolipie scan

# List tools
toolipie list

# Run (uses platform input/output defaults unless overridden)
toolipie run md-to-docx \
  --input input/md-to-docx \
  --output output/md-to-docx \
  --param template=src/toolipie/tools/md_to_docx/assets/presets/word_template.docx
```

## Default I/O

`input/md-to-docx/` → `output/md-to-docx/`
 
Note: If the input folder is missing or contains no matching files, the runner completes without writing outputs and prints: `0 input files identified in '<path>'.`

## Usage

```bash
# Use the built-in reference DOCX template
toolipie run md-to-docx \
  --input input/md-to-docx \
  --output output/md-to-docx

# Provide a custom reference DOCX
toolipie run md-to-docx \
  --input input/md-to-docx \
  --output output/md-to-docx \
  --param template=/path/to/reference.docx
```

## Template

- Default: `assets/presets/word_template.docx` (used if present and non-empty)
- Override with `--template <path>` to control styles (fonts, headings, etc.)

## Common flags

- `--glob` (default for this tool is Markdown patterns from global config)
- `--overwrite` to rewrite existing outputs
- `--workers` is reserved (serial conversion per-file)

## Dependencies

- Python: `pypandoc`
- Pandoc: Auto-handled by `pypandoc` (downloaded privately on first use if missing)

## Notes

- No LaTeX is required for DOCX output.
- If auto-download is blocked (offline/CI), install Pandoc manually or set `PANDOC_PATH`.
