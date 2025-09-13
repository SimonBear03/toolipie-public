# md-to-pdf

Convert Markdown to PDF using an HTML/CSS pipeline (no Pandoc/LaTeX). The flow is:

Markdown (MarkdownIt) → HTML → PDF (WeasyPrint)

## Command

```bash
# Build/refresh registry
toolipie scan

# List tools
toolipie list

# Run with a preset or custom CSS
toolipie run md-to-pdf --input input/md-to-pdf --output output/md-to-pdf --param preset=a4_report
toolipie run md-to-pdf --input input/md-to-pdf --output output/md-to-pdf --param css=/path/to/styles.css
```

## Default I/O

`input/md-to-pdf/` → `output/md-to-pdf/`

Note: If the input folder is missing or contains no matching files, the runner completes without writing outputs and prints: `0 input files identified in '<path>'.`

## Usage

```bash
# Use a built-in CSS preset
toolipie run md-to-pdf \
  --input input/md-to-pdf \
  --output output/md-to-pdf \
  --param preset=a4_report

# Use a custom CSS file
toolipie run md-to-pdf \
  --input input/md-to-pdf \
  --output output/md-to-pdf \
  --param css=/path/to/styles.css
```

## Styling

- Provide CSS via `--css <path>` or `--preset <name>` (files under `assets/presets/`, e.g. `a4_report.css`).
- Paged media: use `@page { size: A4; margin: 1in; }` and other CSS Paged Media rules.
- Common selectors: `h1..h6`, `p`, `pre`, `code`, `table`, `th`, `td`, `blockquote`, `img`.
- Images and relative links are resolved against the Markdown file directory.

## Markdown features

- Parser: `markdown-it` with tables, strikethrough, and linkify enabled.

## Notes

- No external Pandoc/LaTeX required for PDF generation.
- Fonts/styles depend on WeasyPrint CSS support. Use web-safe fonts or ensure fonts are available.
