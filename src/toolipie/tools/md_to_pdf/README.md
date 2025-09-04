# md-to-pdf

Convert Markdown to PDF using an HTML/CSS pipeline (no Pandoc/LaTeX). The flow is:

Markdown (MarkdownIt) → HTML → PDF (WeasyPrint)

## Command

```bash
toolipie md-to-pdf --help
```

## Default I/O

`input/md-to-pdf/` → `output/md-to-pdf/`

## Usage

```bash
# Use a built-in CSS preset
toolipie md-to-pdf \
  --input input/md-to-pdf \
  --output output/md-to-pdf \
  --preset a4_report

# Use a custom CSS file
toolipie md-to-pdf \
  --input input/md-to-pdf \
  --output output/md-to-pdf \
  --css /path/to/styles.css
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
