# pdf-to-png

Convert PDF pages to PNG images.

## Usage

```bash
# Build/refresh registry
toolipie scan

# Run
toolipie run pdf-to-png \
  --input input/pdf-to-png \
  --output output/pdf-to-png \
  --param dpi=300

toolipie run pdf-to-png \
  --input input/pdf-to-png \
  --output output/pdf-to-png \
  --param dpi=300 \
  --workers 0  # auto CPU-1
```

Flags:
- `--param dpi=<int>`: Render DPI (default: 300)
- `--param first_page=<int>`, `--param last_page=<int>`: Page range (1-indexed, inclusive)
- Common flags: `--glob` (defaults to `*.pdf` via manifest), `--overwrite`, `--workers`
  - Parallelism: `--workers N` sets per-page parallelism; `--workers 0` or omitting uses auto = CPU cores minus one (minimum 1).

Output structure:

```
input/pdf-to-png/abc.pdf
└─ output/pdf-to-png/abc/
   ├─ abc_p0001.png
   ├─ abc_p0002.png
   └─ ...
```

## Dependencies

- Python: `pypdfium2`, `Pillow`

## Defaults

- Input folder: `input/pdf-to-png/`
- Output folder: `output/pdf-to-png/`
- Default glob: `*.pdf`
 - If the input folder is missing or contains no matching files, the runner completes without writing outputs and prints: `0 input files identified in '<path>'.`
