# pdf-to-png

Convert PDF pages to PNG images.

## Usage

```bash
toolipie pdf-to-png \
  --input input/pdf-to-png \
  --output output/pdf-to-png \
  --dpi 300
toolipie pdf-to-png \
  --input input/pdf-to-png \
  --output output/pdf-to-png \
  --dpi 300 \
  --workers 0  # auto CPU-1
```

Flags:
- `--dpi`: Render DPI (default: 300)
- `--first-page`, `--last-page`: Page range (1-indexed, inclusive)
This tool outputs PNG only.
- Common flags: `--glob` (defaults to `*.pdf`), `--overwrite`, `--workers`
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

