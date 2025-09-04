# png-prep-ocr

Preprocess PNG images for Amazon Textract (deskew, grayscale, binarize, denoise, rotate).

## Usage

```bash
toolipie png-prep-ocr \
  --input input/png-prep-ocr \
  --output output/png-prep-ocr \
  --glob "**/*.png" \
  --grayscale --deskew --unshear --corner-dewarp --binarize --denoise 0 --rotate 0 --max-size-mb 5
```

- Defaults: deskew=True, unshear=True, grayscale=False, binarize=False, denoise=0, rotate=0.
- Optional: `--corner-dewarp` performs perspective dewarping by detecting a table/page quadrilateral and warping to a rectangle.
- Size capping: `--max-size-mb <float>` adaptively increases PNG compression and downscales in small steps (preserving aspect ratio) until the encoded PNG fits the budget.
- Other options: `--method auto|hough|minrect|sweep`, `--max-abs-angle <deg>`, `--dry-run` to print transforms rather than writing files.

### Parallelism

- Use `--workers N` to set parallel workers. `--workers 0` or omitting the flag uses auto = CPU cores minus one (minimum 1). Applies per-file parallel processing.
- Overwrite behavior: skips existing outputs unless `--overwrite` is provided.

## Outputs

Mirrors input subfolders under the output directory. Example:

```
input/png-prep-ocr/book1/page1.png
input/png-prep-ocr/book1/page2.png
input/png-prep-ocr/book2/sectionA/page1.png

→

output/png-prep-ocr/book1/page1.png
output/png-prep-ocr/book1/page2.png
output/png-prep-ocr/book2/sectionA/page1.png
```

## Dependencies

- Python: `opencv-python-headless`
