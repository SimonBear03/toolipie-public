from __future__ import annotations

import base64
from pathlib import Path

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from toolipie.core import Context, append_run_log
from toolipie.utils.timeit import timeit


_ASPECT_CHOICES = {
    "xMidYMid meet",
    "xMidYMid slice",
    "xMinYMin meet",
    "xMaxYMax meet",
    "none",
}


def _sanitize_aspect_ratio(value: str | None) -> str:
    if not value:
        return "xMidYMid meet"
    trimmed = value.strip()
    if trimmed not in _ASPECT_CHOICES:
        return "xMidYMid meet"
    return trimmed


def _build_svg(width: int, height: int, data_uri: str, aspect_ratio: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        f'  <image href="{data_uri}" xlink:href="{data_uri}" width="{width}" height="{height}" '
        f'preserveAspectRatio="{aspect_ratio}" />\n'
        '</svg>\n'
    )


def run(ctx: Context, aspect_ratio: str = "xMidYMid meet") -> None:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(f"Pillow not available: {exc}")

    out_dir = ctx.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(ctx.files)
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task(f"TOTAL 0/{total}", total=total or 1)
        completed = 0

        for src_path in ctx.files:
            if getattr(ctx, "cancel_event", None) is not None and ctx.cancel_event.is_set():
                break

            src = Path(src_path)
            try:
                rel = src.relative_to(ctx.input_dir)
            except ValueError:
                rel = Path(src.name)
            else:
                rel = Path(rel)

            dst = (out_dir / rel).with_suffix(".svg")
            dst.parent.mkdir(parents=True, exist_ok=True)

            status = "ok"
            error_msg: str | None = None
            elapsed_time = 0.0

            if dst.exists() and not ctx.overwrite:
                status = "skip"
            else:
                with timeit() as elapsed:
                    try:
                        with Image.open(src) as image:
                            image.load()
                            width, height = image.size
                        png_data = src.read_bytes()
                        encoded = base64.b64encode(png_data).decode("ascii")
                        data_uri = f"data:image/png;base64,{encoded}"
                        preserve = _sanitize_aspect_ratio(aspect_ratio)
                        dst.write_text(
                            _build_svg(width, height, data_uri, preserve),
                            encoding="utf-8",
                        )
                    except Exception as exc:
                        status = "error"
                        error_msg = str(exc)
                        try:
                            dst.unlink(missing_ok=True)
                        except Exception:
                            pass
                elapsed_time = round(elapsed(), 4)

            append_run_log(
                ctx.run_log,
                {
                    "input": str(src),
                    "output": str(dst),
                    "status": status,
                    "time": elapsed_time,
                    **({"error": error_msg} if error_msg else {}),
                },
            )

            completed += 1
            progress.update(
                task_id,
                advance=1,
                description=f"TOTAL {completed}/{total}",
            )
