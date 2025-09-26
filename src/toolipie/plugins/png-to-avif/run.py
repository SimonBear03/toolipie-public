from __future__ import annotations

from pathlib import Path

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from toolipie.core import Context, append_run_log
from toolipie.utils.timeit import timeit


def _normalize_image_mode(image):
    """Return image converted to RGB/RGBA depending on alpha channel."""
    try:
        bands = image.getbands()
    except Exception:
        bands = ()
    has_alpha = "A" in bands
    target_mode = "RGBA" if has_alpha else "RGB"
    if image.mode == target_mode:
        return image
    try:
        return image.convert(target_mode)
    except Exception:
        return image.convert("RGBA" if has_alpha else "RGB")


def _clamp_quality(value: int) -> int:
    return max(0, min(int(value), 100))


def _clamp_speed(value: int | None) -> int:
    if value is None:
        return 6
    return max(0, min(int(value), 10))


def run(
    ctx: Context,
    quality: int = 45,
    lossless: bool = False,
    speed: int | None = 6,
    subsampling: str = "4:2:0",
) -> None:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(f"Pillow not available: {exc}")
    try:
        from pillow_avif import AvifImagePlugin  # type: ignore  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"pillow-avif-plugin not available: {exc}")

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

            dst = (out_dir / rel).with_suffix(".avif")
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
                            normalized = _normalize_image_mode(image)
                            save_kwargs: dict[str, object] = {
                                "speed": _clamp_speed(speed),
                            }
                            if lossless:
                                save_kwargs["lossless"] = True
                            else:
                                save_kwargs["quality"] = _clamp_quality(quality)
                            sub = (subsampling or "").strip().lower()
                            if sub and sub != "auto":
                                save_kwargs["chroma_subsampling"] = subsampling
                            normalized.save(dst, format="AVIF", **save_kwargs)
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

