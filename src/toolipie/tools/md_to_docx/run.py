from __future__ import annotations

from pathlib import Path
from typing import Optional

import pypandoc
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from ...core import Context, append_run_log, ensure_pandoc
from ...utils.timeit import timeit


def run(ctx: Context, template: Optional[str] = None) -> None:
    out_dir = ctx.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_pandoc()
    args = ["--standalone"]
    if template is None:
        # Default to tool-local preset if present and non-empty
        default_template = (
            Path(__file__).resolve().parent / "assets" / "presets" / "word_template.docx"
        )
        if default_template.exists() and default_template.stat().st_size > 0:
            template = str(default_template)
    if template:
        args += ["--reference-doc", template]

    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        total = len(ctx.files)
        task_ids = []
        for idx, md in enumerate(ctx.files, start=1):
            task_ids.append(progress.add_task(f"DOCX {idx}/{total} {md.name}", total=1))

        for idx, md in enumerate(ctx.files):
            out_path = out_dir / (md.stem + ".docx")
            with timeit() as elapsed:
                try:
                    pypandoc.convert_file(
                        str(md), "docx", extra_args=args, outputfile=str(out_path)
                    )
                    status = "ok"
                except Exception:
                    status = "error"
            append_run_log(
                ctx.run_log,
                {
                    "input": str(md),
                    "output": str(out_path),
                    "status": status,
                    "time": round(elapsed(), 4),
                },
            )
            progress.update(task_ids[idx], advance=1)
