from __future__ import annotations

from pathlib import Path
from typing import Optional

import pypdfium2 as pdfium
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
import concurrent.futures
import os

from ...core import Context, append_run_log
from ...utils.timeit import timeit


def _render_one_page(
    args: tuple[
        str,  # pdf_path
        str,  # output_dir (per-pdf)
        str,  # base_name
        int,  # zero_based_index
        int,  # page_number (1-indexed for filename)
        int,  # dpi
        bool, # overwrite
    ]
) -> tuple[str, int, bool, Optional[str]]:
    pdf_path, output_dir, base_name, zero_based_index, page_number, dpi, overwrite = args
    try:
        pdf_doc = pdfium.PdfDocument(pdf_path)
        page = pdf_doc[zero_based_index]
        scale = (dpi or 300) / 72.0
        pil_image = page.render(scale=scale).to_pil()
        out_dir_path = Path(output_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        out_path = out_dir_path / f"{base_name}_p{page_number:04d}.png"
        if out_path.exists() and not overwrite:
            return base_name, page_number, False, None
        pil_image.save(str(out_path), format="PNG")
        return base_name, page_number, True, None
    except Exception as e:
        return base_name, page_number, False, str(e)


def run(
    ctx: Context,
    dpi: int | None = 300,
    first_page: int | None = None,
    last_page: int | None = None,
) -> None:
    out_dir = ctx.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # PNG-only output
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        # Determine parallelism: None or 0 -> auto (CPU-1, min 1)
        if ctx.workers and ctx.workers > 0:
            jobs = int(ctx.workers)
        else:
            try:
                cpu = os.cpu_count() or 1
            except Exception:
                cpu = 1
            jobs = max(1, cpu - 1)

        # Discover tasks (per-page) and per-PDF totals
        per_pdf_totals: dict[str, int] = {}
        per_pdf_done: dict[str, int] = {}
        per_pdf_task_id: dict[str, int] = {}
        tasks: list[tuple[str, str, str, int, int, int, bool]] = []
        total_pages_all = 0
        for pdf in ctx.files:
            pdf_path = Path(pdf).resolve()
            base_name = pdf_path.stem
            out_dir_pdf = out_dir / base_name
            out_dir_pdf.mkdir(parents=True, exist_ok=True)
            try:
                pdf_doc = pdfium.PdfDocument(str(pdf_path))
                total_pages = len(pdf_doc)
                # Compute range
                start_page_idx = (first_page - 1) if (first_page and first_page > 0) else 0
                end_page_idx = (last_page - 1) if (last_page and last_page > 0) else (total_pages - 1)
                start_page_idx = max(0, min(start_page_idx, total_pages - 1))
                end_page_idx = max(0, min(end_page_idx, total_pages - 1))
                if end_page_idx < start_page_idx:
                    end_page_idx = start_page_idx
                page_indices = list(range(start_page_idx, end_page_idx + 1))
            except Exception:
                page_indices = []
                total_pages = 0
            per_pdf_totals[base_name] = len(page_indices)
            per_pdf_done[base_name] = 0
            per_pdf_task_id[base_name] = progress.add_task(
                f"{base_name} 0/{len(page_indices)}", total=len(page_indices) or 1
            )
            for zero_idx in page_indices:
                page_number = zero_idx + 1
                tasks.append(
                    (
                        str(pdf_path),
                        str(out_dir_pdf),
                        base_name,
                        zero_idx,
                        page_number,
                        int(dpi or 300),
                        bool(ctx.overwrite),
                    )
                )
                total_pages_all += 1

        overall = progress.add_task(f"TOTAL 0/{total_pages_all}", total=total_pages_all or 1)
        overall_done = 0

        # Execute per-page tasks in a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = [ex.submit(_render_one_page, t) for t in tasks]
            for fut in concurrent.futures.as_completed(futures):
                base_name, page_number, wrote, error = fut.result()
                # Update per-pdf
                per_pdf_done[base_name] = per_pdf_done.get(base_name, 0) + 1
                progress.update(
                    per_pdf_task_id[base_name],
                    advance=1,
                    description=f"{base_name} {per_pdf_done[base_name]}/{per_pdf_totals[base_name]}",
                )
                # Update overall
                overall_done += 1
                progress.update(
                    overall,
                    advance=1,
                    description=f"TOTAL {overall_done}/{total_pages_all}",
                )
        # Append one run record per PDF (summary)
        for pdf in ctx.files:
            pdf_path = Path(pdf).resolve()
            base_name = pdf_path.stem
            append_run_log(
                ctx.run_log,
                {
                    "input": str(pdf_path),
                    "output": str(out_dir / base_name),
                    "status": "ok",
                    "pages": per_pdf_totals.get(base_name, 0),
                    "time": 0.0,
                },
            )


