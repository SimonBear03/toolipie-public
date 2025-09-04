from __future__ import annotations

from pathlib import Path
from typing import Optional

from markdown_it import MarkdownIt
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from weasyprint import CSS, HTML

from ...core import Context, append_run_log
from ...utils.timeit import timeit


def find_css_preset(name: Optional[str]) -> Optional[Path]:
    if not name:
        return None
    # Look within this tool's assets first
    preset_path = Path(__file__).resolve().parent / "assets" / "presets" / f"{name}.css"
    return preset_path if preset_path.exists() else None


def resolve_css(css: Optional[str], preset: Optional[str]) -> list[CSS]:
    styles: list[CSS] = []
    # Explicit CSS path has priority
    if css:
        styles.append(CSS(filename=css))
        return styles
    # Then preset CSS under presets/
    preset_path = find_css_preset(preset)
    if preset_path:
        styles.append(CSS(filename=str(preset_path)))
    # Always ensure some sane defaults if no styles provided
    if not styles:
        base_css = CSS(
            string="""
            @page { size: A4; margin: 1in; }
            html { font-family: sans-serif; font-size: 12pt; }
            h1,h2,h3 { margin: 0.8em 0 0.4em; }
            p { line-height: 1.5; margin: 0.6em 0; }
            code, pre { font-family: Menlo, monospace; font-size: 10pt; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ccc; padding: 4px 6px; }
            blockquote { border-left: 3px solid #ccc; padding-left: 10px; color: #555; }
            img { max-width: 100%; }
            """
        )
        styles.append(base_css)
    return styles


def run(ctx: Context, preset: Optional[str] = None, css: Optional[str] = None) -> None:
    out_dir = ctx.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    styles = resolve_css(css, preset)

    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        total = len(ctx.files)
        task_ids = []
        for idx, md in enumerate(ctx.files, start=1):
            task_ids.append(progress.add_task(f"PDF {idx}/{total} {md.name}", total=1))

        for idx, md in enumerate(ctx.files):
            out_path = out_dir / (md.stem + ".pdf")
            with timeit() as elapsed:
                try:
                    # HTML/CSS pipeline: md -> html -> pdf (no external LaTeX)
                    text = Path(md).read_text(encoding="utf-8")
                    md_parser = (
                        MarkdownIt("commonmark")
                        .enable("table")
                        .enable("strikethrough")
                        .enable("linkify")
                    )
                    html = md_parser.render(text)
                    HTML(string=html, base_url=str(Path(md).parent)).write_pdf(
                        str(out_path), stylesheets=styles
                    )
                    status = "ok"
                except Exception as exc:
                    status = "error"
                    # Best-effort debug HTML dump
                    debug_html = out_dir / (md.stem + ".html")
                    try:
                        debug_html.write_text(html if "html" in locals() else "", encoding="utf-8")
                    except Exception:
                        pass
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
