from __future__ import annotations

from typing import Optional

import typer
from . import __version__

from .core import build_context
from .tools.md_to_docx import run as md_to_docx_tool
from .tools.md_to_pdf import run as md_to_pdf_tool
from .tools.pdf_to_png import run as pdf_to_png_tool
from .tools.png_prep_ocr import run as png_prep_ocr_tool

app = typer.Typer(
    help="Toolipie — personal CLI toolbox", no_args_is_help=True, add_completion=False
)

def _version_callback(value: bool) -> None:
    if value:
        print(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show Toolipie platform version and exit",
        is_eager=True,
        callback=_version_callback,
    )
):
    """Toolipie root command."""


@app.command("md-to-docx")
def md_to_docx(
    input: Optional[str] = typer.Option(None, help="Input directory or file"),
    output: Optional[str] = typer.Option(None, help="Output directory"),
    glob: Optional[str] = typer.Option(None, help="Glob for input files"),
    overwrite: Optional[bool] = typer.Option(None, help="Overwrite outputs if exist"),
    workers: Optional[int] = typer.Option(None, help="Number of workers (reserved)"),
    template: Optional[str] = typer.Option(None, help="Word reference DOCX"),
):
    ctx = build_context("md-to-docx", input, output, glob, overwrite, workers)
    md_to_docx_tool.run(ctx, template=template)


@app.command("md-to-pdf")
def md_to_pdf(
    input: Optional[str] = typer.Option(None, help="Input directory or file"),
    output: Optional[str] = typer.Option(None, help="Output directory"),
    glob: Optional[str] = typer.Option(None, help="Glob for input files"),
    overwrite: Optional[bool] = typer.Option(None, help="Overwrite outputs if exist"),
    workers: Optional[int] = typer.Option(None, help="Number of workers (reserved)"),
    preset: Optional[str] = typer.Option(
        None, help="CSS preset name under presets/ (e.g., a4_report)"
    ),
    css: Optional[str] = typer.Option(None, help="Path to a CSS file to style the PDF"),
):
    ctx = build_context("md-to-pdf", input, output, glob, overwrite, workers)
    md_to_pdf_tool.run(ctx, preset=preset, css=css)


@app.command("pdf-to-png")
def pdf_to_png(
    input: Optional[str] = typer.Option(None, help="Input directory or file"),
    output: Optional[str] = typer.Option(None, help="Output directory"),
    glob: Optional[str] = typer.Option(None, help="Glob for input files (default: *.pdf)"),
    overwrite: Optional[bool] = typer.Option(None, help="Overwrite outputs if exist"),
    workers: Optional[int] = typer.Option(None, help="Number of workers (reserved)"),
    dpi: Optional[int] = typer.Option(300, help="Render DPI (default: 300)"),
    first_page: Optional[int] = typer.Option(None, help="First page to render (1-indexed)"),
    last_page: Optional[int] = typer.Option(None, help="Last page to render (inclusive)"),
):
    # Use *.pdf by default for this tool if user does not provide a glob
    eff_glob = glob or "*.pdf"
    ctx = build_context("pdf-to-png", input, output, eff_glob, overwrite, workers)
    pdf_to_png_tool.run(
        ctx,
        dpi=dpi or 300,
        first_page=first_page,
        last_page=last_page,
    )


 


@app.command("png-prep-ocr")
def png_prep_ocr(
    input: Optional[str] = typer.Option(None, help="Input directory or file (PNG only)"),
    output: Optional[str] = typer.Option(None, help="Output directory"),
    glob: Optional[str] = typer.Option(None, help="Glob for input files (e.g., *.png)"),
    overwrite: Optional[bool] = typer.Option(None, help="Overwrite outputs if exist"),
    workers: Optional[int] = typer.Option(
        None,
        help="Number of parallel workers (0/None = auto: CPU-1)",
    ),
    grayscale: bool = typer.Option(False, help="Convert to grayscale (output)"),
    deskew: bool = typer.Option(True, help="Auto deskew (default on)"),
    binarize: bool = typer.Option(False, help="Otsu binarization (off by default)"),
    denoise: int = typer.Option(0, help="Denoise strength (0 to disable)"),
    rotate: int = typer.Option(0, help="Extra rotate degrees (clockwise) after deskew"),
    max_size_mb: Optional[float] = typer.Option(None, help="Cap output PNG size in MB (adaptive compression/downscale)"),
    method: Optional[str] = typer.Option(
        "auto", help="Skew estimation: auto|hough|minrect|sweep"
    ),
    max_abs_angle: float = typer.Option(8.0, help="Ignore corrections larger than this (°)"),
    unshear: bool = typer.Option(True, help="Level horizontal lines via y-shear after rotation"),
    dry_run: bool = typer.Option(False, help="Only estimate angle; do not write files"),
    corner_dewarp: bool = typer.Option(False, help="Corner dewarp (perspective quad warp)"),
):
    eff_glob = glob or "**/*.png"
    ctx = build_context("png-prep-ocr", input, output, eff_glob, overwrite, workers)
    png_prep_ocr_tool.run(
        ctx,
        grayscale=grayscale,
        deskew=deskew,
        binarize=binarize,
        denoise=denoise,
        rotate=rotate,
        max_size_mb=max_size_mb,
        method=method,
        max_abs_angle=max_abs_angle,
        unshear=unshear,
        dry_run=dry_run,
        corner_dewarp=corner_dewarp,
    )


if __name__ == "__main__":
    app()
