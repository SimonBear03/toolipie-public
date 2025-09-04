from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
import concurrent.futures
import os

from ...core import Context, append_run_log
from ...utils.timeit import timeit


def _read_image(path: Path) -> np.ndarray:
    data = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if data is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return data


def _write_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, buf = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError(f"Failed to encode PNG: {path}")
    path.write_bytes(buf.tobytes())


def _encode_png_bytes(img: np.ndarray, compress_level: int = 3) -> bytes:
    # compress_level: 0 (no compression, largest) .. 9 (max compression, slowest)
    params = [cv2.IMWRITE_PNG_COMPRESSION, int(np.clip(compress_level, 0, 9))]
    ok, buf = cv2.imencode(".png", img, params)
    if not ok:
        raise RuntimeError("Failed to encode PNG to bytes")
    return buf.tobytes()


def _downscale_for_budget(img: np.ndarray, target_bytes: int, max_iters: int = 6) -> np.ndarray:
    """Downscale image adaptively until encoded PNG bytes are <= budget.
    Strategy:
      1) Try increasing PNG compression up to level 9
      2) If still too big, progressively downscale by ~85% each step (keeping aspect ratio)
    """
    # First try compression sweep without resizing
    for level in (3, 5, 7, 9):
        data = _encode_png_bytes(img, compress_level=level)
        if len(data) <= target_bytes:
            # Return decoded to keep consistent pipeline (we only need to write later)
            return img
    # Resize loop
    work = img
    for _ in range(max_iters):
        h, w = work.shape[:2]
        if min(h, w) <= 300:
            break
        scale = 0.85
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        work = cv2.resize(work, new_size, interpolation=cv2.INTER_AREA)
        # Try a couple of compression levels at this size
        for level in (5, 7, 9):
            data = _encode_png_bytes(work, compress_level=level)
            if len(data) <= target_bytes:
                return work
    return work


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _binarize_otsu(gray: np.ndarray) -> np.ndarray:
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def _denoise(img: np.ndarray, strength: int) -> np.ndarray:
    if strength <= 0:
        return img
    return cv2.fastNlMeansDenoising(img, None, h=strength, templateWindowSize=7, searchWindowSize=21)


def _rotate(img: np.ndarray, degrees: int) -> np.ndarray:
    if not degrees:
        return img
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, degrees, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


 


def _calc_skew_angle_hough(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = 255 - bw  # content white

    h = max(1, gray.shape[0] // 200)
    w = max(20, gray.shape[1] // 40)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w, h))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    edges = cv2.Canny(closed, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=120,
        minLineLength=max(60, gray.shape[1] // 5),
        maxLineGap=gray.shape[1] // 30,
    )
    if lines is None or len(lines) == 0:
        return 0.0

    angles: list[float] = []
    weights: list[float] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 5:
            continue
        slope_deg = 90.0 if dx == 0 else float(np.degrees(np.arctan2(dy, dx)))
        folded = ((slope_deg + 45.0) % 90.0) - 45.0
        angles.append(folded)
        weights.append(length)
    if not angles:
        return 0.0

    angles_arr = np.asarray(angles, dtype=np.float32)
    weights_arr = np.asarray(weights, dtype=np.float32)
    bins = np.linspace(-10.0, 10.0, 161)  # 0.125° bins
    hist, edges = np.histogram(angles_arr, bins=bins, weights=weights_arr)
    peak_idx = int(hist.argmax())
    lo, hi = edges[peak_idx], edges[peak_idx + 1]
    mask = (angles_arr >= lo) & (angles_arr <= hi)
    if not np.any(mask):
        order = np.argsort(angles_arr)
        angles_sorted = angles_arr[order]
        weights_sorted = weights_arr[order]
        cumsum = weights_sorted.cumsum()
        cutoff = weights_sorted.sum() / 2.0
        idx = int(np.searchsorted(cumsum, cutoff))
        return float(angles_sorted[min(idx, len(angles_sorted) - 1)])
    a = angles_arr[mask]
    wts = weights_arr[mask]
    order = np.argsort(a)
    a = a[order]
    wts = wts[order]
    cumsum = wts.cumsum()
    cutoff = wts.sum() / 2.0
    return float(a[min(int(np.searchsorted(cumsum, cutoff)), len(a) - 1)])


def _minrect_angle(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = 255 - bw
    ys, xs = np.where(bw > 0)
    if xs.size == 0:
        return 0.0
    coords = np.column_stack((xs, ys)).astype(np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(coords)
    angle = float(rect[-1])
    # Normalize angle into (-45, 45]
    if angle < -45.0:
        angle += 90.0
    elif angle > 45.0:
        angle -= 90.0
    return angle


def _sweep_projection_angle(gray: np.ndarray, search_deg: float = 5.0, step_deg: float = 0.2) -> float:
    h, w = gray.shape[:2]
    scale = 1000.0 / max(h, w)
    if scale < 1.0:
        small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = gray
    small = cv2.GaussianBlur(small, (3, 3), 0)
    sobelx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    best_angle = 0.0
    best_score = -1.0
    for ang in np.arange(-search_deg, search_deg + 1e-6, step_deg):
        rot = _rotate_bound(sobelx, ang)
        col_proj = rot.sum(axis=0).astype(np.float32)
        score = float(col_proj.var())
        if score > best_score:
            best_score = score
            best_angle = float(ang)
    return float(-best_angle)


def _estimate_horizontal_shear(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = 255 - bw
    h = max(1, gray.shape[0] // 200)
    w = max(20, gray.shape[1] // 40)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w, h))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    edges = cv2.Canny(closed, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=120,
        minLineLength=max(60, gray.shape[1] // 5),
        maxLineGap=gray.shape[1] // 30,
    )
    if lines is None or len(lines) == 0:
        return 0.0
    slopes = []
    weights = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        slope = dy / dx
        if abs(slope) <= np.tan(np.radians(20.0)):
            length = float(np.hypot(dx, dy))
            slopes.append(slope)
            weights.append(length)
    if not slopes:
        return 0.0
    slopes = np.asarray(slopes, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    order = np.argsort(slopes)
    slopes = slopes[order]
    weights = weights[order]
    cumsum = weights.cumsum()
    cutoff = weights.sum() / 2.0
    k = float(slopes[min(int(np.searchsorted(cumsum, cutoff)), len(slopes) - 1)])
    # Clamp to ±8° equivalent
    k = float(np.clip(k, -np.tan(np.radians(8.0)), np.tan(np.radians(8.0))))
    return k


def _shear_y_bound(image: np.ndarray, shear_k: float) -> np.ndarray:
    (h, w) = image.shape[:2]
    def map_point(x: float, y: float) -> tuple[float, float]:
        return (x, y - shear_k * x)
    pts = [map_point(0, 0), map_point(w, 0), map_point(0, h), map_point(w, h)]
    ys = [p[1] for p in pts]
    min_y, max_y = min(ys), max(ys)
    trans_y = -min_y
    new_h = int(np.ceil(max_y - min_y))
    M = np.array([[1.0, 0.0, 0.0], [-shear_k, 1.0, trans_y]], dtype=np.float32)
    return cv2.warpAffine(image, M, (w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _shear_y_same_size(image: np.ndarray, shear_k: float) -> np.ndarray:
    """Apply y' = y - k*x without expanding canvas; keep same size."""
    (h, w) = image.shape[:2]
    M = np.array([[1.0, 0.0, 0.0], [-shear_k, 1.0, 0.0]], dtype=np.float32)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _rotate_bound(image: np.ndarray, angle_deg: float) -> np.ndarray:
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


 


def _process_one(
    args: tuple[
        Path,
        Context,
        bool,  # grayscale
        bool,  # deskew
        bool,  # binarize
        int,   # denoise
        int,   # rotate
        Optional[float],  # max_size_mb
        Optional[str],  # method
        float,  # max_abs_angle
        bool,  # unshear
        bool,  # dry_run
        bool,  # corner_dewarp
    ]
) -> tuple[Path, Path, dict]:
    (
        src,
        ctx,
        grayscale,
        deskew,
        binarize,
        denoise,
        rotate,
        max_size_mb,
        method,
        max_abs_angle,
        unshear,
        dry_run,
        corner_dewarp,
    ) = args

    out_dir = ctx.output_dir
    try:
        try:
            rel = src.relative_to(ctx.input_dir)
        except ValueError:
            rel = Path(src.name)
        dst = out_dir / rel
        if not dry_run and dst.exists() and not ctx.overwrite:
            return src, dst, {"status": "skip", "time": 0.0}

        with timeit() as elapsed:
            try:
                img = _read_image(src)
                work = img
                angle_est: float = 0.0
                angle_hough: float = 0.0
                angle_minrect: float = 0.0
                angle_sweep: float = 0.0
                angle_applied: float = 0.0
                unshear_k: float = 0.0
                unshear_applied: bool = False
                dewarp_applied: bool = False
                dewarp_quad: list[list[float]] | None = None
                dewarp_size: tuple[int, int] | None = None

                if deskew:
                    gray = _to_grayscale(work)
                    if method == "hough":
                        angle_hough = _calc_skew_angle_hough(gray)
                        angle_est = angle_hough
                    elif method == "minrect":
                        angle_minrect = _minrect_angle(gray)
                        angle_est = angle_minrect
                    elif method == "sweep":
                        angle_sweep = _sweep_projection_angle(gray)
                        angle_est = angle_sweep
                    else:
                        angle_hough = _calc_skew_angle_hough(gray)
                        angle_est = angle_hough
                        if abs(angle_est) < 1e-3:
                            angle_minrect = _minrect_angle(gray)
                            if abs(angle_minrect) > abs(angle_est):
                                angle_est = angle_minrect
                        if abs(angle_est) < 1e-3:
                            angle_sweep = _sweep_projection_angle(gray)
                            angle_est = angle_sweep
                    if abs(angle_est) <= max_abs_angle:
                        work = _rotate_bound(work, -angle_est)
                        angle_applied = -angle_est
                if rotate:
                    work = _rotate(work, rotate)

                if corner_dewarp:
                    gray_d = _to_grayscale(work)
                    lo = int(max(0, (1.0 - 0.33) * float(np.median(gray_d))))
                    hi = int(min(255, (1.0 + 0.33) * float(np.median(gray_d))))
                    edges = cv2.Canny(gray_d, lo, hi)
                    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), 1)
                    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    quad = None
                    if contours:
                        contours = sorted(contours, key=cv2.contourArea, reverse=True)
                        h_img, w_img = gray_d.shape[:2]
                        for cnt in contours[:10]:
                            area = cv2.contourArea(cnt)
                            if area < 0.2 * (h_img * w_img):
                                continue
                            peri = cv2.arcLength(cnt, True)
                            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                            if len(approx) == 4:
                                quad = approx.reshape(4, 2).astype(np.float32)
                                break
                    if quad is not None:
                        s = quad.sum(axis=1)
                        diff = np.diff(quad, axis=1)
                        tl = quad[np.argmin(s)]
                        br = quad[np.argmax(s)]
                        tr = quad[np.argmin(diff)]
                        bl = quad[np.argmax(diff)]
                        ordered = np.array([tl, tr, br, bl], dtype=np.float32)
                        width_top = np.linalg.norm(tr - tl)
                        width_bottom = np.linalg.norm(br - bl)
                        max_w = int(max(width_top, width_bottom))
                        height_left = np.linalg.norm(bl - tl)
                        height_right = np.linalg.norm(br - tr)
                        max_h = int(max(height_left, height_right))
                        dst_rect = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32)
                        M = cv2.getPerspectiveTransform(ordered, dst_rect)
                        work = cv2.warpPerspective(work, M, (max_w, max_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                        dewarp_applied = True
                        dewarp_quad = ordered.astype(float).tolist()
                        dewarp_size = (max_w, max_h)

                if unshear:
                    gray_u = _to_grayscale(work)
                    k = _estimate_horizontal_shear(gray_u)
                    unshear_k = float(k)
                    if abs(k) > 1e-4:
                        work = _shear_y_bound(work, k)
                        unshear_applied = True
                if denoise:
                    gray_for_denoise = _to_grayscale(work) if work.ndim == 3 else work
                    work = _denoise(gray_for_denoise, denoise)
                if binarize:
                    gray_for_bin = _to_grayscale(work) if work.ndim == 3 else work
                    work = _binarize_otsu(gray_for_bin)
                if grayscale and work.ndim == 3:
                    work = _to_grayscale(work)

                if dry_run:
                    details = [f"est={angle_est:.2f}°", f"applied={angle_applied:.2f}°"]
                    if method == "auto":
                        details.append(f"hough={angle_hough:.2f}°")
                        details.append(f"minrect={angle_minrect:.2f}°")
                        details.append(f"sweep={angle_sweep:.2f}°")
                    if unshear:
                        details.append(f"unshear_k={unshear_k:.5f}{'*' if unshear_applied else ''}")
                    if corner_dewarp:
                        details.append("corner_dewarp=on" if 'quad' in locals() and quad is not None else "corner_dewarp=off")
                    status = "dry"
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if max_size_mb and max_size_mb > 0:
                        budget = int(max_size_mb * 1024 * 1024)
                        sized = _downscale_for_budget(work, budget)
                        data = _encode_png_bytes(sized, compress_level=7)
                        if len(data) > budget:
                            data = _encode_png_bytes(sized, compress_level=9)
                        dst.write_bytes(data)
                        work = sized
                    else:
                        _write_png(dst, work)
                    status = "ok"
            except Exception as e:
                status = "error"
                error_msg = str(e)
        rec: dict = {
            "status": status,
            **({
                "angle": round(angle_est, 4),
                "angle_applied": round(angle_applied, 4),
                **({
                    "angle_hough": round(angle_hough, 4),
                    "angle_minrect": round(angle_minrect, 4),
                    "angle_sweep": round(angle_sweep, 4),
                } if method == "auto" else {}),
                "unshear_k": round(unshear_k, 6),
                "unshear_applied": unshear_applied,
                "corner_dewarp_applied": dewarp_applied,
                **({
                    "corner_dewarp_quad": dewarp_quad,
                    "corner_dewarp_size": dewarp_size,
                } if dewarp_applied else {}),
                **({
                    "max_size_mb": float(max_size_mb),
                    "output_size_bytes": int(dst.stat().st_size) if (not dry_run and dst.exists()) else None,
                } if (max_size_mb and not dry_run and status == "ok") else {}),
                **({"error": error_msg} if status == "error" else {}),
            } if (deskew or unshear or corner_dewarp or status == "error") else {}),
        }
        rec["time"] = round(elapsed(), 4)
        return src, dst, rec
    except Exception:
        # Best-effort: record as error
        return src, ctx.output_dir / src.name, {"status": "error", "time": 0.0}


def run(
    ctx: Context,
    grayscale: bool = False,
    deskew: bool = True,
    binarize: bool = False,
    denoise: int = 0,
    rotate: int = 0,
    max_size_mb: Optional[float] = None,
    method: Optional[str] = "auto",
    max_abs_angle: float = 8.0,
    unshear: bool = False,
    dry_run: bool = False,
    corner_dewarp: bool = False,
) -> None:
    out_dir = ctx.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter to PNG files only (silent ignore for non-PNGs)
    png_files = [p for p in ctx.files if Path(p).suffix.lower() == ".png"]

    with Progress(TextColumn("{task.description}"), BarColumn(), TimeElapsedColumn()) as progress:
        total = len(png_files)
        completed = 0

        # Build per-folder tasks
        folder_to_total: dict[str, int] = {}
        folder_to_done: dict[str, int] = {}
        for p in png_files:
            src = Path(p)
            try:
                rel = src.relative_to(ctx.input_dir)
            except ValueError:
                rel = Path(src.name)
            folder = str(rel.parent) if str(rel.parent) != "." else "(root)"
            folder_to_total[folder] = folder_to_total.get(folder, 0) + 1
        folder_to_task: dict[str, int] = {}
        for folder, tot in sorted(folder_to_total.items()):
            folder_to_done[folder] = 0
            folder_to_task[folder] = progress.add_task(f"{folder} 0/{tot}", total=tot)

        # Add TOTAL at the bottom (after per-folder tasks)
        all_task = progress.add_task(f"TOTAL 0/{total}", total=total or 1)

        # Determine parallelism: None or 0 -> auto (CPU-1, min 1)
        if ctx.workers and ctx.workers > 0:
            jobs = int(ctx.workers)
        else:
            try:
                cpu = os.cpu_count() or 1
            except Exception:
                cpu = 1
            jobs = max(1, cpu - 1)

        def gen_args():
            for p in png_files:
                yield (
                    Path(p),
                    ctx,
                    grayscale,
                    deskew,
                    binarize,
                    denoise,
                    rotate,
                    max_size_mb,
                    method,
                    max_abs_angle,
                    unshear,
                    dry_run,
                    corner_dewarp,
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
            for src, dst, rec in ex.map(_process_one, gen_args()):
                append_run_log(
                    ctx.run_log,
                    {
                        "input": str(src),
                        "output": str(dst),
                        **rec,
                    },
                )
                completed += 1
                progress.update(
                    all_task,
                    advance=1,
                    description=f"TOTAL {completed}/{total}",
                )
                # Update folder task
                try:
                    rel = Path(src).relative_to(ctx.input_dir)
                except ValueError:
                    rel = Path(src.name)
                folder = str(rel.parent) if str(rel.parent) != "." else "(root)"
                if folder in folder_to_task:
                    folder_to_done[folder] += 1
                    progress.update(
                        folder_to_task[folder],
                        advance=1,
                        description=f"{folder} {folder_to_done[folder]}/{folder_to_total[folder]}",
                    )


