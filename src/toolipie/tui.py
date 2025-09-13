from __future__ import annotations

import curses
from typing import Any, List, Optional, Callable
import threading
import queue
import time
import re
import sys
import traceback
import os
import signal
import pty
import select
import textwrap


HELP_ALL = "↑/↓: Move  •  Enter: Select/Run  •  ←/→: Toggle/Cycle  •  ESC: Default/Cancel  •  q: Back  •  Ctrl+C twice: Exit"


def _truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: max(0, width - 1)] + "…"


def _draw_menu(
    stdscr: "curses._CursesWindow",
    items: List[dict[str, str]],
    idx: int,
    note: Optional[str],
    start: int,
    title: str,
) -> None:
    stdscr.clear()
    h, w = stdscr.getmaxyx()

    # Header
    stdscr.attron(curses.A_BOLD)
    stdscr.addstr(0, 0, _truncate(title, w - 1))
    stdscr.attroff(curses.A_BOLD)

    # Help line at bottom (single line)
    help_y = h - 2
    if help_y >= 0:
        line = _truncate(HELP_ALL, w - 1)
        stdscr.addstr(help_y, 0, line)
    if note:
        note_y = h - 3
        if note_y >= 2:
            stdscr.addstr(note_y, 0, _truncate(note, w - 1), curses.A_DIM)

    # Layout: left list, right details
    top_y = 2
    left_w = max(24, min(40, w // 3))
    right_x = left_w + 1
    # Draw vertical separator
    for yy in range(top_y, help_y):
        if 0 <= right_x - 1 < w:
            stdscr.addstr(yy, right_x - 1, "|")

    # Left list: single-line per tool
    y = top_y
    list_area_h = max(0, help_y - top_y)
    max_items = max(1, list_area_h)
    end = min(len(items), start + max_items)
    for i in range(start, end):
        it = items[i]
        marker = "→ " if i == idx else "  "
        text = f"{marker}{it['key']}"
        if i == idx:
            stdscr.attron(curses.A_REVERSE)
        stdscr.addstr(y, 0, _truncate(text, left_w - 1))
        if i == idx:
            stdscr.attroff(curses.A_REVERSE)
        y += 1
        if y >= help_y:
            break

    # Left scrollbar
    total_items = max(1, len(items))
    if total_items > max_items and list_area_h > 0:
        col = left_w - 1
        for yy in range(top_y, top_y + list_area_h):
            if 0 <= col < w:
                stdscr.addstr(yy, col, "|")
        thumb_h = max(1, int(list_area_h * (max_items / total_items)))
        max_start = max(1, total_items - max_items)
        pos = int((start / max_start) * (list_area_h - thumb_h)) if max_start > 0 else 0
        for yy in range(top_y + pos, top_y + pos + thumb_h):
            if top_y <= yy < help_y:
                stdscr.addstr(yy, col, "#")
        if start > 0:
            stdscr.addstr(top_y, col, "^")
        if start + max_items < total_items:
            bottom_y = min(help_y - 1, top_y + list_area_h - 1)
            if bottom_y >= top_y:
                stdscr.addstr(bottom_y, col, "v")

    # Right details: selected tool title + description
    if 0 <= idx < len(items):
        sel_tool = items[idx]
        detail_w = max(1, w - right_x - 1)
        dy = top_y
        # Title
        stdscr.attron(curses.A_BOLD)
        stdscr.addstr(dy, right_x, _truncate(sel_tool.get('title', ''), detail_w))
        stdscr.attroff(curses.A_BOLD)
        dy += 1
        # Description wrapped
        desc = sel_tool.get('desc', '') or ''
        for line in textwrap.wrap(desc, width=detail_w):
            if dy >= help_y:
                break
            stdscr.addstr(dy, right_x, _truncate(line, detail_w))
            dy += 1

    # position indicator
    indicator = f"{idx+1 if (0 <= idx < len(items)) else 0}/{len(items)}"
    x = max(0, w - 1 - len(indicator))
    stdscr.addstr(help_y, x, indicator)

    stdscr.refresh()

def launch_menu(items: List[dict[str, str]], title: str = "Toolipie — Select a Tool", allow_back: bool = False) -> Optional[str]:
    """Launch a simple selection menu.

    - title: header text.
    - allow_back: if True, pressing q/ESC returns None to the caller (acts as Back).
    """
    if not items:
        return None

    def _menu_wrapper(stdscr: "curses._CursesWindow", items: List[dict[str, str]], title: str, allow_back: bool) -> Optional[str]:
        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.keypad(True)
        curses.raw()

        index = 0 if items else -1
        ctrl_c = 0
        note: Optional[str] = None
        start = 0
        _draw_menu(stdscr, items, index, note, start, title)

        while True:
            try:
                ch = stdscr.getch()
            except KeyboardInterrupt:
                ctrl_c += 1
                if ctrl_c >= 2:
                    raise KeyboardInterrupt
                note = "Ctrl+C detected — press again to exit"
                _draw_menu(stdscr, items, index, note, start, title)
                continue
            if ch in (curses.KEY_UP, ord('k')):
                ctrl_c = 0
                if index > 0:
                    index -= 1
                note = None
                h, _ = stdscr.getmaxyx()
                avail = max(1, ((h - 6) // 2))
                if index < start:
                    start = index
                _draw_menu(stdscr, items, index, note, start, title)
            elif ch in (curses.KEY_DOWN, ord('j')):
                ctrl_c = 0
                if index < len(items) - 1:
                    index += 1
                note = None
                h, _ = stdscr.getmaxyx()
                avail = max(1, ((h - 6) // 2))
                if index >= start + avail:
                    start = max(0, index - avail + 1)
                _draw_menu(stdscr, items, index, note, start, title)
            elif ch in (curses.KEY_ENTER, 10, 13):
                return items[index]["key"] if 0 <= index < len(items) else None
            elif ch in (ord('q'), 27):
                if allow_back:
                    return None
                note = "Enter to select, Ctrl+C twice to exit"
                _draw_menu(stdscr, items, index, note, start, title)
            elif ch == 3:  # Ctrl+C
                ctrl_c += 1
                if ctrl_c >= 2:
                    raise KeyboardInterrupt
                note = "Ctrl+C detected — press again to exit"
                _draw_menu(stdscr, items, index, note, start, title)
            else:
                ctrl_c = 0

    return curses.wrapper(_menu_wrapper, items, title, allow_back)


def _input_text(
    stdscr: "curses._CursesWindow", prompt: str, initial: str = "", allow: Optional[set[int]] = None
) -> tuple[Optional[str], bool]:
    h, w = stdscr.getmaxyx()
    buf = list(initial)
    ctrl_c = 0
    curses.curs_set(1)
    while True:
        line = prompt + "".join(buf)
        hint = "Enter: Confirm  •  Esc: Cancel  •  Ctrl+C twice: Exit"
        stdscr.move(h - 2, 0)
        stdscr.clrtoeol()
        stdscr.addstr(h - 2, 0, _truncate(hint, w - 1), curses.A_DIM)
        stdscr.move(h - 1, 0)
        stdscr.clrtoeol()
        stdscr.addstr(h - 1, 0, _truncate(line, w - 1))
        stdscr.refresh()
        ch = stdscr.getch()
        if ch in (10, 13):
            curses.curs_set(0)
            return ("".join(buf), False)
        elif ch in (27,):
            curses.curs_set(0)
            return (None, False)
        elif ch in (curses.KEY_BACKSPACE, 127, 8):
            if buf:
                buf.pop()
        elif ch == 3:
            ctrl_c += 1
            if ctrl_c >= 2:
                curses.curs_set(0)
                raise KeyboardInterrupt
        else:
            if allow is None:
                if 32 <= ch <= 126:
                    buf.append(chr(ch))
            else:
                if ch in allow:
                    buf.append(chr(ch))


def _draw_options_panel(
    stdscr: "curses._CursesWindow",
    title: str,
    desc: str,
    specs: List[dict[str, Any]],
    values: dict,
    effective: dict,
    idx: int,
    note: Optional[str],
    start: int,
) -> None:
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    stdscr.attron(curses.A_BOLD)
    stdscr.addstr(0, 0, _truncate(title, w - 1))
    stdscr.attroff(curses.A_BOLD)
    stdscr.addstr(1, 0, _truncate(desc, w - 1))
    # Selected line
    def _sel() -> str:
        if idx == 0:
            return "Run default"
        if idx == 1:
            return ""
        if idx == 2:
            return "Run custom"
        if 3 <= idx < 3 + len(specs):
            return specs[idx - 3].get("label", specs[idx - 3].get("name", ""))
        return ""
    sel_text = _sel()
    if sel_text:
        stdscr.addstr(2, 0, _truncate(f"Selected: {sel_text}", w - 1), curses.A_DIM)
    help_y = h - 2
    if help_y >= 0:
        stdscr.addstr(help_y, 0, _truncate(HELP_ALL, w - 1))
    if note:
        note_y = h - 3
        if note_y >= 2:
            stdscr.addstr(note_y, 0, _truncate(note, w - 1), curses.A_DIM)

    def fmt(spec: dict, val: Any) -> str:
        kind = spec.get("kind")
        eff_val = effective.get(spec["name"]) if val is None else val
        if kind == "tri":
            if eff_val is True:
                return "On"
            if eff_val is False:
                return "Off"
            return ""
        if eff_val is None:
            return ""
        return str(eff_val)

    total_rows = 3 + len(specs)
    y = 4
    avail_rows = max(0, (h - 2) - y)
    end = min(total_rows, start + avail_rows)

    def row_label(i: int) -> tuple[str, bool]:
        if i == 0:
            return ("Run default", True)
        if i == 1:
            return ("", False)
        if i == 2:
            return ("Run custom", True)
        if 3 <= i <= 2 + len(specs):
            spec = specs[i - 3]
            label = spec.get("label", spec.get("name"))
            val = values.get(spec["name"])  # may be None
            return (f"{label}: {fmt(spec, val)}", True)
        return ("", False)

    for i in range(start, end):
        text, selectable = row_label(i)
        marker = "→ " if (selectable and i == idx) else "  "
        line = marker + text
        if selectable and i == idx:
            stdscr.attron(curses.A_REVERSE)
        stdscr.addstr(y, 0, _truncate(line, w - 1))
        if selectable and i == idx:
            stdscr.attroff(curses.A_REVERSE)
        y += 1

    list_h = avail_rows
    total_rows = 3 + len(specs)
    if total_rows > list_h and list_h > 0:
        col = max(0, w - 2)
        for yy in range(4, 4 + list_h):
            stdscr.addstr(yy, col, "|")
        thumb_h = max(1, int(list_h * (list_h / total_rows)))
        max_start = max(1, total_rows - list_h)
        pos = int((start / max_start) * (list_h - thumb_h)) if max_start > 0 else 0
        for yy in range(4 + pos, 4 + pos + thumb_h):
            if 4 <= yy < h - 2:
                stdscr.addstr(yy, col, "#")
        if start > 0:
            stdscr.addstr(4, col, "^")
        if start + list_h < total_rows:
            bottom_y = min(h - 3, 4 + list_h - 1)
            if bottom_y >= 4:
                stdscr.addstr(bottom_y, col, "v")
    indicator = f"{idx+1}/{total_rows}"
    x = max(0, w - 1 - len(indicator))
    stdscr.addstr(h - 2, x, indicator)
    stdscr.refresh()


def launch_options_panel(
    title: str,
    desc: str,
    specs: List[dict[str, Any]],
    effective: dict,
) -> Optional[dict[str, Any]]:
    def panel(stdscr: "curses._CursesWindow") -> Optional[dict[str, Any]]:
        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.keypad(True)
        curses.raw()

        values: dict[str, Any] = {s["name"]: None for s in specs}
        idx = 0
        start = 0
        ctrl_c = 0
        note: Optional[str] = None
        _draw_options_panel(stdscr, title, desc, specs, values, effective, idx, note, start)

        while True:
            ch = stdscr.getch()
            num_entries = 3 + len(specs)
            if ch in (curses.KEY_UP, ord('k')):
                ctrl_c = 0
                if idx > 0:
                    nxt = idx - 1
                    idx = 0 if nxt == 1 else nxt
                note = None
                h, _ = stdscr.getmaxyx()
                avail = max(1, (h - 6))
                if idx < start:
                    start = idx
                _draw_options_panel(stdscr, title, desc, specs, values, effective, idx, note, start)
            elif ch in (curses.KEY_DOWN, ord('j')):
                ctrl_c = 0
                if idx < num_entries - 1:
                    nxt = idx + 1
                    idx = 2 if nxt == 1 else nxt
                note = None
                h, _ = stdscr.getmaxyx()
                avail = max(1, (h - 6))
                if idx >= start + avail:
                    start = max(0, idx - avail + 1)
                _draw_options_panel(stdscr, title, desc, specs, values, effective, idx, note, start)
            elif ch in (curses.KEY_LEFT, curses.KEY_RIGHT):
                if 3 <= idx < 3 + len(specs):
                    spec = specs[idx - 3]
                    kind = spec.get("kind")
                    name = spec["name"]
                    if kind == "choice":
                        choices: List[str] = spec.get("choices", [])
                        cur = values.get(name) if values.get(name) is not None else effective.get(name)
                        if choices:
                            if ch == curses.KEY_RIGHT:
                                values[name] = choices[0] if cur is None else choices[(choices.index(cur) + 1) % len(choices)] if cur in choices else choices[0]
                            else:
                                values[name] = choices[-1] if cur is None else choices[(choices.index(cur) - 1) % len(choices)] if cur in choices else choices[-1]
                            _draw_options_panel(stdscr, title, desc, specs, values, effective, idx, note, start)
                    elif kind == "tri":
                        values[name] = False if ch == curses.KEY_LEFT else True
                        _draw_options_panel(stdscr, title, desc, specs, values, effective, idx, note, start)
                continue
            elif ch in (curses.KEY_ENTER, 10, 13):
                if idx == 0:
                    return {"action": "run_default"}
                if idx == 2:
                    return {"action": "run_custom", "values": values}
                if 3 <= idx < 3 + len(specs):
                    spec = specs[idx - 3]
                    name = spec["name"]
                    kind = spec.get("kind")
                    base_val = values.get(name)
                    if base_val is None:
                        base_val = effective.get(name)
                    initial = "" if base_val is None else str(base_val)
                    allow = None
                    if kind == "int":
                        allow = set([ord(c) for c in "-0123456789"])
                    elif kind == "float":
                        allow = set([ord(c) for c in "-0123456789."])
                    text, exit_flag = _input_text(stdscr, f"{spec.get('label', name)}: ", initial, allow)
                    if exit_flag:
                        return {"action": "exit"}
                    if text is None or text.strip() == "":
                        values[name] = None
                    else:
                        try:
                            if kind == "int":
                                values[name] = int(text)
                            elif kind == "float":
                                values[name] = float(text)
                            else:
                                values[name] = text
                        except Exception:
                            values[name] = None
                    _draw_options_panel(stdscr, title, desc, specs, values, effective, idx, note, start)
            elif ch == 27:  # ESC reset
                if 3 <= idx < 3 + len(specs):
                    spec = specs[idx - 3]
                    values[spec["name"]] = None
                    _draw_options_panel(stdscr, title, desc, specs, values, effective, idx, note, start)
            elif ch in (ord('q'),):
                return {"action": "back"}
            elif ch == 3:
                ctrl_c += 1
                if ctrl_c >= 2:
                    return {"action": "exit"}
                note = "Ctrl+C detected — press again to exit"
                _draw_options_panel(stdscr, title, desc, specs, values, effective, idx, note, start)
            else:
                ctrl_c = 0
    try:
        return curses.wrapper(panel)
    except KeyboardInterrupt:
        return {"action": "exit"}


def launch_text_prompt(title: str, prompt: str, initial: str = "") -> Optional[str]:
    """Open a simple text prompt in TUI and return the entered string or None if cancelled."""
    def panel(stdscr: "curses._CursesWindow") -> Optional[str]:
        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.keypad(True)
        curses.raw()
        h, w = stdscr.getmaxyx()
        stdscr.clear()
        stdscr.attron(curses.A_BOLD)
        stdscr.addstr(0, 0, _truncate(title, w - 1))
        stdscr.attroff(curses.A_BOLD)
        text, exit_flag = _input_text(stdscr, prompt, initial)
        if exit_flag:
            return None
        if text is None or text.strip() == "":
            return None
        return text.strip()
    return curses.wrapper(panel)


# --- Output / Console panel -------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _strip_ansi(s: str) -> str:
    try:
        return _ANSI_RE.sub("", s)
    except Exception:
        return s


class _CaptureWriter:
    def __init__(self, q: "queue.Queue[Optional[str]]") -> None:
        self.q = q
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        s = s.replace("\r", "\n")
        s = _strip_ansi(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            try:
                self.q.put_nowait(line)
            except Exception:
                pass
        return len(s)

    def flush(self) -> None:  # pragma: no cover - simple
        if self._buf:
            try:
                self.q.put_nowait(self._buf)
            except Exception:
                pass
            self._buf = ""

    def isatty(self) -> bool:  # hint to libraries
        # Treat as TTY so libraries like Rich render progress updates.
        # We strip ANSI and normalize CR to NL above.
        return True


def launch_output_panel(title: str, runner: Callable[[], None], cancel_event: Optional[threading.Event] = None) -> None:
    """Run a callable while showing its stdout/stderr in a scrollable TUI panel."""

    def _panel(stdscr: "curses._CursesWindow") -> None:
        curses.curs_set(0)
        stdscr.nodelay(True)  # non-blocking to refresh while running
        stdscr.keypad(True)
        curses.raw()

        q: "queue.Queue[Optional[str]]" = queue.Queue()
        done = threading.Event()
        lines: List[str] = []
        scroll = 0  # 0 = bottom (auto-follow)

        def worker() -> None:
            out = _CaptureWriter(q)
            err = _CaptureWriter(q)
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = out, err
            try:
                runner()
            except Exception:
                tb = traceback.format_exc()
                try:
                    q.put(tb)
                except Exception:
                    pass
            finally:
                try:
                    out.flush()
                    err.flush()
                except Exception:
                    pass
                sys.stdout, sys.stderr = old_out, old_err
                try:
                    q.put(None)  # sentinel
                except Exception:
                    pass
                done.set()

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        help_base = "↑/↓: Scroll  •  q: Cancel/Back  •  Ctrl+C twice: Abort"
        ctrl_c = 0
        requested_cancel = False

        while True:
            # Drain queue
            while True:
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    # finished signal
                    break
                lines.append(item)
            stdscr.clear()
            h, w = stdscr.getmaxyx()
            stdscr.attron(curses.A_BOLD)
            stdscr.addstr(0, 0, _truncate(title, w - 1))
            stdscr.attroff(curses.A_BOLD)

            if not done.is_set():
                if requested_cancel:
                    info = "Cancelling… waiting for tasks to stop"
                else:
                    info = "Running…"
            else:
                info = "Completed. Press q to return."
            help_y = h - 2
            if help_y >= 0:
                stdscr.addstr(help_y, 0, _truncate(f"{help_base}", w - 1))
            note_y = h - 3
            if note_y >= 2:
                stdscr.addstr(note_y, 0, _truncate(info, w - 1), curses.A_DIM)

            # Viewport
            top_y = 2
            max_rows = max(0, help_y - top_y)
            total = len(lines)
            if scroll == 0:
                start = max(0, total - max_rows)
            else:
                start = max(0, total - max_rows - scroll)
            end = min(total, start + max_rows)
            y = top_y
            for i in range(start, end):
                stdscr.addstr(y, 0, _truncate(lines[i], w - 1))
                y += 1

            # indicator
            indicator = f"{total} lines"
            x = max(0, w - 1 - len(indicator))
            stdscr.addstr(help_y, x, indicator)
            stdscr.refresh()

            ch = stdscr.getch()
            if ch == -1:
                time.sleep(0.05)
                continue
            if ch in (curses.KEY_UP, ord('k')):
                if scroll < max(total - max_rows, 0):
                    scroll += 1
            elif ch in (curses.KEY_DOWN, ord('j')):
                if scroll > 0:
                    scroll -= 1
            elif ch in (ord('q'), 27):
                if done.is_set():
                    return
                # request cancel and signal the runner via event (best-effort)
                requested_cancel = True
                try:
                    if cancel_event is not None:
                        cancel_event.set()
                except Exception:
                    pass
            elif ch == 3:  # Ctrl+C
                ctrl_c += 1
                if ctrl_c >= 2:
                    raise KeyboardInterrupt
            else:
                ctrl_c = 0

    curses.wrapper(_panel)


def launch_process_panel(title: str, argv: list[str]) -> int:
    """Spawn a subprocess (pty) and stream its combined output inside a TUI panel.

    Returns the process exit code.
    q requests graceful termination: send SIGTERM, then SIGKILL after timeout.
    """

    def _panel(stdscr: "curses._CursesWindow") -> int:
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.keypad(True)
        curses.raw()

        master_fd, slave_fd = pty.openpty()
        pid = os.fork()
        if pid == 0:
            # Child
            try:
                os.setsid()
            except Exception:
                pass
            os.dup2(slave_fd, 1)
            os.dup2(slave_fd, 2)
            try:
                os.close(master_fd)
            except Exception:
                pass
            try:
                os.execv(argv[0], argv)
            except Exception:
                os._exit(127)
        # Parent
        try:
            os.close(slave_fd)
        except Exception:
            pass

        buf = ""
        lines: List[str] = [""]
        cur_row = 0
        cur_col = 0
        scroll = 0
        requested_cancel = False
        cancel_started_at: float | None = None
        CANCEL_GRACE = 2.0  # seconds before auto-escalate to SIGKILL
        ctrl_c = 0
        exit_code: Optional[int] = None

        def kill_proc(sig: int) -> None:
            try:
                os.kill(pid, sig)
            except Exception:
                pass

        while True:
            # Read available bytes
            try:
                r, _, _ = select.select([master_fd], [], [], 0.05)
                if master_fd in r:
                    chunk_b = os.read(master_fd, 4096)
                    if chunk_b:
                        buf += chunk_b.decode("utf-8", errors="ignore")
            except Exception:
                pass

            # Minimal ANSI handling: CR/LF, CSI cursor moves, erase in line
            i = 0
            L = len(buf)
            while i < L:
                ch = buf[i]
                if ch == "\x1b":
                    # CSI?
                    if i + 1 < L and buf[i + 1] == "[":
                        j = i + 2
                        params = ""
                        while j < L and not ("A" <= buf[j] <= "Z" or "a" <= buf[j] <= "z"):
                            params += buf[j]
                            j += 1
                        if j < L:
                            cmd = buf[j]
                            parts = [p for p in params.split(";") if p]
                            getn = lambda default=1: int(parts[0]) if parts else default
                            if cmd == "A":  # up
                                cur_row = max(0, cur_row - getn())
                                cur_col = min(cur_col, len(lines[cur_row]))
                            elif cmd == "B":  # down
                                cur_row += getn()
                                while cur_row >= len(lines):
                                    lines.append("")
                                cur_col = min(cur_col, len(lines[cur_row]))
                            elif cmd == "C":  # right
                                cur_col += getn()
                            elif cmd == "D":  # left
                                cur_col = max(0, cur_col - getn())
                            elif cmd == "K":  # erase in line
                                mode = int(parts[0]) if parts else 0
                                if cur_row >= len(lines):
                                    lines.extend([""] * (cur_row - len(lines) + 1))
                                line = lines[cur_row]
                                if mode == 2:
                                    lines[cur_row] = ""
                                    cur_col = 0
                                elif mode == 1:
                                    lines[cur_row] = line[cur_col:]
                                    cur_col = 0
                                else:
                                    lines[cur_row] = line[:cur_col]
                            # ignore other CSI (colors, hide cursor, etc.)
                            i = j + 1
                            continue
                    # Unknown ESC sequence; skip
                    i += 1
                    continue
                elif ch == "\r":
                    cur_col = 0
                    i += 1
                    continue
                elif ch == "\n":
                    cur_row += 1
                    cur_col = 0
                    if cur_row >= len(lines):
                        lines.append("")
                    i += 1
                    continue
                else:
                    # printable
                    if cur_row >= len(lines):
                        lines.extend([""] * (cur_row - len(lines) + 1))
                    line = lines[cur_row]
                    if cur_col > len(line):
                        line = line + (" " * (cur_col - len(line)))
                    if cur_col == len(line):
                        line = line + ch
                    else:
                        line = line[:cur_col] + ch + line[cur_col + 1 :]
                    lines[cur_row] = _strip_ansi(line)
                    cur_col += 1
                    i += 1
            buf = ""

            # Check child status
            if exit_code is None:
                try:
                    pid_done, status = os.waitpid(pid, os.WNOHANG)
                    if pid_done == pid:
                        if os.WIFEXITED(status):
                            exit_code = os.WEXITSTATUS(status)
                        elif os.WIFSIGNALED(status):
                            exit_code = 128 + os.WTERMSIG(status)
                except ChildProcessError:
                    exit_code = 0

            stdscr.clear()
            h, w = stdscr.getmaxyx()
            stdscr.attron(curses.A_BOLD)
            stdscr.addstr(0, 0, _truncate(title, w - 1))
            stdscr.attroff(curses.A_BOLD)

            info: str
            if exit_code is None:
                info = "Cancelling… waiting to terminate" if requested_cancel else "Running…"
            else:
                if requested_cancel:
                    info = "Cancelled."
                elif exit_code == 0:
                    info = "Completed successfully."
                else:
                    info = "Completed with errors."

            help_y = h - 2
            if help_y >= 0:
                stdscr.addstr(help_y, 0, _truncate("↑/↓: Scroll  •  q: Cancel/Back  •  Ctrl+C twice: Abort", w - 1))
            note_y = h - 3
            if note_y >= 2:
                stdscr.addstr(note_y, 0, _truncate(info, w - 1), curses.A_DIM)

            top_y = 2
            max_rows = max(0, help_y - top_y)
            total = len(lines)
            start = max(0, total - max_rows - scroll) if max_rows > 0 else 0
            end = min(total, start + max_rows)
            y = top_y
            for i in range(start, end):
                stdscr.addstr(y, 0, _truncate(lines[i], w - 1))
                y += 1

            indicator = f"{total} lines"
            x = max(0, w - 1 - len(indicator))
            stdscr.addstr(help_y, x, indicator)
            stdscr.refresh()

            ch = stdscr.getch()
            if ch == -1:
                # If cancelling and grace time exceeded, escalate to SIGKILL
                if requested_cancel and cancel_started_at is not None and exit_code is None:
                    if (time.time() - cancel_started_at) >= CANCEL_GRACE:
                        kill_proc(signal.SIGKILL)
                time.sleep(0.05)
                continue
            if ch in (curses.KEY_UP, ord('k')):
                if scroll < max(total - max_rows, 0):
                    scroll += 1
            elif ch in (curses.KEY_DOWN, ord('j')):
                if scroll > 0:
                    scroll -= 1
            elif ch in (ord('q'), 27):
                if exit_code is not None:
                    try:
                        os.close(master_fd)
                    except Exception:
                        pass
                    return exit_code
                if not requested_cancel:
                    requested_cancel = True
                    cancel_started_at = time.time()
                    kill_proc(signal.SIGTERM)
                else:
                    # escalate immediately on second request
                    kill_proc(signal.SIGKILL)
            elif ch == 3:
                ctrl_c += 1
                if ctrl_c >= 2:
                    kill_proc(signal.SIGKILL)
                    try:
                        os.close(master_fd)
                    except Exception:
                        pass
                    return exit_code or 130
            else:
                ctrl_c = 0

            # Auto-exit once the process has terminated
            if exit_code is not None:
                try:
                    os.close(master_fd)
                except Exception:
                    pass
                return exit_code

    return curses.wrapper(_panel)
