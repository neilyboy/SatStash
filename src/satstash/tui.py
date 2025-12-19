from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timedelta, timezone
import json
import socket
from pathlib import Path
import random
import re
import shutil
import subprocess
import signal
import threading
import time
from typing import Callable, List, Optional
from urllib.parse import urlparse

import requests
from requests.exceptions import HTTPError

from mutagen.mp4 import MP4, MP4Cover

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message_pump import NoActiveAppError
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Button, DataTable, Footer, Header, Input, ProgressBar, Static

from satstash.api.client import SxmClient
from satstash.usecases.live import LivePlaybackHandle
from satstash.hls.proxy import HlsProxy
from satstash.hls.variants import select_variant
from satstash.usecases.record import (
    FfmpegRecordHandle,
    RecordHandle,
    probe_dvr_buffer_start_pdt,
    start_ffmpeg_recording,
    start_recording,
    stop_ffmpeg_recording,
    stop_recording_with_options,
)
from satstash.auth.direct import SiriusXMDirectAuth
from satstash.channels import Channel, clear_channels_cache, load_cached_channels, save_cached_channels
from satstash.config import load_settings, save_settings
from satstash.session import NotLoggedInError, SessionExpiredError, clear_session, load_session, save_session
from satstash.usecases.live import start_live_playback

from platformdirs import user_cache_dir, user_config_dir
from PIL import Image as PILImage
from PIL import ImageEnhance
from rich.style import Style
from rich.text import Text


_BUILTIN_LOGO = """SATSTASH
SIRIUSXM
"""


def _scheduled_recordings_path() -> Path:
    cfg_dir = Path(user_config_dir("satstash"))
    try:
        cfg_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return cfg_dir / "scheduled_recordings.json"


def _dvr_queue_path() -> Path:
    cfg_dir = Path(user_config_dir("satstash"))
    try:
        cfg_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return cfg_dir / "dvr_queue.json"


def _load_dvr_queue() -> list[Path]:
    path = _dvr_queue_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = []
    if isinstance(raw, dict):
        items = raw.get("items", [])
    elif isinstance(raw, list):
        items = raw
    out: list[Path] = []
    for x in items or []:
        if not isinstance(x, str) or not x.strip():
            continue
        try:
            out.append(Path(x).expanduser())
        except Exception:
            continue
    return out


def _save_dvr_queue(items: list[Path]) -> None:
    path = _dvr_queue_path()
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        payload = {"items": [str(p) for p in (items or []) if isinstance(p, Path)]}
    except Exception:
        payload = {"items": []}
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _playlists_dir() -> Path:
    cfg_dir = Path(user_config_dir("satstash"))
    try:
        cfg_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    d = cfg_dir / "playlists"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d


def _sanitize_playlist_name(name: str) -> str:
    s = (name or "").strip()
    if not s:
        return ""
    out = "".join(c if c.isalnum() or c in " _-." else "_" for c in s).strip()
    out = out.replace("/", "_").replace("\\", "_")
    return out


def _playlist_path(name: str) -> Optional[Path]:
    safe = _sanitize_playlist_name(name)
    if not safe:
        return None
    return _playlists_dir() / f"{safe}.json"


def _save_playlist(*, name: str, items: list[Path]) -> Optional[Path]:
    p = _playlist_path(name)
    if p is None:
        return None
    tmp = p.with_suffix(p.suffix + ".tmp")
    payload = {
        "name": _sanitize_playlist_name(name),
        "created_at_iso": datetime.now().astimezone().isoformat(),
        "items": [str(x) for x in (items or []) if isinstance(x, Path)],
    }
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)
    return p


def _load_playlist(name: str) -> Optional[list[Path]]:
    p = _playlist_path(name)
    if p is None or not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    items = []
    if isinstance(raw, dict):
        items = raw.get("items", [])
    elif isinstance(raw, list):
        items = raw
    out: list[Path] = []
    for x in items or []:
        if not isinstance(x, str) or not x.strip():
            continue
        try:
            out.append(Path(x).expanduser())
        except Exception:
            continue
    return out


def _load_scheduled_recordings() -> list[dict]:
    path = _scheduled_recordings_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            items = raw.get("schedules", [])
        else:
            items = raw
        return [x for x in items if isinstance(x, dict)]
    except Exception:
        return []


def _save_scheduled_recordings(items: list[dict]) -> None:
    path = _scheduled_recordings_path()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"schedules": items}, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _new_schedule_id() -> str:
    try:
        return hashlib.sha1(f"{time.time()}-{random.random()}".encode("utf-8")).hexdigest()[:12]
    except Exception:
        return str(int(time.time()))


def _schedule_dt_local(s: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat((s or "").strip())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
        return dt
    except Exception:
        return None


class _ScheduleChannelPickerPane(Widget):
    def __init__(self, channels: List[Channel], *, on_pick: Callable[[Optional[Channel]], None]):
        super().__init__()
        self._channels = list(channels)
        self._on_pick = on_pick

    def compose(self) -> ComposeResult:
        yield Static("Pick a channel", id="sched_pick_title")
        with Horizontal(id="sched_pick_actions"):
            yield Button("Select", id="sched_pick_select", variant="primary")
            yield Button("Cancel", id="sched_pick_cancel")
        tbl = ActivatableDataTable(id="sched_pick_table")
        try:
            tbl._activate_callback = lambda: self._pick_selected()
        except Exception:
            pass
        yield tbl

    def on_mount(self) -> None:
        tbl = self.query_one("#sched_pick_table", DataTable)
        tbl.cursor_type = "row"
        tbl.add_column("Channel", width=34)
        tbl.add_column("Ch#", width=6)
        for ch in sorted(self._channels, key=lambda c: (c.number is None, c.number or 9999, c.name)):
            lbl = f"{ch.number if ch.number is not None else ''} {ch.name}".strip() or ch.name
            tbl.add_row(lbl, str(ch.number or ""), key=str(ch.id))
        tbl.focus()

    def _pick_selected(self) -> None:
        tbl = self.query_one("#sched_pick_table", DataTable)
        if tbl.cursor_coordinate is None:
            return
        row_key, _ = tbl.coordinate_to_cell_key(tbl.cursor_coordinate)
        cid = row_key.value if hasattr(row_key, "value") else str(row_key)
        for ch in self._channels:
            if str(ch.id) == str(cid):
                self._on_pick(ch)
                return
        self._on_pick(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        try:
            event.stop()
        except Exception:
            pass
        if event.button.id == "sched_pick_cancel":
            self._on_pick(None)
            return
        if event.button.id == "sched_pick_select":
            self._pick_selected()


class _ScheduleEditorPane(Widget):
    def __init__(
        self,
        *,
        channel: Channel,
        existing: Optional[dict],
        on_save: Callable[[Optional[dict]], None],
    ):
        super().__init__()
        self._channel = channel
        self._existing = dict(existing or {})
        self._on_save = on_save

    def compose(self) -> ComposeResult:
        now = datetime.now().astimezone()
        yield Static(f"System time now: {now.strftime('%Y-%m-%d %I:%M:%S %p %Z')}")
        yield Static(f"Channel: {self._channel.name}")
        yield Static("Date (YYYY-MM-DD)")
        yield Input(id="sched_date")
        yield Static("Start (HH:MM 24h)")
        yield Input(id="sched_start")
        yield Static("End (HH:MM 24h)")
        yield Input(id="sched_end")
        with Horizontal():
            yield Button("Save", id="sched_save", variant="primary")
            yield Button("Cancel", id="sched_cancel")

    def on_mount(self) -> None:
        tz = datetime.now().astimezone().tzinfo
        date_in = self.query_one("#sched_date", Input)
        start_in = self.query_one("#sched_start", Input)
        end_in = self.query_one("#sched_end", Input)

        st = _schedule_dt_local(str(self._existing.get("start_time_iso") or ""))
        en = _schedule_dt_local(str(self._existing.get("end_time_iso") or ""))
        now = datetime.now().astimezone()
        if st is None:
            st = now.replace(second=0, microsecond=0)
        if en is None:
            en = (st + timedelta(minutes=30)).replace(second=0, microsecond=0)
        if st.tzinfo is None:
            st = st.replace(tzinfo=tz)
        if en.tzinfo is None:
            en = en.replace(tzinfo=tz)

        date_in.value = st.astimezone().strftime("%Y-%m-%d")
        start_in.value = st.astimezone().strftime("%H:%M")
        end_in.value = en.astimezone().strftime("%H:%M")
        date_in.focus()

    def _parse_local(self, date_s: str, time_s: str) -> Optional[datetime]:
        tz = datetime.now().astimezone().tzinfo
        ds = (date_s or "").strip()
        ts = (time_s or "").strip()
        if not ds or not ts:
            return None

        # Accept both 24h and 12h inputs.
        # Examples:
        # - 23:15
        # - 4:51 AM
        # - 04:51PM
        candidates = [
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %I:%M %p",
            "%Y-%m-%d %I:%M%p",
        ]
        for fmt in candidates:
            try:
                dt = datetime.strptime(f"{ds} {ts}", fmt)
                return dt.replace(tzinfo=tz)
            except Exception:
                continue
        return None

    def _save(self) -> None:
        date_s = (self.query_one("#sched_date", Input).value or "").strip()
        start_s = (self.query_one("#sched_start", Input).value or "").strip()
        end_s = (self.query_one("#sched_end", Input).value or "").strip()
        st = self._parse_local(date_s, start_s)
        en = self._parse_local(date_s, end_s)
        if st is None or en is None:
            try:
                self.app.notify("Invalid date/time. Use 24h HH:MM (e.g. 23:15) or 12h H:MM AM/PM (e.g. 4:51 AM).")
            except Exception:
                pass
            return
        if en <= st:
            try:
                self.app.notify("End must be after start")
            except Exception:
                pass
            return

        out = dict(self._existing)
        if not out.get("id"):
            out["id"] = _new_schedule_id()
        out["enabled"] = bool(out.get("enabled", True))
        out["channel_id"] = str(self._channel.id)
        out["channel_type"] = str(getattr(self._channel, "channel_type", "channel-linear") or "channel-linear")
        out["channel_name"] = str(self._channel.name)
        out["start_time_iso"] = st.astimezone().isoformat()
        out["end_time_iso"] = en.astimezone().isoformat()
        if not out.get("created_at_iso"):
            out["created_at_iso"] = datetime.now().astimezone().isoformat()
        self._on_save(out)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        try:
            event.stop()
        except Exception:
            pass
        if event.button.id == "sched_cancel":
            self._on_save(None)
            return
        if event.button.id == "sched_save":
            self._save()


class ScheduledRecordingsPane(Widget):
    def compose(self) -> ComposeResult:
        yield Static("Scheduled Recordings", id="sched_title")
        yield Static("", id="sched_now")
        with Horizontal(id="sched_actions"):
            yield Button("Add", id="sched_add")
            yield Button("Edit", id="sched_edit")
            yield Button("Delete", id="sched_delete")
            yield Button("Enable/Disable", id="sched_toggle")
            yield Button("Run now", id="sched_run_now")
            yield Button("Close", id="sched_close")
        yield DataTable(id="sched_table")

    def on_mount(self) -> None:
        try:
            self.set_interval(1.0, self._refresh_now)
        except Exception:
            pass
        try:
            tbl = self.query_one("#sched_table", DataTable)
            tbl.cursor_type = "row"
            tbl.add_column("On", width=4)
            tbl.add_column("Channel", width=28)
            tbl.add_column("Start", width=18)
            tbl.add_column("End", width=18)
            self._reload()
            tbl.focus()
        except Exception:
            pass

    def _refresh_now(self) -> None:
        try:
            now = datetime.now().astimezone()
            self.query_one("#sched_now", Static).update(
                f"System time now: {now.strftime('%Y-%m-%d %I:%M:%S %p %Z')}"
            )
        except Exception:
            pass

    def _reload(self) -> None:
        tbl = self.query_one("#sched_table", DataTable)
        tbl.clear()
        items = _load_scheduled_recordings()
        def key_dt(x: dict) -> float:
            st = _schedule_dt_local(str(x.get("start_time_iso") or ""))
            return float(st.timestamp()) if st else 0.0
        for it in sorted(items, key=key_dt):
            ch_name = str(it.get("channel_name") or "Unknown")
            enabled = bool(it.get("enabled", True))
            st = _schedule_dt_local(str(it.get("start_time_iso") or ""))
            en = _schedule_dt_local(str(it.get("end_time_iso") or ""))
            tbl.add_row(
                "YES" if enabled else "NO",
                ch_name,
                st.astimezone().strftime("%m-%d %H:%M") if st else "?",
                en.astimezone().strftime("%m-%d %H:%M") if en else "?",
                key=str(it.get("id") or ""),
            )

    def _selected_id(self) -> Optional[str]:
        try:
            tbl = self.query_one("#sched_table", DataTable)
            if tbl.cursor_coordinate is None:
                return None
            row_key, _ = tbl.coordinate_to_cell_key(tbl.cursor_coordinate)
            return row_key.value if hasattr(row_key, "value") else str(row_key)
        except Exception:
            return None

    def _upsert_item(self, item: dict) -> None:
        items = _load_scheduled_recordings()
        out: list[dict] = []
        replaced = False
        sid = str(item.get("id") or "")
        for it in items:
            if str(it.get("id") or "") == sid:
                out.append(item)
                replaced = True
            else:
                out.append(it)
        if not replaced:
            out.append(item)
        _save_scheduled_recordings(out)

    def _delete_item(self, sid: str) -> None:
        items = [it for it in _load_scheduled_recordings() if str(it.get("id") or "") != str(sid)]
        _save_scheduled_recordings(items)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        # Prevent bubbling to SatStashApp.on_button_pressed (which would show Not implemented yet).
        try:
            event.stop()
        except Exception:
            pass

        bid = event.button.id
        if bid == "sched_close":
            try:
                self.app._show_right_idle()  # type: ignore[attr-defined]
            except Exception:
                pass
            return

        if bid == "sched_add":
            self._action_add()
            return

        if bid == "sched_edit":
            self._action_edit()
            return

        if bid == "sched_delete":
            sid = self._selected_id()
            if not sid:
                try:
                    self.app.notify("Select a schedule first")
                except Exception:
                    pass
                return
            self._delete_item(sid)
            self._reload()
            return

        if bid == "sched_toggle":
            sid = self._selected_id()
            if not sid:
                try:
                    self.app.notify("Select a schedule first")
                except Exception:
                    pass
                return
            items = _load_scheduled_recordings()
            for it in items:
                if str(it.get("id") or "") == str(sid):
                    it["enabled"] = not bool(it.get("enabled", True))
                    _save_scheduled_recordings(items)
                    break
            self._reload()
            return

        if bid == "sched_run_now":
            sid = self._selected_id()
            if not sid:
                try:
                    self.app.notify("Select a schedule first")
                except Exception:
                    pass
                return
            items = _load_scheduled_recordings()
            now = datetime.now().astimezone()
            for it in items:
                if str(it.get("id") or "") == str(sid):
                    en = _schedule_dt_local(str(it.get("end_time_iso") or ""))
                    if en is None or en <= now:
                        try:
                            self.app.notify("End time already passed")
                        except Exception:
                            pass
                        return
                    it["start_time_iso"] = now.isoformat()
                    _save_scheduled_recordings(items)
                    break
            self._reload()
            return

    def _action_add(self) -> None:
        try:
            app = self.app
            channels = app.fetch_channels(force_refresh=False)
        except Exception as exc:
            try:
                self.app.notify(f"Failed to load channels: {exc}")
            except Exception:
                pass
            return

        def back_to_main() -> None:
            try:
                self.app._set_right_pane(ScheduledRecordingsPane())  # type: ignore[attr-defined]
            except Exception:
                pass

        def after_pick(ch: Optional[Channel]) -> None:
            if ch is None:
                back_to_main()
                return

            def after_edit(payload: Optional[dict]) -> None:
                if payload:
                    self._upsert_item(payload)
                back_to_main()

            try:
                self.app._set_right_pane(_ScheduleEditorPane(channel=ch, existing=None, on_save=after_edit))  # type: ignore[attr-defined]
            except Exception:
                back_to_main()

        try:
            self.app._set_right_pane(_ScheduleChannelPickerPane(channels, on_pick=after_pick))  # type: ignore[attr-defined]
        except Exception:
            pass

    def _action_edit(self) -> None:
        sid = self._selected_id()
        if not sid:
            try:
                self.app.notify("Select a schedule first")
            except Exception:
                pass
            return
        items = _load_scheduled_recordings()
        existing = next((it for it in items if str(it.get("id") or "") == str(sid)), None)
        if not existing:
            try:
                self.app.notify("Schedule not found")
            except Exception:
                pass
            return
        try:
            channels = self.app.fetch_channels(force_refresh=False)
            ch = next((c for c in channels if str(c.id) == str(existing.get("channel_id") or "")), None)
        except Exception:
            ch = None
        if ch is None:
            try:
                self.app.notify("Channel not found; try Add instead")
            except Exception:
                pass
            return

        def back_to_main() -> None:
            try:
                self.app._set_right_pane(ScheduledRecordingsPane())  # type: ignore[attr-defined]
            except Exception:
                pass

        def after_edit(payload: Optional[dict]) -> None:
            if payload:
                self._upsert_item(payload)
            back_to_main()

        try:
            self.app._set_right_pane(_ScheduleEditorPane(channel=ch, existing=existing, on_save=after_edit))  # type: ignore[attr-defined]
        except Exception:
            back_to_main()



def _probe_duration_s(path: Path) -> Optional[float]:
    try:
        if not shutil.which("ffprobe"):
            try:
                mp4 = MP4(str(path))
                dur = float(getattr(getattr(mp4, "info", None), "length", 0.0) or 0.0)
                if dur > 0.0:
                    return dur
            except Exception:
                return None
            return None
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(path),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if not out:
            try:
                mp4 = MP4(str(path))
                dur = float(getattr(getattr(mp4, "info", None), "length", 0.0) or 0.0)
                if dur > 0.0:
                    return dur
            except Exception:
                return None
            return None
        try:
            v = float(out)
            if v > 0.0:
                return v
        except Exception:
            pass
        try:
            mp4 = MP4(str(path))
            dur = float(getattr(getattr(mp4, "info", None), "length", 0.0) or 0.0)
            if dur > 0.0:
                return dur
        except Exception:
            return None
        return None
    except Exception:
        return None


def _sanitize_track_index(*, tracks: list[dict], duration_s: Optional[float]) -> list[dict]:
    # Ensure stable, monotonic offsets for cue/chapters.
    # - sort by offset
    # - clamp to [0, duration)
    # - de-dupe obvious duplicates
    # - enforce strictly increasing offsets (best-effort)
    try:
        max_off: Optional[float] = None
        if duration_s is not None:
            try:
                max_off = max(0.0, float(duration_s) - 0.001)
            except Exception:
                max_off = None

        norm: list[dict] = []
        for t in tracks or []:
            try:
                off = float(t.get("offset_s") or 0.0)
            except Exception:
                off = 0.0
            off = max(0.0, off)
            if max_off is not None:
                off = min(off, max_off)
            nt = dict(t)
            nt["offset_s"] = off
            norm.append(nt)

        norm.sort(key=lambda x: float(x.get("offset_s") or 0.0))

        deduped: list[dict] = []
        seen: set[tuple[str, str, int]] = set()
        for t in norm:
            tid = str(t.get("id") or "")
            disp = str(t.get("display") or t.get("title") or "")
            try:
                off_ms = int(round(float(t.get("offset_s") or 0.0) * 1000.0))
            except Exception:
                off_ms = 0
            key = (tid, disp, off_ms)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(t)

        # Enforce strictly increasing offsets (some players dislike equal INDEX/START).
        out: list[dict] = []
        prev: float = -1.0
        for t in deduped:
            try:
                off = float(t.get("offset_s") or 0.0)
            except Exception:
                off = 0.0
            if off <= prev:
                off = prev + 0.001
            if max_off is not None:
                off = min(off, max_off)
            if off < 0.0:
                off = 0.0
            nt = dict(t)
            nt["offset_s"] = off
            out.append(nt)
            prev = off

        return out
    except Exception:
        return list(tracks or [])


def _fmt_time(seconds: float) -> str:
    try:
        seconds_i = int(seconds)
    except Exception:
        return "0:00"
    m, s = divmod(max(0, seconds_i), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _parse_ffmpeg_size_bytes(text: str) -> tuple[int, str]:
    """Best-effort parse of ffmpeg progress lines (size=...)."""
    try:
        lines = (text or "").splitlines()
    except Exception:
        lines = []
    try:
        tail = "\n".join(lines[-300:])
    except Exception:
        tail = text or ""

    # Prefer the richest unit formats first.
    try:
        mm = re.findall(r"\bsize=\s*([0-9]+(?:\.[0-9]+)?)\s*([KMG])i?B\b", tail)
    except Exception:
        mm = []
    if mm:
        num_s, unit = mm[-1]
        try:
            num = float(num_s)
        except Exception:
            num = 0.0
        mult = 1.0
        if unit == "K":
            mult = 1024.0
        elif unit == "M":
            mult = 1024.0 * 1024.0
        elif unit == "G":
            mult = 1024.0 * 1024.0 * 1024.0
        try:
            return int(num * mult), f"{num_s}{unit}B"
        except Exception:
            return 0, ""

    # Accept older kB format.
    try:
        mk = re.findall(r"\bsize=\s*([0-9]+)kB\b", tail)
    except Exception:
        mk = []
    if mk:
        try:
            kb = int(mk[-1])
            return kb * 1024, f"{kb}kB"
        except Exception:
            return 0, ""

    # Accept raw bytes.
    try:
        mb = re.findall(r"\bsize=\s*([0-9]+)\s*B\b", tail)
    except Exception:
        mb = []
    if mb:
        try:
            b = int(mb[-1])
            return b, f"{b}B"
        except Exception:
            return 0, ""

    return 0, ""


def _parse_ffmpeg_hls_activity(text: str) -> str:
    """Best-effort parse of ffmpeg logs to show what segment/URL it is currently touching."""
    try:
        lines = (text or "").splitlines()
    except Exception:
        lines = []
    try:
        tail = "\n".join(lines[-250:])
    except Exception:
        tail = text or ""

    def _shorten(s: str) -> str:
        ss = (s or "").strip()
        if not ss:
            return ""
        # Extract a stable segment id when possible.
        try:
            m = re.findall(r"\bseg\s*([0-9]+(?:\.[0-9]+)?)\b", ss, flags=re.IGNORECASE)
        except Exception:
            m = []
        if m:
            return f"seg{m[-1]}"
        try:
            m2 = re.findall(r"\bsegment\s*([0-9]+(?:\.[0-9]+)?)\b", ss, flags=re.IGNORECASE)
        except Exception:
            m2 = []
        if m2:
            return f"seg{m2[-1]}"

        # Fallback to filename-ish token and hard trim.
        try:
            ss = ss.split("?")[0]
        except Exception:
            pass
        try:
            name = Path(ss).name
        except Exception:
            name = ss
        name = (name or "").strip()

        # Try to derive a compact seg id from the filename.
        try:
            nm = name
            try:
                nm = nm.split("?")[0]
            except Exception:
                pass
            nums = re.findall(r"([0-9]+)", nm)
            if nums:
                if len(nums) >= 2:
                    return f"seg{nums[-2]}.{nums[-1]}"
                return f"seg{nums[-1]}"
        except Exception:
            pass

        if len(name) > 32:
            return name[:29] + "..."
        return name

    # Common ffmpeg patterns.
    m = None
    try:
        m = re.findall(r"Opening '([^']+)'", tail)
    except Exception:
        m = None
    if m:
        try:
            last = str(m[-1]).strip()
            return _shorten(last)
        except Exception:
            return ""

    try:
        u = re.findall(r"(https?://\S+)", tail)
    except Exception:
        u = []
    if u:
        try:
            last = str(u[-1]).strip().rstrip("')\"]")
            return _shorten(last)
        except Exception:
            return ""

    try:
        f = re.findall(r"\b([\w.-]+\.(?:aac|m4a|m4s|ts|mp4))\b", tail, flags=re.IGNORECASE)
    except Exception:
        f = []
    if f:
        try:
            return _shorten(str(f[-1]))
        except Exception:
            return ""

    return ""


def _parse_pdt(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(s.strip().replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _parse_ffmeta_chapters(meta_path: Path) -> list[dict]:
    tracks: list[dict] = []
    if not meta_path.exists():
        return tracks
    try:
        lines = meta_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return tracks

    in_ch = False
    start_ms: Optional[int] = None
    title: str = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[CHAPTER]"):
            if in_ch and start_ms is not None:
                tracks.append({"offset_s": max(0.0, float(start_ms) / 1000.0), "title": title})
            in_ch = True
            start_ms = None
            title = ""
            continue
        if not in_ch:
            continue
        if line.startswith("START="):
            try:
                start_ms = int(line.split("=", 1)[1].strip())
            except Exception:
                start_ms = None
            continue
        if line.lower().startswith("title="):
            title = line.split("=", 1)[1].strip()
            continue

    if in_ch and start_ms is not None:
        tracks.append({"offset_s": max(0.0, float(start_ms) / 1000.0), "title": title})

    # Dedupe consecutive identical offsets.
    out: list[dict] = []
    last_off: Optional[float] = None
    for t in tracks:
        try:
            off = float(t.get("offset_s") or 0.0)
        except Exception:
            off = 0.0
        if last_off is not None and abs(off - last_off) < 1e-6:
            continue
        last_off = off
        out.append(t)
    return out


def _get_latest_playlist_pdt(url: str) -> Optional[datetime]:
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        tl = _parse_hls_timeline(r.text or "")
        if not tl:
            return None
        # SiriusXM HLS clients typically start near the end of the live window.
        return tl[-1][0]
    except Exception:
        return None


def _get_playlist_pdt_at_or_before(url: str, wall_dt: datetime) -> Optional[datetime]:
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        tl = _parse_hls_timeline(r.text or "")
        if not tl:
            return None

        # Choose the newest segment PDT that is not after the moment we observed
        # audio actually writing. This aligns "file time zero" with real mux start.
        best: Optional[datetime] = None
        for seg_dt, _dur, _cum in tl:
            if seg_dt <= wall_dt and (best is None or seg_dt > best):
                best = seg_dt
        return best or tl[-1][0]
    except Exception:
        return None


def _parse_hls_timeline(playlist_text: str) -> list[tuple[datetime, float, float]]:
    # Returns a list of (segment_start_dt, duration_s, cumulative_s)
    out: list[tuple[datetime, float, float]] = []
    cur_pdt: Optional[datetime] = None
    cum = 0.0
    for raw in (playlist_text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
            cur_pdt = _parse_pdt(line.split(":", 1)[1])
            continue
        if line.startswith("#EXTINF:"):
            if cur_pdt is None:
                continue
            try:
                dur_s = float(line.split(":", 1)[1].split(",", 1)[0].strip())
            except Exception:
                dur_s = 0.0
            out.append((cur_pdt, dur_s, cum))
            cum += max(0.0, dur_s)
            # Do NOT auto-advance PDT: SiriusXM playlists commonly include a PROGRAM-DATE-TIME
            # for every segment. Treat each PDT as authoritative for the *next* EXTINF.
            continue
    return out


def _hls_offset_for_timestamp(timeline: list[tuple[datetime, float, float]], ts: datetime) -> Optional[float]:
    if not timeline:
        return None
    # Find last segment whose PDT <= ts.
    best: Optional[tuple[datetime, float, float]] = None
    for seg_start, dur_s, cum_s in timeline:
        if seg_start <= ts:
            best = (seg_start, dur_s, cum_s)
        else:
            break
    if best is None:
        return None
    seg_start, dur_s, cum_s = best
    delta = (ts - seg_start).total_seconds()
    if delta < 0:
        return None
    # Clamp within the segment duration to avoid jumping past boundaries when timestamps jitter.
    if dur_s > 0:
        delta = min(delta, max(0.0, dur_s))
    return max(0.0, float(cum_s + delta))


def _image_to_ascii(path: Path, *, width: int = 26) -> str:
    # Simple grayscale ASCII render. Keeps dependencies minimal.
    # Avoid a leading space so the result never looks completely blank on dark images.
    ramp = ".:-=+*#%@"
    try:
        img = PILImage.open(path).convert("L")
        w, h = img.size
        if not w or not h:
            return ""

        aspect = h / w
        new_w = max(8, int(width))
        new_h = max(8, int(aspect * new_w * 0.55))
        img = img.resize((new_w, new_h))
        px = list(img.getdata())
        lines = []
        for y in range(new_h):
            row = px[y * new_w : (y + 1) * new_w]
            line = "".join(ramp[int(p / 255 * (len(ramp) - 1))] for p in row)
            lines.append(line)
        return "\n".join(lines)
    except Exception:
        return ""


def _parse_m3u_playlist(path: Path) -> list[Path]:
    try:
        base = path.parent
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    out: list[Path] = []
    for raw in lines:
        s = (raw or "").strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("http://") or s.startswith("https://"):
            continue
        try:
            p = Path(s)
            if not p.is_absolute():
                p = base / p
            p = p.expanduser()
            if p.exists() and p.is_file():
                out.append(p)
        except Exception:
            continue
    return out


def _read_local_tags(path: Path) -> tuple[str, str, str]:
    title = ""
    artist = ""
    album = ""
    try:
        from mutagen import File as MutagenFile

        mf = MutagenFile(str(path))
        if mf is None:
            raise RuntimeError("no tags")
        tags = getattr(mf, "tags", None)
        if tags is None:
            raise RuntimeError("no tags")

        def first(keys: list[str]) -> str:
            for k in keys:
                try:
                    v = tags.get(k)
                    if v is None:
                        continue
                    if isinstance(v, (list, tuple)) and v:
                        return str(v[0] or "").strip()
                    return str(v or "").strip()
                except Exception:
                    continue
            return ""

        title = first(["TIT2", "title", "\xa9nam"])
        artist = first(["TPE1", "artist", "\xa9ART"])
        album = first(["TALB", "album", "\xa9alb"])
    except Exception:
        title, artist, album = "", "", ""
    return title, artist, album


def _track_number_from_tags(path: Path) -> Optional[int]:
    try:
        from mutagen import File as MutagenFile

        mf = MutagenFile(str(path))
        if mf is None:
            return None
        tags = getattr(mf, "tags", None)
        if tags is None:
            return None

        # MP4/M4A often uses 'trkn' => [(track, total)]
        try:
            v = tags.get("trkn")
            if isinstance(v, (list, tuple)) and v:
                first = v[0]
                if isinstance(first, (list, tuple)) and first:
                    n = int(first[0])
                    if n > 0:
                        return n
        except Exception:
            pass

        # MP3 ID3 commonly uses 'TRCK' like '3/12'
        try:
            v2 = tags.get("TRCK")
            if isinstance(v2, (list, tuple)) and v2:
                s = str(v2[0] or "").strip()
                if s:
                    s = s.split("/", 1)[0]
                    n = int(s)
                    if n > 0:
                        return n
        except Exception:
            pass
    except Exception:
        return None
    return None


def _natural_key(s: str) -> list[object]:
    parts = re.split(r"(\d+)", s.lower())
    out: list[object] = []
    for p in parts:
        if not p:
            continue
        if p.isdigit():
            try:
                # Type-tag numeric parts so comparisons never mix int/str.
                out.append((0, int(p)))
            except Exception:
                out.append((1, str(p)))
        else:
            out.append((1, str(p)))
    return out


def _extract_embedded_cover_path(path: Path) -> Optional[Path]:
    try:
        from mutagen import File as MutagenFile
        from mutagen.id3 import ID3

        mf = MutagenFile(str(path))
        if mf is None:
            return None

        data: bytes = b""
        ext = ""
        try:
            if mf.mime and any("mp4" in m or "m4a" in m for m in mf.mime):
                covr = None
                try:
                    covr = mf.tags.get("covr") if getattr(mf, "tags", None) is not None else None
                except Exception:
                    covr = None
                if isinstance(covr, (list, tuple)) and covr:
                    data = bytes(covr[0])
                    ext = "jpg"
        except Exception:
            pass

        if not data:
            try:
                id3 = ID3(str(path))
                apics = list(id3.getall("APIC"))
                if apics:
                    data = bytes(apics[0].data or b"")
                    mt = (apics[0].mime or "").lower()
                    ext = "png" if "png" in mt else "jpg"
            except Exception:
                pass

        if not data:
            return None

        cache_dir = Path(user_cache_dir("satstash")) / "dvr_art"
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(data).hexdigest()[:16]
        out = cache_dir / f"{key}.{ext or 'jpg'}"
        if not out.exists() or out.stat().st_size == 0:
            out.write_bytes(data)
        return out
    except Exception:
        return None


def _image_to_rich_blocks(path: Path, *, width: int = 30) -> Text:
    img = PILImage.open(path).convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageEnhance.Color(img).enhance(1.10)

    w, h = img.size
    if not w or not h:
        return Text("")

    aspect = h / w
    out_rows = max(8, int(aspect * int(width) * 0.55))
    px_h = out_rows * 2
    px_w = max(8, int(width))
    img = img.resize((px_w, px_h), resample=PILImage.Resampling.LANCZOS)

    out = Text()
    pixels = img.load()
    for y in range(0, px_h, 2):
        for x in range(px_w):
            r1, g1, b1 = pixels[x, y]
            r2, g2, b2 = pixels[x, y + 1]
            style = Style(color=f"#{r1:02x}{g1:02x}{b1:02x}", bgcolor=f"#{r2:02x}{g2:02x}{b2:02x}")
            out.append("▀", style=style)
        if y + 2 < px_h:
            out.append("\n")
    return out


def _image_to_rich_braille_fit(path: Path, *, width: int, height: int, bg_rgb: tuple[int, int, int] = (11, 13, 12)) -> Text:
    # Render using Unicode Braille (2x4 dots per character cell). This provides
    # higher effective resolution than half-blocks (1x2).
    try:
        img_rgba = PILImage.open(path).convert("RGBA")
        # Apply enhancements to RGB only, then re-apply alpha before compositing.
        alpha = img_rgba.getchannel("A")
        rgb = img_rgba.convert("RGB")
        rgb = ImageEnhance.Contrast(rgb).enhance(1.10)
        rgb = ImageEnhance.Color(rgb).enhance(1.05)
        rgb.putalpha(alpha)
        bg = PILImage.new("RGBA", rgb.size, (int(bg_rgb[0]), int(bg_rgb[1]), int(bg_rgb[2]), 255))
        img = PILImage.alpha_composite(bg, rgb).convert("RGB")
    except Exception:
        img = PILImage.open(path).convert("RGB")
        img = ImageEnhance.Contrast(img).enhance(1.10)
        img = ImageEnhance.Color(img).enhance(1.05)

    w, h = img.size
    if not w or not h:
        return Text("")

    out_w = max(8, int(width))
    out_h = max(6, int(height))
    px_w = out_w * 2
    px_h = out_h * 4

    # Fit while preserving aspect ratio, then letterbox into the target grid.
    scale = min(px_w / w, px_h / h)
    new_w = max(2, int(w * scale))
    new_h = max(4, int(h * scale))
    new_w = (new_w // 2) * 2
    new_h = (new_h // 4) * 4
    img = img.resize((max(2, new_w), max(4, new_h)), resample=PILImage.Resampling.LANCZOS)

    canvas = PILImage.new("RGB", (px_w, px_h), (int(bg_rgb[0]), int(bg_rgb[1]), int(bg_rgb[2])))
    ox = max(0, (px_w - img.size[0]) // 2)
    oy = max(0, (px_h - img.size[1]) // 2)
    canvas.paste(img, (ox, oy))
    pixels = canvas.load()

    # Braille dot bit mapping.
    # positions within a 2x4 cell (x,y):
    # (0,0)=dot1, (0,1)=dot2, (0,2)=dot3, (0,3)=dot7
    # (1,0)=dot4, (1,1)=dot5, (1,2)=dot6, (1,3)=dot8
    dot_bits = {
        (0, 0): 0x01,
        (0, 1): 0x02,
        (0, 2): 0x04,
        (1, 0): 0x08,
        (1, 1): 0x10,
        (1, 2): 0x20,
        (0, 3): 0x40,
        (1, 3): 0x80,
    }

    def lum(rgb: tuple[int, int, int]) -> int:
        r, g, b = rgb
        return int(0.2126 * r + 0.7152 * g + 0.0722 * b)

    out = Text()
    for cy in range(0, px_h, 4):
        for cx in range(0, px_w, 2):
            cell: list[tuple[int, int, int]] = [
                pixels[cx + dx, cy + dy] for dy in range(4) for dx in range(2)
            ]
            lums = [lum(c) for c in cell]
            thr = sorted(lums)[len(lums) // 2]

            bits = 0
            on_colors: list[tuple[int, int, int]] = []
            off_colors: list[tuple[int, int, int]] = []
            for dy in range(4):
                for dx in range(2):
                    c = pixels[cx + dx, cy + dy]
                    if lum(c) <= thr:
                        bits |= dot_bits[(dx, dy)]
                        on_colors.append(c)
                    else:
                        off_colors.append(c)

            ch = chr(0x2800 + bits)

            def avg(cols: list[tuple[int, int, int]]) -> tuple[int, int, int]:
                if not cols:
                    return (0, 0, 0)
                rs = sum(c[0] for c in cols)
                gs = sum(c[1] for c in cols)
                bs = sum(c[2] for c in cols)
                n = len(cols)
                return (rs // n, gs // n, bs // n)

            fg = avg(on_colors)
            bg = avg(off_colors)
            style = Style(color=f"#{fg[0]:02x}{fg[1]:02x}{fg[2]:02x}", bgcolor=f"#{bg[0]:02x}{bg[1]:02x}{bg[2]:02x}")
            out.append(ch, style=style)
        if cy + 4 < px_h:
            out.append("\n")
    return out


def _image_to_rich_blocks_fit(path: Path, *, width: int, height: int, bg_rgb: tuple[int, int, int] = (11, 13, 12)) -> Text:
    try:
        img_rgba = PILImage.open(path).convert("RGBA")
        # Composite onto the app background so transparent PNGs don't render as black.
        # Match the general UI background color.
        alpha = img_rgba.getchannel("A")
        rgb = img_rgba.convert("RGB")
        rgb = ImageEnhance.Contrast(rgb).enhance(1.15)
        rgb = ImageEnhance.Color(rgb).enhance(1.10)
        rgb.putalpha(alpha)
        bg = PILImage.new("RGBA", rgb.size, (int(bg_rgb[0]), int(bg_rgb[1]), int(bg_rgb[2]), 255))
        img = PILImage.alpha_composite(bg, rgb).convert("RGB")
    except Exception:
        img = PILImage.open(path).convert("RGB")
        img = ImageEnhance.Contrast(img).enhance(1.15)
        img = ImageEnhance.Color(img).enhance(1.10)

    w, h = img.size
    if not w or not h:
        return Text("")

    box_w = max(8, int(width))
    box_h = max(8, int(height))
    px_w = box_w
    px_h = box_h * 2

    # Fit inside the box, preserving aspect ratio; letterbox to fill.
    scale = min(px_w / w, px_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = img.resize((new_w, new_h), resample=PILImage.Resampling.LANCZOS)

    canvas = PILImage.new("RGB", (px_w, px_h), (int(bg_rgb[0]), int(bg_rgb[1]), int(bg_rgb[2])))
    ox = max(0, (px_w - new_w) // 2)
    oy = max(0, (px_h - new_h) // 2)
    canvas.paste(resized, (ox, oy))

    out = Text()
    pixels = canvas.load()
    for y in range(0, px_h, 2):
        for x in range(px_w):
            r1, g1, b1 = pixels[x, y]
            r2, g2, b2 = pixels[x, y + 1]
            style = Style(color=f"#{r1:02x}{g1:02x}{b1:02x}", bgcolor=f"#{r2:02x}{g2:02x}{b2:02x}")
            out.append("▀", style=style)
        if y + 2 < px_h:
            out.append("\n")
    return out


def _wrap_text(text: str, *, width: int = 28) -> str:
    if not text:
        return ""
    out = []
    s = str(text)
    for i in range(0, len(s), width):
        out.append(s[i : i + width])
    return "\n".join(out)


def _normalize_art_url(url_or_path: str) -> str:
    s = (url_or_path or "").strip()
    if not s:
        return ""
    if s.startswith("http://") or s.startswith("https://"):
        return s
    rel = s.lstrip("/")
    if rel.startswith("entity-management/"):
        return "https://siriusxm-prd.imgix.net/" + rel
    return "https://siriusxm.imgix.net/" + rel


def _imgsrv_url_from_key(key: str, *, width: int = 300, height: int = 300) -> str:
    payload = {
        "key": key.lstrip("/"),
        "edits": [
            {"format": {"type": "jpeg"}},
            {"resize": {"height": int(height), "width": int(width)}},
        ],
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    token = base64.urlsafe_b64encode(raw).decode("ascii")
    return "https://imgsrv-sxm-prod-device.streaming.siriusxm.com/" + token


def _pick_image_url(root: dict) -> Optional[str]:
    try:
        tile = (root.get("tile") or {})
        a11 = (tile.get("aspect_1x1") or {})
        pref = (a11.get("preferredImage") or {}).get("url")
        if isinstance(pref, str) and pref:
            return pref
        default = (a11.get("defaultImage") or {}).get("url")
        if isinstance(default, str) and default:
            return default
        a169 = (tile.get("aspect_16x9") or {})
        pref2 = (a169.get("preferredImage") or {}).get("url")
        if isinstance(pref2, str) and pref2:
            return pref2
        default2 = (a169.get("defaultImage") or {}).get("url")
        if isinstance(default2, str) and default2:
            return default2
    except Exception:
        return None


def _parse_cue_tracks(cue_path: Path) -> list[dict]:
    tracks: list[dict] = []
    if not cue_path.exists():
        return tracks
    current: Optional[dict] = None

    def flush() -> None:
        nonlocal current
        if current:
            tracks.append(current)
        current = None

    try:
        for raw in cue_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line:
                continue
            up = line.upper()
            if up.startswith("TRACK ") and " AUDIO" in up:
                flush()
                parts = line.split()
                num = parts[1] if len(parts) > 1 else ""
                current = {"track": num, "title": "", "artist": "", "offset_s": 0.0}
                continue
            if current is None:
                continue
            if up.startswith("TITLE "):
                v = line.split(" ", 1)[1] if " " in line else ""
                current["title"] = v.strip().strip('"')
                continue
            if up.startswith("PERFORMER "):
                v = line.split(" ", 1)[1] if " " in line else ""
                current["artist"] = v.strip().strip('"')
                continue
            if up.startswith("INDEX 01 "):
                v = line.split(" ", 2)[2] if len(line.split()) >= 3 else ""
                try:
                    mm_s, ss_s, ff_s = v.strip().split(":")
                    mm = int(mm_s)
                    ss = int(ss_s)
                    ff = int(ff_s)
                    current["offset_s"] = max(0.0, float(mm * 60 + ss) + float(ff) / 75.0)
                except Exception:
                    current["offset_s"] = 0.0
                continue
        flush()
    except Exception:
        return []

    # De-dup by offset (cue may repeat INDEX lines)
    out: list[dict] = []
    seen = set()
    for t in tracks:
        key = (t.get("track"), int(round(float(t.get("offset_s") or 0.0) * 1000.0)))
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out



def _find_first_url(node: object) -> Optional[str]:
    try:
        if isinstance(node, str):
            s = node.strip()
            if s.startswith("http://") or s.startswith("https://"):
                return s
            if s.startswith("//"):
                return "https:" + s
            if "/" in s and not s.startswith("{") and not s.startswith("["):
                if any(ext in s.lower() for ext in (".jpg", ".jpeg", ".png", ".webp")):
                    key = s.lstrip("/")
                    return _imgsrv_url_from_key(key, width=300, height=300)
            return None
        if isinstance(node, dict):
            for k in (
                "url",
                "imageUrl",
                "imageURL",
                "uri",
                "imageKey",
                "key",
                "assetKey",
                "path",
            ):
                v = node.get(k)
                if isinstance(v, str):
                    u = _find_first_url(v)
                    if u:
                        return u
            for v in node.values():
                u = _find_first_url(v)
                if u:
                    return u
            return None
        if isinstance(node, list):
            for v in node:
                u = _find_first_url(v)
                if u:
                    return u
            return None
    except Exception:
        return None
    return None


def _extract_art_url_from_live_item(item: dict, *, channel_logo_url: Optional[str] = None) -> str:
    img_url = _pick_image_url(item.get("images") or {})
    if not img_url:
        img_url = _pick_image_url(item.get("artistImages") or {})
    if not img_url:
        img_url = _find_first_url(item.get("images") or {})
    if not img_url:
        img_url = _find_first_url(item.get("artistImages") or {})
    if not img_url:
        img_url = _find_first_url(item)
    if not img_url and channel_logo_url:
        img_url = channel_logo_url
    if isinstance(img_url, str) and img_url:
        return _normalize_art_url(img_url)
    return ""


def _fetch_image_bytes(url: str, *, timeout: float = 15.0) -> tuple[bytes, str]:
    if not url:
        return b"", ""
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "image/*,*/*;q=0.8",
    }
    try:
        r = requests.get(url, timeout=timeout, headers=headers)
        r.raise_for_status()
        ctype = (r.headers.get("content-type") or "").lower()
        return (r.content or b""), ctype
    except HTTPError as exc:
        try:
            resp = getattr(exc, "response", None)
            status = int(getattr(resp, "status_code", 0) or 0)
        except Exception:
            status = 0
        if status == 410:
            try:
                p = urlparse(url)
                host = (p.netloc or "").lower()
                path = (p.path or "").lstrip("/")
                if host.endswith("siriusxm.imgix.net") and path and any(ext in path.lower() for ext in (".jpg", ".jpeg", ".png", ".webp")):
                    alt = _imgsrv_url_from_key(path, width=300, height=300)
                    r2 = requests.get(alt, timeout=timeout, headers=headers)
                    r2.raise_for_status()
                    ctype2 = (r2.headers.get("content-type") or "").lower()
                    return (r2.content or b""), ctype2
            except Exception:
                pass
        raise


def _fetch_itunes_cover_bytes(*, artist: str, title: str, timeout: float = 12.0) -> tuple[bytes, str]:
    try:
        a = (artist or "").strip()
        t = (title or "").strip()
    except Exception:
        return b"", ""
    if not a or not t:
        return b"", ""

    def norm(s: str) -> str:
        try:
            s2 = (s or "").lower()
        except Exception:
            s2 = ""
        out = []
        for ch in s2:
            if ch.isalnum() or ch.isspace():
                out.append(ch)
        return " ".join("".join(out).split())

    qa = norm(a)
    qt = norm(t)
    if not qa or not qt:
        return b"", ""
    try:
        term = requests.utils.quote(f"{a} {t}")
        url = f"https://itunes.apple.com/search?term={term}&media=music&entity=song&limit=1"
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.json() if r.content else {}
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list) or not results:
            return b"", ""
        res = results[0] if isinstance(results[0], dict) else {}

        # Reject low-confidence matches (better no art than wrong art).
        try:
            ra = norm(str(res.get("artistName") or ""))
            rt = norm(str(res.get("trackName") or ""))
        except Exception:
            ra = ""
            rt = ""
        if not ra or not rt:
            return b"", ""
        # Require substantial overlap.
        if not (qa in ra or ra in qa):
            return b"", ""
        if not (qt in rt or rt in qt):
            return b"", ""

        art_url = (res.get("artworkUrl100") or res.get("artworkUrl60") or "").strip()
        if not art_url:
            return b"", ""
        art_url = art_url.replace("100x100bb", "600x600bb")
        return _fetch_image_bytes(art_url, timeout=timeout)
    except Exception:
        return b"", ""


def _sniff_image_type(data: bytes, ctype: str) -> str:
    ct = (ctype or "").lower()
    if "png" in ct:
        return "png"
    if "jpeg" in ct or "jpg" in ct:
        return "jpeg"
    if "webp" in ct:
        return "webp"
    try:
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        if len(data) >= 3 and data[0:3] == b"\xff\xd8\xff":
            return "jpeg"
        # RIFF....WEBP
        if len(data) >= 12 and data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "webp"
    except Exception:
        pass
    return ""


def _coerce_cover_image_for_mp4(cover_bytes: bytes, cover_content_type: str) -> tuple[bytes, str]:
    if not cover_bytes:
        return b"", ""
    kind = _sniff_image_type(cover_bytes, cover_content_type)
    if kind in ("jpeg", "png"):
        return cover_bytes, (cover_content_type or ("image/" + kind))

    # VLC often doesn't display WEBP (or unknown) inside MP4 'covr'. Convert to PNG.
    try:
        if shutil.which("ffmpeg"):
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                "pipe:0",
                "-frames:v",
                "1",
                "-f",
                "image2",
                "-vcodec",
                "png",
                "pipe:1",
            ]
            p = subprocess.run(cmd, input=cover_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p.returncode == 0 and p.stdout:
                return p.stdout, "image/png"
    except Exception:
        pass

    # Fallback: return original bytes (may be ignored by some players).
    return cover_bytes, (cover_content_type or "")


def _is_bumper_like(*, title: str, artist: str) -> bool:
    try:
        title_l = (title or "").strip().lower()
        artist_l = (artist or "").strip().lower()
    except Exception:
        return False
    try:
        if title_l.startswith("#") or title_l.startswith("@"): 
            return True
        if artist_l.startswith("#") or artist_l.startswith("@"): 
            return True
        if title_l and any(k in title_l for k in ["fb.com", "twitter", "instagram", "sirius", "sxm"]):
            return True
        if artist_l and any(k in artist_l for k in ["sirius", "sxm"]):
            return True
    except Exception:
        return False
    return False


def _looks_like_show_metadata(*, title: str, artist: str) -> bool:
    try:
        t = (title or "").strip().lower()
        a = (artist or "").strip().lower()
    except Exception:
        return False
    s = f"{a} {t}".strip()
    if not s:
        return False
    # Dates/locations/hashtags/handles tend to produce terrible iTunes matches.
    if any(ch in s for ch in ("#", "@")):
        return True
    if "," in s and any(st in s for st in (" il", " ny", " ca", " tx", " fl", " nj", " ma", " pa", " va", " nc", " sc", " ga")):
        return True
    # Many live show titles start with digits or contain year-like tokens.
    try:
        if any(tok.isdigit() and len(tok) >= 2 for tok in s.split()):
            return True
    except Exception:
        pass
    return False


def _safe_filename_component(s: str) -> str:
    try:
        s2 = (s or "").strip()
    except Exception:
        s2 = ""
    if not s2:
        return ""
    return "".join(c if c.isalnum() or c in " _-." else "_" for c in s2).strip()


def _safe_filename(s: str) -> str:
    """Sanitize a full filename token (defensive against path separators and odd unicode)."""
    try:
        out = _safe_filename_component(s)
    except Exception:
        out = ""
    # Prevent accidental directory traversal / separators on any platform.
    try:
        out = out.replace("/", "_").replace("\\", "_")
    except Exception:
        pass
    try:
        while "__" in out:
            out = out.replace("__", "_")
    except Exception:
        pass
    return (out or "").strip(" ._")


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suf = path.suffix
    for i in range(1, 1000):
        p2 = path.with_name(f"{stem}_{i:02d}{suf}")
        if not p2.exists():
            return p2
    return path.with_name(f"{stem}_{int(time.time())}{suf}")


def _output_category_dir(settings: "Settings", category: str) -> Path:
    # Prefer output_dir; fall back to recordings_dir for older configs.
    base_raw = getattr(settings, "output_dir", None) or ""
    if not str(base_raw).strip():
        base_raw = getattr(settings, "recordings_dir", "~/Music/SiriusXM")
    base = Path(str(base_raw)).expanduser()
    return base / str(category)


def _ffmpeg_trim_m4a(*, src: Path, dst: Path, start_s: float, duration_s: float, mode: str = "copy") -> None:
    if not src.exists():
        raise RuntimeError("trim input missing")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found")
    start_s = max(0.0, float(start_s))
    duration_s = max(0.0, float(duration_s))
    # Trimming modes:
    # - mode='copy' keeps audio bit-perfect; must use input seeking (-ss before -i)
    #   or ffmpeg may produce empty/invalid outputs with some container+codec combos.
    # - mode='aac' re-encodes for tighter, more accurate cuts (uses output seeking).
    codec_args: list[str]
    if mode == "aac":
        codec_args = ["-c:a", "aac"]
    else:
        # Copy without re-encode.
        codec_args = ["-c:a", "copy"]
        # Only apply ADTS->ASC when the input is ADTS AAC.
        try:
            suf = (src.suffix or "").lower()
        except Exception:
            suf = ""
        if suf in {".aac", ".adts"}:
            codec_args += ["-bsf:a", "aac_adtstoasc"]

    # Ensure the muxer is MP4/M4A when writing m4a/mp4.
    out_fmt: list[str] = []
    try:
        dsuf = (dst.suffix or "").lower()
    except Exception:
        dsuf = ""
    if dsuf in {".m4a", ".mp4"}:
        out_fmt = ["-f", "mp4"]

    # Build command with correct seek placement.
    if mode == "aac":
        # Output seeking for accuracy when decoding anyway.
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(src),
            "-ss",
            f"{start_s:.3f}",
            "-t",
            f"{duration_s:.3f}",
            "-vn",
            *codec_args,
            "-movflags",
            "+faststart",
            *out_fmt,
            str(dst),
        ]
    else:
        # Input seeking for copy mode.
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-ss",
            f"{start_s:.3f}",
            "-i",
            str(src),
            "-t",
            f"{duration_s:.3f}",
            "-vn",
            *codec_args,
            "-movflags",
            "+faststart",
            *out_fmt,
            str(dst),
        ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()[-10:]
        raise RuntimeError("ffmpeg trim failed: " + " ".join([*cmd]) + " :: " + " | ".join(tail))


def _ffmpeg_faststart_copy(*, src: Path, dst: Path) -> None:
    if not src.exists():
        raise RuntimeError("faststart input missing")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(src),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()[-10:]
        raise RuntimeError("ffmpeg faststart failed: " + " | ".join(tail))


def _ffmpeg_mux_chapters_from_ffmeta(*, src: Path, ffmeta: Path, dst: Path) -> None:
    if not src.exists():
        raise RuntimeError("chapters input missing")
    if not ffmeta.exists():
        raise RuntimeError("ffmetadata missing")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(src),
        "-f",
        "ffmetadata",
        "-i",
        str(ffmeta),
        "-map",
        "0",
        "-map_metadata",
        "0",
        "-map_chapters",
        "1",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(dst),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()[-10:]
        raise RuntimeError("ffmpeg chapter mux failed: " + " | ".join(tail))


def _ffmpeg_extract_adts_aac(*, src: Path, dst: Path) -> None:
    if not src.exists():
        raise RuntimeError("aac extract input missing")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-c:a",
        "copy",
        "-f",
        "adts",
        str(dst),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()[-10:]
        raise RuntimeError("ffmpeg aac extract failed: " + " | ".join(tail))


def _ffmpeg_trim_adts_aac(*, src: Path, dst: Path, start_s: float, duration_s: float) -> None:
    if not src.exists():
        raise RuntimeError("trim input missing")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found")
    start_s = max(0.0, float(start_s))
    duration_s = max(0.0, float(duration_s))
    # For bit-perfect trims, use input seeking.
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(src),
        "-t",
        f"{duration_s:.3f}",
        "-vn",
        "-c:a",
        "copy",
        "-f",
        "adts",
        str(dst),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()[-10:]
        raise RuntimeError("ffmpeg aac trim failed: " + " ".join([*cmd]) + " :: " + " | ".join(tail))


def _ffmpeg_remux_aac_to_m4a(*, src: Path, dst: Path) -> None:
    if not src.exists():
        raise RuntimeError("remux input missing")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-c:a",
        "copy",
        "-bsf:a",
        "aac_adtstoasc",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(dst),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()[-10:]
        raise RuntimeError("ffmpeg aac remux failed: " + " ".join([*cmd]) + " :: " + " | ".join(tail))


def _tag_m4a(
    *,
    path: Path,
    title: str,
    artist: str,
    album: str,
    cover_bytes: bytes,
    cover_content_type: str,
) -> None:
    if not path.exists() or path.suffix.lower() != ".m4a":
        return

    audio = MP4(str(path))
    if title:
        audio["\xa9nam"] = [title]
    if artist:
        audio["\xa9ART"] = [artist]
        audio["aART"] = [artist]
    if album:
        audio["\xa9alb"] = [album]
    if cover_bytes:
        cb, ct = _coerce_cover_image_for_mp4(cover_bytes, cover_content_type)
        if cb:
            fmt = MP4Cover.FORMAT_JPEG
            if "png" in (ct or ""):
                fmt = MP4Cover.FORMAT_PNG
            audio["covr"] = [MP4Cover(cb, imageformat=fmt)]
    audio.save()


def _cue_time(seconds: float) -> str:
    try:
        s = max(0.0, float(seconds))
    except Exception:
        s = 0.0
    total_frames = int(round(s * 75.0))
    mm, rem = divmod(total_frames, 75 * 60)
    ss, ff = divmod(rem, 75)
    return f"{mm:02d}:{ss:02d}:{ff:02d}"


def _write_cue(*, audio_path: Path, tracks: list[dict], duration_s: Optional[float] = None) -> Optional[Path]:
    if not audio_path.exists():
        return None
    cue_path = audio_path.with_suffix(audio_path.suffix + ".cue")
    try:
        lines: list[str] = []
        lines.append(f'FILE "{audio_path.name}" MP4')
        for i, t in enumerate(tracks, start=1):
            title = str(t.get("title") or "")
            artist = str(t.get("artist") or "")
            title_cue = title.replace('"', "'")
            artist_cue = artist.replace('"', "'")
            try:
                offset_s = float(t.get("offset_s") or 0.0)
            except Exception:
                offset_s = 0.0
            if duration_s is not None:
                try:
                    offset_s = max(0.0, min(float(offset_s), max(0.0, float(duration_s) - 0.001)))
                except Exception:
                    offset_s = max(0.0, float(offset_s))
            lines.append(f"  TRACK {i:02d} AUDIO")
            if title:
                lines.append(f'    TITLE "{title_cue}"')
            if artist:
                lines.append(f'    PERFORMER "{artist_cue}"')
            lines.append(f"    INDEX 01 {_cue_time(offset_s)}")
        cue_path.write_text("\n".join(lines) + "\n", encoding="utf-8", errors="replace")
        return cue_path
    except Exception:
        return None


def _write_ffmetadata(*, audio_path: Path, tracks: list[dict], duration_s: Optional[float] = None) -> Optional[Path]:
    if not audio_path.exists():
        return None
    meta_path = audio_path.with_suffix(audio_path.suffix + ".ffmeta")
    try:
        lines: list[str] = [";FFMETADATA1"]

        dur_ms: Optional[int] = None
        if duration_s is not None:
            try:
                dur_ms = max(0, int(round(float(duration_s) * 1000.0)))
            except Exception:
                dur_ms = None

        starts_ms: list[int] = []
        titles: list[str] = []
        for t in tracks:
            try:
                starts_ms.append(max(0, int(round(float(t.get("offset_s") or 0.0) * 1000.0))))
            except Exception:
                starts_ms.append(0)
            titles.append(str(t.get("display") or t.get("title") or ""))

        if dur_ms is not None and dur_ms > 0:
            try:
                max_start = max(0, dur_ms - 1)
                starts_ms = [min(max(0, int(s)), max_start) for s in starts_ms]
            except Exception:
                pass

        for i, start_ms in enumerate(starts_ms):
            if i + 1 < len(starts_ms):
                end_ms = max(start_ms, starts_ms[i + 1] - 1)
            else:
                if dur_ms is not None and dur_ms > 0:
                    end_ms = max(start_ms, dur_ms - 1)
                else:
                    end_ms = start_ms + (12 * 60 * 60 * 1000)
            if dur_ms is not None and dur_ms > 0:
                try:
                    end_ms = max(start_ms, min(int(end_ms), dur_ms - 1))
                except Exception:
                    end_ms = max(start_ms, int(end_ms))
            title = titles[i].replace("\n", " ").strip()
            lines.append("[CHAPTER]")
            lines.append("TIMEBASE=1/1000")
            lines.append(f"START={start_ms}")
            lines.append(f"END={end_ms}")
            if title:
                lines.append(f"title={title}")

        meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8", errors="replace")
        return meta_path
    except Exception:
        return None


class LoginScreen(Screen[bool]):
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container():
            yield Static("Login", id="title")
            yield Input(placeholder="Email/Username", id="username")
            yield Input(placeholder="Password", password=True, id="password")
            with Horizontal():
                yield Button("Login", id="do_login", variant="primary")
                yield Button("Cancel", id="cancel")
        yield Footer()

    def on_mount(self) -> None:
        # Pre-fill from Settings (optional; stored in plaintext in config.json).
        try:
            settings = getattr(self.app, "settings", None) or load_settings()
        except Exception:
            settings = None
        try:
            u = (getattr(settings, "auth_username", "") or "").strip() if settings else ""
            p = getattr(settings, "auth_password", "") or "" if settings else ""
            if u:
                self.query_one("#username", Input).value = u
            if p:
                self.query_one("#password", Input).value = p
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "cancel":
            self.dismiss(False)
            return
        if event.button.id != "do_login":
            return

        username = (self.query_one("#username", Input).value or "").strip()
        password = self.query_one("#password", Input).value or ""
        if not username or not password:
            self.app.notify("Username and password required")
            return

        try:
            result = SiriusXMDirectAuth().authenticate(username, password)
            save_session(result.bearer_token, result.cookies, lifetime_hours=12)
            self.dismiss(True)
        except Exception as exc:
            self.app.notify(f"Login failed: {exc}")


class LiveSelectScreen(Screen[None]):
    DEFAULT_CSS = """
    LiveSelectScreen {
        layout: vertical;
    }
    LiveSelectScreen Container {
        layout: vertical;
        height: 1fr;
    }
    LiveSelectScreen #channels {
        height: 1fr;
    }
    LiveSelectScreen #actions {
        height: auto;
    }
    """

    def __init__(self, channels: List[Channel]):
        super().__init__()
        self.channels = sorted(channels, key=lambda c: (c.number is None, c.number or 9999, c.name))
        self.filtered: List[Channel] = self.channels
        self._channels_by_id: dict[str, Channel] = {}
        self._selected_id: Optional[str] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container():
            yield Static("Listen Live", id="title")
            yield Input(placeholder="Search channel name/number", id="search")
            yield DataTable(id="channels")
            with Horizontal(id="actions"):
                yield Button("Play Selected", id="play", variant="primary")
                yield Button("Refresh", id="refresh")
                yield Button("Back", id="back")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#channels", DataTable)
        table.cursor_type = "row"
        table.add_columns("Ch", "Name")
        self._render_results()
        self.query_one("#search", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "search":
            return
        # Enter in the search box plays the first match.
        if not self.filtered:
            self.app.notify("No matches")
            return
        self._selected_id = self.filtered[0].id
        self.play_selected()
        self.query_one("#search", Input).focus()

    def _render_results(self) -> None:
        table = self.query_one("#channels", DataTable)
        table.clear()
        self._channels_by_id = {}
        self._selected_id = None

        if not self.filtered:
            table.add_row("-", "No matches")
            return

        for ch in self.filtered:
            num = "-" if ch.number is None else str(ch.number)
            table.add_row(num, ch.name, key=ch.id)
            self._channels_by_id[ch.id] = ch

        if self.filtered:
            self._selected_id = self.filtered[0].id

        table.cursor_coordinate = (0, 0)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "search":
            return
        q = (event.value or "").strip().lower()
        if not q:
            self.filtered = self.channels
        else:
            q_is_digit = q.isdigit()
            self.filtered = [
                ch
                for ch in self.channels
                if q in ch.name.lower()
                or (
                    q_is_digit
                    and ch.number is not None
                    and (q == str(ch.number) or str(ch.number).startswith(q))
                )
            ]
        self._render_results()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        # Prevent the App-level button handler from treating these as main-menu buttons.
        event.stop()
        if event.button.id == "back":
            self.go_back()
            return
        if event.button.id == "refresh":
            self.refresh_channels()
            return
        if event.button.id != "play":
            return

        self.play_selected()

    def on_key(self, event) -> None:
        if getattr(event, "key", None) == "r":
            event.stop()
            self.refresh_channels()
            return

        # Support Enter-to-play when the table has focus.
        if getattr(event, "key", None) != "enter":
            return
        table = self.query_one("#channels", DataTable)
        if not table.has_focus:
            return
        event.stop()

        # Use the highlighted row (cursor) rather than only relying on _selected_id.
        # _selected_id is updated on RowSelected, but cursor movement alone doesn't select.
        try:
            row_index = int(getattr(table, "cursor_row"))
        except Exception:
            row_index = 0
        try:
            ordered = getattr(table, "ordered_rows", [])
            row = ordered[row_index] if ordered and 0 <= row_index < len(ordered) else None
            row_key = getattr(row, "key", None)
            key_val = getattr(row_key, "value", None)
            if isinstance(key_val, str) and key_val:
                self._selected_id = key_val
        except Exception:
            pass

        ch = self._selected_channel()
        if not ch:
            return
        try:
            self._start_playback_worker(ch)
        except Exception as exc:
            self.app.notify(f"Playback failed: {exc}")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        key = getattr(event.row_key, "value", None)
        if isinstance(key, str) and key:
            self._selected_id = key

    def refresh_channels(self) -> None:
        self.app.notify("Refreshing channel list...")
        self.app.run_worker(self._refresh_worker, thread=True, exclusive=True)

    def _refresh_worker(self) -> None:
        clear_channels_cache()
        try:
            channels = self.app.fetch_channels(force_refresh=True)
        except (NotLoggedInError, SessionExpiredError):
            def ui():
                self.app.notify("Session expired. Please login again.")
                self.app._prompt_login_and_retry(after_login=lambda: self.refresh_channels())

            self.app.call_from_thread(ui)
            return
        # Update state back on UI thread
        def apply():
            self.channels = sorted(channels, key=lambda c: (c.number is None, c.number or 9999, c.name))
            self.filtered = self.channels
            self._render_results()
            self.query_one("#search", Input).value = ""
            self.query_one("#search", Input).focus()
            self.app.notify(f"Loaded {len(channels)} channels")
        self.app.call_from_thread(apply)

    def _start_playback_worker(self, ch: Channel) -> None:
        label = f"{ch.number if ch.number is not None else ''} {ch.name}".strip()
        self.app.notify(f"Starting playback: {label}")

        def work():
            try:
                self.app.start_live(channel=ch)
            except HTTPError as exc:
                def ui():
                    self.app.notify(f"Playback failed: {exc}")

                self.app.call_from_thread(ui)
            except (NotLoggedInError, SessionExpiredError):
                def ui():
                    self.app.notify("Session expired. Please login again.")

                    def after_login() -> None:
                        try:
                            self.app.start_live(channel=ch, push_screen=False)
                        except Exception as exc2:
                            self.app.call_from_thread(lambda: self.app.notify(f"Playback failed: {exc2}"))

                    self.app._prompt_login_and_retry(after_login=after_login)

                self.app.call_from_thread(ui)
            except Exception as exc:

                def ui():
                    self.app.notify(f"Playback failed: {exc}")

                self.app.call_from_thread(ui)

        self.app.run_worker(work, thread=True, exclusive=True)

    def play_selected(self) -> None:
        ch = self._selected_channel()
        if not ch:
            self.app.notify("No channel selected")
            return
        try:
            self._start_playback_worker(ch)
        except Exception as exc:
            self.app.notify(f"Playback failed: {exc}")

    def go_back(self) -> None:
        self.app.pop_screen()

    def _selected_channel(self) -> Optional[Channel]:
        if self._selected_id and self._selected_id in self._channels_by_id:
            return self._channels_by_id[self._selected_id]
        return self.filtered[0] if self.filtered else None


class LiveSelectPane(Widget):
    def __init__(self, channels: List[Channel]):
        super().__init__()
        self.channels = sorted(channels, key=lambda c: (c.number is None, c.number or 9999, c.name))
        self.filtered: List[Channel] = self.channels
        self._channels_by_id: dict[str, Channel] = {}
        self._selected_id: Optional[str] = None

    def compose(self) -> ComposeResult:
        with Container(id="live_pane"):
            yield Static("Listen Live", id="live_title")
            yield Input(placeholder="Search channel name/number", id="live_search")
            yield DataTable(id="live_channels")
            with Horizontal(id="live_actions"):
                yield Button("Play", id="live_play", variant="primary")
                yield Button("Refresh", id="live_refresh")
                yield Button("Close", id="live_close")

    def on_mount(self) -> None:
        table = self.query_one("#live_channels", DataTable)
        table.cursor_type = "row"
        table.add_columns("Ch", "Name")
        self._render_results()
        self.query_one("#live_search", Input).focus()

    def _selected_channel(self) -> Optional[Channel]:
        if self._selected_id and self._selected_id in self._channels_by_id:
            return self._channels_by_id[self._selected_id]
        return self.filtered[0] if self.filtered else None

    def _render_results(self) -> None:
        table = self.query_one("#live_channels", DataTable)
        table.clear()
        self._channels_by_id = {}
        self._selected_id = None

        if not self.filtered:
            table.add_row("-", "No matches")
            return

        for ch in self.filtered:
            num = "-" if ch.number is None else str(ch.number)
            table.add_row(num, ch.name, key=ch.id)
            self._channels_by_id[ch.id] = ch

        if self.filtered:
            self._selected_id = self.filtered[0].id

        table.cursor_coordinate = (0, 0)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "live_search":
            return
        q = (event.value or "").strip().lower()
        if not q:
            self.filtered = self.channels
        else:
            q_is_digit = q.isdigit()
            self.filtered = [
                ch
                for ch in self.channels
                if q in ch.name.lower()
                or (
                    q_is_digit
                    and ch.number is not None
                    and (q == str(ch.number) or str(ch.number).startswith(q))
                )
            ]
        self._render_results()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "live_search":
            return
        if not self.filtered:
            try:
                self.app.notify("No matches")
            except Exception:
                pass
            return
        self._selected_id = self.filtered[0].id
        self._play_selected()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        key = getattr(event.row_key, "value", None)
        if isinstance(key, str) and key:
            self._selected_id = key

    def on_key(self, event) -> None:
        # Support Enter-to-play when the table has focus.
        if getattr(event, "key", None) != "enter":
            return
        table = self.query_one("#live_channels", DataTable)
        if not table.has_focus:
            return
        event.stop()
        try:
            row_index = int(getattr(table, "cursor_row"))
        except Exception:
            row_index = 0
        try:
            ordered = getattr(table, "ordered_rows", [])
            row = ordered[row_index] if ordered and 0 <= row_index < len(ordered) else None
            row_key = getattr(row, "key", None)
            key_val = getattr(row_key, "value", None)
            if isinstance(key_val, str) and key_val:
                self._selected_id = key_val
        except Exception:
            pass
        self._play_selected()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        bid = event.button.id
        if bid == "live_close":
            try:
                getattr(self.app, "_clear_right_pane", lambda: None)()
            except Exception:
                pass
            return
        if bid == "live_refresh":
            self._refresh_channels()
            return
        if bid == "live_play":
            self._play_selected()
            return

    def _refresh_channels(self) -> None:
        try:
            self.app.notify("Refreshing channel list...")
        except Exception:
            pass

        def work() -> None:
            clear_channels_cache()
            try:
                channels = self.app.fetch_channels(force_refresh=True)
            except (NotLoggedInError, SessionExpiredError):
                def ui():
                    self.app.notify("Session expired. Please login again.")
                    self.app._prompt_login_and_retry(after_login=lambda: self._refresh_channels())
                self.app.call_from_thread(ui)
                return
            except Exception as exc:
                self.app.call_from_thread(lambda: self.app.notify(f"Refresh failed: {exc}"))
                return

            def apply() -> None:
                self.channels = sorted(channels, key=lambda c: (c.number is None, c.number or 9999, c.name))
                self.filtered = self.channels
                self._render_results()
                try:
                    self.query_one("#live_search", Input).value = ""
                    self.query_one("#live_search", Input).focus()
                except Exception:
                    pass
                self.app.notify(f"Loaded {len(channels)} channels")

            self.app.call_from_thread(apply)

        self.app.run_worker(work, thread=True, exclusive=True)

    def _play_selected(self) -> None:
        ch = self._selected_channel()
        if not ch:
            try:
                self.app.notify("No channel selected")
            except Exception:
                pass
            return

        label = f"{ch.number if ch.number is not None else ''} {ch.name}".strip()
        try:
            self.app.notify(f"Starting playback: {label}")
        except Exception:
            pass

        def work() -> None:
            try:
                self.app.start_live(channel=ch, push_screen=False)
                def ui_idle() -> None:
                    try:
                        getattr(self.app, "_show_right_idle", lambda: None)()
                    except Exception:
                        pass
                try:
                    self.app.call_from_thread(ui_idle)
                except Exception:
                    pass
            except (NotLoggedInError, SessionExpiredError):
                def ui():
                    self.app.notify("Session expired. Please login again.")

                    def after_login() -> None:
                        try:
                            self.app.start_live(channel=ch, push_screen=False)
                        except Exception as exc2:
                            self.app.call_from_thread(lambda: self.app.notify(f"Playback failed: {exc2}"))

                    self.app._prompt_login_and_retry(after_login=after_login)

                self.app.call_from_thread(ui)
            except Exception as exc:
                self.app.call_from_thread(lambda: self.app.notify(f"Playback failed: {exc}"))

        self.app.run_worker(work, thread=True, exclusive=True)


class VodPane(Container):
    DEFAULT_CSS = """
    VodPane #vod_meta {
        height: 1fr;
        border: solid #16783a;
        padding: 1 2;
    }

    VodPane #vod_progress {
        height: auto;
        padding: 0 2;
    }
    """

    def __init__(self):
        super().__init__()
        self._variants: list[dict] = []
        self._selected_url: Optional[str] = None
        self._handle: Optional[FfmpegRecordHandle] = None
        self._last_inspect_ctx: dict = {}
        self._progress_ctx: dict = {}
        self._progress_timer = None

    def render(self):
        try:
            from rich.text import Text

            return Text("")
        except Exception:
            return ""

    def compose(self) -> ComposeResult:
        with Container(id="vod_pane"):
            yield Static("VOD", id="vod_title")
            yield Input(placeholder="Paste .m3u8 URL (master or media)", id="vod_url")
            with Horizontal(id="vod_actions"):
                yield Button("Inspect", id="vod_inspect", variant="primary")
                yield Button("Download", id="vod_download")
                yield Button("Download Split", id="vod_download_split")
                yield Button("Stop", id="vod_stop")
                yield Button("Close", id="vod_close")
            yield Static("", id="vod_status")
            yield Static("", id="vod_progress")
            yield Static("", id="vod_meta")
            tbl = ActivatableDataTable(id="vod_table")
            try:
                tbl._activate_callback = lambda: self._select_highlighted()
            except Exception:
                pass
            yield tbl

    def on_mount(self) -> None:
        try:
            tbl = self.query_one("#vod_table", DataTable)
            tbl.cursor_type = "row"
            tbl.add_columns("Type", "Quality", "Bandwidth", "Codec")
        except Exception:
            pass
        try:
            self.query_one("#vod_url", Input).focus()
        except Exception:
            try:
                self.query_one("#vod_table", DataTable).focus()
            except Exception:
                pass
        try:
            self._refresh_buttons()
        except Exception:
            pass

    def _refresh_buttons(self) -> None:
        active = self._handle is not None
        try:
            self.query_one("#vod_stop", Button).disabled = not active
        except Exception:
            pass
        try:
            self.query_one("#vod_download", Button).disabled = active or (not bool(self._selected_url))
        except Exception:
            pass
        try:
            self.query_one("#vod_download_split", Button).disabled = active or (not bool(self._selected_url))
        except Exception:
            pass

    def _set_status(self, msg: str) -> None:
        try:
            self.query_one("#vod_status", Static).update(str(msg))
        except Exception:
            pass

    def _set_progress(self, msg: str) -> None:
        try:
            self.query_one("#vod_progress", Static).update(str(msg or ""))
        except Exception:
            pass

    def _set_meta(self, msg: str) -> None:
        try:
            self.query_one("#vod_meta", Static).update(str(msg or ""))
        except Exception:
            pass

    def _fmt_inspect_meta(self, ctx: dict) -> str:
        md = ctx.get("tunesource_metadata") if isinstance(ctx, dict) else None
        if not isinstance(md, dict):
            return ""

        block = None
        if isinstance(md.get("aod"), dict):
            block = md.get("aod")
        elif isinstance(md.get("vod"), dict):
            block = md.get("vod")
        if not isinstance(block, dict):
            return ""

        ep = block.get("episode") if isinstance(block.get("episode"), dict) else {}
        items = block.get("items") if isinstance(block.get("items"), list) else []
        first_item = items[0] if items and isinstance(items[0], dict) else {}

        show = str(block.get("channelName") or ep.get("showName") or "").strip()
        title = str(ep.get("name") or "").strip()
        item_title = str(first_item.get("name") or "").strip()
        artist = str(first_item.get("artistName") or ep.get("artistName") or "").strip()
        dt_raw = str(ep.get("originalAirTimestamp") or ep.get("startTimestamp") or "").strip()
        dur_ms = ep.get("duration")
        try:
            if dur_ms is None:
                dur_ms = first_item.get("duration")
        except Exception:
            pass

        ymd = ""
        if dt_raw:
            try:
                from datetime import datetime

                dd = datetime.fromisoformat(dt_raw.replace("Z", "+00:00"))
                ymd = dd.date().isoformat()
            except Exception:
                ymd = ""

        dur_s = None
        try:
            if dur_ms is not None:
                dur_s = int(float(dur_ms) / 1000.0)
        except Exception:
            dur_s = None

        dur_str = ""
        if dur_s is not None and dur_s >= 0:
            h, rem = divmod(dur_s, 3600)
            m, s = divmod(rem, 60)
            dur_str = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

        lines: list[str] = []
        if show:
            lines.append(f"Show: {show}")
        if artist:
            lines.append(f"Artist: {artist}")
        if title:
            lines.append(f"Episode: {title}")
        if item_title and item_title != title:
            lines.append(f"Track: {item_title}")
        if ymd:
            lines.append(f"Date: {ymd}")
        if dur_str:
            lines.append(f"Duration: {dur_str}")
        if not lines:
            return ""
        return "\n".join(lines)

    def _start_progress_timer(self) -> None:
        try:
            if self._progress_timer is not None:
                self._progress_timer.stop()
        except Exception:
            pass

        def tick() -> None:
            ctx = dict(getattr(self, "_progress_ctx", {}) or {})
            tmp_path = ctx.get("tmp_path")
            started = ctx.get("started")
            log_path = ctx.get("log_path")
            if not tmp_path or not started:
                return
            try:
                from pathlib import Path
                import time

                def _tail_lines(path: str, *, n: int = 4) -> list[str]:
                    try:
                        lp = str(path or "")
                        if not lp:
                            return []
                        p2 = Path(lp)
                        if not p2.exists():
                            return []
                        # Read only the tail of the file to avoid expensive reads of huge logs.
                        with open(p2, "rb") as f:
                            try:
                                f.seek(0, 2)
                                size2 = f.tell()
                                f.seek(max(0, size2 - 16_384), 0)
                            except Exception:
                                try:
                                    f.seek(0)
                                except Exception:
                                    return []
                            raw = f.read() or b""
                        txt = raw.decode("utf-8", errors="replace")
                        txt = txt.replace("\r", "\n")
                        lines2 = [ln.strip() for ln in txt.splitlines() if ln.strip()]
                        if not lines2:
                            return []
                        out2 = lines2[-n:]
                        # Keep UI tidy.
                        trimmed = []
                        for ln in out2:
                            if len(ln) > 180:
                                trimmed.append(ln[:177] + "...")
                            else:
                                trimmed.append(ln)
                        return trimmed
                    except Exception:
                        return []

                p = Path(str(tmp_path))
                if not p.exists():
                    return
                sz = p.stat().st_size
                elapsed = max(0.001, float(time.time() - float(started)))
                mb = sz / (1024.0 * 1024.0)
                rate = mb / elapsed
                m, s = divmod(int(elapsed), 60)
                h, m = divmod(m, 60)
                t = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

                if sz < 1024:
                    size_str = f"{sz} B"
                elif sz < 1024 * 1024:
                    size_str = f"{(sz / 1024.0):.1f} KB"
                else:
                    size_str = f"{mb:.1f} MB"

                if mb < 0.01:
                    rate_str = "0.00 MB/s"
                else:
                    rate_str = f"{rate:.2f} MB/s"

                hint = ""
                if sz < 1024 and elapsed >= 10:
                    hint = "  |  waiting for first segments..."

                lp = f"  |  log: {log_path}" if log_path else ""
                tail = _tail_lines(str(log_path or ""), n=4) if log_path else []
                tail_block = ("\n" + "\n".join(tail)) if tail else ""
                self._set_progress(f"Progress: {size_str} written  |  {rate_str}  |  elapsed {t}{hint}{lp}{tail_block}")
            except Exception:
                return

        try:
            self._progress_timer = self.set_interval(1.0, tick)
        except Exception:
            self._progress_timer = None

    def _stop_progress_timer(self) -> None:
        try:
            if self._progress_timer is not None:
                self._progress_timer.stop()
        except Exception:
            pass
        self._progress_timer = None
        self._progress_ctx = {}
        try:
            self._set_progress("")
        except Exception:
            pass

    def _abs_url(self, base_url: str, maybe_rel: str) -> str:
        try:
            from urllib.parse import urljoin

            return urljoin(base_url, maybe_rel)
        except Exception:
            return maybe_rel

    def _parse_attrs(self, s: str) -> dict:
        # Best-effort parse for EXT-X-STREAM-INF attribute lists.
        out: dict[str, str] = {}
        raw = (s or "").strip()
        if raw.startswith("#EXT-X-STREAM-INF:"):
            raw = raw.split(":", 1)[1]
        parts: list[str] = []
        buf = ""
        in_q = False
        for ch in raw:
            if ch == '"':
                in_q = not in_q
            if ch == "," and not in_q:
                parts.append(buf)
                buf = ""
            else:
                buf += ch
        if buf:
            parts.append(buf)
        for p in parts:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            k = k.strip().upper()
            v = v.strip().strip('"')
            if k:
                out[k] = v
        return out

    def _inspect(self) -> None:
        raw_url = (self.query_one("#vod_url", Input).value or "").strip()
        if not raw_url:
            self._set_status("Paste a URL first")
            return

        # Ensure we have an authenticated session before we start resolving/fetching.
        try:
            self.app._get_client()
        except (NotLoggedInError, SessionExpiredError):
            try:
                self.app._prompt_login_and_retry(after_login=self._inspect)
            except Exception:
                self._set_status("Login required")
            return

        self._set_status("Fetching playlist...")
        self._variants = []
        self._selected_url = None
        self._refresh_buttons()

        def work() -> None:
            import requests
            import re
            from datetime import datetime

            def normalize_url(u: str) -> str:
                # Some returned URLs may contain embedded whitespace/newlines.
                # This breaks requests and commonly yields 404.
                try:
                    return "".join(str(u or "").strip().split())
                except Exception:
                    return str(u or "").strip()

            def resolve_to_m3u8(u: str) -> str:
                # Accept:
                # - direct m3u8 URLs
                # - SiriusXM episode page URLs: /player/episode-audio/entity/<uuid> and /player/episode-video/entity/<uuid>
                su = (u or "").strip()
                if su.lower().endswith(".m3u8") or ".m3u8?" in su.lower():
                    return su

                m = re.search(r"/player/(episode-(?:audio|video))/entity/([0-9a-fA-F-]{10,})", su)
                if not m:
                    return su
                kind = m.group(1)
                eid = m.group(2)
                entity_type = "episode-audio" if kind == "episode-audio" else "episode-video"

                # Episode pages do not embed m3u8; resolve via authenticated tuneSource.
                # (playback-state only contains progress/markAsPlayed for these entity types.)
                try:
                    client = self.app._get_client()
                    # For episodes, WEB manifests frequently point at non-existent keys (NoSuchKey).
                    # FULL + HLS V3 is consistently fetchable for both audio and video episodes.
                    ts = client.tune_source(
                        entity_id=eid,
                        entity_type=entity_type,
                        manifest_variant="FULL",
                        hls_version="V3",
                    )
                    master = ts.master_url()
                    try:
                        md = (((ts.raw or {}).get("streams") or [])[0] or {}).get("metadata")
                        if isinstance(md, dict) and md:
                            return str(master).strip(), md
                    except Exception:
                        pass
                except (NotLoggedInError, SessionExpiredError) as exc:
                    raise RuntimeError(f"Login required to resolve episode URL: {exc}")
                except Exception as exc:
                    raise RuntimeError(f"Failed to resolve episode to stream URL: {exc}")

                if not master:
                    raise RuntimeError("Failed to resolve episode to stream URL (no master_url)")
                return str(master).strip(), {}

            inspect_ctx: dict = {"input_url": raw_url}
            try:
                resolved, md = resolve_to_m3u8(raw_url)
                url = normalize_url(resolved)
                inspect_ctx["resolved_url"] = url
                if isinstance(md, dict) and md:
                    inspect_ctx["tunesource_metadata"] = md
            except Exception as exc:
                self.app.call_from_thread(lambda: self._set_status(str(exc)))
                return

            try:
                # Prefer the authenticated session (correct UA/cookies) but allow fallback.
                r = None
                try:
                    client = self.app._get_client()
                    r = client.session.get(
                        url,
                        timeout=20,
                        headers={
                            "Origin": "https://www.siriusxm.com",
                            "Referer": "https://www.siriusxm.com/",
                            "Accept": "application/vnd.apple.mpegurl,application/x-mpegURL,text/plain,*/*",
                        },
                    )
                except Exception:
                    r = None
                if r is None:
                    r = requests.get(url, timeout=20)
                r.raise_for_status()
                text = r.text or ""
            except Exception as exc:
                msg = str(exc)
                if "404" in msg:
                    snippet = ""
                    try:
                        if "r" in locals() and r is not None:
                            snippet = (r.text or "").strip().replace("\n", " ")[:240]
                    except Exception:
                        snippet = ""
                    extra = f" Response: {snippet}" if snippet else ""
                    msg = (
                        "Fetch failed (404). This often indicates missing entitlement/auth on the streaming CDN. "
                        f"Details: {exc}{extra}"
                    )
                self.app.call_from_thread(lambda: self._set_status(msg))
                return

            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            is_master = any(ln.startswith("#EXT-X-STREAM-INF") for ln in lines)

            variants: list[dict] = []
            if is_master:
                i = 0
                while i < len(lines):
                    ln = lines[i]
                    if ln.startswith("#EXT-X-STREAM-INF"):
                        attrs = self._parse_attrs(ln)
                        nxt = ""
                        if i + 1 < len(lines):
                            nxt = lines[i + 1].strip()
                        if nxt and not nxt.startswith("#"):
                            vurl = self._abs_url(url, nxt)
                            bw = attrs.get("BANDWIDTH") or attrs.get("AVERAGE-BANDWIDTH") or ""
                            res = attrs.get("RESOLUTION") or ""
                            codecs = attrs.get("CODECS") or ""
                            t = "video" if res else "audio"
                            qual = res or (f"{bw}bps" if bw else "(variant)")
                            variants.append(
                                {
                                    "type": t,
                                    "quality": qual,
                                    "bandwidth": bw,
                                    "codec": codecs,
                                    "url": vurl,
                                }
                            )
                        i += 1
                    i += 1
            else:
                variants.append(
                    {
                        "type": "media",
                        "quality": "(single)",
                        "bandwidth": "",
                        "codec": "",
                        "url": url,
                    }
                )

            def _variant_sort_key(v: dict) -> tuple:
                try:
                    t = str(v.get("type") or "")
                    bw_raw = str(v.get("bandwidth") or "").strip()
                    try:
                        bw = int(bw_raw) if bw_raw else 0
                    except Exception:
                        bw = 0
                    res = str(v.get("quality") or "")
                    w = 0
                    h = 0
                    try:
                        if "x" in res:
                            ws, hs = res.lower().split("x", 1)
                            w = int("".join(ch for ch in ws if ch.isdigit()) or 0)
                            h = int("".join(ch for ch in hs if ch.isdigit()) or 0)
                    except Exception:
                        w = 0
                        h = 0
                    tprio = 2
                    if t == "video":
                        tprio = 0
                    elif t == "audio":
                        tprio = 1
                    return (tprio, -(h or 0), -(w or 0), -(bw or 0), res)
                except Exception:
                    return (9, 0, 0, 0, "")

            try:
                variants = sorted(variants, key=_variant_sort_key)
            except Exception:
                pass

            def ui_ok() -> None:
                self._variants = variants
                self._render_variants()
                if variants:
                    self._selected_url = str(variants[0].get("url") or "")
                self._last_inspect_ctx = dict(inspect_ctx or {})
                self._set_meta(self._fmt_inspect_meta(self._last_inspect_ctx))
                self._refresh_buttons()
                if raw_url != url:
                    self._set_status(f"Resolved to m3u8 and found {len(variants)} variant(s)")
                else:
                    self._set_status(f"Found {len(variants)} variant(s)")

            self.app.call_from_thread(ui_ok)

        self.app.run_worker(work, thread=True, exclusive=True)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "vod_url":
            return
        self._inspect()

    def _render_variants(self) -> None:
        try:
            tbl = self.query_one("#vod_table", DataTable)
        except Exception:
            return
        try:
            tbl.clear()
        except Exception:
            pass
        if not self._variants:
            try:
                tbl.add_row("-", "No variants", "", "")
            except Exception:
                pass
            return
        for idx, v in enumerate(self._variants):
            try:
                tbl.add_row(
                    str(v.get("type") or ""),
                    str(v.get("quality") or ""),
                    str(v.get("bandwidth") or ""),
                    str(v.get("codec") or ""),
                    key=str(idx),
                )
            except Exception:
                continue
        try:
            tbl.cursor_coordinate = (0, 0)
        except Exception:
            pass

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        try:
            if getattr(event.data_table, "id", None) != "vod_table":
                return
        except Exception:
            return
        self._select_highlighted(quiet=True)

    def _select_highlighted(self, *, quiet: bool = False) -> None:
        try:
            tbl = self.query_one("#vod_table", DataTable)
        except Exception:
            return
        try:
            row_idx = int(getattr(tbl, "cursor_row", 0) or 0)
        except Exception:
            row_idx = 0
        if row_idx < 0 or row_idx >= len(self._variants):
            return
        try:
            self._selected_url = str(self._variants[row_idx].get("url") or "")
        except Exception:
            self._selected_url = None
        self._refresh_buttons()
        if (not quiet) and self._selected_url:
            self._set_status(f"Selected: {self._selected_url}")

    def _suggest_ext(self) -> str:
        try:
            u = str(self._selected_url or "")
            if "audio" in u.lower():
                return ".m4a"
            v = None
            for it in self._variants:
                if str(it.get("url") or "") == u:
                    v = it
                    break
            if v and str(v.get("type") or "") == "audio":
                return ".m4a"
        except Exception:
            pass
        return ".ts"

    def _download_selected(self, *, split_tracks: bool = False) -> None:
        if self._handle is not None:
            self._set_status("Download already running")
            return

        try:
            if self._variants:
                self._select_highlighted(quiet=True)
        except Exception:
            pass

        url = str(self._selected_url or "").strip()
        if not url:
            self._set_status("Select a variant first")
            return

        out_base = _output_category_dir(self.app.settings, "VOD")
        try:
            out_base.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        tmp_dir = out_base / ".satstash_tmp"
        try:
            tmp_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        ext = self._suggest_ext()
        tmp_path = tmp_dir / f"vod-{ts}{ext}.part"
        final_path = out_base / f"vod-{ts}{ext}"
        log_dir = Path(user_cache_dir("satstash"))
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        log_path = log_dir / f"ffmpeg-vod-{ts}.log"

        self._set_status("Starting download...")
        self._refresh_buttons()

        def work() -> None:
            try:
                proxy_info = None
                proxy = None
                input_url = url
                ff_headers = None
                ff_user_agent = "Mozilla/5.0"
                try:
                    # Route downloads through the same HLS proxy used by Live/DVR.
                    # This ensures AES keys are fetched with auth and served as raw bytes.
                    from satstash.hls.proxy import HlsProxy

                    client = self.app._get_client()
                    # Build auth headers for ffmpeg so segment URLs can be fetched directly
                    # when using key-only proxy mode.
                    try:
                        cookie_header = "".join(
                            [f"{k}={v}; " for (k, v) in (client.session.cookies.get_dict() or {}).items()]
                        ).strip()
                    except Exception:
                        cookie_header = ""
                    try:
                        auth = str((client.session.headers or {}).get("Authorization") or "").strip()
                    except Exception:
                        auth = ""
                    hdr_lines = [
                        "Origin: https://www.siriusxm.com",
                        "Referer: https://www.siriusxm.com/",
                        "Accept: application/vnd.apple.mpegurl,application/x-mpegURL,text/plain,*/*",
                    ]
                    if cookie_header:
                        hdr_lines.append(f"Cookie: {cookie_header}")
                    if auth:
                        hdr_lines.append(f"Authorization: {auth}")
                    try:
                        ff_headers = "\r\n".join(hdr_lines) + "\r\n"
                    except Exception:
                        ff_headers = None

                    # Prefer key-only proxy for speed (segments direct, key proxied).
                    try:
                        proxy = HlsProxy(client=client, variant_url=url, proxy_segments=False)
                        proxy_info = proxy.start()
                        input_url = proxy_info.url
                    except Exception:
                        proxy = HlsProxy(client=client, variant_url=url)
                        proxy_info = proxy.start()
                        input_url = proxy_info.url

                    # Preflight the local proxy playlist so we fail fast if the proxy
                    # can't fetch/decrypt segments (instead of ffmpeg hanging).
                    try:
                        import requests

                        rr = requests.get(input_url, timeout=10)
                        rr.raise_for_status()
                        body = (rr.text or "").strip()
                        if "#EXTM3U" not in body:
                            raise RuntimeError(f"Proxy did not return a valid m3u8 (len={len(body)})")
                    except Exception as exc:
                        raise RuntimeError(f"Proxy preflight failed: {exc}")
                except Exception as exc:
                    try:
                        if proxy is not None and proxy_info is not None:
                            proxy.stop(proxy_info)
                    except Exception:
                        pass
                    raise RuntimeError(f"Proxy start failed: {exc}")

                container = "mp4-aac" if ext == ".m4a" else "mpegts"
                handle = start_ffmpeg_recording(
                    input_url=input_url,
                    tmp_path=tmp_path,
                    final_path=final_path,
                    log_path=log_path,
                    container=container,
                    debug=True,
                    preroll_s=1.0,
                    duration_s=None,
                    headers=ff_headers,
                    user_agent=ff_user_agent,
                )
            except Exception as exc:
                self.app.call_from_thread(lambda: self._set_status(f"ffmpeg start failed: {exc}"))
                self.app.call_from_thread(lambda: self._refresh_buttons())
                return

            def ui_set_handle() -> None:
                self._handle = handle
                self._refresh_buttons()
                self._set_status(f"Downloading... ({final_path.name})")
                try:
                    import time

                    self._progress_ctx = {
                        "tmp_path": str(tmp_path),
                        "started": float(time.time()),
                        "log_path": str(log_path),
                        "proxy_url": str(getattr(proxy_info, "url", "") or ""),
                    }
                    self._start_progress_timer()
                except Exception:
                    pass

            self.app.call_from_thread(ui_set_handle)

            try:
                rc = handle.process.wait()
            except Exception:
                rc = None

            try:
                if proxy is not None and proxy_info is not None:
                    proxy.stop(proxy_info)
            except Exception:
                pass

            out_path = None
            try:
                out_path = stop_ffmpeg_recording(handle, finalize=True)
            except Exception:
                out_path = None

            # Apply archival naming + artwork + JSON sidecar.
            final_out = out_path if out_path is not None else final_path
            meta_path = None
            art_path = None
            try:
                import json

                def safe_name(s: str) -> str:
                    s2 = "".join(c if (c.isalnum() or c in " _-().") else "_" for c in (s or "")).strip()
                    s2 = " ".join(s2.split())
                    return s2[:180] if s2 else ""

                def pick_episode_info(ctx: dict) -> tuple[str, str, str, str, dict]:
                    md = ctx.get("tunesource_metadata") if isinstance(ctx, dict) else None
                    if not isinstance(md, dict):
                        return "", "", "", "", {}
                    block = None
                    if isinstance(md.get("aod"), dict):
                        block = md.get("aod")
                    elif isinstance(md.get("vod"), dict):
                        block = md.get("vod")
                    if not isinstance(block, dict):
                        return "", "", "", "", md

                    ep = block.get("episode") if isinstance(block.get("episode"), dict) else {}
                    show = safe_name(str(block.get("channelName") or ep.get("showName") or ""))
                    title = safe_name(str(ep.get("name") or ""))
                    dt_raw = str(ep.get("originalAirTimestamp") or ep.get("startTimestamp") or "").strip()
                    ymd = ""
                    if dt_raw:
                        try:
                            dd = datetime.fromisoformat(dt_raw.replace("Z", "+00:00"))
                            ymd = dd.date().isoformat()
                        except Exception:
                            ymd = ""
                    art_key = ""
                    try:
                        # aod: episode.showImages.tile.aspect_1x1
                        # vod: episode.images.tile.aspect_1x1
                        tile = None
                        if isinstance(ep.get("showImages"), dict):
                            tile = (ep.get("showImages") or {}).get("tile")
                        if not isinstance(tile, dict) and isinstance(ep.get("images"), dict):
                            tile = (ep.get("images") or {}).get("tile")
                        if not isinstance(tile, dict):
                            tile = {}
                        a11 = tile.get("aspect_1x1") if isinstance(tile.get("aspect_1x1"), dict) else {}
                        art_key = (
                            (a11.get("preferredImage") or {}).get("url")
                            or (a11.get("defaultImage") or {}).get("url")
                            or ""
                        )
                    except Exception:
                        art_key = ""
                    return show, title, ymd, str(art_key or "").strip(), md

                ctx = dict(getattr(self, "_last_inspect_ctx", {}) or {})
                show, title, ymd, art_key, md_all = pick_episode_info(ctx)

                # Rename output file to Option A: Show - Episode - YYYY-MM-DD
                new_out = final_out
                if show and title and ymd:
                    base = f"{show} - {title} - {ymd}".strip()
                    base = safe_name(base)
                    if base:
                        candidate = final_out.with_name(base + final_out.suffix)
                        if candidate != final_out:
                            try:
                                if not candidate.exists():
                                    final_out.replace(candidate)
                                    new_out = candidate
                                else:
                                    # Collision: suffix with short timestamp.
                                    candidate2 = final_out.with_name(base + f" - {ts}" + final_out.suffix)
                                    if not candidate2.exists():
                                        final_out.replace(candidate2)
                                        new_out = candidate2
                            except Exception:
                                new_out = final_out

                # Download artwork next to the media file.
                if art_key and new_out and new_out.exists():
                    try:
                        art_out = new_out.with_suffix(".jpg")
                        if not art_out.exists():
                            # Try common SiriusXM img hosts (mirrors artwork logic elsewhere).
                            candidates = []
                            if art_key.startswith("http"):
                                candidates.append(art_key)
                            else:
                                key = art_key.lstrip("/")
                                try:
                                    candidates.append(_imgsrv_url_from_key(key, width=800, height=800))
                                except Exception:
                                    pass
                                try:
                                    candidates.append(_imgix_url_from_key(key))
                                except Exception:
                                    pass
                                candidates.append(f"https://siriusxm-prd.imgix.net/{key}")
                                candidates.append(f"https://siriusxm.imgix.net/{key}")
                            last_exc = None
                            for au in candidates:
                                try:
                                    rr = requests.get(au, timeout=20, headers={"User-Agent": "Mozilla/5.0", "Accept": "image/*,*/*;q=0.8"})
                                    if rr.status_code == 410:
                                        last_exc = RuntimeError("410 Gone")
                                        continue
                                    rr.raise_for_status()
                                    art_out.write_bytes(rr.content)
                                    art_path = art_out
                                    break
                                except Exception as e:
                                    last_exc = e
                                    continue
                            if art_path is None and last_exc is not None:
                                pass
                        else:
                            art_path = art_out
                    except Exception:
                        art_path = None

                # Write archival metadata sidecar.
                chosen = None
                for it in self._variants:
                    if str(it.get("url") or "") == url:
                        chosen = dict(it)
                        break

                meta = {
                    "ts": ts,
                    "final_path": str(new_out),
                    "selected_url": url,
                    "selected_variant": chosen,
                    "inspect": ctx,
                    "tunesource_metadata": md_all,
                    "artwork_key": art_key,
                    "artwork_path": str(art_path) if art_path else None,
                }
                meta_path = Path(str(new_out) + ".json")
                meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

                # For audio episodes, generate cue/chapters from tuneSource cut metadata.
                try:
                    def _to_cue_time(ms: int) -> str:
                        v = max(0, int(ms))
                        total_s, rem_ms = divmod(v, 1000)
                        m, s = divmod(total_s, 60)
                        frames = int(round((rem_ms / 1000.0) * 75.0))
                        if frames >= 75:
                            frames = 74
                        return f"{m:02d}:{s:02d}:{frames:02d}"

                    def _safe_cue_str(s: str) -> str:
                        try:
                            return str(s or "").replace('"', "'").strip()
                        except Exception:
                            return ""

                    if new_out and new_out.suffix.lower() == ".m4a":
                        block = None
                        if isinstance(md_all, dict) and isinstance(md_all.get("aod"), dict):
                            block = md_all.get("aod")
                        elif isinstance(md_all, dict) and isinstance(md_all.get("vod"), dict):
                            block = md_all.get("vod")
                        items = None
                        if isinstance(block, dict) and isinstance(block.get("items"), list):
                            items = [x for x in block.get("items") if isinstance(x, dict)]
                        if items:
                            items2 = []
                            for it in items:
                                try:
                                    off = int(it.get("offset") or 0)
                                    dur = int(it.get("duration") or 0)
                                except Exception:
                                    off = 0
                                    dur = 0
                                nm = _safe_cue_str(it.get("name") or "")
                                art = _safe_cue_str(it.get("artistName") or "")
                                if nm:
                                    items2.append({"offset": off, "duration": dur, "name": nm, "artist": art})
                            items2.sort(key=lambda x: (x.get("offset") or 0, x.get("name") or ""))

                            cue_path = new_out.with_suffix(".cue")
                            if not cue_path.exists():
                                cue_lines: list[str] = []
                                cue_lines.append(f'FILE "{new_out.name}" M4A')
                                n = 0
                                for it in items2:
                                    n += 1
                                    cue_lines.append(f"  TRACK {n:02d} AUDIO")
                                    cue_lines.append(f"    TITLE \"{_safe_cue_str(it.get('name') or '')}\"")
                                    if it.get("artist"):
                                        cue_lines.append(f"    PERFORMER \"{_safe_cue_str(it.get('artist') or '')}\"")
                                    cue_lines.append(f"    INDEX 01 {_to_cue_time(int(it.get('offset') or 0))}")
                                cue_path.write_text("\n".join(cue_lines) + "\n", encoding="utf-8")

                            ffm_path = new_out.with_suffix(".ffmetadata")
                            if not ffm_path.exists():
                                ffm_lines: list[str] = [";FFMETADATA1"]
                                for it in items2:
                                    try:
                                        st = int(it.get("offset") or 0)
                                        en = st + int(it.get("duration") or 0)
                                    except Exception:
                                        st = 0
                                        en = 0
                                    if en <= st:
                                        continue
                                    ffm_lines.append("[CHAPTER]")
                                    ffm_lines.append("TIMEBASE=1/1000")
                                    ffm_lines.append(f"START={st}")
                                    ffm_lines.append(f"END={en}")
                                    ffm_lines.append(f"title={_safe_cue_str(it.get('name') or '')}")
                                ffm_path.write_text("\n".join(ffm_lines) + "\n", encoding="utf-8")

                            try:
                                import shutil
                                import subprocess

                                if ffm_path.exists() and shutil.which("ffmpeg") and new_out.exists():
                                    tmp_ch = new_out.with_suffix(new_out.suffix + ".chapters.tmp")
                                    try:
                                        if tmp_ch.exists():
                                            tmp_ch.unlink()
                                    except Exception:
                                        pass
                                    argv_ch = [
                                        "ffmpeg",
                                        "-nostdin",
                                        "-y",
                                        "-loglevel",
                                        "error",
                                        "-i",
                                        str(new_out),
                                        "-i",
                                        str(ffm_path),
                                        "-map_metadata",
                                        "1",
                                        "-c",
                                        "copy",
                                        "-movflags",
                                        "+faststart",
                                        str(tmp_ch),
                                    ]
                                    rc_ch = subprocess.call(argv_ch, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                    if rc_ch == 0 and tmp_ch.exists() and tmp_ch.stat().st_size > 0:
                                        try:
                                            tmp_ch.replace(new_out)
                                        except Exception:
                                            try:
                                                if tmp_ch.exists():
                                                    tmp_ch.unlink()
                                            except Exception:
                                                pass
                                    else:
                                        try:
                                            if tmp_ch.exists():
                                                tmp_ch.unlink()
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                except Exception:
                    pass

                final_out = new_out
            except Exception:
                pass

            # Optional: export individual track files for audio episodes.
            split_ok = True
            split_summary = ""
            split_error = ""
            split_log_path = ""
            if split_tracks and final_out and str(final_out).lower().endswith(".m4a"):
                try:
                    import shutil
                    import subprocess

                    # Write a dedicated log for split ffmpeg subprocesses.
                    try:
                        lp = Path(str(log_path))
                        split_log_path = str(lp.with_name(f"{lp.stem}-split.log"))
                        with open(split_log_path, "w", encoding="utf-8", errors="replace") as fsl:
                            fsl.write("satstash vod split log\n")
                            fsl.write(f"input={final_out}\n")
                    except Exception:
                        split_log_path = ""

                    if not shutil.which("ffmpeg"):
                        raise RuntimeError("ffmpeg not found (required for track splitting)")

                    ctx2 = dict(getattr(self, "_last_inspect_ctx", {}) or {})
                    md2 = ctx2.get("tunesource_metadata") if isinstance(ctx2, dict) else None
                    block2 = None
                    if isinstance(md2, dict) and isinstance(md2.get("aod"), dict):
                        block2 = md2.get("aod")
                    elif isinstance(md2, dict) and isinstance(md2.get("vod"), dict):
                        block2 = md2.get("vod")

                    items_raw = None
                    if isinstance(block2, dict) and isinstance(block2.get("items"), list):
                        items_raw = [x for x in block2.get("items") if isinstance(x, dict)]
                    if not items_raw:
                        raise RuntimeError("No tuneSource items available for splitting")

                    def should_keep_item(it: dict) -> bool:
                        try:
                            if bool(it.get("isInterstitial")):
                                return False
                        except Exception:
                            pass
                        try:
                            flags = it.get("cutFlags")
                            if isinstance(flags, list) and flags:
                                # Prefer SONG markers when present.
                                if "SONG" in flags:
                                    return True
                                # Otherwise, keep NAVIGABLE cuts.
                                if "NAVIGABLE" in flags:
                                    return True
                        except Exception:
                            pass
                        return True

                    items2 = []
                    for it in items_raw:
                        if not should_keep_item(it):
                            continue
                        try:
                            off_ms = int(it.get("offset") or 0)
                            dur_ms = int(it.get("duration") or 0)
                        except Exception:
                            off_ms = 0
                            dur_ms = 0
                        nm = _safe_filename(str(it.get("name") or "")).strip()
                        if not nm:
                            continue
                        if dur_ms <= 0:
                            continue
                        items2.append({"offset_ms": off_ms, "duration_ms": dur_ms, "name": nm})
                    items2.sort(key=lambda x: (x.get("offset_ms") or 0, x.get("name") or ""))
                    if not items2:
                        raise RuntimeError("No usable items to split")

                    out_dir = final_out.with_suffix("")
                    try:
                        out_dir.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        pass

                    try:
                        if out_dir.exists() and (not out_dir.is_dir()):
                            raise RuntimeError(f"Split output path is not a directory: {out_dir}")
                    except Exception:
                        raise

                    def ui(msg: str) -> None:
                        try:
                            self._set_status(msg)
                        except Exception:
                            pass

                    self.app.call_from_thread(lambda: ui(f"Splitting into {len(items2)} track(s)..."))

                    created = 0
                    for idx, it in enumerate(items2, start=1):
                        start_s = float(it.get("offset_ms") or 0) / 1000.0
                        dur_s = float(it.get("duration_ms") or 0) / 1000.0
                        if dur_s <= 0.01:
                            continue
                        base = _safe_filename(it.get("name") or "")
                        if not base:
                            base = f"Track {idx:02d}"
                        # Keep names filesystem-safe and ffmpeg-friendly.
                        safe_base = _safe_filename(f"{idx:02d} - {base}")
                        safe_base = (safe_base or f"Track {idx:02d}")[:180]
                        dst = _unique_path(out_dir / f"{safe_base}.m4a")
                        if dst.exists():
                            continue
                        # IMPORTANT: ffmpeg guesses muxer from extension. Ensure tmp still ends with .m4a.
                        tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
                        try:
                            if tmp.exists():
                                tmp.unlink()
                        except Exception:
                            pass

                        # For stream copy, use INPUT seeking (-ss before -i) for cleaner boundaries.
                        # Output seeking with -c copy is often inaccurate and can cause overlap/duplication.
                        argv_copy = [
                            "ffmpeg",
                            "-nostdin",
                            "-y",
                            "-loglevel",
                            "error",
                            "-ss",
                            f"{start_s:.3f}",
                            "-i",
                            str(final_out),
                            "-t",
                            f"{dur_s:.3f}",
                            "-c",
                            "copy",
                            "-movflags",
                            "+faststart",
                            "-f",
                            "mp4",
                            str(tmp),
                        ]
                        try:
                            p2 = subprocess.run(argv_copy, capture_output=True, text=True)
                            rc2 = int(p2.returncode)
                            err2 = (p2.stderr or "")[-4000:]
                            try:
                                if split_log_path:
                                    with open(split_log_path, "a", encoding="utf-8", errors="replace") as fsl:
                                        fsl.write(f"\n# track {idx:02d} copy start={start_s:.3f} dur={dur_s:.3f}\n")
                                        fsl.write("cmd=" + " ".join(argv_copy) + "\n")
                                        fsl.write(f"rc={rc2}\n")
                                        if err2:
                                            fsl.write("stderr=\n" + err2 + "\n")
                            except Exception:
                                pass
                        except Exception as exc:
                            rc2 = 1
                            err2 = str(exc)

                        if rc2 != 0 or (not tmp.exists()) or tmp.stat().st_size <= 0:
                            # Fallback: re-encode AAC for maximum compatibility.
                            try:
                                if tmp.exists():
                                    tmp.unlink()
                            except Exception:
                                pass
                            argv_enc = [
                                "ffmpeg",
                                "-nostdin",
                                "-y",
                                "-loglevel",
                                "error",
                                "-i",
                                str(final_out),
                                "-ss",
                                f"{start_s:.3f}",
                                "-t",
                                f"{dur_s:.3f}",
                                "-vn",
                                "-c:a",
                                "aac",
                                "-b:a",
                                "256k",
                                "-movflags",
                                "+faststart",
                                "-f",
                                "mp4",
                                str(tmp),
                            ]
                            try:
                                p3 = subprocess.run(argv_enc, capture_output=True, text=True)
                                rc3 = int(p3.returncode)
                                err3 = (p3.stderr or "")[-4000:]
                                try:
                                    if split_log_path:
                                        with open(split_log_path, "a", encoding="utf-8", errors="replace") as fsl:
                                            fsl.write(f"\n# track {idx:02d} aac start={start_s:.3f} dur={dur_s:.3f}\n")
                                            fsl.write("cmd=" + " ".join(argv_enc) + "\n")
                                            fsl.write(f"rc={rc3}\n")
                                            if err3:
                                                fsl.write("stderr=\n" + err3 + "\n")
                                except Exception:
                                    pass
                            except Exception as exc:
                                rc3 = 1
                                err3 = str(exc)
                            if rc3 != 0 or (not tmp.exists()) or tmp.stat().st_size <= 0:
                                if not split_error:
                                    split_error = (err3 or err2 or "split failed").strip().splitlines()[-1] if (err3 or err2) else "split failed"
                                try:
                                    if tmp.exists():
                                        tmp.unlink()
                                except Exception:
                                    pass
                                continue

                        try:
                            tmp.replace(dst)
                            created += 1
                        except Exception as exc:
                            if not split_error:
                                split_error = str(exc)
                            try:
                                if tmp.exists():
                                    tmp.unlink()
                            except Exception:
                                pass

                    split_summary = f"Split complete: wrote {created}/{len(items2)} track(s)"
                    self.app.call_from_thread(lambda: ui(split_summary))
                    if created <= 0:
                        raise RuntimeError("Split produced 0 tracks")
                except Exception as exc:
                    split_ok = False
                    if not split_error:
                        split_error = str(exc)
                    try:
                        self.app.call_from_thread(lambda: self._set_status(f"Split failed: {exc}"))
                    except Exception:
                        pass

            # Delete ffmpeg log on success (keep logs on failure for debugging).
            try:
                if rc == 0 and split_ok and final_out and Path(str(final_out)).exists():
                    try:
                        Path(str(log_path)).unlink()
                    except Exception:
                        pass
            except Exception:
                pass

            def ui_done() -> None:
                self._handle = None
                self._stop_progress_timer()
                self._refresh_buttons()
                if rc == 0 and split_ok and final_out.exists():
                    extra = f"\n{split_summary}" if split_summary else ""
                    self._set_status(f"Done: {final_out}{extra}")
                else:
                    extra = f" | split_error: {split_error}" if split_error else ""
                    extra2 = f" | split_log: {split_log_path}" if split_log_path else ""
                    self._set_status(f"Download finished (rc={rc}). Log: {log_path}{extra}{extra2}")

            self.app.call_from_thread(ui_done)

        self.app.run_worker(work, thread=True, exclusive=True)

    def _stop_download(self) -> None:
        h = self._handle
        if h is None:
            self._set_status("No download running")
            return

        def work() -> None:
            try:
                stop_ffmpeg_recording(h, finalize=True)
            except Exception:
                pass

            def ui() -> None:
                self._handle = None
                self._refresh_buttons()
                self._set_status("Stopped")

            self.app.call_from_thread(ui)

        self.app.run_worker(work, thread=True, exclusive=False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        bid = event.button.id
        if bid == "vod_close":
            try:
                if self._handle is not None:
                    self._stop_download()
            except Exception:
                pass
            try:
                getattr(self.app, "_show_right_idle", lambda: None)()
            except Exception:
                pass
            return
        if bid == "vod_inspect":
            self._inspect()
            return
        if bid == "vod_download":
            self._download_selected()
            return
        if bid == "vod_download_split":
            self._download_selected(split_tracks=True)
            return
        if bid == "vod_stop":
            self._stop_download()
            return


class RecordSelectScreen(Screen[None]):
    DEFAULT_CSS = """
    RecordSelectScreen {
        layout: vertical;
    }
    RecordSelectScreen Container {
        layout: vertical;
        height: 1fr;
    }
    RecordSelectScreen #channels {
        height: 1fr;
    }
    RecordSelectScreen #actions {
        height: auto;
    }
    """

    def __init__(self, channels: List[Channel]):
        super().__init__()
        self.channels = sorted(channels, key=lambda c: (c.number is None, c.number or 9999, c.name))
        self.filtered: List[Channel] = self.channels
        self._channels_by_id: dict[str, Channel] = {}
        self._selected_id: Optional[str] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container():
            yield Static("Record Now", id="title")
            yield Static("", id="rec_status")
            yield Input(placeholder="Search channel name/number", id="search")
            yield DataTable(id="channels")
            with Horizontal(id="actions"):
                yield Button("Record Single File", id="record_single", variant="primary")
                yield Button("Record Split Tracks", id="record_split")
                yield Button("Record & Listen", id="record_listen")
                yield Button("Stop Recording", id="stop_record")
                yield Button("Back", id="back")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#channels", ActivatableDataTable)
        try:
            table.can_focus = True
        except Exception:
            pass
        try:
            table._activate_callback = self._activate_highlighted
        except Exception:
            pass
        table.cursor_type = "row"
        table.add_columns("Ch", "Name")
        self._render_results()
        self.query_one("#search", Input).focus()
        self._refresh_recording_ui()
        self.set_interval(1.0, self._refresh_recording_ui)

    def _refresh_recording_ui(self) -> None:
        try:
            rec = getattr(self.app, "_record_handle", None)
            ch = getattr(self.app, "_record_channel", None)
            started = getattr(self.app, "_record_started_at", None)
            pending = bool(getattr(self.app, "_record_pending", False))

            is_active = (rec is not None) or pending

            try:
                self.query_one("#record_single", Button).disabled = is_active
            except Exception:
                pass
            try:
                self.query_one("#record_split", Button).disabled = is_active
            except Exception:
                pass
            try:
                self.query_one("#stop_record", Button).disabled = not is_active
            except Exception:
                pass

            if not ch or not started or not is_active:
                self.query_one("#rec_status", Static).update("")
                return

            elapsed = max(0, int(time.time() - float(started)))
            m, s = divmod(elapsed, 60)
            h, m = divmod(m, 60)
            t = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
            label = f"{ch.number if ch.number is not None else ''} {ch.name}".strip()

            if rec is None:
                self.query_one("#rec_status", Static).update(
                    f"Recording: {label} ({t}) (starting...)"
                )
                return

            self.query_one("#rec_status", Static).update(
                f"Recording: {label} ({t})\n{rec.final_path}"
            )
        except Exception:
            pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "search":
            return
        # Enter in the search box plays the first match.
        if not self.filtered:
            self.app.notify("No matches")
            return
        self._selected_id = self.filtered[0].id
        self._start_record_selected()
        self.query_one("#search", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "search":
            return
        q = (event.value or "").strip().lower()
        if not q:
            self.filtered = self.channels
        else:
            q_is_digit = q.isdigit()
            self.filtered = [
                ch
                for ch in self.channels
                if q in ch.name.lower()
                or (
                    q_is_digit
                    and ch.number is not None
                    and (q == str(ch.number) or str(ch.number).startswith(q))
                )
            ]
        self._render_results()

    def _render_results(self) -> None:
        table = self.query_one("#channels", DataTable)
        table.clear()
        self._channels_by_id = {}
        self._selected_id = None

        if not self.filtered:
            table.add_row("-", "No matches")
            return

        for ch in self.filtered:
            num = "-" if ch.number is None else str(ch.number)
            table.add_row(num, ch.name, key=ch.id)
            self._channels_by_id[ch.id] = ch

        self._selected_id = self.filtered[0].id
        table.cursor_coordinate = (0, 0)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        key = getattr(event.row_key, "value", None)
        if isinstance(key, str) and key:
            self._selected_id = key

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "back":
            # Avoid leaving an orphan ffmpeg process running.
            try:
                self.app.stop_record_now(silent=True)
            except Exception:
                pass
            self.app.pop_screen()
            return
        if event.button.id == "stop_record":
            self._stop_recording()
            return
        if event.button.id == "record_listen":
            self._record_and_listen_selected()
            return
        if event.button.id == "record_single":
            self._start_record_selected_single()
            return
        if event.button.id == "record_split":
            self._start_record_selected_split()
            return
        return

    def _selected_channel(self) -> Optional[Channel]:
        if self._selected_id and self._selected_id in self._channels_by_id:
            return self._channels_by_id[self._selected_id]
        return self.filtered[0] if self.filtered else None

    def _start_record_selected_single(self) -> None:
        ch = self._selected_channel()
        if not ch:
            self.app.notify("No channel selected")
            return
        self.app.start_record_single(channel=ch)

    def _start_record_selected_split(self) -> None:
        ch = self._selected_channel()
        if not ch:
            self.app.notify("No channel selected")
            return
        self.app.start_record_tracks(channel=ch)

    def _record_and_listen_selected(self) -> None:
        ch = self._selected_channel()
        if not ch:
            self.app.notify("No channel selected")
            return
        # Start recording first (sets UI pending state), then start playback.
        self.app.start_record_single(channel=ch)
        label = f"{ch.number if ch.number is not None else ''} {ch.name}".strip()
        self.app.notify(f"Starting playback: {label}")

        def work() -> None:
            try:
                self.app.start_live(channel=ch)
            except Exception as exc:
                self.app.call_from_thread(lambda: self.app.notify(f"Playback failed: {exc}"))

        self.app.run_worker(work, thread=True, exclusive=True)

    def _stop_recording(self) -> None:
        self.app.stop_record_now()


class ActivatableDataTable(DataTable):
    BINDINGS = [("enter", "activate", "")]

    def action_activate(self) -> None:
        cb = getattr(self, "_activate_callback", None)
        if cb is None:
            return
        try:
            cb()
        except Exception:
            return


class RecordSelectPane(Container):
    DEFAULT_CSS = """
    RecordSelectPane {
        layout: vertical;
        height: 1fr;
    }
    RecordSelectPane #record_pane {
        layout: vertical;
        height: 1fr;
    }
    RecordSelectPane #channels {
        height: 1fr;
    }
    RecordSelectPane #actions {
        height: auto;
    }
    """

    def __init__(self, channels: List[Channel]):
        super().__init__()
        self.channels = sorted(channels, key=lambda c: (c.number is None, c.number or 9999, c.name))
        self.filtered: List[Channel] = self.channels
        self._channels_by_id: dict[str, Channel] = {}
        self._selected_id: Optional[str] = None

    def render(self):
        try:
            from rich.text import Text

            return Text("")
        except Exception:
            return ""

    def compose(self) -> ComposeResult:
        with Container(id="record_pane"):
            yield Static("Record Now", id="title")
            yield Static("", id="rec_status")
            yield Input(placeholder="Search channel name/number", id="search")
            yield ActivatableDataTable(id="channels")
            with Horizontal(id="actions"):
                yield Button("Record Single File", id="record_single", variant="primary")
                yield Button("Record Split Tracks", id="record_split")
                yield Button("Record & Listen", id="record_listen")
                yield Button("Stop Recording", id="stop_record")
                yield Button("Close", id="close")

    def on_mount(self) -> None:
        table = self.query_one("#channels", ActivatableDataTable)
        try:
            table.can_focus = True
        except Exception:
            pass
        try:
            table._activate_callback = self._activate_highlighted
        except Exception:
            pass
        table.cursor_type = "row"
        table.add_columns("Ch", "Name")
        self._render_results()
        self.query_one("#search", Input).focus()
        self._refresh_recording_ui()
        self.set_interval(1.0, self._refresh_recording_ui)

    def _refresh_recording_ui(self) -> None:
        try:
            rec = getattr(self.app, "_record_handle", None)
            ch = getattr(self.app, "_record_channel", None)
            started = getattr(self.app, "_record_started_at", None)
            pending = bool(getattr(self.app, "_record_pending", False))

            is_active = (rec is not None) or pending

            try:
                self.query_one("#record_single", Button).disabled = is_active
            except Exception:
                pass
            try:
                self.query_one("#record_split", Button).disabled = is_active
            except Exception:
                pass
            try:
                self.query_one("#stop_record", Button).disabled = not is_active
            except Exception:
                pass

            if not ch or not started or not is_active:
                self.query_one("#rec_status", Static).update("")
                return

            elapsed = max(0, int(time.time() - float(started)))
            m, s = divmod(elapsed, 60)
            h, m = divmod(m, 60)
            t = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
            label = f"{ch.number if ch.number is not None else ''} {ch.name}".strip()

            if rec is None:
                self.query_one("#rec_status", Static).update(
                    f"Recording: {label} ({t}) (starting...)"
                )
                return

            self.query_one("#rec_status", Static).update(
                f"Recording: {label} ({t})\n{rec.final_path}"
            )
        except Exception:
            pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "search":
            return
        if not self.filtered:
            self.app.notify("No matches")
            return
        self._selected_id = self.filtered[0].id
        self._start_record_selected_single()
        self.query_one("#search", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "search":
            return
        q = (event.value or "").strip().lower()
        if not q:
            self.filtered = self.channels
        else:
            q_is_digit = q.isdigit()
            self.filtered = [
                ch
                for ch in self.channels
                if q in ch.name.lower()
                or (
                    q_is_digit
                    and ch.number is not None
                    and (q == str(ch.number) or str(ch.number).startswith(q))
                )
            ]
        self._render_results()

    def _render_results(self) -> None:
        table = self.query_one("#channels", DataTable)
        table.clear()
        self._channels_by_id = {}
        self._selected_id = None

        if not self.filtered:
            table.add_row("-", "No matches")
            return

        for ch in self.filtered:
            num = "-" if ch.number is None else str(ch.number)
            table.add_row(num, ch.name, key=ch.id)
            self._channels_by_id[ch.id] = ch

        self._selected_id = self.filtered[0].id
        table.cursor_coordinate = (0, 0)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        key = getattr(event.row_key, "value", None)
        if isinstance(key, str) and key:
            self._selected_id = key

    def _activate_highlighted(self) -> None:
        table = self.query_one("#channels", DataTable)
        try:
            row_index = int(getattr(table, "cursor_row", 0) or 0)
        except Exception:
            row_index = 0
        try:
            ordered = getattr(table, "ordered_rows", [])
            row = ordered[row_index] if ordered and 0 <= row_index < len(ordered) else None
            row_key = getattr(row, "key", None)
            key_val = getattr(row_key, "value", None)
            if isinstance(key_val, str) and key_val:
                self._selected_id = key_val
        except Exception:
            pass
        self._start_record_selected_single()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "close":
            try:
                getattr(self.app, "_show_right_idle", lambda: None)()
            except Exception:
                pass
            return
        if event.button.id == "stop_record":
            self._stop_recording()
            return
        if event.button.id == "record_listen":
            self._record_and_listen_selected()
            return
        if event.button.id == "record_single":
            self._start_record_selected_single()
            return
        if event.button.id == "record_split":
            self._start_record_selected_split()
            return

    def _selected_channel(self) -> Optional[Channel]:
        if self._selected_id and self._selected_id in self._channels_by_id:
            return self._channels_by_id[self._selected_id]
        return self.filtered[0] if self.filtered else None

    def _start_record_selected_single(self) -> None:
        ch = self._selected_channel()
        if not ch:
            self.app.notify("No channel selected")
            return
        self.app.start_record_single(channel=ch)

    def _start_record_selected_split(self) -> None:
        ch = self._selected_channel()
        if not ch:
            self.app.notify("No channel selected")
            return
        self.app.start_record_tracks(channel=ch)

    def _record_and_listen_selected(self) -> None:
        ch = self._selected_channel()
        if not ch:
            self.app.notify("No channel selected")
            return
        self.app.start_record_single(channel=ch)
        label = f"{ch.number if ch.number is not None else ''} {ch.name}".strip()
        self.app.notify(f"Starting playback: {label}")

        def work() -> None:
            try:
                self.app.start_live(channel=ch)
            except Exception as exc:
                self.app.call_from_thread(lambda: self.app.notify(f"Playback failed: {exc}"))

        self.app.run_worker(work, thread=True, exclusive=True)

    def _stop_recording(self) -> None:
        self.app.stop_record_now()


class CatchUpSelectScreen(Screen[None]):
    DEFAULT_CSS = """
    CatchUpSelectScreen {
        layout: vertical;
    }
    CatchUpSelectScreen Container {
        layout: vertical;
        height: 1fr;
    }
    CatchUpSelectScreen #channels {
        height: 1fr;
    }
    CatchUpSelectScreen #actions {
        height: auto;
    }
    """

    def __init__(self, channels: List[Channel]):
        super().__init__()
        self.channels = sorted(channels, key=lambda c: (c.number is None, c.number or 9999, c.name))
        self.filtered: List[Channel] = self.channels
        self._channels_by_id: dict[str, Channel] = {}
        self._selected_id: Optional[str] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container():
            yield Static("Catch Up", id="title")
            yield Input(placeholder="Search channel name/number", id="search")
            yield DataTable(id="channels")
            with Horizontal(id="actions"):
                yield Button("Open", id="open", variant="primary")
                yield Button("Back", id="back")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#channels", DataTable)
        try:
            table.can_focus = True
        except Exception:
            pass
        table.cursor_type = "row"
        table.add_columns("Ch", "Name")
        self._render_results()
        self.query_one("#search", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "search":
            return
        if not self.filtered:
            self.app.notify("No matches")
            return
        self._selected_id = self.filtered[0].id
        self._open_selected()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "search":
            return
        q = (event.value or "").strip().lower()
        if not q:
            self.filtered = self.channels
        else:
            q_is_digit = q.isdigit()
            self.filtered = [
                ch
                for ch in self.channels
                if q in ch.name.lower()
                or (
                    q_is_digit
                    and ch.number is not None
                    and (q == str(ch.number) or str(ch.number).startswith(q))
                )
            ]
        self._render_results()

    def _render_results(self) -> None:
        table = self.query_one("#channels", DataTable)
        table.clear()
        self._channels_by_id = {}
        self._selected_id = None

        if not self.filtered:
            table.add_row("-", "No matches")
            return

        for ch in self.filtered:
            num = "-" if ch.number is None else str(ch.number)
            table.add_row(num, ch.name, key=ch.id)
            self._channels_by_id[ch.id] = ch

        self._selected_id = self.filtered[0].id
        table.cursor_coordinate = (0, 0)

    def _selected_channel(self) -> Optional[Channel]:
        if self._selected_id and self._selected_id in self._channels_by_id:
            return self._channels_by_id[self._selected_id]
        return self.filtered[0] if self.filtered else None

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        key = getattr(event.row_key, "value", None)
        if isinstance(key, str) and key:
            self._selected_id = key

    def _activate_highlighted(self) -> None:
        table = self.query_one("#channels", DataTable)
        try:
            row_index = int(getattr(table, "cursor_row", 0) or 0)
        except Exception:
            row_index = 0
        try:
            ordered = getattr(table, "ordered_rows", [])
            row = ordered[row_index] if ordered and 0 <= row_index < len(ordered) else None
            row_key = getattr(row, "key", None)
            key_val = getattr(row_key, "value", None)
            if isinstance(key_val, str) and key_val:
                self._selected_id = key_val
        except Exception:
            pass
        self._open_selected()

    def on_key(self, event) -> None:
        if getattr(event, "key", None) != "enter":
            return
        focused = getattr(self, "focused", None)
        if getattr(focused, "id", None) == "search":
            return
        table = self.query_one("#channels", DataTable)
        if not getattr(table, "has_focus", False):
            return
        event.stop()
        try:
            row_index = int(getattr(table, "cursor_row", 0) or 0)
        except Exception:
            row_index = 0
        try:
            ordered = getattr(table, "ordered_rows", [])
            row = ordered[row_index] if ordered and 0 <= row_index < len(ordered) else None
            row_key = getattr(row, "key", None)
            key_val = getattr(row_key, "value", None)
            if isinstance(key_val, str) and key_val:
                self._selected_id = key_val
        except Exception:
            pass
        self._open_selected()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "back":
            self.app.pop_screen()
            return
        if event.button.id == "open":
            self._play_selected()
            return

    def _open_selected(self) -> None:
        ch = self._selected_channel()
        if not ch:
            self.app.notify("No channel selected")
            return
        self.app.push_screen(CatchUpScreen(channel=ch))

    def _play_selected(self) -> None:
        ch = self._selected_channel()
        if not ch:
            self.app.notify("No channel selected")
            return
        self.app.start_live(channel=ch, push_screen=False)


class CatchUpSelectPane(Container):
    DEFAULT_CSS = """
    CatchUpSelectPane {
        layout: vertical;
        height: 1fr;
    }
    CatchUpSelectPane #catchup_select_pane {
        layout: vertical;
        height: 1fr;
    }
    CatchUpSelectPane #channels {
        height: 1fr;
    }
    CatchUpSelectPane #actions {
        height: auto;
    }
    """

    def __init__(self, channels: List[Channel]):
        super().__init__()
        self.channels = sorted(channels, key=lambda c: (c.number is None, c.number or 9999, c.name))
        self.filtered: List[Channel] = self.channels
        self._channels_by_id: dict[str, Channel] = {}
        self._selected_id: Optional[str] = None

    def render(self):
        try:
            from rich.text import Text

            return Text("")
        except Exception:
            return ""

    def compose(self) -> ComposeResult:
        with Container(id="catchup_select_pane"):
            yield Static("Catch Up", id="title")
            yield Input(placeholder="Search channel name/number", id="search")
            yield ActivatableDataTable(id="channels")
            with Horizontal(id="actions"):
                yield Button("Open", id="open", variant="primary")
                yield Button("Close", id="close")

    def on_mount(self) -> None:
        table = self.query_one("#channels", ActivatableDataTable)
        try:
            table.can_focus = True
        except Exception:
            pass
        try:
            table._activate_callback = self._activate_highlighted
        except Exception:
            pass
        table.cursor_type = "row"
        table.add_columns("Ch", "Name")
        self._render_results()
        self.query_one("#search", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "search":
            return
        if not self.filtered:
            self.app.notify("No matches")
            return
        self._selected_id = self.filtered[0].id
        self._open_selected()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "search":
            return
        q = (event.value or "").strip().lower()
        if not q:
            self.filtered = self.channels
        else:
            q_is_digit = q.isdigit()
            self.filtered = [
                ch
                for ch in self.channels
                if q in ch.name.lower()
                or (
                    q_is_digit
                    and ch.number is not None
                    and (q == str(ch.number) or str(ch.number).startswith(q))
                )
            ]
        self._render_results()

    def _render_results(self) -> None:
        table = self.query_one("#channels", DataTable)
        table.clear()
        self._channels_by_id = {}
        self._selected_id = None

        if not self.filtered:
            table.add_row("-", "No matches")
            return

        for ch in self.filtered:
            num = "-" if ch.number is None else str(ch.number)
            table.add_row(num, ch.name, key=ch.id)
            self._channels_by_id[ch.id] = ch

        self._selected_id = self.filtered[0].id
        table.cursor_coordinate = (0, 0)

    def _selected_channel(self) -> Optional[Channel]:
        if self._selected_id and self._selected_id in self._channels_by_id:
            return self._channels_by_id[self._selected_id]
        return self.filtered[0] if self.filtered else None

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        key = getattr(event.row_key, "value", None)
        if isinstance(key, str) and key:
            self._selected_id = key

    def _activate_highlighted(self) -> None:
        table = self.query_one("#channels", DataTable)
        try:
            row_index = int(getattr(table, "cursor_row", 0) or 0)
        except Exception:
            row_index = 0
        try:
            ordered = getattr(table, "ordered_rows", [])
            row = ordered[row_index] if ordered and 0 <= row_index < len(ordered) else None
            row_key = getattr(row, "key", None)
            key_val = getattr(row_key, "value", None)
            if isinstance(key_val, str) and key_val:
                self._selected_id = key_val
        except Exception:
            pass
        self._open_selected()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "close":
            try:
                getattr(self.app, "_show_right_idle", lambda: None)()
            except Exception:
                pass
            return
        if event.button.id == "open":
            self._open_selected()
            return

    def _open_selected(self) -> None:
        ch = self._selected_channel()
        if not ch:
            self.app.notify("No channel selected")
            return
        try:
            self.app._set_right_pane(CatchUpPane(channel=ch))
        except Exception:
            self.app.push_screen(CatchUpScreen(channel=ch))


class CatchUpPane(Container):
    DEFAULT_CSS = """
    CatchUpPane {
        layout: vertical;
        height: 1fr;
    }
    CatchUpPane #catchup_pane {
        layout: vertical;
        height: 1fr;
    }
    CatchUpPane #tracks {
        height: 1fr;
    }
    CatchUpPane #export_status {
        height: auto;
        background: darkgreen;
        color: white;
        text-style: bold;
    }
    CatchUpPane #toggle_numbering {
        width: 14;
    }
    CatchUpPane #actions {
        height: auto;
    }
    """

    def __init__(self, *, channel: Channel):
        super().__init__()
        self._channel = channel
        self._items: list[dict] = []
        self._rows: list[dict] = []
        self._start_key: Optional[str] = None
        self._end_key: Optional[str] = None
        self._export_mode: str = "tracks"
        self._range_mode: str = "track"
        self._minutes: int = 30
        self._track_numbering: bool = True
        self._loading: bool = False
        self._cancel_export = threading.Event()

    def render(self):
        try:
            from rich.text import Text

            return Text("")
        except Exception:
            return ""

    def cancel_export(self) -> None:
        try:
            self._cancel_export.set()
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        with Container(id="catchup_pane"):
            label = f"{self._channel.number if self._channel.number is not None else ''} {self._channel.name}".strip()
            yield Static(f"Catch Up: {label}", id="title")
            yield Static("", id="mode")
            yield Static("Start: (none)   End: (none)", id="range")
            yield Static("", id="export_status")
            yield DataTable(id="tracks")
            with Horizontal(id="actions"):
                yield Button("Toggle Range", id="toggle_range")
                yield Button("Toggle Export", id="toggle_export")
                yield Button("Numbering: ON", id="toggle_numbering")
                yield Button("Export", id="export", variant="primary")
                yield Button("Refresh", id="refresh")
                yield Button("Close", id="close")

    def on_mount(self) -> None:
        table = self.query_one("#tracks", DataTable)
        table.cursor_type = "row"
        table.add_columns("", "Time", "Artist", "Title")
        self._refresh_mode_label()
        self._load_items_async()

    def _refresh_mode_label(self) -> None:
        exp = "Individual tracks" if self._export_mode == "tracks" else "Single file"
        rng = "Track range" if self._range_mode == "track" else f"Time range ({self._minutes}m)"
        num = "ON" if self._track_numbering else "OFF"
        extra = f"   Numbering: {num}" if self._export_mode == "tracks" else ""
        msg = f"Range: {rng}   Export: {exp}{extra}   Keys: S=start  E=end"
        self.query_one("#mode", Static).update(msg)
        try:
            btn = self.query_one("#toggle_numbering", Button)
            btn.disabled = self._export_mode != "tracks"
            btn.label = f"Numbering: {num}"
        except Exception:
            pass

    def _load_items_async(self) -> None:
        if self._loading:
            return
        self._loading = True
        self.query_one("#tracks", DataTable).clear()
        self.query_one("#tracks", DataTable).add_row("", "", "Loading…", "")

        def work() -> None:
            items: list[dict] = []
            try:
                client = self.app._get_client()
                start_ts = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat().replace("+00:00", "Z")
                data = client.live_update(channel_id=self._channel.id, start_timestamp=start_ts)
                raw = data.get("items") or []
                if isinstance(raw, list):
                    items = [it for it in raw if isinstance(it, dict)]
            except Exception as exc:
                def ui_fail() -> None:
                    self._loading = False
                    self.app.notify(f"Catch-up load failed: {exc}")
                    self.query_one("#tracks", DataTable).clear()
                self.app.call_from_thread(ui_fail)
                return

            def parse_item_dt(item: dict) -> Optional[datetime]:
                try:
                    ts = item.get("timestamp")
                    if isinstance(ts, str) and ts:
                        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
                except Exception:
                    return None
                return None

            parsed: list[tuple[datetime, dict]] = []
            for it in items:
                dt = parse_item_dt(it)
                if dt is None:
                    continue
                parsed.append((dt, it))
            parsed.sort(key=lambda x: x[0])

            buffer_start: Optional[datetime] = None
            try:
                buffer_start = probe_dvr_buffer_start_pdt(
                    client=client,
                    channel_id=self._channel.id,
                    channel_type=getattr(self._channel, "channel_type", "channel-linear") or "channel-linear",
                    preferred_quality=self.app.settings.preferred_quality or "256k",
                    lookback_hours=5,
                )
            except Exception:
                buffer_start = None

            rows: list[dict] = []
            for dt, it in parsed:
                if isinstance(buffer_start, datetime):
                    try:
                        if dt < buffer_start:
                            continue
                    except Exception:
                        pass
                tid = it.get("id")
                if not tid:
                    tid = f"{dt.isoformat()}:{it.get('name') or ''}:{it.get('artistName') or ''}"
                artist = (it.get("artistName") or "").strip()
                title = (it.get("name") or "").strip()

                dur_raw = it.get("duration") or 0
                dur_ms = 0
                try:
                    dv = float(dur_raw)
                    if 0 < dv <= 24 * 60 * 60:
                        dur_ms = int(round(dv * 1000.0))
                    elif dv > 0:
                        dur_ms = int(round(dv))
                except Exception:
                    dur_ms = 0
                rows.append(
                    {
                        "key": str(tid),
                        "dt": dt,
                        "artist": artist,
                        "title": title,
                        "album": (it.get("albumName") or "").strip(),
                        "duration_ms": dur_ms,
                        "raw": it,
                    }
                )

            def ui_ok() -> None:
                self._items = items
                self._rows = rows
                self._loading = False
                self._render_table()

            self.app.call_from_thread(ui_ok)

        self.app.run_worker(work, thread=True, exclusive=True)

    def _render_table(self) -> None:
        table = self.query_one("#tracks", DataTable)
        preserve_key: Optional[str] = None
        preserve_row: int = 0
        try:
            preserve_row = int(getattr(table, "cursor_row", 0) or 0)
            ordered = getattr(table, "ordered_rows", [])
            if ordered and 0 <= preserve_row < len(ordered):
                preserve_key = getattr(getattr(ordered[preserve_row], "key", None), "value", None)
                if not isinstance(preserve_key, str) or not preserve_key:
                    preserve_key = None
        except Exception:
            preserve_key = None
            preserve_row = 0

        table.clear()
        if not self._rows:
            table.add_row("", "", "(no items)", "")
            return

        for r in self._rows:
            dt: datetime = r["dt"]
            t = dt.astimezone().strftime("%H:%M:%S")
            mark = ""
            if self._start_key and r["key"] == self._start_key:
                mark = "S"
            if self._end_key and r["key"] == self._end_key:
                mark = (mark + "E") if mark else "E"
            table.add_row(mark, t, r.get("artist") or "", r.get("title") or "", key=r["key"])

        try:
            if preserve_key:
                ordered2 = getattr(table, "ordered_rows", [])
                found = None
                for i, rr in enumerate(ordered2):
                    k = getattr(getattr(rr, "key", None), "value", None)
                    if k == preserve_key:
                        found = i
                        break
                if found is not None:
                    table.cursor_coordinate = (found, 0)
                else:
                    table.cursor_coordinate = (min(max(0, preserve_row), max(0, len(ordered2) - 1)), 0)
            else:
                table.cursor_coordinate = (0, 0)
        except Exception:
            pass

        self._refresh_range_label()

    def _refresh_range_label(self) -> None:
        def fmt_key(k: Optional[str]) -> str:
            if not k:
                return "(none)"
            for r in self._rows:
                if r.get("key") == k:
                    dt: datetime = r["dt"]
                    t = dt.astimezone().strftime("%H:%M:%S")
                    return f"{t} {r.get('artist') or ''} - {r.get('title') or ''}".strip()
            return "(unknown)"

        self.query_one("#range", Static).update(f"Start: {fmt_key(self._start_key)}\nEnd:   {fmt_key(self._end_key)}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "close":
            try:
                getattr(self.app, "_show_right_idle", lambda: None)()
            except Exception:
                pass
            return
        if event.button.id == "refresh":
            self._load_items_async()
            return
        if event.button.id == "toggle_export":
            self._export_mode = "single" if self._export_mode == "tracks" else "tracks"
            self._refresh_mode_label()
            return
        if event.button.id == "toggle_numbering":
            self._track_numbering = not bool(self._track_numbering)
            try:
                event.button.label = f"Numbering: {'ON' if self._track_numbering else 'OFF'}"
            except Exception:
                pass
            self._refresh_mode_label()
            return
        if event.button.id == "toggle_range":
            self._range_mode = "time" if self._range_mode == "track" else "track"
            self._refresh_mode_label()
            return
        if event.button.id == "export":
            self._export()
            return

    def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        key = getattr(event, "key", "")
        if key in ("s", "S"):
            self._set_marker("start")
            event.stop()
            return
        if key in ("e", "E"):
            self._set_marker("end")
            event.stop()
            return
        if key in ("d", "D"):
            try:
                cur = bool(getattr(self.app, "_debug_enabled", False))
            except Exception:
                cur = False
            try:
                setattr(self.app, "_debug_enabled", (not cur))
            except Exception:
                pass
            try:
                self.app.notify(f"Debug: {'ON' if not cur else 'OFF'}")
            except Exception:
                pass
            event.stop()
            return

    def _set_marker(self, which: str) -> None:
        table = self.query_one("#tracks", DataTable)
        try:
            i = int(getattr(table, "cursor_row", 0) or 0)
            ordered = getattr(table, "ordered_rows", [])
            if not ordered or i < 0 or i >= len(ordered):
                return
            row_key = getattr(getattr(ordered[i], "key", None), "value", None)
            if not isinstance(row_key, str) or not row_key:
                return
        except Exception:
            return

        if which == "start":
            self._start_key = row_key
        else:
            self._end_key = row_key

        self._render_table()

    def _export(self) -> None:
        # Determine export window.
        if self._range_mode == "track":
            if not self._start_key or not self._end_key:
                self.app.notify("Pick start (S) and end (E) tracks")
                return
            start = None
            end = None
            start_idx = None
            end_idx = None
            for i, r in enumerate(self._rows):
                if r.get("key") == self._start_key:
                    start = r
                    start_idx = i
                if r.get("key") == self._end_key:
                    end = r
                    end_idx = i
            if start is None or end is None or start_idx is None or end_idx is None:
                self.app.notify("Invalid start/end selection")
                return
            if start_idx > end_idx:
                start, end = end, start
                start_idx, end_idx = end_idx, start_idx

            # Build selected rows by index so we never export tracks outside the chosen range.
            # Also include one extra "boundary-only" row after the end so bumpers at the end
            # can clamp to the next metadata boundary even though that next track isn't exported.
            selected_rows = self._rows[start_idx : end_idx + 1]
            boundary_row = None
            try:
                if end_idx + 1 < len(self._rows):
                    boundary_row = self._rows[end_idx + 1]
            except Exception:
                boundary_row = None

            start_pdt = start["dt"].astimezone(timezone.utc)
            end_pdt = end["dt"].astimezone(timezone.utc)
            # End boundary must be robust against bogus duration_ms (common for bumpers/IDs).
            # Prefer next metadata boundary when available and only trust duration_ms if sane.
            title = str(end.get("title") or "")
            artist = str(end.get("artist") or "")
            bumper_like = False
            try:
                bumper_like = bool(_is_bumper_like(title=title, artist=artist))
            except Exception:
                bumper_like = False

            cand: Optional[datetime] = None
            try:
                if end_idx + 1 < len(self._rows):
                    cand = self._rows[end_idx + 1]["dt"].astimezone(timezone.utc)
            except Exception:
                cand = None

            gap_s = 0.0
            if isinstance(cand, datetime):
                try:
                    gap_s = float((cand - end_pdt).total_seconds())
                except Exception:
                    gap_s = 0.0

            dur_ms = 0
            try:
                dur_ms = int(end.get("duration_ms") or 0)
            except Exception:
                dur_ms = 0

            dur_s = 0.0
            if dur_ms > 0:
                try:
                    dur_s = float(dur_ms) / 1000.0
                except Exception:
                    dur_s = 0.0

            # Duration is "trusted" only when it is within a sane bound AND doesn't wildly
            # disagree with the next metadata boundary.
            trust_duration = False
            try:
                if dur_s > 0.0 and dur_s <= 30 * 60:
                    trust_duration = True
                if isinstance(cand, datetime) and gap_s > 0.0:
                    # If duration is much larger than the observed boundary gap, don't trust it.
                    if dur_s > gap_s + 10.0:
                        trust_duration = False
            except Exception:
                trust_duration = False

            if bumper_like:
                # Bumpers/IDs should never extend beyond a tight cap.
                cap_end = end_pdt + timedelta(seconds=25.0)
                if isinstance(cand, datetime):
                    cap_end = min(cap_end, cand)
                end_pdt = cap_end
            elif isinstance(cand, datetime) and 0.5 <= gap_s <= 10 * 60:
                # If the next boundary is reasonably close, prefer it.
                end_pdt = cand
            elif trust_duration:
                end_pdt = end_pdt + timedelta(milliseconds=dur_ms)
            else:
                # Last resort: keep it bounded.
                end_pdt = end_pdt + timedelta(minutes=6)
        else:
            minutes = max(1, int(self._minutes or 30))
            start_pdt = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).astimezone(timezone.utc)
            end_pdt = datetime.now(timezone.utc).astimezone(timezone.utc)

            selected_rows = self._rows
            boundary_row = None

        # Build track index for cue/chapter and for individual-track export.
        # Include the boundary-only row if present (export=False) to provide next_dt.
        tracks: list[dict] = []
        rows_for_tracks = list(selected_rows)
        try:
            if boundary_row is not None:
                rows_for_tracks.append(boundary_row)
        except Exception:
            pass

        for r in rows_for_tracks:
            dt = r["dt"].astimezone(timezone.utc)
            if dt < start_pdt or dt > end_pdt:
                continue
            try:
                off = (dt - start_pdt).total_seconds()
            except Exception:
                off = 0.0
            exportable = True
            try:
                if boundary_row is not None and r is boundary_row:
                    exportable = False
            except Exception:
                exportable = True

            tracks.append(
                {
                    "id": r.get("key"),
                    "offset_s": max(0.0, float(off)),
                    "artist": r.get("artist") or "",
                    "title": r.get("title") or "",
                    "album": r.get("album") or "",
                    "dt": dt,
                    "duration_ms": r.get("duration_ms") or 0,
                    "export": exportable,
                    "raw": r.get("raw") or {},
                }
            )

        if self._export_mode == "tracks":
            self._export_tracks_window(start_pdt=start_pdt, end_pdt=end_pdt, tracks=tracks)
            return

        self._export_window(start_pdt=start_pdt, end_pdt=end_pdt, track_index=tracks)

    def _export_window(self, *, start_pdt: datetime, end_pdt: datetime, track_index: list[dict]) -> None:
        # Starting a new export should clear any previous cancel state.
        try:
            self._cancel_export.clear()
        except Exception:
            pass

        label = f"{self._channel.number if self._channel.number is not None else ''} {self._channel.name}".strip() or "channel"
        safe_chan = "".join(c if c.isalnum() or c in " _-." else "_" for c in label).strip() or "channel"
        base_dir = _output_category_dir(self.app.settings, "CatchUp") / safe_chan
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        out_dir = base_dir / ts
        out_dir.mkdir(parents=True, exist_ok=True)

        safe_title = "CatchUp"
        try:
            span_s = max(0.0, float((end_pdt - start_pdt).total_seconds()))
        except Exception:
            span_s = 0.0
        self.app.notify(f"Exporting catch-up: {safe_chan} ({_fmt_time(span_s)})")

        def set_status(msg: str) -> None:
            def ui() -> None:
                try:
                    self.query_one("#export_status", Static).update(str(msg))
                except Exception:
                    pass

            try:
                self.app.call_from_thread(ui)
            except Exception:
                pass

        set_status("Preparing export...")

        def work() -> None:
            handle: Optional[RecordHandle] = None
            try:
                client = self.app._get_client()

                def progress(msg: str) -> None:
                    try:
                        self.app.call_from_thread(lambda: self.app.notify(str(msg)))
                    except Exception:
                        pass
                    try:
                        set_status(str(msg))
                    except Exception:
                        pass

                handle = start_recording(
                    client=client,
                    channel_id=self._channel.id,
                    channel_type=getattr(self._channel, "channel_type", "channel-linear") or "channel-linear",
                    preferred_quality=self.app.settings.preferred_quality or "256k",
                    out_dir=out_dir,
                    title=safe_title,
                    progress=progress,
                    debug=bool(getattr(self.app, "_debug_enabled", False)),
                    start_pdt=start_pdt,
                    end_pdt=end_pdt,
                )

                if self._cancel_export.is_set():
                    out_path = stop_recording_with_options(handle, finalize_on_stop=True, progress=progress)

                    def ui_cancel() -> None:
                        self.app.notify(f"Catch-up canceled: {out_path}")
                        try:
                            self.app.notify(f"ffmpeg log: {handle.log_path}")
                        except Exception:
                            pass

                    self.app.call_from_thread(ui_cancel)
                    return

                # Finite catch-up download: proxy will emit #EXT-X-ENDLIST so ffmpeg can
                # download as fast as possible and exit on its own.
                started_at = time.time()
                last_tick = 0.0
                while True:
                    if self._cancel_export.is_set():
                        out_path = stop_recording_with_options(handle, finalize_on_stop=True, progress=progress)

                        def ui_cancel2() -> None:
                            self.app.notify(f"Catch-up canceled: {out_path}")
                            try:
                                self.app.notify(f"ffmpeg log: {handle.log_path}")
                            except Exception:
                                pass

                        self.app.call_from_thread(ui_cancel2)
                        return

                    rc = handle.process.poll()
                    if rc is not None:
                        break

                    now = time.time()
                    if now - last_tick >= 1.0:
                        last_tick = now
                        elapsed = max(0.0, now - started_at)
                        size_bytes = 0
                        diag = ""
                        diag_match = ""
                        diag_log = ""
                        seg_hint = ""
                        # Track the largest file size among likely ffmpeg output candidates.
                        try:
                            candidates: list[Path] = []
                            try:
                                candidates.append(handle.tmp_path)
                            except Exception:
                                pass
                            try:
                                candidates.append(handle.final_path)
                            except Exception:
                                pass
                            try:
                                td = out_dir / ".satstash_tmp"
                                if td.exists():
                                    for p in td.iterdir():
                                        candidates.append(p)
                            except Exception:
                                pass
                            try:
                                for p in out_dir.glob("__window-*.m4a"):
                                    candidates.append(p)
                            except Exception:
                                pass

                            best = 0
                            best_path: Optional[Path] = None
                            for p in candidates:
                                try:
                                    if p.exists() and p.is_file():
                                        sz = int(p.stat().st_size)
                                        if sz >= best:
                                            best = sz
                                            best_path = p
                                except Exception:
                                    continue
                            size_bytes = int(best)
                            try:
                                if bool(getattr(self.app, "_debug_enabled", False)):
                                    parts = []
                                    for pp in candidates[:8]:
                                        try:
                                            parts.append(f"{pp.name}={int(pp.stat().st_size) if pp.exists() else 0}B")
                                        except Exception:
                                            parts.append(f"{pp.name}=?")
                                    bp = best_path.name if best_path is not None else "(none)"
                                    diag = f" files[{', '.join(parts)}] best={bp}"
                            except Exception:
                                diag = ""
                        except Exception:
                            size_bytes = 0

                        # If file size stays tiny (common with fragmented MP4), parse ffmpeg log for size=.
                        if size_bytes < 1024:
                            try:
                                lp = getattr(handle, "log_path", None)
                                if lp and Path(lp).exists():
                                    tail = Path(lp).read_text(encoding="utf-8", errors="replace")
                                    b2, m2 = _parse_ffmpeg_size_bytes(tail)
                                    if b2 > size_bytes:
                                        size_bytes = b2
                                    if m2:
                                        diag_match = m2
                                    try:
                                        seg_hint = _parse_ffmpeg_hls_activity(tail)
                                    except Exception:
                                        seg_hint = ""
                                    diag_log = str(lp)
                            except Exception:
                                pass

                        if bool(getattr(self.app, "_debug_enabled", False)):
                            try:
                                extra = []
                                if diag_match:
                                    extra.append(f"size={diag_match}")
                                if diag_log:
                                    extra.append(f"log={Path(diag_log).name}")
                                if extra:
                                    diag = (diag + " " + " ".join(extra)).strip()
                            except Exception:
                                pass

                        show_size = bool(size_bytes >= 1024)
                        if size_bytes >= 1024 * 1024:
                            size_disp = f"{(float(size_bytes) / (1024.0 * 1024.0)):.1f} MB"
                        elif size_bytes >= 1024:
                            size_disp = f"{(float(size_bytes) / 1024.0):.0f} KB"
                        else:
                            size_disp = f"{int(size_bytes)} B"

                        msg = f"Downloading... elapsed {_fmt_time(elapsed)}"
                        if show_size:
                            msg = msg + f"  ({size_disp})"
                        if seg_hint:
                            msg = msg + f"  seg:{seg_hint}"
                        if diag:
                            msg = msg + "\n" + diag
                        set_status(msg)
                    time.sleep(0.25)

                set_status("Finalizing...")
                out_path = stop_recording_with_options(handle, finalize_on_stop=True, progress=progress)
                cue_path = None
                ffmeta_path = None
                # Try to embed cover art into the exported window so DVR playback shows art.
                try:
                    cover_bytes, cover_type = (b"", "")
                    logo_url = None
                    try:
                        if getattr(self._channel, "logo_url", None):
                            logo_url = str(self._channel.logo_url)
                    except Exception:
                        logo_url = None

                    # Prefer first track item's artwork (more specific than channel logo).
                    art_url = ""
                    try:
                        for t in track_index or []:
                            if not isinstance(t, dict):
                                continue
                            if not bool(t.get("export", True)):
                                continue
                            raw = t.get("raw") if isinstance(t.get("raw"), dict) else {}
                            art_url = _extract_art_url_from_live_item(raw or {}, channel_logo_url=logo_url)
                            if art_url:
                                break
                    except Exception:
                        art_url = ""

                    if not art_url and logo_url:
                        try:
                            art_url = _normalize_art_url(logo_url)
                        except Exception:
                            art_url = logo_url

                    if art_url:
                        try:
                            cover_bytes, cover_type = _fetch_image_bytes(art_url)
                        except Exception:
                            cover_bytes, cover_type = (b"", "")

                    # As a last resort, iTunes fallback using the first exportable track's artist/title.
                    if not cover_bytes:
                        try:
                            for t in track_index or []:
                                if not isinstance(t, dict):
                                    continue
                                if not bool(t.get("export", True)):
                                    continue
                                artist = str(t.get("artist") or "").strip()
                                title = str(t.get("title") or "").strip()
                                if not artist or not title:
                                    continue
                                if _looks_like_show_metadata(title=title, artist=artist):
                                    continue
                                cover_bytes, cover_type = _fetch_itunes_cover_bytes(artist=artist, title=title)
                                if cover_bytes:
                                    break
                        except Exception:
                            cover_bytes, cover_type = (b"", "")

                    # Tag only if we got something.
                    if out_path.exists() and out_path.suffix.lower() == ".m4a" and cover_bytes:
                        _tag_m4a(
                            path=out_path,
                            title=safe_title,
                            artist=safe_chan,
                            album=safe_chan,
                            cover_bytes=cover_bytes,
                            cover_content_type=cover_type,
                        )
                except Exception:
                    pass
                try:
                    if out_path.exists() and out_path.suffix.lower() == ".m4a" and track_index:
                        dur_s2 = _probe_duration_s(out_path)
                        safe_tracks = _sanitize_track_index(tracks=track_index, duration_s=dur_s2)
                        cue_path = _write_cue(audio_path=out_path, tracks=safe_tracks, duration_s=dur_s2)
                        ffmeta_path = _write_ffmetadata(audio_path=out_path, tracks=safe_tracks, duration_s=dur_s2)
                        # Embed chapters into the m4a for better player compatibility (VLC, etc.).
                        try:
                            if ffmeta_path and ffmeta_path.exists() and shutil.which("ffmpeg"):
                                tmp_ch = _unique_path(out_dir / f"__chapters-{out_path.stem}.m4a")
                                _ffmpeg_mux_chapters_from_ffmeta(src=out_path, ffmeta=ffmeta_path, dst=tmp_ch)
                                try:
                                    if tmp_ch.exists() and tmp_ch.stat().st_size > 0:
                                        tmp_ch.replace(out_path)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    cue_path = None
                    ffmeta_path = None

                def ui_done() -> None:
                    try:
                        self.query_one("#export_status", Static).update("")
                    except Exception:
                        pass
                    if out_path == handle.final_path:
                        self.app.notify(f"Catch-up saved: {out_path}")
                    else:
                        self.app.notify(f"Catch-up partial: {out_path}")
                        self.app.notify(f"ffmpeg log: {handle.log_path}")
                    try:
                        if cue_path and cue_path.exists() and cue_path.stat().st_size > 0:
                            self.app.notify(f"CUE: {cue_path}")
                    except Exception:
                        pass
                    try:
                        if ffmeta_path and ffmeta_path.exists() and ffmeta_path.stat().st_size > 0:
                            self.app.notify(f"Chapters: {ffmeta_path}")
                    except Exception:
                        pass

                self.app.call_from_thread(ui_done)
            except Exception as exc:
                def ui_fail() -> None:
                    try:
                        self.query_one("#export_status", Static).update("")
                    except Exception:
                        pass
                    self.app.notify(f"Catch-up export failed: {exc}")
                    try:
                        if handle is not None:
                            self.app.notify(f"ffmpeg log: {handle.log_path}")
                    except Exception:
                        pass
                self.app.call_from_thread(ui_fail)
                return

        self.app.run_worker(work, thread=True, exclusive=True)

    def _export_tracks_window(self, *, start_pdt: datetime, end_pdt: datetime, tracks: list[dict]) -> None:
        try:
            self._cancel_export.clear()
        except Exception:
            pass

        label = f"{self._channel.number if self._channel.number is not None else ''} {self._channel.name}".strip() or "channel"
        safe_chan = "".join(c if c.isalnum() or c in " _-." else "_" for c in label).strip() or "channel"
        base_dir = _output_category_dir(self.app.settings, "CatchUp") / safe_chan
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        out_dir = base_dir / ts
        out_dir.mkdir(parents=True, exist_ok=True)

        ordered: list[dict] = []
        try:
            ordered = [t for t in tracks if isinstance(t, dict) and isinstance(t.get("dt"), datetime)]
            ordered.sort(key=lambda x: x.get("dt"))
        except Exception:
            ordered = list(tracks or [])

        usable = [t for t in ordered if bool(t.get("export", True)) and (t.get("title") or t.get("artist") or "").strip()]
        if not usable:
            self.app.notify("No tracks in selected window")
            return

        def set_status(msg: str) -> None:
            def ui() -> None:
                try:
                    self.query_one("#export_status", Static).update(str(msg))
                except Exception:
                    pass

            try:
                self.app.call_from_thread(ui)
            except Exception:
                pass

        self.app.notify(f"Exporting {len(usable)} tracks: {safe_chan}")
        set_status(f"Preparing {len(usable)} tracks...")

        def work() -> None:
            try:
                client = self.app._get_client()
            except Exception as exc:
                self.app.call_from_thread(lambda: self.app.notify(f"Catch-up export failed: {exc}"))
                return

            buffer_start = probe_dvr_buffer_start_pdt(
                client=client,
                channel_id=self._channel.id,
                channel_type=getattr(self._channel, "channel_type", "channel-linear") or "channel-linear",
                preferred_quality=self.app.settings.preferred_quality or "256k",
                lookback_hours=5,
            )

            if isinstance(buffer_start, datetime):
                try:
                    bs_local = buffer_start.astimezone().strftime("%H:%M:%S")
                except Exception:
                    bs_local = "(unknown)"
                self.app.call_from_thread(lambda: self.app.notify(f"DVR buffer starts at {bs_local}"))

                # If the selected range begins before the buffer start, warn that Option B will skip.
                try:
                    if start_pdt < buffer_start:
                        self.app.call_from_thread(lambda: self.app.notify("Selected range begins before DVR buffer; out-of-buffer tracks will be skipped"))
                except Exception:
                    pass

            saved = 0
            failed = 0
            skipped = 0
            playlist_entries: list[str] = []

            out_i = 0
            for i, t in enumerate(ordered):
                if not bool(t.get("export", True)):
                    continue
                if self._cancel_export.is_set():
                    self.app.call_from_thread(lambda: self.app.notify("Catch-up track export canceled"))
                    break

                if not (t.get("title") or t.get("artist") or "").strip():
                    continue

                out_i += 1

                dt = t.get("dt")
                if not isinstance(dt, datetime):
                    continue
                track_start = dt.astimezone(timezone.utc)

                track_end: Optional[datetime] = None
                next_dt: Optional[datetime] = None
                try:
                    if i + 1 < len(ordered):
                        nd = ordered[i + 1].get("dt")
                        if isinstance(nd, datetime):
                            next_dt = nd.astimezone(timezone.utc)
                except Exception:
                    next_dt = None
                try:
                    dur_ms = int(t.get("duration_ms") or 0)
                except Exception:
                    dur_ms = 0
                try:
                    if dur_ms > 0 and dur_ms <= 30 * 60 * 1000:
                        track_end = track_start + timedelta(milliseconds=dur_ms)
                except Exception:
                    track_end = None

                # Prefer next metadata boundary when it's reasonable.
                if isinstance(next_dt, datetime):
                    try:
                        gap_s = float((next_dt - track_start).total_seconds())
                        if 0.5 <= gap_s <= 10 * 60:
                            track_end = next_dt
                    except Exception:
                        pass

                if track_end is None:
                    track_end = track_start + timedelta(minutes=6)

                # Clamp to requested window.
                if track_start < start_pdt:
                    skipped += 1
                    continue
                if track_start > end_pdt:
                    break
                if track_end > end_pdt:
                    track_end = end_pdt

                # If buffer start exists, skip out-of-buffer items.
                if isinstance(buffer_start, datetime):
                    try:
                        if track_start < buffer_start:
                            skipped += 1
                            continue
                    except Exception:
                        pass

                safe_artist = "".join(c if c.isalnum() or c in " _-." else "_" for c in (t.get("artist") or "")).strip()
                safe_title = "".join(c if c.isalnum() or c in " _-." else "_" for c in (t.get("title") or "")).strip()
                if self._track_numbering:
                    name = f"{out_i:03d} - {safe_artist} - {safe_title}".strip(" -") or f"{out_i:03d}"
                else:
                    base = f"{safe_artist} - {safe_title}".strip(" -")
                    if not base:
                        base = safe_title or safe_artist or f"track-{out_i}"
                    name = base
                out_path = _unique_path(out_dir / f"{name}.m4a")

                try:
                    self.app.call_from_thread(lambda: set_status(f"[{out_i}/{len(usable)}] {safe_title or safe_artist}"))
                except Exception:
                    pass

                try:
                    handle = start_recording(
                        client=client,
                        channel_id=self._channel.id,
                        channel_type=getattr(self._channel, "channel_type", "channel-linear") or "channel-linear",
                        preferred_quality=self.app.settings.preferred_quality or "256k",
                        out_dir=out_dir,
                        title=name,
                        progress=None,
                        debug=bool(getattr(self.app, "_debug_enabled", False)),
                        start_pdt=track_start,
                        end_pdt=track_end,
                    )
                    # Wait for ffmpeg to finish.
                    started_at = time.time()
                    last_tick = 0.0
                    while True:
                        if self._cancel_export.is_set():
                            stop_recording_with_options(handle, finalize_on_stop=True)
                            break
                        rc = handle.process.poll()
                        if rc is not None:
                            break
                        now = time.time()
                        if now - last_tick >= 1.0:
                            last_tick = now
                            elapsed = max(0.0, now - started_at)
                            size_bytes = 0
                            seg_hint = ""
                            try:
                                candidates: list[Path] = []
                                try:
                                    candidates.append(handle.tmp_path)
                                except Exception:
                                    pass
                                try:
                                    candidates.append(handle.final_path)
                                except Exception:
                                    pass
                                best = 0
                                for p in candidates:
                                    try:
                                        if p.exists() and p.is_file():
                                            sz = int(p.stat().st_size)
                                            if sz >= best:
                                                best = sz
                                    except Exception:
                                        continue
                                size_bytes = int(best)
                            except Exception:
                                size_bytes = 0

                            if size_bytes < 1024:
                                try:
                                    lp = getattr(handle, "log_path", None)
                                    if lp and Path(lp).exists():
                                        tail = Path(lp).read_text(encoding="utf-8", errors="replace")
                                        b2, _m2 = _parse_ffmpeg_size_bytes(tail)
                                        if b2 > size_bytes:
                                            size_bytes = b2
                                        seg_hint = _parse_ffmpeg_hls_activity(tail)
                                except Exception:
                                    seg_hint = ""

                            show_size = bool(size_bytes >= 1024)
                            if size_bytes >= 1024 * 1024:
                                size_disp = f"{(float(size_bytes) / (1024.0 * 1024.0)):.1f} MB"
                            elif size_bytes >= 1024:
                                size_disp = f"{(float(size_bytes) / 1024.0):.0f} KB"
                            else:
                                size_disp = f"{int(size_bytes)} B"

                            msg = f"[{out_i}/{len(usable)}] Downloading... elapsed {_fmt_time(elapsed)}"
                            if show_size:
                                msg = msg + f"  ({size_disp})"
                            msg = msg + f"\n{safe_title or safe_artist}"
                            if seg_hint:
                                msg = msg + f"  | seg:{seg_hint}"
                            try:
                                set_status(msg)
                            except Exception:
                                pass
                        time.sleep(0.2)
                    final_path = stop_recording_with_options(handle, finalize_on_stop=True)

                    # Move/rename to desired name if needed.
                    try:
                        if final_path.exists() and final_path != out_path:
                            final_path.replace(out_path)
                            final_path = out_path
                    except Exception:
                        pass

                    # Embed tags + cover art so DVR playback shows artwork.
                    try:
                        raw = t.get("raw") if isinstance(t.get("raw"), dict) else {}
                        logo_url = None
                        try:
                            if getattr(self._channel, "logo_url", None):
                                logo_url = str(self._channel.logo_url)
                        except Exception:
                            logo_url = None

                        art_url = _extract_art_url_from_live_item(raw or {}, channel_logo_url=logo_url)
                        cover_bytes, cover_type = (b"", "")
                        if art_url:
                            try:
                                cover_bytes, cover_type = _fetch_image_bytes(art_url)
                            except Exception:
                                cover_bytes, cover_type = (b"", "")

                        if not cover_bytes:
                            try:
                                if not _looks_like_show_metadata(title=safe_title, artist=safe_artist):
                                    cover_bytes, cover_type = _fetch_itunes_cover_bytes(artist=safe_artist, title=safe_title)
                            except Exception:
                                cover_bytes, cover_type = (b"", "")

                        if final_path.exists() and final_path.suffix.lower() == ".m4a":
                            _tag_m4a(
                                path=final_path,
                                title=safe_title,
                                artist=safe_artist,
                                album=str(t.get("album") or "").strip(),
                                cover_bytes=cover_bytes,
                                cover_content_type=cover_type,
                            )
                    except Exception:
                        pass

                    if final_path.exists() and final_path.stat().st_size > 0:
                        saved += 1
                        playlist_entries.append(str(final_path))
                    else:
                        failed += 1
                except Exception:
                    failed += 1
                    continue

            def ui_done() -> None:
                try:
                    self.query_one("#export_status", Static).update("")
                except Exception:
                    pass
                self.app.notify(f"Catch-up export done: saved={saved} failed={failed} skipped={skipped}")
                if playlist_entries:
                    try:
                        pl = out_dir / "playlist.m3u"
                        pl.write_text("\n".join(playlist_entries) + "\n", encoding="utf-8", errors="replace")
                        self.app.notify(f"Playlist: {pl}")
                    except Exception:
                        pass

            self.app.call_from_thread(ui_done)

        self.app.run_worker(work, thread=True, exclusive=True)


class RightIdleLogoPane(Widget):
    def compose(self) -> ComposeResult:
        with Container(id="right_idle"):
            yield Static("", id="right_idle_logo")
            yield Static("Select an option from the menu below", id="right_idle_hint")

    def on_mount(self) -> None:
        self._rerender()

    def on_resize(self, event) -> None:
        try:
            event.stop()
        except Exception:
            pass
        self._rerender()

    def _rerender(self) -> None:
        ww = 0
        wh = 0
        try:
            # Use the widget's allocated size; the Static can be shrink-wrapped to content.
            sz = getattr(self, "size", None)
            ww = int(getattr(sz, "width", 0) or 0) if sz is not None else 0
            wh = int(getattr(sz, "height", 0) or 0) if sz is not None else 0
        except Exception:
            ww, wh = (0, 0)

        if ww <= 0 or wh <= 0:
            try:
                logo_w = self.query_one("#right_idle_logo", Static)
                wdg_sz = getattr(logo_w, "size", None)
                ww = int(getattr(wdg_sz, "width", 0) or 0) if wdg_sz is not None else 0
                wh = int(getattr(wdg_sz, "height", 0) or 0) if wdg_sz is not None else 0
            except Exception:
                ww, wh = (0, 0)

        if ww <= 0 or wh <= 0:
            # Initial mount can report 0x0; retry shortly.
            try:
                self.set_timer(0.15, self._rerender)
            except Exception:
                pass
            return

        def work() -> None:
            rendered: object = ""
            try:
                here = Path(__file__).resolve()
                candidates = [
                    here.parent / "logo.png",  # packaged install (src/satstash/logo.png)
                    here.parents[2] / "logo.png",  # repo checkout (logo.png at project root)
                ]
                for logo_path in candidates:
                    if logo_path.exists():
                        try:
                            rendered = _image_to_rich_blocks_fit(logo_path, width=ww, height=wh, bg_rgb=(15, 18, 17))
                            break
                        except Exception:
                            rendered = ""
            except Exception:
                rendered = ""

            if not rendered:
                rendered = _BUILTIN_LOGO

            def ui() -> None:
                try:
                    self.query_one("#right_idle_logo", Static).update(rendered or "")
                except Exception:
                    pass

            try:
                self.app.call_from_thread(ui)
            except Exception:
                pass

        try:
            self.app.run_worker(work, thread=True)
        except Exception:
            pass


class BrowseDvrPane(Container):
    def __init__(self, recordings_dir: Path):
        super().__init__()
        self.recordings_dir = recordings_dir
        self._mode: str = "folders"  # folders|files|queue
        self._folder: Optional[Path] = None
        self._query: str = ""
        self._rows: list[Path] = []
        self._visible_rows: list[Path] = []
        self._selected: Optional[Path] = None
        self._return_mode: Optional[str] = None
        self._return_folder: Optional[Path] = None

    def render(self):
        # Composite widget; children provide the UI. Return an empty renderable.
        try:
            from rich.text import Text

            return Text("")
        except Exception:
            return ""

    def compose(self) -> ComposeResult:
        with Container(id="dvr_pane"):
            yield Static("Browse DVR", id="dvr_title")
            yield Static("", id="dvr_path")
            yield Input(placeholder="Search", id="dvr_search")
            yield Static("Selected: (none)", id="dvr_selected")
            yield DataTable(id="dvr_table")
            with Horizontal(id="dvr_actions"):
                yield Button("Open/Play", id="dvr_open", variant="primary")
                yield Button("Back", id="dvr_back")
                yield Button("Refresh", id="dvr_refresh")
                yield Button("Close", id="dvr_close")

    def on_mount(self) -> None:
        table = self.query_one("#dvr_table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Name", "Info")
        self._load_folders()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "dvr_search":
            return
        self._query = (event.value or "").strip().lower()
        self._render_table()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        key = getattr(event.row_key, "value", None)
        if isinstance(key, str) and key:
            try:
                self._selected = Path(key)
            except Exception:
                self._selected = None
        self._refresh_selected_label()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        # Keep selection in sync with cursor movement (arrow keys).
        try:
            event.stop()
        except Exception:
            pass
        self._sync_selected_from_cursor()

    def _sync_selected_from_cursor(self) -> None:
        try:
            table = self.query_one("#dvr_table", DataTable)
        except Exception:
            return

        # Always map selection from the cursor row into the currently-rendered rows.
        # This avoids relying on RowSelected events (which don't fire on arrow movement)
        # and avoids DataTable row-key edge cases.
        row_idx = 0
        try:
            cc = getattr(table, "cursor_coordinate", None)
            if cc is not None:
                row_idx = int(getattr(cc, "row", 0) or 0)
            else:
                row_idx = int(getattr(table, "cursor_row", 0) or 0)
        except Exception:
            row_idx = 0

        rows = list(getattr(self, "_visible_rows", None) or [])
        if not rows:
            rows = list(getattr(self, "_rows", None) or [])
        if not rows:
            self._selected = None
            self._refresh_selected_label()
            return

        if row_idx < 0:
            row_idx = 0
        if row_idx >= len(rows):
            row_idx = max(0, len(rows) - 1)

        try:
            self._selected = rows[row_idx]
        except Exception:
            self._selected = None
        self._refresh_selected_label()

    def on_key(self, event) -> None:
        k = getattr(event, "key", None)
        focused = None
        try:
            focused = getattr(self.app, "focused", None)
        except Exception:
            focused = None
        if focused is None:
            focused = getattr(self, "focused", None)

        # Ignore hotkeys while typing in the search box.
        if getattr(focused, "id", None) == "dvr_search":
            return

        # Toggle queue view.
        if k == "q":
            try:
                table = self.query_one("#dvr_table", DataTable)
            except Exception:
                table = None
            table_focused = False
            try:
                table_focused = (focused is table) or isinstance(focused, DataTable) or (getattr(focused, "id", None) == "dvr_table")
            except Exception:
                table_focused = (focused is table) or (getattr(focused, "id", None) == "dvr_table")
            if not table_focused:
                return

            event.stop()
            if self._mode == "queue":
                self._exit_queue_view()
            else:
                self._enter_queue_view()
            return

        # Queue-only actions.
        if k == "x" and self._mode == "queue":
            try:
                table = self.query_one("#dvr_table", DataTable)
            except Exception:
                table = None
            table_focused = False
            try:
                table_focused = (focused is table) or isinstance(focused, DataTable) or (getattr(focused, "id", None) == "dvr_table")
            except Exception:
                table_focused = (focused is table) or (getattr(focused, "id", None) == "dvr_table")
            if not table_focused:
                return

            event.stop()
            try:
                self.app.toggle_dvr_shuffle()
            except Exception:
                pass
            try:
                self._load_queue()
            except Exception:
                pass
            return

        # Queue-only actions: save/load playlists.
        if k in {"w", "l"} and self._mode == "queue":
            try:
                table = self.query_one("#dvr_table", DataTable)
            except Exception:
                table = None
            table_focused = False
            try:
                table_focused = (focused is table) or isinstance(focused, DataTable) or (getattr(focused, "id", None) == "dvr_table")
            except Exception:
                table_focused = (focused is table) or (getattr(focused, "id", None) == "dvr_table")
            if not table_focused:
                return

            event.stop()

            mode = "save" if k == "w" else "load"

            def done(name: Optional[str]) -> None:
                if not name:
                    return
                if mode == "save":
                    try:
                        q = getattr(self.app, "_dvr_queue", None)
                    except Exception:
                        q = None
                    items: list[Path] = []
                    if isinstance(q, list):
                        items = [p for p in q if isinstance(p, Path)]
                    try:
                        saved = _save_playlist(name=name, items=items)
                        if saved is None:
                            self.app.notify("Playlist save failed")
                        else:
                            self.app.notify(f"Playlist saved: {saved.name}")
                    except Exception as exc:
                        try:
                            self.app.notify(f"Playlist save failed: {exc}")
                        except Exception:
                            pass
                    return

                # load
                try:
                    loaded = _load_playlist(name)
                except Exception:
                    loaded = None
                if not loaded:
                    try:
                        self.app.notify("Playlist not found or empty")
                    except Exception:
                        pass
                    return

                try:
                    self.app._dvr_queue = loaded
                    self.app._dvr_queue_index = 0
                    self.app._dvr_chapters = []
                    self.app._dvr_chapter_index = 0
                    self.app._dvr_shuffle_order = []
                except Exception:
                    pass
                try:
                    _save_dvr_queue([p for p in loaded if isinstance(p, Path)])
                except Exception:
                    pass
                try:
                    self.app.notify(f"Playlist loaded: {name} ({len(loaded)})")
                except Exception:
                    pass
                try:
                    self._load_queue()
                except Exception:
                    pass

            try:
                self.app.push_screen(PlaylistNameScreen(mode=mode), done)
            except Exception:
                try:
                    self.app.notify("Playlist prompt failed")
                except Exception:
                    pass
            return

        # Queue-only actions: edit queue.
        if self._mode == "queue" and k in {"d", "c", "J", "K"}:
            try:
                table = self.query_one("#dvr_table", DataTable)
            except Exception:
                table = None
            table_focused = False
            try:
                table_focused = (focused is table) or isinstance(focused, DataTable) or (getattr(focused, "id", None) == "dvr_table")
            except Exception:
                table_focused = (focused is table) or (getattr(focused, "id", None) == "dvr_table")
            if not table_focused:
                return

            event.stop()
            self._sync_selected_from_cursor()

            def persist(items: list[Path]) -> None:
                try:
                    _save_dvr_queue([p for p in (items or []) if isinstance(p, Path)])
                except Exception:
                    pass

            def set_queue(items: list[Path]) -> None:
                try:
                    self.app._dvr_queue = [p for p in (items or []) if isinstance(p, Path)]
                    self.app._dvr_shuffle_order = []
                except Exception:
                    pass

            def reload_keep(row: int) -> None:
                try:
                    self._load_queue()
                except Exception:
                    return
                try:
                    tbl2 = self.query_one("#dvr_table", DataTable)
                    try:
                        tbl2.cursor_row = int(max(0, row))
                    except Exception:
                        pass
                except Exception:
                    pass
                self._sync_selected_from_cursor()

            try:
                q = list(getattr(self.app, "_dvr_queue", []) or [])
            except Exception:
                q = []
            items0 = [p for p in q if isinstance(p, Path)]
            if not items0:
                return

            # Remove selected from queue.
            if k == "d":
                if self._selected is None:
                    return
                try:
                    idx = items0.index(self._selected)
                except Exception:
                    return
                idx = int(idx)
                try:
                    items0.pop(idx)
                except Exception:
                    return
                set_queue(items0)
                try:
                    cur_i = int(getattr(self.app, "_dvr_queue_index", 0) or 0)
                    if idx < cur_i:
                        self.app._dvr_queue_index = max(0, cur_i - 1)
                    if self.app._dvr_queue_index >= len(items0):
                        self.app._dvr_queue_index = max(0, len(items0) - 1)
                except Exception:
                    pass
                persist(items0)
                reload_keep(min(idx, max(0, len(items0) - 1)))
                return

            # Clear queue (confirm).
            if k == "c":

                async def go_clear() -> None:
                    ok = await self.app.push_screen_wait(ConfirmClearQueueScreen())
                    if not ok:
                        return
                    try:
                        self.app._dvr_queue = []
                        self.app._dvr_queue_index = 0
                        self.app._dvr_shuffle_order = []
                    except Exception:
                        pass
                    persist([])
                    try:
                        self._load_queue()
                    except Exception:
                        pass

                try:
                    self.app.call_after_refresh(go_clear)
                except Exception:
                    pass
                return

            # Reorder (J=down, K=up).
            if self._selected is None:
                return
            try:
                idx = items0.index(self._selected)
            except Exception:
                return
            idx = int(idx)
            new_idx = idx + (1 if k == "J" else -1)
            if new_idx < 0 or new_idx >= len(items0):
                return
            items0[idx], items0[new_idx] = items0[new_idx], items0[idx]
            set_queue(items0)
            try:
                cur_i = int(getattr(self.app, "_dvr_queue_index", 0) or 0)
                if cur_i == idx:
                    self.app._dvr_queue_index = new_idx
                elif cur_i == new_idx:
                    self.app._dvr_queue_index = idx
            except Exception:
                pass
            persist(items0)
            reload_keep(new_idx)
            return

        # Only treat Enter as open/play when the table is focused.
        if k == "enter":
            try:
                table = self.query_one("#dvr_table", DataTable)
            except Exception:
                table = None
            # Ignore Enter while typing in the search box or when an action button is focused.
            if getattr(focused, "id", None) in {"dvr_search", "dvr_open", "dvr_back", "dvr_refresh", "dvr_close"}:
                return
            table_focused = False
            try:
                table_focused = (focused is table) or isinstance(focused, DataTable) or (getattr(focused, "id", None) == "dvr_table")
            except Exception:
                table_focused = (focused is table) or (getattr(focused, "id", None) == "dvr_table")
            if table is not None and table_focused:
                event.stop()
                self._sync_selected_from_cursor()
                self._open_selected()
            return

        # Enqueue / play-folder shortcuts (keyboard-only workflows).
        if k in {"e", "E", "f", "F"}:
            try:
                table = self.query_one("#dvr_table", DataTable)
            except Exception:
                table = None
            table_focused = False
            try:
                table_focused = (focused is table) or isinstance(focused, DataTable) or (getattr(focused, "id", None) == "dvr_table")
            except Exception:
                table_focused = (focused is table) or (getattr(focused, "id", None) == "dvr_table")
            if not table_focused:
                return

            event.stop()
            self._sync_selected_from_cursor()
            target = self._selected
            if target is None:
                try:
                    self.app.notify("No selection")
                except Exception:
                    pass
                return

            # F: play folder (immediate), E: enqueue folder, e: enqueue selected.
            if k in {"f", "F"}:
                try:
                    if target.is_dir():
                        self.app.start_dvr_folder(folder=target, start_at=None)
                        try:
                            self.app._home_player_flash_action("DVR Folder")
                        except Exception:
                            pass
                        return
                except Exception as exc:
                    try:
                        self.app.notify(f"Play folder failed: {exc}")
                    except Exception:
                        pass
                    return

            enqueue_folder = k == "E"
            try:
                self._enqueue_target(target, folder_mode=enqueue_folder)
            except Exception as exc:
                try:
                    self.app.notify(f"Enqueue failed: {exc}")
                except Exception:
                    pass
            return

        # Back navigation shortcuts.
        if k in {"backspace", "escape"}:
            # Don't steal backspace while typing in the search box.
            if getattr(focused, "id", None) == "dvr_search":
                return
            event.stop()
            if self._mode == "queue":
                self._exit_queue_view()
            else:
                self._go_back()
            return

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        bid = event.button.id
        if bid == "dvr_close":
            try:
                getattr(self.app, "_show_right_idle", lambda: None)()
            except Exception:
                pass
            return
        if bid == "dvr_refresh":
            self._reload()
            return
        if bid == "dvr_back":
            self._go_back()
            return
        if bid == "dvr_open":
            self._open_selected()
            return

    def _set_path_label(self) -> None:
        try:
            if self._mode == "queue":
                self.query_one("#dvr_title", Static).update("DVR Queue")
                self.query_one("#dvr_path", Static).update(
                    "Queue  (Enter=Play  x=Shuffle  w=Save  l=Load  d=Remove  c=Clear  J/K=Move  q=Back)"
                )
            elif self._mode == "folders":
                self.query_one("#dvr_title", Static).update("Browse DVR")
                self.query_one("#dvr_path", Static).update(f"Folder: {self.recordings_dir}")
            else:
                self.query_one("#dvr_title", Static).update("Browse DVR")
                self.query_one("#dvr_path", Static).update(f"Folder: {self._folder}")
        except Exception:
            pass

    def _refresh_selected_label(self) -> None:
        try:
            if self._selected is None:
                self.query_one("#dvr_selected", Static).update("Selected: (none)")
            else:
                self.query_one("#dvr_selected", Static).update(f"Selected: {self._selected.name}")
        except Exception:
            pass

    def _reload(self) -> None:
        if self._mode == "queue":
            self._load_queue()
            return
        if self._mode == "folders":
            self._load_folders()
            return
        self._load_files(folder=self._folder)

    def _go_back(self) -> None:
        if self._mode == "files":
            # Go up one directory when nested; only return to the root folder list
            # when we're already at the recordings root.
            cur = None
            try:
                cur = self._folder
            except Exception:
                cur = None

            base = None
            try:
                base = self.recordings_dir.expanduser().resolve(strict=False)
            except Exception:
                base = self.recordings_dir

            parent = None
            try:
                if cur is not None:
                    parent = cur.expanduser().resolve(strict=False).parent
            except Exception:
                try:
                    parent = cur.parent if cur is not None else None
                except Exception:
                    parent = None

            at_root = False
            try:
                if cur is None:
                    at_root = True
                else:
                    at_root = (cur.expanduser().resolve(strict=False) == base)
            except Exception:
                try:
                    at_root = (cur == base)
                except Exception:
                    at_root = True

            if (not at_root) and parent is not None:
                self._folder = parent
                self._selected = None
                self._rows = []
                try:
                    self.query_one("#dvr_search", Input).value = ""
                except Exception:
                    pass
                self._load_files(folder=parent)
                return

            # At root: switch back to top-level folders view.
            self._mode = "folders"
            self._folder = None
            self._selected = None
            self._rows = []
            try:
                self.query_one("#dvr_search", Input).value = ""
            except Exception:
                pass
            self._load_folders()
            return
        try:
            getattr(self.app, "_show_right_idle", lambda: None)()
        except Exception:
            pass

    def _enter_queue_view(self) -> None:
        try:
            self._return_mode = self._mode
            self._return_folder = self._folder
        except Exception:
            self._return_mode = None
            self._return_folder = None

        self._mode = "queue"
        self._folder = None
        try:
            self.query_one("#dvr_search", Input).value = ""
        except Exception:
            pass
        self._query = ""
        self._load_queue()

    def _exit_queue_view(self) -> None:
        m = None
        f = None
        try:
            m = self._return_mode
            f = self._return_folder
        except Exception:
            m = None
            f = None

        if m in {"folders", "files"}:
            self._mode = m
            self._folder = f if m == "files" else None
        else:
            self._mode = "folders"
            self._folder = None

        self._selected = None
        self._rows = []
        try:
            self.query_one("#dvr_search", Input).value = ""
        except Exception:
            pass
        self._query = ""

        if self._mode == "files":
            self._load_files(folder=self._folder)
        else:
            self._load_folders()

    def _load_queue(self) -> None:
        self._set_path_label()

        def ui() -> None:
            q = getattr(self.app, "_dvr_queue", None)
            rows: list[Path] = []
            if isinstance(q, list):
                rows = [p for p in q if isinstance(p, Path)]
            self._rows = rows
            self._selected = rows[0] if rows else None
            self._render_table()
            self._refresh_selected_label()
            self._set_path_label()
            try:
                self.query_one("#dvr_table", DataTable).focus()
            except Exception:
                pass

        try:
            self.app.call_from_thread(ui)
        except Exception:
            ui()

    def _open_selected(self) -> None:
        p = self._selected
        if p is None:
            try:
                self.app.notify("No selection")
            except Exception:
                pass
            return

        if self._mode == "queue":
            # Jump/play selected queue item.
            try:
                q = getattr(self.app, "_dvr_queue", None)
                if isinstance(q, list):
                    for i, x in enumerate(q):
                        try:
                            if x == p:
                                self.app._dvr_queue_index = int(i)
                                break
                        except Exception:
                            continue
                self.app._dvr_chapters = []
                self.app._dvr_chapter_index = 0
                self.app._play_dvr_queue_item(offset_s=0.0, source_label="Queue")
                try:
                    self.app._home_player_flash_action("DVR Queue")
                except Exception:
                    pass
                self._load_queue()
            except Exception as exc:
                try:
                    self.app.notify(f"Queue play failed: {exc}")
                except Exception:
                    pass
            return

        if self._mode == "folders":
            self._mode = "files"
            self._folder = p
            self._selected = None
            self._rows = []
            try:
                self.query_one("#dvr_search", Input).value = ""
            except Exception:
                pass
            self._load_files(folder=p)
            return

        # files mode: drill into folders, or play files
        try:
            if p.is_dir():
                self._folder = p
                self._selected = None
                self._rows = []
                try:
                    self.query_one("#dvr_search", Input).value = ""
                except Exception:
                    pass
                self._load_files(folder=p)
                return
        except Exception:
            pass

        # play selected file
        try:
            self.app.start_dvr_source(source_path=p, offset_s=None)
            try:
                self.app._home_player_flash_action("DVR Play")
            except Exception:
                pass
        except Exception as exc:
            try:
                self.app.notify(f"Play failed: {exc}")
            except Exception:
                pass

    def _enqueue_target(self, p: Path, *, folder_mode: bool) -> None:
        # folder_mode=True means enqueue folder contents; otherwise enqueue the selected item.
        target = p
        if folder_mode:
            if not target.is_dir():
                raise RuntimeError("Selection is not a folder")
            paths = self.app._dvr_build_folder_queue(target)
        else:
            paths = self._expand_enqueue_target(target)

        if not paths:
            raise RuntimeError("Nothing to enqueue")

        # Append to queue without interrupting current playback.
        q = getattr(self.app, "_dvr_queue", None)
        if not isinstance(q, list):
            q = []
        try:
            before = len(q)
        except Exception:
            before = 0

        # De-dupe while preserving order (avoid repeats when enqueueing whole folders).
        def norm_path(x: Path) -> str:
            try:
                # Use non-strict resolve so missing/relative paths don't collapse into errors
                # (which can cause every item to look like the same "duplicate").
                return str(x.expanduser().resolve(strict=False))
            except Exception:
                try:
                    s = str(x)
                    if s:
                        return s
                    return ""
                except Exception:
                    return ""

        existing: set[str]
        try:
            existing = {norm_path(x) for x in q if isinstance(x, Path)}
        except Exception:
            existing = set()

        attempted = 0
        dupes = 0
        for x in paths:
            if not isinstance(x, Path):
                continue
            attempted += 1
            nx = norm_path(x)
            if not nx:
                # Extremely defensive; treat as unique so we don't block enqueues.
                nx = f"__unresolved__:{id(x)}:{str(x)}"
            if nx in existing:
                dupes += 1
                continue
            try:
                q.append(x.expanduser().resolve(strict=False))
            except Exception:
                q.append(x)
            existing.add(nx)

        self.app._dvr_queue = q
        try:
            _save_dvr_queue([x for x in q if isinstance(x, Path)])
        except Exception:
            pass
        try:
            self.app._dvr_shuffle_order = []
        except Exception:
            pass

        added = max(0, len(q) - before)
        total = 0
        try:
            total = len(q)
        except Exception:
            total = 0

        try:
            if added <= 0 and attempted > 0:
                # Common when enqueueing a playlist then enqueueing a track from it.
                msg = "Already queued"
                if dupes:
                    msg = f"Already queued ({dupes} duplicate{'s' if dupes != 1 else ''})"
                self.app.notify(f"{msg}  ·  Queue: {total} item{'s' if total != 1 else ''}")
            else:
                self.app.notify(
                    f"Enqueued {added} item{'s' if added != 1 else ''}  ·  Queue: {total} item{'s' if total != 1 else ''}"
                )
        except Exception:
            pass

    def _expand_enqueue_target(self, p: Path) -> list[Path]:
        # Convert selection into queue items (Paths that can be played directly).
        try:
            if p.is_dir():
                return self.app._dvr_build_folder_queue(p)
        except Exception:
            pass

        suf = ""
        try:
            suf = p.suffix.lower()
        except Exception:
            suf = ""

        if suf in {".m3u", ".m3u8"}:
            try:
                items = _parse_m3u_playlist(p)
                return [x for x in items if isinstance(x, Path)]
            except Exception:
                return []

        if suf in {".cue", ".ffmeta"}:
            # Enqueue the matching audio file (same basename).
            try:
                base = p.with_suffix("")
                if base.exists() and base.is_file():
                    return [base]
            except Exception:
                pass
            return []

        return [p]

    def _load_folders(self) -> None:
        self._set_path_label()

        def work() -> None:
            base = self.recordings_dir.expanduser()
            folders: list[Path] = []
            try:
                base.mkdir(parents=True, exist_ok=True)
                for p in base.iterdir():
                    if p.is_dir() and p.name != ".satstash_tmp":
                        folders.append(p)
                folders.sort(key=lambda x: x.name.lower())
            except Exception:
                folders = []

            def ui() -> None:
                self._rows = folders
                self._selected = folders[0] if folders else None
                self._render_table()
                self._refresh_selected_label()
                self._set_path_label()
                try:
                    self.query_one("#dvr_table", DataTable).focus()
                except Exception:
                    pass

            self.app.call_from_thread(ui)

        self.app.run_worker(work, thread=True, exclusive=False)

    def _load_files(self, *, folder: Optional[Path]) -> None:
        self._set_path_label()

        def work() -> None:
            rows: list[Path] = []
            last_err: Optional[Exception] = None

            def is_candidate_file(p: Path) -> bool:
                try:
                    if p.name.startswith("."):
                        return False
                    if p.name.endswith(".part"):
                        return False
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".txt", ".nfo", ".json", ".log", ".md"}:
                        return False
                    # Allow playlists/metadata helpers explicitly.
                    if p.suffix.lower() in {".m3u", ".m3u8", ".cue", ".ffmeta"}:
                        return True
                    # Default: accept most files (lets mpv/ffplay decide), but avoid common non-media.
                    if p.suffix.lower() in {".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar"}:
                        return False
                    return True
                except Exception:
                    return False
            try:
                f = (folder or self.recordings_dir).expanduser()
                try:
                    if not f.exists() or not f.is_dir():
                        f = self.recordings_dir.expanduser()
                except Exception:
                    f = self.recordings_dir.expanduser()
                # Show subfolders first (common layout: channel/show/date/...)
                dirs: list[Path] = []
                files: list[Path] = []
                for p in f.iterdir():
                    try:
                        if p.name == ".satstash_tmp":
                            continue
                    except Exception:
                        pass
                    if p.is_dir():
                        dirs.append(p)
                        continue
                    if not p.is_file():
                        continue
                    if is_candidate_file(p):
                        files.append(p)
                dirs.sort(key=lambda x: x.name.lower())
                files.sort(key=lambda x: _natural_key(x.name))
                rows = dirs + files
            except Exception as exc:
                last_err = exc
                rows = []

            def ui() -> None:
                self._rows = rows
                self._selected = rows[0] if rows else None
                self._render_table()
                self._refresh_selected_label()
                self._set_path_label()
                if (not rows) and (last_err is not None):
                    try:
                        self.app.notify(f"DVR refresh failed: {last_err}")
                    except Exception:
                        pass
                try:
                    self.query_one("#dvr_table", DataTable).focus()
                except Exception:
                    pass

            self.app.call_from_thread(ui)

        self.app.run_worker(work, thread=True, exclusive=False)

    def _render_table(self) -> None:
        try:
            q = (self._query or "").strip().lower()
            rows = self._rows
            if q:
                rows = [p for p in rows if q in p.name.lower()]
        except Exception:
            rows = list(self._rows)

        # Remember exactly what we're rendering so actions can map cursor row -> Path.
        try:
            self._visible_rows = list(rows)
        except Exception:
            self._visible_rows = []

        try:
            table = self.query_one("#dvr_table", DataTable)
            table.clear()
        except Exception:
            return

        if not rows:
            try:
                table.add_row("(no matches)", "")
            except Exception:
                pass
            try:
                self._visible_rows = []
            except Exception:
                pass
            return

        cur_idx = -1
        try:
            if self._mode == "queue":
                cur_idx = int(getattr(self.app, "_dvr_queue_index", -1) or -1)
        except Exception:
            cur_idx = -1

        for idx, p in enumerate(rows):
            info = ""
            try:
                if self._mode == "queue":
                    try:
                        info = "playing" if idx == cur_idx else ""
                    except Exception:
                        info = ""
                elif self._mode == "folders":
                    # Count recursively; recordings are often nested (show/date/...)
                    n = 0
                    for f in p.rglob("*"):
                        if not f.is_file():
                            continue
                        if ".satstash_tmp" in f.parts:
                            continue
                        if f.name.endswith(".part"):
                            continue
                        if f.name.startswith("."):
                            continue
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".txt", ".nfo", ".json", ".log", ".md"}:
                            continue
                        if f.suffix.lower() in {".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar"}:
                            continue
                        # Treat almost anything else as a playable candidate.
                        n += 1
                    info = f"{n} tracks"
                else:
                    try:
                        if p.is_dir():
                            info = "folder"
                        else:
                            info = p.suffix.lower().lstrip(".")
                    except Exception:
                        info = p.suffix.lower().lstrip(".")
            except Exception:
                info = ""
            try:
                table.add_row(p.name, info, key=str(p))
            except Exception:
                pass

        try:
            table.cursor_coordinate = (0, 0)
        except Exception:
            pass

        # Ensure our internal selection matches the highlighted row.
        self._sync_selected_from_cursor()


class CatchUpScreen(Screen[None]):
    DEFAULT_CSS = """
    CatchUpScreen {
        layout: vertical;
    }
    CatchUpScreen Container {
        layout: vertical;
        height: 1fr;
    }
    CatchUpScreen #export_status {
        background: darkgreen;
        color: white;
        text-style: bold;
        padding: 0 1;
    }
    CatchUpScreen #tracks {
        height: 1fr;
    }
    #actions {
        height: auto;
    }
    """

    def __init__(self, *, channel: Channel):
        super().__init__()
        self._channel = channel
        self._items: list[dict] = []
        self._rows: list[dict] = []
        self._start_key: Optional[str] = None
        self._end_key: Optional[str] = None
        self._export_mode: str = "tracks"  # default per your preference; implemented: single
        self._range_mode: str = "track"  # track | time
        self._minutes: int = 30
        self._loading: bool = False
        self._cancel_export = threading.Event()

    def cancel_export(self) -> None:
        try:
            self._cancel_export.set()
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container():
            label = f"{self._channel.number if self._channel.number is not None else ''} {self._channel.name}".strip()
            yield Static(f"Catch Up: {label}", id="title")
            yield Static("", id="mode")
            yield Static("Start: (none)   End: (none)", id="range")
            yield Static("", id="export_status")
            yield DataTable(id="tracks")
            with Horizontal(id="actions"):
                yield Button("Toggle Range", id="toggle_range")
                yield Button("Toggle Export", id="toggle_export")
                yield Button("Export", id="export", variant="primary")
                yield Button("Refresh", id="refresh")
                yield Button("Back", id="back")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#tracks", DataTable)
        table.cursor_type = "row"
        table.add_columns("", "Time", "Artist", "Title")
        self._refresh_mode_label()
        self._load_items_async()

    def _refresh_mode_label(self) -> None:
        exp = "Individual tracks" if self._export_mode == "tracks" else "Single file"
        rng = "Track range" if self._range_mode == "track" else f"Time range ({self._minutes}m)"
        msg = f"Range: {rng}   Export: {exp}   Keys: S=start  E=end"
        self.query_one("#mode", Static).update(msg)

    def _load_items_async(self) -> None:
        if self._loading:
            return
        self._loading = True
        self.query_one("#tracks", DataTable).clear()
        self.query_one("#tracks", DataTable).add_row("", "", "Loading…", "")

        def work() -> None:
            items: list[dict] = []
            try:
                client = self.app._get_client()
                start_ts = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat().replace("+00:00", "Z")
                data = client.live_update(channel_id=self._channel.id, start_timestamp=start_ts)
                raw = data.get("items") or []
                if isinstance(raw, list):
                    items = [it for it in raw if isinstance(it, dict)]
            except Exception as exc:
                def ui_fail() -> None:
                    self._loading = False
                    self.app.notify(f"Catch-up load failed: {exc}")
                    self.query_one("#tracks", DataTable).clear()
                self.app.call_from_thread(ui_fail)
                return

            def parse_item_dt(item: dict) -> Optional[datetime]:
                try:
                    ts = item.get("timestamp")
                    if isinstance(ts, str) and ts:
                        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
                except Exception:
                    return None
                return None

            parsed: list[tuple[datetime, dict]] = []
            for it in items:
                dt = parse_item_dt(it)
                if dt is None:
                    continue
                parsed.append((dt, it))
            parsed.sort(key=lambda x: x[0])

            # Probe actual DVR buffer start (timeshifted) so the list doesn't show expired tracks.
            buffer_start: Optional[datetime] = None
            try:
                buffer_start = probe_dvr_buffer_start_pdt(
                    client=client,
                    channel_id=self._channel.id,
                    channel_type=getattr(self._channel, "channel_type", "channel-linear") or "channel-linear",
                    preferred_quality=self.app.settings.preferred_quality or "256k",
                    lookback_hours=5,
                )
            except Exception:
                buffer_start = None

            rows: list[dict] = []
            for dt, it in parsed:
                if isinstance(buffer_start, datetime):
                    try:
                        if dt < buffer_start:
                            continue
                    except Exception:
                        pass
                tid = it.get("id")
                if not tid:
                    tid = f"{dt.isoformat()}:{it.get('name') or ''}:{it.get('artistName') or ''}"
                artist = (it.get("artistName") or "").strip()
                title = (it.get("name") or "").strip()

                dur_raw = it.get("duration") or 0
                dur_ms = 0
                try:
                    dv = float(dur_raw)
                    # live_update duration is often in seconds, despite being used like ms elsewhere.
                    # Heuristic:
                    # - if it's a small number (<= 24h), treat as seconds
                    # - if it's huge, assume it's already milliseconds
                    if 0 < dv <= 24 * 60 * 60:
                        dur_ms = int(round(dv * 1000.0))
                    elif dv > 0:
                        dur_ms = int(round(dv))
                except Exception:
                    dur_ms = 0
                rows.append(
                    {
                        "key": str(tid),
                        "dt": dt,
                        "artist": artist,
                        "title": title,
                        "album": (it.get("albumName") or "").strip(),
                        "duration_ms": dur_ms,
                        "raw": it,
                    }
                )

            def ui_ok() -> None:
                self._items = items
                self._rows = rows
                self._loading = False
                self._render_table()

            self.app.call_from_thread(ui_ok)

        self.app.run_worker(work, thread=True, exclusive=True)

    def _render_table(self) -> None:
        table = self.query_one("#tracks", DataTable)
        # Preserve current cursor selection so S/E doesn't jump you back to the top.
        preserve_key: Optional[str] = None
        preserve_row: int = 0
        try:
            preserve_row = int(getattr(table, "cursor_row", 0) or 0)
            ordered = getattr(table, "ordered_rows", [])
            if ordered and 0 <= preserve_row < len(ordered):
                preserve_key = getattr(getattr(ordered[preserve_row], "key", None), "value", None)
                if not isinstance(preserve_key, str) or not preserve_key:
                    preserve_key = None
        except Exception:
            preserve_key = None
            preserve_row = 0

        table.clear()
        if not self._rows:
            table.add_row("", "", "(no items)", "")
            return

        for r in self._rows:
            dt: datetime = r["dt"]
            t = dt.astimezone().strftime("%H:%M:%S")
            mark = ""
            if self._start_key and r["key"] == self._start_key:
                mark = "S"
            if self._end_key and r["key"] == self._end_key:
                mark = (mark + "E") if mark else "E"
            table.add_row(mark, t, r.get("artist") or "", r.get("title") or "", key=r["key"])

        try:
            if preserve_key:
                ordered2 = getattr(table, "ordered_rows", [])
                found = None
                for i, rr in enumerate(ordered2):
                    k = getattr(getattr(rr, "key", None), "value", None)
                    if k == preserve_key:
                        found = i
                        break
                if found is not None:
                    table.cursor_coordinate = (found, 0)
                else:
                    table.cursor_coordinate = (min(max(0, preserve_row), max(0, len(ordered2) - 1)), 0)
            else:
                table.cursor_coordinate = (0, 0)
        except Exception:
            pass

        self._refresh_range_label()

    def _refresh_range_label(self) -> None:
        def fmt_key(k: Optional[str]) -> str:
            if not k:
                return "(none)"
            for r in self._rows:
                if r.get("key") == k:
                    dt: datetime = r["dt"]
                    t = dt.astimezone().strftime("%H:%M:%S")
                    return f"{t} {r.get('artist') or ''} - {r.get('title') or ''}".strip()
            return "(unknown)"

        self.query_one("#range", Static).update(f"Start: {fmt_key(self._start_key)}\nEnd:   {fmt_key(self._end_key)}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "back":
            self.app.pop_screen()
            return
        if event.button.id == "refresh":
            self._load_items_async()
            return
        if event.button.id == "toggle_export":
            self._export_mode = "single" if self._export_mode == "tracks" else "tracks"
            self._refresh_mode_label()
            return
        if event.button.id == "toggle_range":
            self._range_mode = "time" if self._range_mode == "track" else "track"
            self._refresh_mode_label()
            return
        if event.button.id == "export":
            self._export()
            return

    def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        key = getattr(event, "key", "")
        if key in ("s", "S"):
            self._set_marker("start")
            event.stop()
            return
        if key in ("e", "E"):
            self._set_marker("end")
            event.stop()
            return
        if key in ("d", "D"):
            try:
                cur = bool(getattr(self.app, "_debug_enabled", False))
            except Exception:
                cur = False
            try:
                setattr(self.app, "_debug_enabled", (not cur))
            except Exception:
                pass
            try:
                self.app.notify(f"Debug: {'ON' if not cur else 'OFF'}")
            except Exception:
                pass
            event.stop()
            return

    def _set_marker(self, which: str) -> None:
        table = self.query_one("#tracks", DataTable)
        try:
            i = int(getattr(table, "cursor_row", 0) or 0)
            ordered = getattr(table, "ordered_rows", [])
            if not ordered or i < 0 or i >= len(ordered):
                return
            row_key = getattr(getattr(ordered[i], "key", None), "value", None)
            if not isinstance(row_key, str) or not row_key:
                return
        except Exception:
            return

        if which == "start":
            self._start_key = row_key
        else:
            self._end_key = row_key

        self._render_table()

    def _export(self) -> None:
        # Determine export window.
        if self._range_mode == "track":
            if not self._start_key or not self._end_key:
                self.app.notify("Pick start (S) and end (E) tracks")
                return
            start = None
            end = None
            start_idx = None
            end_idx = None
            for i, r in enumerate(self._rows):
                if r.get("key") == self._start_key:
                    start = r
                    start_idx = i
                if r.get("key") == self._end_key:
                    end = r
                    end_idx = i
            if start is None or end is None or start_idx is None or end_idx is None:
                self.app.notify("Invalid start/end selection")
                return
            if start_idx > end_idx:
                start, end = end, start
                start_idx, end_idx = end_idx, start_idx

            # Build selected rows by index so we never export tracks outside the chosen range.
            # Also include one extra "boundary-only" row after the end so bumpers at the end
            # can clamp to the next metadata boundary even though that next track isn't exported.
            selected_rows = self._rows[start_idx : end_idx + 1]
            boundary_row = None
            try:
                if end_idx + 1 < len(self._rows):
                    boundary_row = self._rows[end_idx + 1]
            except Exception:
                boundary_row = None

            start_pdt = start["dt"].astimezone(timezone.utc)
            end_pdt = end["dt"].astimezone(timezone.utc)
            # End boundary must be robust against bogus duration_ms (common for bumpers/IDs).
            # Prefer next metadata boundary when available and only trust duration_ms if sane.
            title = str(end.get("title") or "")
            artist = str(end.get("artist") or "")
            bumper_like = False
            try:
                bumper_like = bool(_is_bumper_like(title=title, artist=artist))
            except Exception:
                bumper_like = False

            cand: Optional[datetime] = None
            try:
                if end_idx + 1 < len(self._rows):
                    cand = self._rows[end_idx + 1]["dt"].astimezone(timezone.utc)
            except Exception:
                cand = None

            gap_s = 0.0
            if isinstance(cand, datetime):
                try:
                    gap_s = float((cand - end_pdt).total_seconds())
                except Exception:
                    gap_s = 0.0

            dur_ms = 0
            try:
                dur_ms = int(end.get("duration_ms") or 0)
            except Exception:
                dur_ms = 0

            dur_s = 0.0
            if dur_ms > 0:
                try:
                    dur_s = float(dur_ms) / 1000.0
                except Exception:
                    dur_s = 0.0

            # Duration is "trusted" only when it is within a sane bound AND doesn't wildly
            # disagree with the next metadata boundary.
            trust_duration = False
            try:
                if dur_s > 0.0 and dur_s <= 30 * 60:
                    trust_duration = True
                if isinstance(cand, datetime) and gap_s > 0.0:
                    # If duration is much larger than the observed boundary gap, don't trust it.
                    if dur_s > gap_s + 10.0:
                        trust_duration = False
            except Exception:
                trust_duration = False

            if bumper_like:
                # Bumpers/IDs should never extend beyond a tight cap.
                cap_end = end_pdt + timedelta(seconds=25.0)
                if isinstance(cand, datetime):
                    cap_end = min(cap_end, cand)
                end_pdt = cap_end
            elif isinstance(cand, datetime) and 0.5 <= gap_s <= 10 * 60:
                # If the next boundary is reasonably close, prefer it.
                end_pdt = cand
            elif trust_duration:
                end_pdt = end_pdt + timedelta(milliseconds=dur_ms)
            else:
                # Last resort: keep it bounded.
                end_pdt = end_pdt + timedelta(minutes=6)
        else:
            minutes = max(1, int(self._minutes or 30))
            start_pdt = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).astimezone(timezone.utc)
            end_pdt = datetime.now(timezone.utc).astimezone(timezone.utc)

            selected_rows = self._rows

        # Build track index for cue/chapter and for individual-track export.
        # Include the boundary-only row if present (export=False) to provide next_dt.
        tracks: list[dict] = []
        rows_for_tracks = list(selected_rows)
        try:
            if boundary_row is not None:
                rows_for_tracks.append(boundary_row)
        except Exception:
            pass

        for r in rows_for_tracks:
            dt = r["dt"].astimezone(timezone.utc)
            if dt < start_pdt or dt > end_pdt:
                continue
            try:
                off = (dt - start_pdt).total_seconds()
            except Exception:
                off = 0.0
            exportable = True
            try:
                if boundary_row is not None and r is boundary_row:
                    exportable = False
            except Exception:
                exportable = True

            tracks.append(
                {
                    "id": r.get("key"),
                    "offset_s": max(0.0, float(off)),
                    "artist": r.get("artist") or "",
                    "title": r.get("title") or "",
                    "album": r.get("album") or "",
                    "dt": dt,
                    "duration_ms": r.get("duration_ms") or 0,
                    "export": exportable,
                    "raw": r.get("raw") or {},
                }
            )

        if self._export_mode == "tracks":
            self._export_tracks_window(start_pdt=start_pdt, end_pdt=end_pdt, tracks=tracks)
            return

        self._export_window(start_pdt=start_pdt, end_pdt=end_pdt, track_index=tracks)

    def _export_window(self, *, start_pdt: datetime, end_pdt: datetime, track_index: list[dict]) -> None:
        # Starting a new export should clear any previous cancel state.
        try:
            self._cancel_export.clear()
        except Exception:
            pass

        label = f"{self._channel.number if self._channel.number is not None else ''} {self._channel.name}".strip() or "channel"
        safe_chan = "".join(c if c.isalnum() or c in " _-." else "_" for c in label).strip() or "channel"
        base_dir = _output_category_dir(self.app.settings, "CatchUp") / safe_chan
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        out_dir = base_dir / ts
        out_dir.mkdir(parents=True, exist_ok=True)

        safe_title = "CatchUp"
        try:
            span_s = max(0.0, float((end_pdt - start_pdt).total_seconds()))
        except Exception:
            span_s = 0.0
        self.app.notify(f"Exporting catch-up: {safe_chan} ({_fmt_time(span_s)})")

        def set_status(msg: str) -> None:
            def ui() -> None:
                try:
                    self.query_one("#export_status", Static).update(str(msg))
                except Exception:
                    pass

            try:
                self.app.call_from_thread(ui)
            except Exception:
                pass

        set_status("Preparing export...")

        def work() -> None:
            handle: Optional[RecordHandle] = None
            try:
                client = self.app._get_client()

                def progress(msg: str) -> None:
                    try:
                        self.app.call_from_thread(lambda: self.app.notify(str(msg)))
                    except Exception:
                        pass
                    try:
                        set_status(str(msg))
                    except Exception:
                        pass

                handle = start_recording(
                    client=client,
                    channel_id=self._channel.id,
                    channel_type=getattr(self._channel, "channel_type", "channel-linear") or "channel-linear",
                    preferred_quality=self.app.settings.preferred_quality or "256k",
                    out_dir=out_dir,
                    title=safe_title,
                    progress=progress,
                    debug=bool(getattr(self.app, "_debug_enabled", False)),
                    start_pdt=start_pdt,
                    end_pdt=end_pdt,
                )

                if self._cancel_export.is_set():
                    out_path = stop_recording_with_options(handle, finalize_on_stop=True, progress=progress)

                    def ui_cancel() -> None:
                        self.app.notify(f"Catch-up canceled: {out_path}")
                        try:
                            self.app.notify(f"ffmpeg log: {handle.log_path}")
                        except Exception:
                            pass

                    self.app.call_from_thread(ui_cancel)
                    return

                # Finite catch-up download: proxy will emit #EXT-X-ENDLIST so ffmpeg can
                # download as fast as possible and exit on its own.
                started_at = time.time()
                last_tick = 0.0
                while True:
                    if self._cancel_export.is_set():
                        out_path = stop_recording_with_options(handle, finalize_on_stop=True, progress=progress)

                        def ui_cancel2() -> None:
                            self.app.notify(f"Catch-up canceled: {out_path}")
                            try:
                                self.app.notify(f"ffmpeg log: {handle.log_path}")
                            except Exception:
                                pass

                        self.app.call_from_thread(ui_cancel2)
                        return

                    rc = handle.process.poll()
                    if rc is not None:
                        break

                    now = time.time()
                    if now - last_tick >= 1.0:
                        last_tick = now
                        elapsed = max(0.0, now - started_at)
                        size_bytes = 0
                        diag = ""
                        diag_match = ""
                        diag_log = ""
                        # Track the largest file size among likely ffmpeg output candidates.
                        try:
                            candidates: list[Path] = []
                            try:
                                candidates.append(handle.tmp_path)
                            except Exception:
                                pass
                            try:
                                candidates.append(handle.final_path)
                            except Exception:
                                pass
                            try:
                                td = out_dir / ".satstash_tmp"
                                if td.exists():
                                    for p in td.iterdir():
                                        candidates.append(p)
                            except Exception:
                                pass
                            try:
                                for p in out_dir.glob("__window-*.m4a"):
                                    candidates.append(p)
                            except Exception:
                                pass

                            best = 0
                            best_path: Optional[Path] = None
                            for p in candidates:
                                try:
                                    if p.exists() and p.is_file():
                                        sz = int(p.stat().st_size)
                                        if sz >= best:
                                            best = sz
                                            best_path = p
                                except Exception:
                                    continue
                            size_bytes = int(best)
                            try:
                                if bool(getattr(self.app, "_debug_enabled", False)):
                                    parts = []
                                    for pp in candidates[:8]:
                                        try:
                                            parts.append(f"{pp.name}={int(pp.stat().st_size) if pp.exists() else 0}B")
                                        except Exception:
                                            parts.append(f"{pp.name}=?")
                                    bp = best_path.name if best_path is not None else "(none)"
                                    diag = f" files[{', '.join(parts)}] best={bp}"
                            except Exception:
                                diag = ""
                        except Exception:
                            size_bytes = 0

                        # If file size stays tiny (common with fragmented MP4), parse ffmpeg log for size=.
                        if size_bytes < 1024:
                            try:
                                lp = getattr(handle, "log_path", None)
                                if lp and Path(lp).exists():
                                    tail = Path(lp).read_text(encoding="utf-8", errors="replace")
                                    b2, m2 = _parse_ffmpeg_size_bytes(tail)
                                    if b2 > size_bytes:
                                        size_bytes = b2
                                    if m2:
                                        diag_match = m2
                                    diag_log = str(lp)
                            except Exception:
                                pass

                        if bool(getattr(self.app, "_debug_enabled", False)):
                            try:
                                extra = []
                                if diag_match:
                                    extra.append(f"size={diag_match}")
                                if diag_log:
                                    extra.append(f"log={Path(diag_log).name}")
                                if extra:
                                    diag = (diag + " " + " ".join(extra)).strip()
                            except Exception:
                                pass

                        if size_bytes >= 1024 * 1024:
                            size_disp = f"{(float(size_bytes) / (1024.0 * 1024.0)):.1f} MB"
                        elif size_bytes >= 1024:
                            size_disp = f"{(float(size_bytes) / 1024.0):.0f} KB"
                        else:
                            size_disp = f"{int(size_bytes)} B"

                        msg = f"Downloading... elapsed {_fmt_time(elapsed)}  ({size_disp})"
                        if diag:
                            msg = msg + "\n" + diag
                        set_status(msg)
                    time.sleep(0.25)

                set_status("Finalizing...")
                out_path = stop_recording_with_options(handle, finalize_on_stop=True, progress=progress)
                cue_path = None
                ffmeta_path = None
                try:
                    if out_path.exists() and out_path.suffix.lower() == ".m4a" and track_index:
                        dur_s2 = _probe_duration_s(out_path)
                        safe_tracks = _sanitize_track_index(tracks=track_index, duration_s=dur_s2)
                        cue_path = _write_cue(audio_path=out_path, tracks=safe_tracks, duration_s=dur_s2)
                        ffmeta_path = _write_ffmetadata(audio_path=out_path, tracks=safe_tracks, duration_s=dur_s2)
                except Exception:
                    cue_path = None
                    ffmeta_path = None

                def ui_done() -> None:
                    try:
                        self.query_one("#export_status", Static).update("")
                    except Exception:
                        pass
                    if out_path == handle.final_path:
                        self.app.notify(f"Catch-up saved: {out_path}")
                    else:
                        self.app.notify(f"Catch-up partial: {out_path}")
                        self.app.notify(f"ffmpeg log: {handle.log_path}")
                    try:
                        if cue_path and cue_path.exists() and cue_path.stat().st_size > 0:
                            self.app.notify(f"CUE: {cue_path}")
                    except Exception:
                        pass
                    try:
                        if ffmeta_path and ffmeta_path.exists() and ffmeta_path.stat().st_size > 0:
                            self.app.notify(f"Chapters: {ffmeta_path}")
                    except Exception:
                        pass

                self.app.call_from_thread(ui_done)
            except Exception as exc:
                def ui_fail() -> None:
                    try:
                        self.query_one("#export_status", Static).update("")
                    except Exception:
                        pass
                    self.app.notify(f"Catch-up export failed: {exc}")
                    try:
                        if handle is not None:
                            self.app.notify(f"ffmpeg log: {handle.log_path}")
                    except Exception:
                        pass
                self.app.call_from_thread(ui_fail)
                return

        self.app.run_worker(work, thread=True, exclusive=True)

    def _export_tracks_window(self, *, start_pdt: datetime, end_pdt: datetime, tracks: list[dict]) -> None:
        try:
            self._cancel_export.clear()
        except Exception:
            pass

        label = f"{self._channel.number if self._channel.number is not None else ''} {self._channel.name}".strip() or "channel"
        safe_chan = "".join(c if c.isalnum() or c in " _-." else "_" for c in label).strip() or "channel"
        base_dir = _output_category_dir(self.app.settings, "CatchUp") / safe_chan
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        out_dir = base_dir / ts
        out_dir.mkdir(parents=True, exist_ok=True)

        ordered: list[dict] = []
        try:
            ordered = [t for t in tracks if isinstance(t, dict) and isinstance(t.get("dt"), datetime)]
            ordered.sort(key=lambda x: x.get("dt"))
        except Exception:
            ordered = list(tracks or [])

        usable = [t for t in ordered if bool(t.get("export", True)) and (t.get("title") or t.get("artist") or "").strip()]
        if not usable:
            self.app.notify("No tracks in selected window")
            return

        def set_status(msg: str) -> None:
            def ui() -> None:
                try:
                    self.query_one("#export_status", Static).update(str(msg))
                except Exception:
                    pass

            try:
                self.app.call_from_thread(ui)
            except Exception:
                pass

        self.app.notify(f"Exporting {len(usable)} tracks: {safe_chan}")
        set_status(f"Preparing {len(usable)} tracks...")

        def work() -> None:
            try:
                client = self.app._get_client()
            except Exception as exc:
                self.app.call_from_thread(lambda: self.app.notify(f"Catch-up export failed: {exc}"))
                return

            buffer_start = probe_dvr_buffer_start_pdt(
                client=client,
                channel_id=self._channel.id,
                channel_type=getattr(self._channel, "channel_type", "channel-linear") or "channel-linear",
                preferred_quality=self.app.settings.preferred_quality or "256k",
                lookback_hours=5,
            )

            if isinstance(buffer_start, datetime):
                try:
                    bs_local = buffer_start.astimezone().strftime("%H:%M:%S")
                except Exception:
                    bs_local = "(unknown)"
                self.app.call_from_thread(lambda: self.app.notify(f"DVR buffer starts at {bs_local}"))

                # If the selected range begins before the buffer start, warn that Option B will skip.
                try:
                    if start_pdt < buffer_start:
                        self.app.call_from_thread(lambda: self.app.notify("Selected range begins before DVR buffer; out-of-buffer tracks will be skipped"))
                except Exception:
                    pass

            saved = 0
            failed = 0
            skipped = 0
            playlist_entries: list[str] = []
            debug_trim: list[dict] = []

            total_exportable = 0
            try:
                total_exportable = sum(1 for tt in ordered if bool(tt.get("export", True)) and (tt.get("title") or tt.get("artist") or "").strip())
            except Exception:
                total_exportable = len(usable)

            out_i = 0
            for i, t in enumerate(ordered):
                if not bool(t.get("export", True)):
                    continue
                if self._cancel_export.is_set():
                    self.app.call_from_thread(lambda: self.app.notify("Catch-up track export canceled"))
                    break

                if not (t.get("title") or t.get("artist") or "").strip():
                    continue

                out_i += 1

                dt = t.get("dt")
                if not isinstance(dt, datetime):
                    continue
                track_start = dt.astimezone(timezone.utc)

                track_end: Optional[datetime] = None
                next_dt: Optional[datetime] = None
                try:
                    if i + 1 < len(ordered):
                        nd = ordered[i + 1].get("dt")
                        if isinstance(nd, datetime):
                            next_dt = nd.astimezone(timezone.utc)
                except Exception:
                    next_dt = None
                try:
                    dur_ms = int(t.get("duration_ms") or 0)
                except Exception:
                    dur_ms = 0
                try:
                    title_s = str(t.get("title") or "")
                    artist_s = str(t.get("artist") or "")
                except Exception:
                    title_s = ""
                    artist_s = ""

                bumper_like = False
                try:
                    bumper_like = bool(_is_bumper_like(title=title_s, artist=artist_s))
                except Exception:
                    bumper_like = False

                # For continuity exports, prefer the next metadata boundary when available.
                if isinstance(next_dt, datetime):
                    track_end = next_dt
                elif dur_ms > 0:
                    track_end = track_start + timedelta(milliseconds=dur_ms)
                else:
                    track_end = track_start + timedelta(seconds=25.0)

                # Strong bumper clamp even if duration_ms is present but wrong.
                if bumper_like:
                    cap_end = track_start + timedelta(seconds=25.0)
                    if isinstance(next_dt, datetime):
                        cap_end = min(cap_end, next_dt)
                    if track_end is None or cap_end < track_end:
                        track_end = cap_end
                else:
                    # Only apply a safety cap when duration_ms is missing/bogus.
                    # If duration_ms is present, trust it (and still clamp to next_dt below).
                    try:
                        if dur_ms <= 0 and track_end is not None and track_end > track_start + timedelta(minutes=30):
                            track_end = track_start + timedelta(minutes=30)
                    except Exception:
                        if dur_ms <= 0:
                            track_end = track_start + timedelta(minutes=30)

                # If the next metadata boundary exists and is earlier than our computed end,
                # clamp to it. This fixes cases where duration_ms is wrong (common for bumpers/IDs).
                try:
                    if isinstance(next_dt, datetime) and track_end is not None and next_dt < track_end:
                        track_end = next_dt
                except Exception:
                    pass

                if track_end > end_pdt:
                    track_end = end_pdt
                if track_end <= track_start:
                    continue

                try:
                    dur_s = float((track_end - track_start).total_seconds())
                except Exception:
                    failed += 1
                    continue
                if dur_s <= 0.01:
                    continue

                if bool(getattr(self.app, "_debug_enabled", False)):
                    try:
                        debug_trim.append(
                            {
                                "i": i,
                                "title": title_s,
                                "artist": artist_s,
                                "duration_ms": dur_ms,
                                "bumper_like": bumper_like,
                                "track_start": track_start.isoformat(),
                                "next_dt": next_dt.isoformat() if isinstance(next_dt, datetime) else None,
                                "track_end": track_end.isoformat() if isinstance(track_end, datetime) else None,
                                "window_start": track_start.isoformat(),
                                "start_off_s": 0.0,
                                "dur_s": dur_s,
                            }
                        )
                    except Exception:
                        pass

                artist = str(t.get("artist") or "").strip()
                title = str(t.get("title") or "").strip()
                album = str(t.get("album") or "").strip()
                base_name = " - ".join([p for p in [_safe_filename(artist), _safe_filename(title)] if p])
                if not base_name:
                    base_name = "Track"
                desired = _unique_path(out_dir / f"{base_name}.m4a")

                try:
                    set_status(f"{out_i}/{max(1, total_exportable)} {base_name}: downloading...")

                    # Skip out-of-buffer tracks.
                    if isinstance(buffer_start, datetime) and track_start < buffer_start:
                        skipped += 1
                        continue

                    handle_t: Optional[RecordHandle] = None
                    try:
                        handle_t = start_recording(
                            client=client,
                            channel_id=self._channel.id,
                            channel_type=getattr(self._channel, "channel_type", "channel-linear") or "channel-linear",
                            preferred_quality=self.app.settings.preferred_quality or "256k",
                            out_dir=out_dir,
                            title=base_name,
                            progress=lambda m, bn=base_name, oi=out_i, te=total_exportable: set_status(f"{oi}/{max(1, te)} {bn}: {m}"),
                            debug=bool(getattr(self.app, "_debug_enabled", False)),
                            start_pdt=track_start,
                            end_pdt=track_end,
                        )

                        started_wall = time.time()
                        last_tick = 0.0
                        while handle_t.process.poll() is None:
                            if self._cancel_export.is_set():
                                break
                            now = time.time()
                            if now - last_tick >= 0.8:
                                last_tick = now
                                try:
                                    elapsed = max(0.0, float(now - started_wall))
                                except Exception:
                                    elapsed = 0.0
                                extra = ""
                                try:
                                    lpdt = handle_t.proxy.playhead_pdt(behind_segments=3)
                                    if isinstance(lpdt, datetime):
                                        try:
                                            off = float((lpdt - track_start).total_seconds())
                                        except Exception:
                                            off = 0.0
                                        extra = f"  seg@+{off:.1f}s"
                                except Exception:
                                    extra = ""
                                set_status(f"{out_i}/{max(1, total_exportable)} {base_name}: downloading... {_fmt_time(elapsed)}{extra}")
                            time.sleep(0.20)

                        out_path = stop_recording_with_options(handle_t, finalize_on_stop=True)
                    finally:
                        if self._cancel_export.is_set() and handle_t is not None:
                            try:
                                stop_recording_with_options(handle_t, finalize_on_stop=True)
                            except Exception:
                                pass

                    if self._cancel_export.is_set():
                        break

                    if not out_path.exists() or out_path.suffix.lower() != ".m4a":
                        raise RuntimeError("track download produced no output")

                    tmp_track = _unique_path(out_dir / f"{base_name}.__tmp.m4a")
                    try:
                        if tmp_track.exists():
                            tmp_track.unlink(missing_ok=True)
                    except Exception:
                        pass
                    out_path.replace(tmp_track)

                    # HLS windows end on segment boundaries, which can include a small tail
                    # past the intended end_pdt. Trim the downloaded m4a to the exact
                    # requested duration to prevent overlaps between consecutive tracks.
                    try:
                        tmp_cut = _unique_path(out_dir / f"{base_name}.__cut.m4a")
                        _ffmpeg_trim_m4a(src=tmp_track, dst=tmp_cut, start_s=0.0, duration_s=dur_s, mode="copy")
                        try:
                            tmp_track.unlink(missing_ok=True)
                        except Exception:
                            pass
                        tmp_cut.replace(tmp_track)
                    except Exception:
                        # If trimming fails, keep the downloaded file; better to have slight
                        # overlap than to fail the export.
                        pass

                    raw = t.get("raw") if isinstance(t.get("raw"), dict) else {}
                    logo_url = None
                    try:
                        if getattr(self._channel, "logo_url", None):
                            logo_url = str(self._channel.logo_url)
                    except Exception:
                        logo_url = None
                    art_url = _extract_art_url_from_live_item(raw or {}, channel_logo_url=logo_url)
                    cover_bytes, cover_type = (b"", "")
                    if art_url:
                        try:
                            cover_bytes, cover_type = _fetch_image_bytes(art_url)
                        except Exception:
                            cover_bytes, cover_type = (b"", "")
                    if not cover_bytes:
                        try:
                            if not _looks_like_show_metadata(title=title, artist=artist):
                                cover_bytes, cover_type = _fetch_itunes_cover_bytes(artist=artist, title=title)
                        except Exception:
                            cover_bytes, cover_type = (b"", "")
                    _tag_m4a(
                        path=tmp_track,
                        title=title,
                        artist=artist,
                        album=album,
                        cover_bytes=cover_bytes,
                        cover_content_type=cover_type,
                    )

                    tmp_fast = _unique_path(out_dir / f"{base_name}.__fast.m4a")
                    _ffmpeg_faststart_copy(src=tmp_track, dst=tmp_fast)

                    try:
                        if desired.exists():
                            desired.unlink(missing_ok=True)
                    except Exception:
                        pass
                    tmp_fast.replace(desired)
                    try:
                        tmp_track.unlink(missing_ok=True)
                    except Exception:
                        pass

                    self.app.call_from_thread(lambda p=desired: self.app.notify(f"Saved: {p.name}"))
                    try:
                        playlist_entries.append(desired.name)
                    except Exception:
                        pass
                    saved += 1
                except Exception as exc:
                    failed += 1
                    try:
                        self.app.call_from_thread(lambda: self.app.notify(f"Track export failed ({base_name}): {exc}"))
                        if handle is not None:
                            self.app.call_from_thread(lambda: self.app.notify(f"ffmpeg log: {handle.log_path}"))
                    except Exception:
                        pass
                    try:
                        for p in out_dir.glob(f"{base_name}.__tmp*.m4a"):
                            p.unlink(missing_ok=True)
                        for p in out_dir.glob(f"{base_name}.__fast*.m4a"):
                            p.unlink(missing_ok=True)
                    except Exception:
                        pass
                    continue

            try:
                if window_path is not None:
                    window_path.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                if aac_window is not None:
                    aac_window.unlink(missing_ok=True)
            except Exception:
                pass

            # Write a playlist for quick continuity checking.
            try:
                if playlist_entries:
                    pl_path = out_dir / "playlist.m3u8"
                    lines: list[str] = ["#EXTM3U"]
                    for name in playlist_entries:
                        lines.append(name)
                    pl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    self.app.call_from_thread(lambda p=pl_path: self.app.notify(f"Playlist: {p.name}"))
            except Exception:
                pass

            # Write debug trim for diagnostics.
            try:
                if bool(getattr(self.app, "_debug_enabled", False)) and debug_trim:
                    dbg = out_dir / "debug-trim.jsonl"
                    dbg.write_text("\n".join(json.dumps(x) for x in debug_trim) + "\n", encoding="utf-8")
                    self.app.call_from_thread(lambda p=dbg: self.app.notify(f"Debug: {p.name}"))
            except Exception:
                pass
            try:
                for p in out_dir.glob("__window-*.m4a"):
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass
            except Exception:
                pass

            def ui_done() -> None:
                try:
                    self.query_one("#export_status", Static).update("")
                except Exception:
                    pass
                if self._cancel_export.is_set():
                    return
                buf = ""
                try:
                    if isinstance(buffer_start, datetime):
                        buf = f" (buffer starts {buffer_start.astimezone().strftime('%H:%M:%S')})"
                except Exception:
                    buf = ""
                parts = [
                    f"saved {saved}",
                    f"skipped {skipped}",
                    f"failed {failed}",
                ]
                self.app.notify("Catch-up tracks: " + ", ".join(parts) + buf)

                # If nothing was saved and nothing failed, the user effectively got "no output".
                if saved == 0 and failed == 0 and skipped > 0:
                    self.app.notify("Catch-up export produced no files (all selected tracks were skipped)")

                # Remove empty output folder to avoid confusion.
                try:
                    if out_dir.exists():
                        any_files = any(out_dir.iterdir())
                        if not any_files:
                            out_dir.rmdir()
                except Exception:
                    pass

            self.app.call_from_thread(ui_done)

        self.app.run_worker(work, thread=True, exclusive=True)


class ConfirmDeleteScreen(Screen[bool]):
    def __init__(self, *, target: Path):
        super().__init__()
        self._target = target

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container():
            yield Static("Confirm Delete", id="title")
            yield Static(f"Delete this file?\n{self._target.name}", id="confirm_text")
            with Horizontal():
                yield Button("Delete", id="confirm_delete", variant="error")
                yield Button("Cancel", id="cancel", variant="primary")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "cancel":
            self.dismiss(False)
            return
        if event.button.id == "confirm_delete":
            self.dismiss(True)
            return


class ConfirmClearQueueScreen(Screen[bool]):
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container():
            yield Static("Confirm", id="title")
            yield Static("Clear the DVR queue?", id="confirm_text")
            with Horizontal():
                yield Button("Clear", id="confirm_clear", variant="error")
                yield Button("Cancel", id="cancel", variant="primary")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "cancel":
            self.dismiss(False)
            return
        if event.button.id == "confirm_clear":
            self.dismiss(True)
            return


class HelpOverlayScreen(Screen[None]):
    def __init__(
        self,
        *,
        title: str,
        sections: dict[str, list[str]],
        active_section: str,
    ):
        super().__init__()
        self._title = str(title or "Help")
        self._sections = {str(k): [str(x) for x in (v or [])] for k, v in (sections or {}).items()}
        self._active = str(active_section or "").strip()

    def _fmt(self, title: str, items: list[str]) -> str:
        # Keep this plain and readable (no Unicode bullets).
        out: list[str] = []
        for it in items:
            out.append(f"- {it}")
        return "\n".join(out).strip() + "\n"

    def _panel(self, key: str, title: str) -> ComposeResult:
        body = self._fmt(title, list(self._sections.get(key, []) or []))
        with Container(id=f"help_panel_{key}", classes="help_panel"):
            yield Static(title, classes="help_panel_title")
            yield Static(body or "(none)", classes="help_panel_body")

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="help_root"):
            with Container(id="help_card"):
                yield Static(self._title, id="help_title")
                with Vertical(id="help_grid"):
                    with Horizontal(classes="help_row"):
                        yield from self._panel("global", "Global")
                        yield from self._panel("home", "Home")
                        yield from self._panel("live", "Live")
                        yield from self._panel("catchup", "Catch Up")
                    with Horizontal(classes="help_row"):
                        yield from self._panel("dvr_browse", "Browse DVR")
                        yield from self._panel("dvr_queue", "DVR Queue")
                        yield from self._panel("playlists", "Playlists")
                        yield from self._panel("schedule", "Schedule")
                    with Horizontal(classes="help_row"):
                        yield from self._panel("record", "Record Now")
                        yield from self._panel("now_playing", "Now Playing")
                        yield from self._panel("dvr_now_playing", "DVR Now Playing")
                        yield from self._panel("misc", "Misc")
                yield Static("? / Esc: close", id="help_hint")
        yield Footer()

    def on_mount(self) -> None:
        try:
            if self._active:
                self.query_one(f"#help_panel_{self._active}").add_class("active")
        except Exception:
            pass

    def on_key(self, event) -> None:
        k = getattr(event, "key", None)
        if k in {"?", "escape"}:
            try:
                event.stop()
            except Exception:
                pass
            try:
                self.app.pop_screen()
            except Exception:
                pass
            return


class PlaylistNameScreen(Screen[Optional[str]]):
    def __init__(self, *, mode: str):
        super().__init__()
        self._mode = str(mode or "").strip().lower() or "save"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container():
            title = "Save playlist" if self._mode == "save" else "Load playlist"
            yield Static(title, id="title")
            yield Static("Playlist name", id="playlist_name_label")
            yield Input(placeholder="e.g. Roadtrip", id="playlist_name")
            with Horizontal():
                yield Button("OK", id="playlist_ok", variant="primary")
                yield Button("Cancel", id="playlist_cancel")
        yield Footer()

    def on_mount(self) -> None:
        try:
            self.query_one("#playlist_name", Input).focus()
        except Exception:
            pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "playlist_name":
            return
        try:
            name = (event.value or "").strip()
        except Exception:
            name = ""
        if not name:
            self.dismiss(None)
            return
        self.dismiss(name)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "playlist_cancel":
            self.dismiss(None)
            return
        if event.button.id == "playlist_ok":
            try:
                name = (self.query_one("#playlist_name", Input).value or "").strip()
            except Exception:
                name = ""
            if not name:
                self.dismiss(None)
                return
            self.dismiss(name)


class PlaylistRenameScreen(Screen[Optional[str]]):
    def __init__(self, *, current: str):
        super().__init__()
        self._current = str(current or "").strip()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container():
            yield Static("Rename playlist", id="title")
            yield Static(f"Current: {self._current}", id="playlist_rename_current")
            yield Static("New name", id="playlist_rename_label")
            yield Input(value=self._current, placeholder="New name", id="playlist_rename")
            with Horizontal():
                yield Button("OK", id="playlist_rename_ok", variant="primary")
                yield Button("Cancel", id="playlist_rename_cancel")
        yield Footer()

    def on_mount(self) -> None:
        try:
            i = self.query_one("#playlist_rename", Input)
            i.focus()
            try:
                i.action_end()
            except Exception:
                pass
        except Exception:
            pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "playlist_rename":
            return
        try:
            name = (event.value or "").strip()
        except Exception:
            name = ""
        if not name:
            self.dismiss(None)
            return
        self.dismiss(name)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "playlist_rename_cancel":
            self.dismiss(None)
            return
        if event.button.id == "playlist_rename_ok":
            try:
                name = (self.query_one("#playlist_rename", Input).value or "").strip()
            except Exception:
                name = ""
            if not name:
                self.dismiss(None)
                return
            self.dismiss(name)


class PlaylistManagerScreen(Screen[None]):
    def __init__(self) -> None:
        super().__init__()
        self._rows: list[Path] = []
        self._selected: Optional[Path] = None
        self._query: str = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="playlist_root"):
            yield Static("Playlists", id="playlist_title")
            yield Static(f"Folder: {_playlists_dir()}", id="playlist_folder")
            yield Static("Keys: Enter Load   a Append   n New-from-Queue   r Rename   d Delete   / Search   Esc Back", id="playlist_keys")
            yield Input(placeholder="Search playlists", id="playlist_search")
            yield Static("Selected: (none)", id="playlist_selected")
            yield DataTable(id="playlist_table")
            with Horizontal(id="playlist_actions"):
                yield Button("Load", id="pl_load", variant="primary")
                yield Button("Append", id="pl_append")
                yield Button("Delete", id="pl_delete", variant="error")
                yield Button("Back", id="pl_back")

            # Bottom status / now-playing strip (mirrors home screen IDs so _refresh_status updates it).
            with Horizontal(id="bottom_bar"):
                with Vertical(id="bottom_left"):
                    yield Static("", id="status")
                    yield Static("", id="recording_status")
                with Horizontal(id="bottom_right"):
                    yield Button("Home", id="pl_home")
                    yield Button("Quit", id="pl_quit")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#playlist_table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Name", "Count", "Modified")
        self._reload()
        try:
            self.query_one("#playlist_search", Input).focus()
        except Exception:
            pass

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "playlist_search":
            return
        self._query = (event.value or "").strip().lower()
        self._render()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        key = getattr(event.row_key, "value", None)
        if isinstance(key, str) and key:
            try:
                self._selected = Path(key)
            except Exception:
                self._selected = None
        self._refresh_selected()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        try:
            event.stop()
        except Exception:
            pass
        self._sync_selected_from_cursor()

    def _sync_selected_from_cursor(self) -> None:
        try:
            table = self.query_one("#playlist_table", DataTable)
        except Exception:
            return
        row_idx = 0
        try:
            cc = getattr(table, "cursor_coordinate", None)
            if cc is not None:
                row_idx = int(getattr(cc, "row", 0) or 0)
            else:
                row_idx = int(getattr(table, "cursor_row", 0) or 0)
        except Exception:
            row_idx = 0
        if not self._rows:
            self._selected = None
            self._refresh_selected()
            return
        row_idx = max(0, min(row_idx, len(self._rows) - 1))
        try:
            self._selected = self._rows[row_idx]
        except Exception:
            self._selected = None
        self._refresh_selected()

    def _refresh_selected(self) -> None:
        if self._selected is None:
            self.query_one("#playlist_selected", Static).update("Selected: (none)")
            return
        self.query_one("#playlist_selected", Static).update(f"Selected: {self._selected.name}")

    def _reload(self) -> None:
        try:
            d = _playlists_dir()
            files = [p for p in d.glob("*.json") if p.is_file()]
        except Exception:
            files = []
        files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
        self._all = files  # type: ignore[attr-defined]
        self._render()

    def _render(self) -> None:
        try:
            all_rows = list(getattr(self, "_all", []) or [])
        except Exception:
            all_rows = []
        q = (self._query or "").strip().lower()
        rows = []
        for p in all_rows:
            if not q or q in p.stem.lower() or q in p.name.lower():
                rows.append(p)
        self._rows = rows
        try:
            table = self.query_one("#playlist_table", DataTable)
            table.clear()
        except Exception:
            return
        for p in rows:
            name = p.stem
            cnt = 0
            try:
                loaded = _load_playlist(name)
                if loaded:
                    cnt = len(loaded)
            except Exception:
                cnt = 0
            mod = ""
            try:
                mod = datetime.fromtimestamp(p.stat().st_mtime).astimezone().strftime("%Y-%m-%d %H:%M")
            except Exception:
                mod = ""
            table.add_row(name, str(cnt), mod, key=str(p))
        try:
            table.focus()
        except Exception:
            pass
        self._sync_selected_from_cursor()

    def _load_into_queue(self, *, append: bool) -> None:
        if self._selected is None:
            try:
                self.app.notify("No playlist selected")
            except Exception:
                pass
            return
        name = self._selected.stem
        try:
            items = _load_playlist(name) or []
        except Exception:
            items = []
        if not items:
            try:
                self.app.notify("Playlist empty")
            except Exception:
                pass
            return
        if append:
            try:
                cur = list(getattr(self.app, "_dvr_queue", []) or [])
            except Exception:
                cur = []
            merged = [p for p in cur if isinstance(p, Path)] + [p for p in items if isinstance(p, Path)]
            self.app._dvr_queue = merged
        else:
            self.app._dvr_queue = [p for p in items if isinstance(p, Path)]
            try:
                self.app._dvr_queue_index = 0
                self.app._dvr_chapters = []
                self.app._dvr_chapter_index = 0
                self.app._dvr_shuffle_order = []
            except Exception:
                pass
        try:
            _save_dvr_queue([p for p in getattr(self.app, "_dvr_queue", []) if isinstance(p, Path)])
        except Exception:
            pass
        try:
            self.app.notify(f"{'Appended' if append else 'Loaded'} playlist: {name} ({len(items)})")
        except Exception:
            pass

    def on_key(self, event) -> None:
        k = getattr(event, "key", None)
        if k in {"escape"}:
            event.stop()
            self.app.pop_screen()
            return
        if k == "/":
            event.stop()
            try:
                self.query_one("#playlist_search", Input).focus()
            except Exception:
                pass
            return
        if k == "R":
            event.stop()
            self._reload()
            return
        if k == "enter":
            event.stop()
            self._load_into_queue(append=False)
            return
        if k == "a":
            event.stop()
            self._load_into_queue(append=True)
            return
        if k == "n":
            event.stop()
            try:
                q = list(getattr(self.app, "_dvr_queue", []) or [])
            except Exception:
                q = []
            items = [p for p in q if isinstance(p, Path)]
            if not items:
                try:
                    self.app.notify("Queue is empty")
                except Exception:
                    pass
                return

            def done(name: Optional[str]) -> None:
                if not name:
                    return
                try:
                    saved = _save_playlist(name=name, items=items)
                except Exception:
                    saved = None
                if saved is None:
                    try:
                        self.app.notify("Playlist save failed")
                    except Exception:
                        pass
                    return
                try:
                    self.app.notify(f"Playlist saved: {saved.name}")
                except Exception:
                    pass
                self._reload()

            try:
                self.app.push_screen(PlaylistNameScreen(mode="save"), done)
            except Exception:
                pass
            return
        if k == "r":
            event.stop()
            if self._selected is None:
                return
            cur = self._selected
            cur_name = cur.stem

            def done2(new_name: Optional[str]) -> None:
                if not new_name:
                    return
                safe = _sanitize_playlist_name(new_name)
                if not safe:
                    return
                dst = _playlists_dir() / f"{safe}.json"
                try:
                    if dst.exists():
                        self.app.notify("That playlist name already exists")
                        return
                except Exception:
                    pass
                try:
                    cur.replace(dst)
                except Exception as exc:
                    try:
                        self.app.notify(f"Rename failed: {exc}")
                    except Exception:
                        pass
                    return
                try:
                    self.app.notify(f"Renamed: {cur_name} -> {safe}")
                except Exception:
                    pass
                self._reload()

            try:
                self.app.push_screen(PlaylistRenameScreen(current=cur_name), done2)
            except Exception:
                pass
            return
        if k == "d":
            event.stop()
            if self._selected is None:
                return
            target = self._selected

            async def go() -> None:
                ok = await self.app.push_screen_wait(ConfirmDeleteScreen(target=target))
                if ok:
                    try:
                        target.unlink(missing_ok=True)
                    except Exception as exc:
                        try:
                            self.app.notify(f"Delete failed: {exc}")
                        except Exception:
                            pass
                    self._reload()

            try:
                self.app.call_after_refresh(go)
            except Exception:
                pass
            return

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        bid = event.button.id
        if bid == "pl_back":
            self.app.pop_screen()
            return
        if bid == "pl_home":
            try:
                self.app.action_home()
            except Exception:
                self.app.pop_screen()
            return
        if bid == "pl_quit":
            try:
                self.app.exit()
            except Exception:
                pass
            return
        if bid == "pl_load":
            self._load_into_queue(append=False)
            return
        if bid == "pl_append":
            self._load_into_queue(append=True)
            return
        if bid == "pl_delete":
            if self._selected is None:
                return
            target = self._selected

            async def go2() -> None:
                ok = await self.app.push_screen_wait(ConfirmDeleteScreen(target=target))
                if ok:
                    try:
                        target.unlink(missing_ok=True)
                    except Exception as exc:
                        try:
                            self.app.notify(f"Delete failed: {exc}")
                        except Exception:
                            pass
                    self._reload()

            try:
                self.app.call_after_refresh(go2)
            except Exception:
                pass
            return


class _SplitRecordShim:
    def __init__(
        self,
        *,
        proxy: HlsProxy,
        proxy_info: object,
        ff: FfmpegRecordHandle,
    ):
        self.proxy = proxy
        self.proxy_info = proxy_info
        self.process = ff.process
        self.tmp_path = ff.tmp_path
        self.final_path = ff.final_path
        self.log_path = ff.log_path


class RecordingsLibraryScreen(Screen[None]):
    def __init__(self, recordings_dir: Path):
        super().__init__()
        self.recordings_dir = recordings_dir
        self._selected: Optional[Path] = None
        self._folders: list[Path] = []
        self._rows: list[Path] = []
        self._query: str = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container():
            yield Static("Recordings", id="title")
            yield Static(f"Folder: {self.recordings_dir}", id="folder")
            yield Input(placeholder="Search folders", id="search_folders")
            yield Static("Selected: (none)", id="selected")
            yield DataTable(id="folders")
            with Horizontal(id="actions"):
                yield Button("Open Folder", id="open", variant="primary")
                yield Button("All Recordings", id="all")
                yield Button("Refresh", id="refresh")
                yield Button("Back", id="back")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#folders", DataTable)
        table.cursor_type = "row"
        table.add_columns("Channel Folder", "Count")
        self.reload()
        try:
            self.query_one("#search_folders", Input).focus()
        except Exception:
            pass

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "search_folders":
            return
        self._query = (event.value or "").strip().lower()
        self._apply_filter_and_render()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "back":
            self.app.pop_screen()
            return
        if event.button.id == "refresh":
            self.reload()
            return
        if event.button.id == "all":
            self.app.push_screen(BrowseDvrScreen(self.recordings_dir.expanduser()))
            return
        if event.button.id == "open":
            if not self._selected:
                self.app.notify("No folder selected")
                return
            self.app.push_screen(BrowseDvrScreen(self._selected))
            return

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        key = getattr(event.row_key, "value", None)
        if isinstance(key, str):
            try:
                self._selected = Path(key)
            except Exception:
                self._selected = None
        self._refresh_selected_label()

    def _apply_filter_and_render(self) -> None:
        try:
            q = (self._query or "").strip().lower()
            if q:
                rows: list[Path] = []
                for p in self._folders:
                    if q in p.name.lower():
                        rows.append(p)
                self._rows = rows
            else:
                self._rows = list(self._folders)

            table = self.query_one("#folders", DataTable)
            table.clear()
            self._selected = None
            self._refresh_selected_label()

            if not self._rows:
                table.add_row("(no matches)", "")
                self.app.notify("No matching folders")
                return

            for p in self._rows:
                try:
                    count = 0
                    for f in p.rglob("*"):
                        if not f.is_file():
                            continue
                        if ".satstash_tmp" in f.parts:
                            continue
                        if f.name.endswith(".part"):
                            continue
                        if f.suffix.lower() in {".ts", ".m4a", ".mp3", ".mp4", ".mkv"}:
                            count += 1
                    count_s = str(count)
                except Exception:
                    count_s = "?"
                table.add_row(p.name, count_s, key=str(p))

            self._selected = self._rows[0]
            table.cursor_coordinate = (0, 0)
            self._refresh_selected_label()
        except Exception:
            pass

    def _refresh_selected_label(self) -> None:
        try:
            if self._selected is not None:
                self.query_one("#selected", Static).update(f"Selected: {self._selected.name}")
            else:
                self.query_one("#selected", Static).update("Selected: (none)")
        except Exception:
            pass

    def reload(self) -> None:
        self.app.notify("Loading folders...")
        self.app.run_worker(self._refresh_worker, thread=True, exclusive=True)

    def _refresh_worker(self) -> None:
        folders: list[Path] = []
        try:
            base = self.recordings_dir.expanduser()
            base.mkdir(parents=True, exist_ok=True)
            for p in base.iterdir():
                if p.is_dir():
                    if p.name == ".satstash_tmp":
                        continue
                    folders.append(p)
            folders.sort(key=lambda x: x.name.lower())
        except Exception:
            folders = []

        def apply() -> None:
            self._folders = folders
            if not folders:
                table = self.query_one("#folders", DataTable)
                table.clear()
                self._rows = []
                self._selected = None
                self._refresh_selected_label()
                table.add_row("(no channel folders yet)", "")
                self.app.notify("No channel folders found")
                return

            self.app.notify(f"Loaded {len(folders)} folders")
            self._apply_filter_and_render()

        self.app.call_from_thread(apply)


class BrowseDvrScreen(Screen[None]):
    DEFAULT_CSS = """
    BrowseDvrScreen {
        layout: vertical;
    }
    BrowseDvrScreen Container {
        layout: vertical;
        height: 1fr;
    }
    BrowseDvrScreen #files {
        height: 1fr;
    }
    BrowseDvrScreen #actions {
        height: auto;
    }
    """

    def __init__(self, recordings_dir: Path):
        super().__init__()
        self.recordings_dir = recordings_dir
        self._selected: Optional[Path] = None
        self._all_files: list[Path] = []
        self._rows: list[Path] = []
        self._last_empty: Optional[bool] = None
        self._query: str = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container():
            yield Static("Browse DVR", id="title")
            yield Static(f"Folder: {self.recordings_dir}", id="folder")
            yield Input(placeholder="Search recordings (name/path)", id="search_recordings")
            yield Static("Selected: (none)", id="selected")
            yield DataTable(id="files")
            with Horizontal(id="actions"):
                yield Button("Play", id="play", variant="primary")
                yield Button("Play Folder", id="play_folder")
                yield Button("Shuffle", id="shuffle")
                yield Button("Repeat", id="repeat")
                yield Button("Pause/Resume", id="pause")
                yield Button("Stop", id="stop")
                yield Button("Jump", id="jump")
                yield Button("Delete", id="delete")
                yield Button("Open Folder", id="open_folder")
                yield Button("Refresh", id="refresh")
                yield Button("Back", id="back")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#files", DataTable)
        table.cursor_type = "row"
        table.add_columns("Name", "Size", "Modified")
        self.reload()
        self._refresh_selected_label()
        try:
            self.query_one("#search_recordings", Input).focus()
        except Exception:
            pass

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "search_recordings":
            return
        self._query = (event.value or "").strip().lower()
        self._apply_filter_and_render()

    def _open_folder(self) -> None:
        p = self.recordings_dir.expanduser()
        self.app.notify(f"Opening folder: {p}")

        def work() -> None:
            try:
                import subprocess

                subprocess.Popen(["xdg-open", str(p)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as exc:
                self.app.call_from_thread(lambda: self.app.notify(f"Open folder failed: {exc}"))

        self.app.run_worker(work, thread=True, exclusive=False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "back":
            self.app.pop_screen()
            return
        if event.button.id == "refresh":
            self.reload()
            return
        if event.button.id == "play":
            self._play_selected()
            return
        if event.button.id == "play_folder":
            self._play_folder()
            return
        if event.button.id == "shuffle":
            try:
                self.app.toggle_dvr_shuffle()
            except Exception:
                pass
            self._refresh_selected_label()
            return
        if event.button.id == "repeat":
            try:
                self.app.cycle_dvr_repeat()
            except Exception:
                pass
            self._refresh_selected_label()
            return
        if event.button.id == "pause":
            try:
                self.app.toggle_dvr_pause()
            except Exception:
                pass
            return
        if event.button.id == "stop":
            try:
                self.app.stop_dvr_playback()
            except Exception:
                pass
            return
        if event.button.id == "jump":
            self._jump_selected()
            return
        if event.button.id == "delete":
            self._delete_selected()
            return
        if event.button.id == "open_folder":
            self._open_folder()
            return

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        key = getattr(event.row_key, "value", None)
        if isinstance(key, str):
            try:
                self._selected = Path(key)
            except Exception:
                self._selected = None
        self._refresh_selected_label()

    def _refresh_selected_label(self) -> None:
        try:
            if self._selected is not None:
                self.query_one("#selected", Static).update(f"Selected: {self._selected.name}")
            else:
                self.query_one("#selected", Static).update("Selected: (none)")

            can_act = self._selected is not None and self._selected.exists() and self._selected.is_file()
            try:
                self.query_one("#play", Button).disabled = not can_act
            except Exception:
                pass
            try:
                self.query_one("#play_folder", Button).disabled = self._selected is None
            except Exception:
                pass
            try:
                sh = bool(getattr(self.app, "_dvr_shuffle", False))
                self.query_one("#shuffle", Button).label = "Shuffle: ON" if sh else "Shuffle: OFF"
            except Exception:
                pass
            try:
                rep = str(getattr(self.app, "_dvr_repeat", "off") or "off")
                rep = rep.lower().strip()
                if rep not in {"off", "one", "all"}:
                    rep = "off"
                self.query_one("#repeat", Button).label = f"Repeat: {rep.upper()}"
            except Exception:
                pass
            try:
                cue_ok = False
                if can_act and self._selected is not None:
                    suf = self._selected.suffix.lower()
                    if suf in {".cue", ".ffmeta"}:
                        cue_ok = True
                    else:
                        cue_ok = self._selected.with_suffix(self._selected.suffix + ".cue").exists() or self._selected.with_suffix(
                            self._selected.suffix + ".ffmeta"
                        ).exists()
                self.query_one("#jump", Button).disabled = not cue_ok
            except Exception:
                pass
            try:
                self.query_one("#delete", Button).disabled = not can_act
            except Exception:
                pass
        except Exception:
            pass

    def _play_folder(self) -> None:
        p = self._selected
        if p is None:
            self.app.notify("No selection")
            return
        folder = None
        try:
            if p.exists() and p.is_dir():
                folder = p
            else:
                folder = p.parent
        except Exception:
            folder = None
        if folder is None:
            self.app.notify("Invalid folder")
            return
        try:
            start_at = None
            try:
                if p.exists() and p.is_file():
                    start_at = p
            except Exception:
                start_at = None
            self.app.start_dvr_folder(folder=folder, start_at=start_at)
            try:
                if not isinstance(self.app.screen, DvrNowPlayingScreen):
                    self.app.push_screen(DvrNowPlayingScreen())
            except Exception:
                pass
        except Exception as exc:
            self.app.notify(f"Play folder failed: {exc}")

    def _jump_selected(self) -> None:
        p = self._selected
        if not p or not p.exists():
            self.app.notify("No recording selected")
            return
        meta_path = p.with_suffix(p.suffix + ".ffmeta")
        cue_path = p.with_suffix(p.suffix + ".cue")
        tracks: list[dict] = []
        if meta_path.exists():
            tracks = _parse_ffmeta_chapters(meta_path)
        if (not tracks) and cue_path.exists():
            tracks = _parse_cue_tracks(cue_path)
        if not tracks:
            self.app.notify("No track index found (.ffmeta/.cue)")
            return

        def done(offset_s: Optional[float]) -> None:
            if offset_s is None:
                return
            self._play_selected_with_offset(offset_s=offset_s)

        self.app.push_screen(CueJumpScreen(audio_path=p, cue_path=meta_path if meta_path.exists() else cue_path, tracks=tracks), done)

    def reload(self) -> None:
        self.app.notify("Loading recordings...")
        self.app.run_worker(self._refresh_worker, thread=True, exclusive=True)

    def _apply_filter_and_render(self) -> None:
        try:
            q = (self._query or "").strip().lower()
            if q:
                filtered: list[Path] = []
                base = self.recordings_dir.expanduser()
                for p in self._all_files:
                    try:
                        rel = str(p.relative_to(base))
                    except Exception:
                        rel = p.name
                    if q in rel.lower():
                        filtered.append(p)
                self._rows = filtered
            else:
                self._rows = list(self._all_files)

            table = self.query_one("#files", DataTable)
            table.clear()

            if not self._rows:
                table.add_row("(none)", "", "")
                self._selected = None
                self._refresh_selected_label()
                if self._last_empty is not True:
                    self.app.notify("No recordings found")
                self._last_empty = True
                return

            base = self.recordings_dir.expanduser()
            for p in self._rows:
                try:
                    st = p.stat()
                    size = st.st_size
                    if size > 1024 * 1024:
                        size_s = f"{size / (1024 * 1024):.1f} MB"
                    elif size > 1024:
                        size_s = f"{size / 1024:.1f} KB"
                    else:
                        size_s = f"{size} B"
                    mod = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
                except Exception:
                    size_s = "?"
                    mod = "?"

                try:
                    name = str(p.relative_to(base))
                except Exception:
                    name = p.name
                table.add_row(name, size_s, mod, key=str(p))

            self._selected = self._rows[0]
            table.cursor_coordinate = (0, 0)
            self._refresh_selected_label()
            self._last_empty = False

            if q:
                self.app.notify(f"Showing {len(self._rows)} matches")
            else:
                self.app.notify(f"Loaded {len(self._rows)} recordings")
        except Exception:
            pass

    def _refresh_worker(self) -> None:
        files: list[Path] = []
        try:
            d = self.recordings_dir.expanduser()
            d.mkdir(parents=True, exist_ok=True)
            for p in d.rglob("*"):
                if not p.is_file():
                    continue
                if ".satstash_tmp" in p.parts:
                    continue
                if p.name.endswith(".part"):
                    continue
                if p.suffix.lower() in {".ts", ".m4a", ".mp3", ".mp4", ".mkv", ".m3u", ".m3u8", ".cue", ".ffmeta"}:
                    files.append(p)
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        except Exception:
            files = []

        def apply() -> None:
            self._all_files = files
            self._apply_filter_and_render()

        self.app.call_from_thread(apply)

    def _play_selected(self) -> None:
        self._play_selected_with_offset(offset_s=None)

    def _play_selected_with_offset(self, *, offset_s: Optional[float]) -> None:
        p = self._selected
        if not p:
            self.app.notify("No recording selected")
            return

        try:
            self.app.start_dvr_source(source_path=p, offset_s=offset_s)
            try:
                if not isinstance(self.app.screen, DvrNowPlayingScreen):
                    self.app.push_screen(DvrNowPlayingScreen())
            except Exception:
                pass
        except Exception as exc:
            self.app.notify(f"Play failed: {exc}")

    def _delete_selected(self) -> None:
        p = self._selected
        if not p:
            self.app.notify("No recording selected")
            return

        def done(ok: bool) -> None:
            if not ok:
                self.app.notify("Delete cancelled")
                return
            try:
                p.unlink()
                self.app.notify(f"Deleted: {p.name}")
            except Exception as exc:
                self.app.notify(f"Delete failed: {exc}")
                return
            self.reload()

        self.app.push_screen(ConfirmDeleteScreen(target=p), done)


class CueJumpScreen(Screen[Optional[float]]):
    def __init__(self, *, audio_path: Path, cue_path: Path, tracks: list[dict]):
        super().__init__()
        self._audio_path = audio_path
        self._cue_path = cue_path
        self._tracks = tracks
        self._selected_idx: Optional[int] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container():
            yield Static("Jump to Track", id="title")
            yield Static(f"File: {self._audio_path.name}", id="subtitle")
            yield DataTable(id="tracks")
            with Horizontal(id="actions"):
                yield Button("Play", id="play", variant="primary")
                yield Button("Cancel", id="cancel")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#tracks", DataTable)
        table.cursor_type = "row"
        table.add_columns("#", "Time", "Track")
        for idx, t in enumerate(self._tracks):
            try:
                off = float(t.get("offset_s") or 0.0)
            except Exception:
                off = 0.0
            label = " - ".join([x for x in [str(t.get("artist") or "").strip(), str(t.get("title") or "").strip()] if x]).strip()
            if not label:
                label = str(t.get("title") or t.get("track") or "")
            table.add_row(str(idx + 1), _fmt_time(off), label, key=str(idx))
        if self._tracks:
            self._selected_idx = 0
            table.cursor_coordinate = (0, 0)
        try:
            self.query_one("#play", Button).disabled = self._selected_idx is None
        except Exception:
            pass

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        key = getattr(event.row_key, "value", None)
        if isinstance(key, str) and key.isdigit():
            self._selected_idx = int(key)
        try:
            self.query_one("#play", Button).disabled = self._selected_idx is None
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        if event.button.id != "play":
            return
        idx = self._selected_idx
        if idx is None or idx < 0 or idx >= len(self._tracks):
            self.dismiss(None)
            return
        try:
            off = float(self._tracks[idx].get("offset_s") or 0.0)
        except Exception:
            off = 0.0
        self.dismiss(off)


class DvrNowPlayingScreen(Screen[None]):
    DEFAULT_CSS = """
    DvrNowPlayingScreen {
        layout: vertical;
    }
    DvrNowPlayingScreen #main {
        layout: horizontal;
        height: 1fr;
    }
    DvrNowPlayingScreen #art {
        width: 7fr;
        height: 1fr;
        overflow: hidden;
    }
    DvrNowPlayingScreen #right {
        layout: vertical;
        height: 1fr;
        width: 3fr;
    }
    DvrNowPlayingScreen #meta {
        layout: vertical;
        height: 1fr;
    }
    DvrNowPlayingScreen #controls {
        height: auto;
    }
    DvrNowPlayingScreen #controls_row1, DvrNowPlayingScreen #controls_row2 {
        height: auto;
    }
    DvrNowPlayingScreen #progress {
        width: 1fr;
    }
    DvrNowPlayingScreen #shortcuts {
        height: auto;
        color: #7fa88a;
    }
    """

    def __init__(self):
        super().__init__()
        self._poll = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container():
            yield Static("DVR Playback", id="title")
            yield Static("", id="source")
            with Horizontal(id="main"):
                yield Static("", id="art")
                with Vertical(id="right"):
                    with Vertical(id="meta"):
                        yield Static("", id="track")
                        yield Static("", id="album")
                        yield ProgressBar(total=1, show_eta=False, id="progress")
                        yield Static("", id="timing")
                        yield Static("Keys: space Play/Pause   n Next   b Prev   [ ] Seek   s Stop   p Playlists   h Home   S Schedule", id="shortcuts")
                    with Vertical(id="controls"):
                        with Horizontal(id="controls_row1"):
                            yield Button("Prev", id="prev")
                            yield Button("Play/Pause", id="play_pause", variant="primary")
                            yield Button("Next", id="next")
                            yield Button("Shuffle", id="shuffle")
                            yield Button("Repeat", id="repeat")
                        with Horizontal(id="controls_row2"):
                            yield Button("Pause/Resume", id="pause")
                            yield Button("Stop", id="stop", variant="primary")
                            yield Button("Home", id="home")
                            yield Button("Back", id="back")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh()
        self._poll = self.set_interval(1.0, self._refresh)

    def on_unmount(self) -> None:
        try:
            if self._poll is not None:
                self._poll.stop()
        except Exception:
            pass

    def _refresh(self) -> None:
        try:
            aw = 0
            ah = 0
            try:
                art = self.query_one("#art", Static)
                aw = int(getattr(getattr(art, "size", None), "width", 0) or 0)
                ah = int(getattr(getattr(art, "size", None), "height", 0) or 0)
            except Exception:
                aw = 0
                ah = 0
            info = self.app.get_dvr_now_playing(art_width=aw, art_height=ah)
        except Exception:
            info = {}
        try:
            src = str(info.get("source") or "").strip()
            self.query_one("#source", Static).update(src)
        except Exception:
            pass
        try:
            sh = bool(getattr(self.app, "_dvr_shuffle", False))
            self.query_one("#shuffle", Button).label = "Shuffle: ON" if sh else "Shuffle: OFF"
        except Exception:
            pass
        try:
            rep = str(getattr(self.app, "_dvr_repeat", "off") or "off").lower().strip()
            if rep not in {"off", "one", "all"}:
                rep = "off"
            self.query_one("#repeat", Button).label = f"Repeat: {rep.upper()}"
        except Exception:
            pass
        try:
            track = str(info.get("track") or "").strip() or "(no metadata)"
            album = str(info.get("album") or "").strip()
            self.query_one("#track", Static).update(track)
            self.query_one("#album", Static).update(album)
        except Exception:
            pass

        try:
            pb = self.query_one("#progress", ProgressBar)
            total = float(info.get("duration_s") or 0.0)
            pos = float(info.get("position_s") or 0.0)
            if total <= 0:
                total = 1.0
            pos = max(0.0, min(pos, total))
            pb.update(total=total, progress=pos)
            self.query_one("#timing", Static).update(f"{_fmt_time(pos)} / {_fmt_time(total)}")
        except Exception:
            pass

        try:
            art = info.get("art")
            if art is not None:
                self.query_one("#art", Static).update(art)
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        bid = event.button.id
        if bid == "back":
            self.app.pop_screen()
            return
        if bid == "home":
            try:
                self.app.action_home()
            except Exception:
                pass
            return
        if bid == "stop":
            try:
                self.app.stop_dvr_playback()
            except Exception:
                pass
            return
        if bid == "pause":
            try:
                self.app.toggle_dvr_pause()
            except Exception:
                pass
            return
        if bid == "shuffle":
            try:
                self.app.toggle_dvr_shuffle()
            except Exception:
                pass
            self._refresh()
            return
        if bid == "repeat":
            try:
                self.app.cycle_dvr_repeat()
            except Exception:
                pass
            self._refresh()
            return
        if bid == "next":
            try:
                self.app.dvr_next()
            except Exception as exc:
                self.app.notify(f"Next failed: {exc}")
            return
        if bid == "prev":
            try:
                self.app.dvr_prev()
            except Exception as exc:
                self.app.notify(f"Prev failed: {exc}")
            return


class SatStashApp(App[None]):
    BINDINGS = [
        ("Q", "quit", "Quit"),
        ("h", "home", "Home"),
        ("?", "help", "Help"),
        ("space", "play_pause", "Play/Pause"),
        ("s", "stop_playback", "Stop"),
        ("n", "next_track", "Next"),
        ("b", "prev_track", "Prev"),
        ("p", "playlists", "Playlists"),
        ("[", "seek_back", "Seek -10s"),
        ("]", "seek_forward", "Seek +10s"),
        ("S", "schedule", "Schedule"),
    ]

    CSS = """
    Screen {
        background: #0b0d0c;
        color: #d6d6d6;

        scrollbar-background: #0f1211;
        scrollbar-color: #16783a;
        scrollbar-color-hover: #1d8f45;
        scrollbar-color-active: #28b35a;
    }

    /* Textual scrollbar widgets differ by version; cover the common cases. */
    ScrollBar, VerticalScrollBar, HorizontalScrollBar {
        background: #0f1211;
        color: #16783a;
    }

    ScrollBar:hover, VerticalScrollBar:hover, HorizontalScrollBar:hover {
        color: #1d8f45;
    }

    /* Thumb/track classes used by newer Textual builds. */
    ScrollBar > .scrollbar--thumb, VerticalScrollBar > .scrollbar--thumb, HorizontalScrollBar > .scrollbar--thumb {
        background: #16783a;
    }

    ScrollBar > .scrollbar--thumb:hover, VerticalScrollBar > .scrollbar--thumb:hover, HorizontalScrollBar > .scrollbar--thumb:hover {
        background: #1d8f45;
    }

    /* NOTE: Textual CSS doesn't support :active; hover/focus-only. */

    Input {
        background: #0f1211;
        border: solid #2a2f2e;
        color: #e6e6e6;
    }

    Input:focus {
        border: solid #16783a;
    }

    DataTable {
        background: #0f1211;
        color: #d6d6d6;

        scrollbar-background: #0f1211;
        scrollbar-color: #16783a;
        scrollbar-color-hover: #1d8f45;
        scrollbar-color-active: #28b35a;
    }

    DataTable ScrollBar, DataTable VerticalScrollBar, DataTable HorizontalScrollBar {
        background: #0f1211;
        color: #16783a;
    }

    DataTable ScrollBar:hover, DataTable VerticalScrollBar:hover, DataTable HorizontalScrollBar:hover {
        color: #1d8f45;
    }

    DataTable ScrollBar > .scrollbar--thumb, DataTable VerticalScrollBar > .scrollbar--thumb, DataTable HorizontalScrollBar > .scrollbar--thumb {
        background: #16783a;
    }

    DataTable ScrollBar > .scrollbar--thumb:hover, DataTable VerticalScrollBar > .scrollbar--thumb:hover, DataTable HorizontalScrollBar > .scrollbar--thumb:hover {
        background: #1d8f45;
    }

    DataTable > .datatable--header {
        background: #121817;
        color: #e6e6e6;
        text-style: bold;
    }

    DataTable > .datatable--cursor {
        background: #0f5a2a;
        color: #ffffff;
    }

    DataTable > .datatable--highlight {
        background: #153a24;
        color: #ffffff;
    }

    Button {
        background: #1f2322;
        color: #e6e6e6;
        border: solid #2d3231;
    }

    Button.-primary {
        background: #1f2322;
        border: solid #3a4a46;
        color: #ffffff;
    }

    Button:focus {
        background: #0f5a2a;
        border: solid #16783a;
        color: #ffffff;
    }

    #main_menu_root {
        layout: vertical;
        height: 1fr;
    }

    #menu_center {
        layout: vertical;
        height: 1fr;
        align: left top;
    }

    #home_body {
        layout: horizontal;
        height: 1fr;
        width: 1fr;
    }

    #home_player {
        width: 1fr;
        background: #0f1211;
        border: solid #2a2f2e;
        padding: 1 2;
        margin: 0 2 0 0;
    }

    #right_idle {
        layout: vertical;
        height: 1fr;
        width: 1fr;
        content-align: center middle;
        overflow: hidden;
    }

    #right_idle_logo {
        height: 1fr;
        width: 1fr;
        content-align: center middle;
        overflow: hidden;
    }

    #home_player_art {
        height: 1fr;
        content-align: center middle;
        overflow: hidden;
        margin: 0;
    }

    #home_player_title {
        content-align: left middle;
        text-style: bold;
        margin: 0 0 1 0;
    }

    #home_player_track {
        text-style: bold;
    }

    #home_player_progress {
        width: 1fr;
        margin: 1 0 0 0;
    }

    #home_player_timing {
        color: #9fb7a8;
    }

    #home_player_action {
        color: #28b35a;
        text-style: bold;
        margin: 0 0 0 0;
        height: 1;
    }

    #home_player_hint {
        color: #7fa88a;
        margin: 1 0 0 0;
    }

    #right_content {
        height: 1fr;
        min-height: 18;
        width: 1fr;
        margin: 0;
        background: #0f1211;
        border: solid #2a2f2e;
        padding: 1 2;
        overflow: hidden;
    }

    #home_menu_row {
        layout: horizontal;
        height: auto;
        background: #0f1211;
        border-top: solid #2a2f2e;
        padding: 0 1;
        overflow-x: auto;
    }

    #home_menu_row Button {
        width: auto;
        min-width: 12;
        margin: 0 1 0 0;
    }

    #live_pane {
        layout: vertical;
        height: 1fr;
    }

    #live_channels {
        height: 1fr;
        min-height: 8;
    }

    #live_actions {
        height: auto;
    }

    #dvr_pane {
        layout: vertical;
        height: 1fr;
    }

    #help_root {
        layout: vertical;
        height: 1fr;
        width: 1fr;
        align: center middle;
        background: #0b0d0c;
    }

    #help_card {
        width: 90%;
        max-width: 120;
        background: #0f1211;
        border: heavy #2a2f2e;
        padding: 1 2;
    }

    #help_title {
        text-style: bold;
        content-align: center middle;
        margin: 0 0 1 0;
        color: #e6e6e6;
    }

    #help_grid {
        layout: vertical;
        height: auto;
    }

    .help_row {
        layout: horizontal;
        height: auto;
    }

    .help_panel {
        width: 1fr;
        min-height: 10;
        background: #0b0d0c;
        border: solid #2a2f2e;
        padding: 1 2;
        margin: 0 1 1 0;
    }

    .help_panel_title {
        text-style: bold;
        color: #28b35a;
        margin: 0 0 1 0;
        content-align: left middle;
        height: 1;
    }

    .help_panel_body {
        color: #d6d6d6;
    }

    .help_panel.active {
        border: heavy #28b35a;
        background: #0f1211;
    }

    #help_hint {
        content-align: center middle;
        margin: 1 0 0 0;
        color: #7fa88a;
    }

    #dvr_table {
        height: 1fr;
        min-height: 8;
    }

    #dvr_actions {
        height: auto;
    }

    #menu_left, #menu_right {
        layout: vertical;
        width: 1fr;
    }

    #menu_left Button, #menu_right Button {
        width: 1fr;
        margin: 0 0 1 0;
    }

    #bottom_bar {
        height: auto;
        max-height: 3;
        background: #102015;
        border-top: solid #1d3a26;
        padding: 0 1;
    }

    Footer {
        background: #102015;
        color: #d6d6d6;
        border-top: solid #1d3a26;
    }

    #bottom_left {
        width: 1fr;
        height: auto;
        overflow: hidden;
    }

    #bottom_right {
        width: 38;
        layout: horizontal;
        align: right middle;
        height: auto;
    }

    #bottom_right Button {
        width: 12;
        min-width: 12;
        padding: 0 0;
        margin: 0 0 0 1;
        height: auto;
    }

    #status, #subtitle, #subtitle_player, #recording_status {
        color: #d6d6d6;
        overflow: hidden;
    }

    #recording_status {
        color: #9fd3aa;
    }
    """

    def __init__(self, *, debug: bool = False):
        super().__init__()
        self._debug_enabled = debug
        self.settings = load_settings()
        self._client: Optional[SxmClient] = None
        self._live_handle: Optional[LivePlaybackHandle] = None
        self._live_channel: Optional[Channel] = None
        self._live_paused: bool = False
        self._record_handle: Optional[RecordHandle] = None
        self._record_channel: Optional[Channel] = None
        self._record_started_at: Optional[float] = None
        self._record_pending: bool = False
        self._record_mode: str = "single"
        self._record_stop: Optional[threading.Event] = None
        self._record_split_proxy: Optional[tuple[HlsProxy, object]] = None
        self._record_single_started_wall: Optional[datetime] = None
        self._record_single_track_index: list[dict] = []
        self._record_single_track_lock = threading.Lock()
        self._auth_lock = threading.Lock()

        self._dvr_player_proc: Optional[subprocess.Popen] = None
        self._dvr_player_kind: Optional[str] = None
        self._dvr_player_paused: bool = False
        self._dvr_mpv_ipc_path: Optional[Path] = None

        try:
            self._dvr_queue = _load_dvr_queue()
        except Exception:
            self._dvr_queue = []
        self._dvr_queue_index: int = 0
        self._dvr_shuffle: bool = False
        self._dvr_repeat: str = "off"
        self._dvr_shuffle_order: list[int] = []
        self._dvr_advancing: bool = False
        self._dvr_chapters: list[dict] = []
        self._dvr_chapter_index: int = 0
        self._dvr_now_source: str = ""
        self._dvr_now_track: str = ""
        self._dvr_now_album: str = ""
        self._dvr_now_path: Optional[Path] = None
        self._dvr_now_art_box: tuple[int, int] = (0, 0)
        self._dvr_now_duration_s: Optional[float] = None
        self._dvr_now_position_s: float = 0.0
        self._dvr_now_art: object = None
        self._dvr_now_started_wall: Optional[float] = None
        self._dvr_now_offset_s: float = 0.0

    def _dvr_render_art_for_path(self, p: Path, *, art_width: int = 0, art_height: int = 0) -> object:
        try:
            cover = _extract_embedded_cover_path(p)
            if cover is None:
                return "(no art)"
            w = 40
            h = 18
            try:
                if int(art_width or 0) > 0 and int(art_height or 0) > 0:
                    w = max(8, int(art_width))
                    h = max(8, int(art_height))
            except Exception:
                pass
            if w == 40 and h == 18:
                try:
                    sz = getattr(self, "size", None)
                    if sz is not None:
                        tw = int(getattr(sz, "width", 0) or 0)
                        th = int(getattr(sz, "height", 0) or 0)
                        if tw > 0 and th > 0:
                            w = max(22, min(120, int(tw * 0.65)))
                            h = max(10, min(48, int(th * 0.70)))
                except Exception:
                    w = 40
                    h = 18
            try:
                mode = (getattr(getattr(self, "settings", None), "art_render_mode", None) or "halfblock").strip().lower()
            except Exception:
                mode = "halfblock"
            if mode == "braille":
                return _image_to_rich_braille_fit(cover, width=w, height=h)
            return _image_to_rich_blocks_fit(cover, width=w, height=h)
        except Exception:
            return "(no art)"

    def _dvr_set_now_playing(self, *, source: str, file_path: Path, offset_s: float = 0.0) -> None:
        try:
            title, artist, album = _read_local_tags(file_path)
            label = " - ".join([x for x in [artist, title] if x]).strip()
            if not label:
                label = file_path.name
            self._dvr_now_source = source
            self._dvr_now_track = label
            self._dvr_now_album = album
            self._dvr_now_path = file_path
            self._dvr_now_art_box = (0, 0)
            self._dvr_now_offset_s = float(offset_s or 0.0)
            self._dvr_now_position_s = float(offset_s or 0.0)
            self._dvr_now_duration_s = _probe_duration_s(file_path)
            self._dvr_now_started_wall = time.time()
            self._dvr_now_art = self._dvr_render_art_for_path(file_path)
        except Exception:
            self._dvr_now_source = source
            self._dvr_now_track = file_path.name
            self._dvr_now_album = ""
            self._dvr_now_path = file_path
            self._dvr_now_art_box = (0, 0)
            self._dvr_now_offset_s = float(offset_s or 0.0)
            self._dvr_now_position_s = float(offset_s or 0.0)
            self._dvr_now_duration_s = _probe_duration_s(file_path)
            self._dvr_now_started_wall = time.time()
            self._dvr_now_art = "(no art)"

    def get_dvr_now_playing(self, *, art_width: int = 0, art_height: int = 0) -> dict:
        # Detect end-of-track.
        # Only auto-advance for mpv where we have reliable EOF/idle signals via IPC.
        # For ffplay/vlc, proc.poll() can be noisy and has caused unexpected jumps
        # when users enqueue items during playback.
        try:
            proc = self._dvr_player_proc
            kind = (self._dvr_player_kind or "").strip().lower()

            if kind == "mpv":
                if proc is not None and proc.poll() is not None:
                    self._dvr_on_track_end()
            else:
                # Safe auto-advance for ffplay/vlc only when we're clearly at the end.
                if proc is not None and proc.poll() is not None:
                    dur = None
                    pos = None
                    try:
                        dur = self._dvr_now_duration_s
                    except Exception:
                        dur = None
                    try:
                        pos = self._dvr_now_position_s
                    except Exception:
                        pos = None

                    near_end = False
                    try:
                        if dur is not None:
                            d = float(dur)
                            p = float(pos or 0.0)
                            if d > 0 and p >= max(0.0, d - 0.60):
                                near_end = True
                    except Exception:
                        near_end = False

                    if near_end:
                        self._dvr_on_track_end()
        except Exception:
            pass

        # If mpv IPC is available, use real playback position/duration/pause and EOF.
        try:
            if self._dvr_player_kind == "mpv" and self._dvr_mpv_ipc_path is not None:
                paused = self._mpv_get_property("pause")
                if paused is not None:
                    self._dvr_player_paused = bool(paused)

                # mpv-specific EOF/idle signals are more reliable than process.poll()
                # and allow us to auto-advance quickly.
                eof = self._mpv_get_property("eof-reached")
                idle = self._mpv_get_property("idle-active")
                try:
                    if bool(eof) or bool(idle):
                        self._dvr_on_track_end()
                except Exception:
                    pass

                tpos = self._mpv_get_property("time-pos")
                dur = self._mpv_get_property("duration")
                if dur is not None:
                    try:
                        self._dvr_now_duration_s = float(dur)
                    except Exception:
                        pass
                if tpos is not None:
                    try:
                        self._dvr_now_position_s = float(tpos)
                    except Exception:
                        pass
            elif self._dvr_now_started_wall is not None and (not bool(self._dvr_player_paused)):
                # Fallback progress estimation only while playing.
                self._dvr_now_position_s = float(self._dvr_now_offset_s) + max(0.0, float(time.time() - float(self._dvr_now_started_wall)))
        except Exception:
            try:
                if self._dvr_now_started_wall is not None and (not bool(self._dvr_player_paused)):
                    self._dvr_now_position_s = float(self._dvr_now_offset_s) + max(0.0, float(time.time() - float(self._dvr_now_started_wall)))
            except Exception:
                pass

        # Clamp position to duration when we know duration.
        try:
            if self._dvr_now_duration_s is not None:
                d = float(self._dvr_now_duration_s)
                if d > 0:
                    self._dvr_now_position_s = max(0.0, min(float(self._dvr_now_position_s), d))
        except Exception:
            pass

        # Re-render cover art to match the actual #art widget size when available.
        try:
            p = self._dvr_now_path
            w = int(art_width or 0)
            h = int(art_height or 0)
            if p is not None and p.exists() and w > 0 and h > 0:
                if (w, h) != tuple(self._dvr_now_art_box or (0, 0)):
                    self._dvr_now_art = self._dvr_render_art_for_path(p, art_width=w, art_height=h)
                    self._dvr_now_art_box = (w, h)
        except Exception:
            pass
        return {
            "source": self._dvr_now_source,
            "track": self._dvr_now_track,
            "album": self._dvr_now_album,
            "duration_s": self._dvr_now_duration_s,
            "position_s": self._dvr_now_position_s,
            "art": self._dvr_now_art,
        }

    def start_dvr_source(self, *, source_path: Path, offset_s: Optional[float] = None) -> None:
        # DVR and Live should be mutually exclusive.
        try:
            self.stop_live()
        except Exception:
            pass

        p = source_path.expanduser()
        if not p.exists():
            raise RuntimeError("Path not found")

        if p.is_dir():
            self.start_dvr_folder(folder=p, start_at=None)
            return

        cue_tracks: list[dict] = []
        if p.suffix.lower() in {".m3u", ".m3u8"}:
            self._dvr_queue = _parse_m3u_playlist(p)
            self._dvr_queue_index = 0
            self._dvr_chapters = []
            self._dvr_chapter_index = 0
            if not self._dvr_queue:
                raise RuntimeError("Playlist is empty")
            self._play_dvr_queue_item(offset_s=0.0, source_label=f"Playlist: {p.name}")
            return

        if p.suffix.lower() in {".cue", ".ffmeta"}:
            audio = None
            try:
                if p.suffix.lower() == ".ffmeta":
                    cue_tracks = _parse_ffmeta_chapters(p)
                else:
                    cue_tracks = _parse_cue_tracks(p)
            except Exception:
                cue_tracks = []
            try:
                base = p.with_suffix("")
                if base.exists() and base.is_file():
                    audio = base
            except Exception:
                audio = None
            if audio is None:
                raise RuntimeError("Cue/ffmeta has no matching audio file")
            self._dvr_queue = [audio]
            self._dvr_queue_index = 0
            self._dvr_chapters = cue_tracks
            self._dvr_chapter_index = 0
            off = 0.0
            if offset_s is not None:
                off = float(offset_s)
            self._play_dvr_queue_item(offset_s=off, source_label=f"Chapters: {p.name}")
            return

        # Plain audio file.
        self._dvr_queue = [p]
        try:
            _save_dvr_queue([p])
        except Exception:
            pass
        self._dvr_queue_index = 0
        self._dvr_shuffle_order = []
        self._dvr_chapters = []
        self._dvr_chapter_index = 0
        off2 = float(offset_s or 0.0)
        self._play_dvr_queue_item(offset_s=off2, source_label=f"File: {p.name}")

    def toggle_dvr_shuffle(self) -> None:
        self._dvr_shuffle = not bool(self._dvr_shuffle)
        if not self._dvr_shuffle:
            self._dvr_shuffle_order = []
            self.notify("Shuffle: OFF")
        else:
            self._dvr_shuffle_order = []
            self.notify("Shuffle: ON")

    def cycle_dvr_repeat(self) -> None:
        cur = (self._dvr_repeat or "off").lower().strip()
        nxt = "off"
        if cur == "off":
            nxt = "all"
        elif cur == "all":
            nxt = "one"
        else:
            nxt = "off"
        self._dvr_repeat = nxt
        self.notify(f"Repeat: {nxt.upper()}")

    def _dvr_build_folder_queue(self, folder: Path) -> list[Path]:
        files: list[Path] = []

        def is_candidate(p: Path) -> bool:
            try:
                if p.name.startswith("."):
                    return False
                if ".satstash_tmp" in p.parts:
                    return False
                if p.name.endswith(".part"):
                    return False
                # Exclude helper/metadata files from queue building.
                if p.suffix.lower() in {".m3u", ".m3u8", ".cue", ".ffmeta"}:
                    return False
                # Exclude obvious non-media.
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".txt", ".nfo", ".json", ".log", ".md"}:
                    return False
                if p.suffix.lower() in {".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar"}:
                    return False
                return True
            except Exception:
                return False

        try:
            for p in folder.rglob("*"):
                if not p.is_file():
                    continue
                if is_candidate(p):
                    files.append(p)
        except Exception:
            return []

        def sort_key(x: Path):
            try:
                return _natural_key(str(x.relative_to(folder)))
            except Exception:
                try:
                    return str(x.name).lower()
                except Exception:
                    return ""

        try:
            files.sort(key=sort_key)
        except Exception:
            try:
                files.sort(key=lambda x: str(x).lower())
            except Exception:
                pass
        return files

    def start_dvr_folder(self, *, folder: Path, start_at: Optional[Path]) -> None:
        f = folder.expanduser()
        if not f.exists() or not f.is_dir():
            raise RuntimeError("Folder not found")
        q = self._dvr_build_folder_queue(f)
        if not q:
            raise RuntimeError("Folder has no playable files")
        self._dvr_queue = q
        try:
            _save_dvr_queue([x for x in q if isinstance(x, Path)])
        except Exception:
            pass
        self._dvr_chapters = []
        self._dvr_chapter_index = 0
        self._dvr_shuffle_order = []

        idx = 0
        if start_at is not None:
            try:
                for i, p in enumerate(q):
                    if p == start_at:
                        idx = i
                        break
            except Exception:
                idx = 0
        self._dvr_queue_index = idx
        self._play_dvr_queue_item(offset_s=0.0, source_label=f"Folder: {f.name}")

    def _play_dvr_queue_item(self, *, offset_s: float, source_label: str) -> None:
        if not self._dvr_queue:
            raise RuntimeError("Empty queue")
        idx = max(0, min(int(self._dvr_queue_index), len(self._dvr_queue) - 1))
        p = self._dvr_queue[idx]
        # If chapter mode is active and no explicit offset is provided, use chapter offset.
        off = float(offset_s or 0.0)
        if self._dvr_chapters:
            ci = max(0, min(int(self._dvr_chapter_index), len(self._dvr_chapters) - 1))
            try:
                off = float(self._dvr_chapters[ci].get("offset_s") or 0.0)
            except Exception:
                off = 0.0
            try:
                t = self._dvr_chapters[ci]
                label = " - ".join([x for x in [str(t.get("artist") or "").strip(), str(t.get("title") or "").strip()] if x]).strip()
                if label:
                    source_label = source_label + f"   Track {ci + 1}/{len(self._dvr_chapters)}"
                    self._dvr_now_track = label
            except Exception:
                pass

        # Start playback first (this may stop a previous player and clear timer state),
        # then set now-playing metadata/timers so progress estimation remains correct.
        self.start_dvr_playback(path=p, offset_s=off)
        self._dvr_set_now_playing(source=source_label, file_path=p, offset_s=off)

    def _dvr_shuffle_next_index(self) -> int:
        if not self._dvr_queue:
            return 0
        n = len(self._dvr_queue)
        if n <= 1:
            return 0
        if not self._dvr_shuffle_order:
            self._dvr_shuffle_order = list(range(n))
            random.shuffle(self._dvr_shuffle_order)
            # Avoid repeating current track immediately when possible.
            try:
                if self._dvr_shuffle_order and self._dvr_shuffle_order[0] == self._dvr_queue_index:
                    self._dvr_shuffle_order.append(self._dvr_shuffle_order.pop(0))
            except Exception:
                pass
        try:
            return int(self._dvr_shuffle_order.pop(0))
        except Exception:
            return (self._dvr_queue_index + 1) % n

    def _dvr_on_track_end(self) -> None:
        if self._dvr_advancing:
            return
        self._dvr_advancing = True
        try:
            rep = (self._dvr_repeat or "off").lower().strip()
            if rep == "one":
                self._play_dvr_queue_item(offset_s=0.0, source_label=self._dvr_now_source or "DVR")
                return
            if self._dvr_chapters and self._dvr_chapter_index + 1 < len(self._dvr_chapters):
                self._dvr_chapter_index += 1
                self._play_dvr_queue_item(offset_s=0.0, source_label=self._dvr_now_source or "DVR")
                return

            if self._dvr_queue:
                if bool(self._dvr_shuffle):
                    self._dvr_queue_index = self._dvr_shuffle_next_index()
                    self._dvr_chapter_index = 0
                    self._play_dvr_queue_item(offset_s=0.0, source_label=self._dvr_now_source or "DVR")
                    return
                if self._dvr_queue_index + 1 < len(self._dvr_queue):
                    self._dvr_queue_index += 1
                    self._dvr_chapter_index = 0
                    self._play_dvr_queue_item(offset_s=0.0, source_label=self._dvr_now_source or "DVR")
                    return
                if rep == "all":
                    self._dvr_queue_index = 0
                    self._dvr_chapter_index = 0
                    self._play_dvr_queue_item(offset_s=0.0, source_label=self._dvr_now_source or "DVR")
                    return
        finally:
            self._dvr_advancing = False

    def dvr_next(self) -> None:
        if self._dvr_chapters and len(self._dvr_chapters) > 0:
            if self._dvr_chapter_index + 1 < len(self._dvr_chapters):
                self._dvr_chapter_index += 1
                self._play_dvr_queue_item(offset_s=0.0, source_label=self._dvr_now_source or "DVR")
                return
        if self._dvr_queue and self._dvr_queue_index + 1 < len(self._dvr_queue):
            self._dvr_queue_index += 1
            self._dvr_chapter_index = 0
            self._play_dvr_queue_item(offset_s=0.0, source_label=self._dvr_now_source or "DVR")
            return
        if self._dvr_queue and bool(self._dvr_shuffle):
            self._dvr_queue_index = self._dvr_shuffle_next_index()
            self._dvr_chapter_index = 0
            self._play_dvr_queue_item(offset_s=0.0, source_label=self._dvr_now_source or "DVR")
            return
        self.notify("End of queue")

    def dvr_prev(self) -> None:
        if self._dvr_chapters and len(self._dvr_chapters) > 0:
            if self._dvr_chapter_index > 0:
                self._dvr_chapter_index -= 1
                self._play_dvr_queue_item(offset_s=0.0, source_label=self._dvr_now_source or "DVR")
                return
        if self._dvr_queue and self._dvr_queue_index > 0:
            self._dvr_queue_index -= 1
            self._dvr_chapter_index = 0
            self._play_dvr_queue_item(offset_s=0.0, source_label=self._dvr_now_source or "DVR")
            return
        # restart current
        if self._dvr_queue:
            self._play_dvr_queue_item(offset_s=0.0, source_label=self._dvr_now_source or "DVR")

    def stop_dvr_playback(self, *, notify_stop: bool = True) -> None:
        proc = self._dvr_player_proc
        if proc is None:
            return
        stopped = False
        try:
            # Prefer a graceful stop first.
            try:
                proc.send_signal(signal.SIGINT)
            except Exception:
                pass
            try:
                proc.wait(timeout=1.5)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=1.5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            try:
                stopped = proc.poll() is not None
            except Exception:
                stopped = False
        finally:
            if not stopped:
                # Don't desync UI state if the player ignored stop signals.
                return
            try:
                ipc = self._dvr_mpv_ipc_path
                if ipc is not None and ipc.exists():
                    ipc.unlink(missing_ok=True)
            except Exception:
                pass
            self._dvr_player_proc = None
            self._dvr_player_kind = None
            self._dvr_player_paused = False
            self._dvr_now_started_wall = None
            self._dvr_mpv_ipc_path = None
            if bool(notify_stop):
                self.notify("Playback stopped")

    def _mpv_ipc_request(self, payload: dict) -> Optional[dict]:
        ipc = self._dvr_mpv_ipc_path
        if not ipc:
            return None
        try:
            if self._dvr_player_kind != "mpv":
                return None
        except Exception:
            return None
        try:
            if not ipc.exists():
                return None
        except Exception:
            return None

        try:
            req = (json.dumps(payload) + "\n").encode("utf-8", errors="replace")
        except Exception:
            return None

        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                s.settimeout(0.25)
                s.connect(str(ipc))
                s.sendall(req)
                buf = b""
                while b"\n" not in buf and len(buf) < 1024 * 1024:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
            finally:
                try:
                    s.close()
                except Exception:
                    pass
            line = buf.split(b"\n", 1)[0].strip()
            if not line:
                return None
            return json.loads(line.decode("utf-8", errors="replace"))
        except Exception:
            return None

    def _mpv_get_property(self, name: str) -> Optional[object]:
        resp = self._mpv_ipc_request({"command": ["get_property", name]})
        try:
            if not resp:
                return None
            if resp.get("error") != "success":
                return None
            return resp.get("data")
        except Exception:
            return None

    def _mpv_command(self, args: list[object]) -> bool:
        resp = self._mpv_ipc_request({"command": args})
        try:
            return bool(resp and resp.get("error") == "success")
        except Exception:
            return False

    def toggle_live_pause(self) -> None:
        handle = self._live_handle
        if not handle:
            self.notify("No live playback active")
            return
        try:
            if not self._live_paused:
                handle.process.send_signal(signal.SIGSTOP)
                self._live_paused = True
                try:
                    # Freeze progress timing while paused.
                    self._home_live_paused_audio_now = self._home_live_audio_now_estimated() or None
                except Exception:
                    self._home_live_paused_audio_now = None
                self.notify("Live playback paused")
            else:
                handle.process.send_signal(signal.SIGCONT)
                self._live_paused = False
                try:
                    self._home_live_paused_audio_now = None
                    # Re-anchor so interpolation resumes cleanly.
                    self._home_live_audio_anchor_pdt = None
                    self._home_live_audio_anchor_wall = None
                except Exception:
                    pass
                self.notify("Live playback resumed")
        except Exception as exc:
            self.notify(f"Live pause/resume failed: {exc}")

    def _playback_summary(self) -> str:
        try:
            if self._dvr_player_proc is not None:
                state = "DVR"
                label = (self._dvr_now_track or "").strip() or "(playing)"
                paused = bool(self._dvr_player_paused)
                tail = "paused" if paused else "playing"
                return f"{state}: {label} [{tail}]"
        except Exception:
            pass
        try:
            if self._live_handle is not None and self._live_channel is not None:
                ch = self._live_channel
                label = f"{ch.number if ch.number is not None else ''} {ch.name}".strip() or "Live"
                paused = bool(self._live_paused)
                tail = "paused" if paused else "playing"
                return f"LIVE: {label} [{tail}]"
        except Exception:
            pass
        return ""

    def action_home(self) -> None:
        # Return to the main menu without stopping playback.
        try:
            while True:
                try:
                    stack = getattr(self, "screen_stack", None)
                    if stack is None:
                        break
                    if len(stack) <= 1:
                        break
                except Exception:
                    break
                try:
                    self.pop_screen()
                except Exception:
                    break
        except Exception:
            pass
        try:
            self._show_right_idle()
        except Exception:
            pass

    def action_play_pause(self) -> None:
        # Prefer DVR controls when active.
        if self._dvr_player_proc is not None:
            try:
                self.toggle_dvr_pause()
            except Exception:
                pass
            try:
                self._home_player_flash_action("Play/Pause")
            except Exception:
                pass
            return
        if self._live_handle is not None:
            try:
                self.toggle_live_pause()
            except Exception:
                pass
            try:
                self._home_player_flash_action("Paused" if bool(self._live_paused) else "Playing")
            except Exception:
                pass

    def _home_player_flash_action(self, msg: str, *, ttl_s: float = 1.2) -> None:
        try:
            text = str(msg or "").strip()
        except Exception:
            text = ""
        if not text:
            return

        try:
            self.query_one("#home_player_action", Static).update(text)
        except Exception:
            return

        try:
            self._home_player_action_nonce = int(getattr(self, "_home_player_action_nonce", 0) or 0) + 1
            nonce = self._home_player_action_nonce
        except Exception:
            nonce = 0

        def clear() -> None:
            try:
                if int(getattr(self, "_home_player_action_nonce", 0) or 0) != nonce:
                    return
                self.query_one("#home_player_action", Static).update("")
            except Exception:
                pass

        try:
            self.set_timer(ttl_s, clear)
        except Exception:
            pass

    def action_stop_playback(self) -> None:
        if self._dvr_player_proc is not None:
            try:
                self.stop_dvr_playback()
            except Exception:
                pass
            try:
                self._home_player_flash_action("Stopped")
            except Exception:
                pass
            return
        if self._live_handle is not None:
            try:
                self.stop_live()
            except Exception:
                pass
            try:
                self._home_player_flash_action("Stopped")
            except Exception:
                pass

    def action_next_track(self) -> None:
        if self._dvr_player_proc is not None:
            try:
                self.dvr_next()
            except Exception:
                pass
            try:
                self._home_player_flash_action("Next")
            except Exception:
                pass
            return
        if self._live_handle is not None:
            try:
                self.notify("Live: next not supported")
            except Exception:
                pass
            try:
                self._home_player_flash_action("Next")
            except Exception:
                pass

    def action_prev_track(self) -> None:
        if self._dvr_player_proc is not None:
            try:
                self.dvr_prev()
            except Exception:
                pass
            try:
                self._home_player_flash_action("Prev")
            except Exception:
                pass
            return
        if self._live_handle is not None:
            try:
                self.notify("Live: prev not supported")
            except Exception:
                pass
            try:
                self._home_player_flash_action("Prev")
            except Exception:
                pass

    def action_playlists(self) -> None:
        try:
            self.push_screen(PlaylistManagerScreen())
        except Exception as exc:
            try:
                self.notify(f"Playlists failed: {exc}")
            except Exception:
                pass

    def action_help(self) -> None:
        global_keys: list[str] = [
            "Q = Quit",
            "h = Home",
            "? = Help",
            "p = Playlists",
            "space = Play/Pause",
            "s = Stop",
            "n = Next",
            "b = Prev",
            "[ / ] = Seek -10s / +10s",
            "S = Schedule",
        ]

        active = "home"
        try:
            scr = getattr(self, "screen", None)
            scr_name = getattr(scr, "__class__", type("X", (), {})).__name__
        except Exception:
            scr_name = ""

        if scr_name == "PlaylistManagerScreen":
            active = "playlists"
        elif scr_name == "DvrNowPlayingScreen":
            active = "dvr_now_playing"
        elif scr_name == "NowPlayingScreen":
            active = "now_playing"
        elif scr_name == "CatchUpSelectScreen":
            active = "catchup"
        elif scr_name in {"LiveSelectScreen", "RecordSelectScreen"}:
            # These are older full-screen selectors; we still mark context.
            active = "live" if scr_name == "LiveSelectScreen" else "record"
        else:
            try:
                rp = getattr(self, "_right_pane", None)
                rp_name = getattr(rp, "__class__", type("X", (), {})).__name__
            except Exception:
                rp_name = ""
            if rp_name == "BrowseDvrPane":
                try:
                    mode = str(getattr(rp, "_mode", "") or "")
                except Exception:
                    mode = ""
                active = "dvr_queue" if mode == "queue" else "dvr_browse"
            elif rp_name == "ScheduledRecordingsPane":
                active = "schedule"
            elif rp_name in {"RecordSelectPane"}:
                active = "record"
            elif rp_name in {"CatchUpSelectPane", "CatchUpPane"}:
                active = "catchup"
            else:
                # Heuristic based on widget ids.
                try:
                    focused = getattr(self, "focused", None)
                    if getattr(focused, "id", None) == "tracks":
                        active = "catchup"
                except Exception:
                    pass

        sections: dict[str, list[str]] = {
            "global": global_keys,
            "home": [
                "Use on-screen buttons",
                "h = Home (from anywhere)",
                "p = Playlists",
            ],
            "live": [
                "Enter = Play first match (search focused)",
                "Enter = Play highlighted row (table focused)",
                "r = Refresh channel list (Live full-screen selector)",
                "Type in search box to filter channels",
                "Buttons: Play / Refresh / Close",
                "space = Play/Pause",
                "[ / ] = Seek",
                "s = Stop",
            ],
            "catchup": [
                "Enter = Open/Play (table focused)",
                "Enter = Open first match (search focused)",
                "Type in search box to filter",
                "S = Set start marker (export)",
                "E = Set end marker (export)",
                "D = Toggle debug (export)",
                "Buttons: Toggle Range / Toggle Export / Numbering / Export / Refresh / Close",
                "space = Play/Pause",
                "s = Stop",
            ],
            "dvr_browse": [
                "q = Queue view (table focused)",
                "Enter = Open folder / Play file",
                "Backspace / Esc = Back (not while typing in search)",
                "e = Enqueue selected",
                "E = Enqueue folder",
                "f / F = Play folder now",
            ],
            "dvr_queue": [
                "q = Back to Browse",
                "Enter = Play selected queue item",
                "x = Shuffle ON/OFF",
                "w = Save queue to playlist",
                "l = Load playlist into queue",
                "d = Remove selected",
                "c = Clear queue (confirm)",
                "J / K = Move down / up",
            ],
            "playlists": [
                "Enter = Load playlist",
                "a = Append playlist",
                "n = New from current queue",
                "r = Rename playlist",
                "d = Delete playlist (confirm)",
                "/ = Search",
                "R = Refresh",
                "Esc = Back",
            ],
            "schedule": [
                "S = Open Schedule",
                "Use on-screen buttons: Add / Edit / Delete",
                "Use on-screen buttons: Enable/Disable / Run now / Close",
                "Enter = Select (channel picker table)",
                "Save/Cancel buttons (schedule editor)",
                "Esc = Back",
            ],
            "record": [
                "Enter = Record first match (search focused)",
                "Enter = Record highlighted row (table focused)",
                "Type in search box to filter",
                "Use buttons: Record Single / Split / Record&Listen / Stop",
            ],
            "now_playing": [
                "Uses global playback keys",
                "Buttons: Stop / Start-Stop Recording / Back",
            ],
            "dvr_now_playing": [
                "Uses global playback keys",
                "(Also has on-screen buttons: shuffle/repeat/next/prev)",
            ],
            "misc": [
                "Some screens are button-driven only",
                "If a key does nothing, click the table to focus it",
            ],
        }

        try:
            self.push_screen(
                HelpOverlayScreen(
                    title="Help",
                    sections=sections,
                    active_section=active,
                )
            )
        except Exception as exc:
            try:
                self.notify(f"Help failed: {exc}")
            except Exception:
                pass

    def _dvr_seek_relative(self, delta_s: float) -> None:
        if self._dvr_player_proc is None:
            return

        try:
            focused = getattr(self, "focused", None)
            if focused is not None and focused.__class__.__name__ == "Input":
                return
        except Exception:
            pass

        try:
            delta = float(delta_s or 0.0)
        except Exception:
            delta = 0.0
        if delta == 0.0:
            return

        # Prefer mpv IPC when available.
        try:
            if self._dvr_player_kind == "mpv" and self._dvr_mpv_ipc_path is not None:
                ok = self._mpv_command(["seek", float(delta), "relative"])
                if ok:
                    try:
                        self._home_player_flash_action("Seek")
                    except Exception:
                        pass
                    return
        except Exception:
            pass

        # Fallback: restart playback at a new absolute offset.
        p = self._dvr_now_path
        if p is None:
            return

        try:
            pos = float(getattr(self, "_dvr_now_position_s", 0.0) or 0.0)
        except Exception:
            pos = 0.0
        try:
            dur = self._dvr_now_duration_s
            dur_f = float(dur) if dur is not None else None
        except Exception:
            dur_f = None

        new_pos = max(0.0, pos + delta)
        if dur_f is not None and dur_f > 0:
            new_pos = max(0.0, min(new_pos, max(0.0, dur_f - 0.35)))

        try:
            self.start_dvr_playback(path=p, offset_s=float(new_pos), notify_start=False, notify_stop=False)
        except Exception:
            return
        try:
            self._dvr_set_now_playing(source=self._dvr_now_source or "DVR", file_path=p, offset_s=float(new_pos))
        except Exception:
            pass
        try:
            self._home_player_flash_action("Seek")
        except Exception:
            pass

    def action_seek_back(self) -> None:
        if self._dvr_player_proc is None:
            return
        self._dvr_seek_relative(-10.0)

    def action_seek_forward(self) -> None:
        if self._dvr_player_proc is None:
            return
        self._dvr_seek_relative(10.0)

    def toggle_dvr_pause(self) -> None:
        proc = self._dvr_player_proc
        if not proc:
            return
        if proc is None:
            self.notify("No playback active")
            return

        # Prefer mpv IPC when available.
        try:
            if self._dvr_player_kind == "mpv" and self._dvr_mpv_ipc_path is not None:
                ok = self._mpv_command(["cycle", "pause"])
                if ok:
                    paused = self._mpv_get_property("pause")
                    self._dvr_player_paused = bool(paused)
                    self.notify("Playback paused" if self._dvr_player_paused else "Playback resumed")
                    return
        except Exception:
            pass

        try:
            if not self._dvr_player_paused:
                proc.send_signal(signal.SIGSTOP)
                self._dvr_player_paused = True
                self.notify("Playback paused")
            else:
                proc.send_signal(signal.SIGCONT)
                self._dvr_player_paused = False
                self.notify("Playback resumed")
        except Exception as exc:
            self.notify(f"Pause/resume failed: {exc}")

    def start_dvr_playback(self, *, path: Path, offset_s: Optional[float], notify_start: bool = True, notify_stop: bool = True) -> None:
        # Ensure only one playback process at a time.
        try:
            self.stop_dvr_playback(notify_stop=bool(notify_stop))
        except Exception:
            pass

        p = path
        pref = (self.settings.player_preference or "auto").strip().lower()
        mpv_ok = bool(shutil.which("mpv"))
        ffplay_ok = bool(shutil.which("ffplay"))
        cvlc_ok = bool(shutil.which("cvlc"))
        vlc_ok = bool(shutil.which("vlc"))

        def build_ffplay_argv() -> list[str]:
            argv = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"]
            if offset_s is not None:
                dur = _probe_duration_s(p)
                ss = float(offset_s)
                if dur is not None and dur > 0:
                    ss = max(0.0, min(ss, max(0.0, float(dur) - 0.35)))
                argv += ["-ss", str(ss)]
            argv += [str(p)]
            return argv

        def build_mpv_argv() -> list[str]:
            ipc: Optional[Path] = None
            try:
                log_dir = Path(user_cache_dir("satstash"))
                log_dir.mkdir(parents=True, exist_ok=True)
                ipc = log_dir / f"mpv-ipc-{int(time.time() * 1000)}.sock"
            except Exception:
                ipc = None

            # Ensure stale sockets never block a new mpv instance.
            try:
                if ipc is not None and ipc.exists():
                    ipc.unlink(missing_ok=True)
            except Exception:
                pass

            argv = ["mpv", "--no-terminal", "--msg-level=all=fatal"]
            if ipc is not None:
                argv.append(f"--input-ipc-server={ipc}")
                self._dvr_mpv_ipc_path = ipc
            else:
                self._dvr_mpv_ipc_path = None

            if offset_s is not None:
                try:
                    argv.append(f"--start={float(offset_s):.3f}")
                except Exception:
                    pass
            argv.append(str(p))
            return argv

        def build_vlc_argv() -> list[str]:
            bin_name = "cvlc" if cvlc_ok else "vlc"
            argv = [bin_name, "--play-and-exit", "--quiet"]
            if offset_s is not None:
                try:
                    argv.append(f"--start-time={int(float(offset_s))}")
                except Exception:
                    pass
            argv.append(str(p))
            return argv

        def spawn_with_optional_log(argv: list[str], *, player: str) -> subprocess.Popen:
            log_path: Optional[Path] = None
            try:
                log_dir = Path(user_cache_dir("satstash"))
                log_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                log_path = log_dir / f"{player}-play-{ts}.log"
            except Exception:
                log_path = None

            if log_path is not None and player in {"ffplay", "mpv"}:
                with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
                    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, text=True)
            else:
                proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL)

            try:
                time.sleep(0.6)
                rc = proc.poll()
                if rc is not None and rc != 0:
                    tail = ""
                    if log_path is not None:
                        try:
                            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
                        except Exception:
                            tail = ""
                    raise RuntimeError(f"{player} exited quickly (code {rc})." + (f" Log: {log_path}\n{tail}" if tail else f" Log: {log_path}" if log_path else ""))
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
                raise

            return proc

        primary: str
        if pref in {"mpv", "ffplay", "vlc", "cvlc"}:
            primary = pref
        else:
            primary = "mpv"

        candidates: list[str] = []
        if primary == "mpv":
            candidates = ["mpv", "ffplay"]
        elif primary == "ffplay":
            candidates = ["ffplay", "mpv"]
        elif primary in {"vlc", "cvlc"}:
            candidates = ["cvlc", "vlc", "mpv", "ffplay"]

        last_err: Optional[Exception] = None
        for i, player in enumerate(candidates):
            if player == "mpv" and not mpv_ok:
                continue
            if player == "ffplay" and not ffplay_ok:
                continue
            if player == "cvlc" and not cvlc_ok:
                continue
            if player == "vlc" and not vlc_ok:
                continue

            try:
                if player == "mpv":
                    argv = build_mpv_argv()
                elif player == "ffplay":
                    self._dvr_mpv_ipc_path = None
                    argv = build_ffplay_argv()
                else:
                    self._dvr_mpv_ipc_path = None
                    argv = build_vlc_argv()

                if bool(notify_start):
                    msg = f"Playing with {player}: {p.name}"
                    if i > 0:
                        msg = msg + " (fallback)"
                    self.notify(msg)

                proc = spawn_with_optional_log(argv, player=player)
                self._dvr_player_proc = proc
                self._dvr_player_kind = player
                self._dvr_player_paused = False
                return
            except Exception as exc:
                last_err = exc
                continue

        raise RuntimeError(f"No supported player found (mpv/ffplay/vlc). {last_err}".strip())

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="main_menu_root"):
            with Container(id="menu_center"):
                with Horizontal(id="home_body"):
                    with Vertical(id="home_player"):
                        yield Static("Now Playing", id="home_player_title")
                        yield Static("", id="home_player_art")
                        yield Static("", id="home_player_source")
                        yield Static("", id="home_player_track")
                        yield Static("", id="home_player_album")
                        yield ProgressBar(total=1, show_eta=False, id="home_player_progress")
                        yield Static("", id="home_player_timing")
                        yield Static("", id="home_player_action")
                        yield Static("", id="home_player_hint")
                    with Container(id="right_content"):
                        yield RightIdleLogoPane()

            with Horizontal(id="home_menu_row"):
                yield Button("Listen Live", id="listen_live")
                yield Button("Schedule", id="schedule")
                yield Button("Catch Up", id="catch_up")
                yield Button("VOD", id="vod")
                yield Button("Browse DVR", id="browse_dvr")
                yield Button("Record Now", id="record_now")
                yield Button("Settings", id="settings")

            with Horizontal(id="bottom_bar"):
                with Vertical(id="bottom_left"):
                    yield Static("", id="status")
                    yield Static("", id="recording_status")
                with Horizontal(id="bottom_right"):
                    yield Button("Login", id="login", variant="primary")
                    yield Button("Logout", id="logout")
                    yield Button("Quit", id="quit")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_status()
        # Attempt silent login once on startup if the session is missing/expired.
        try:
            self._startup_autologin_attempted = False
        except Exception:
            pass

        try:
            sess = load_session()
            needs_login = (sess is None) or (not sess.is_valid())
        except Exception:
            needs_login = True

        if needs_login and not bool(getattr(self, "_startup_autologin_attempted", False)):
            self._startup_autologin_attempted = True

            def work() -> None:
                ok = False
                try:
                    ok = bool(self._try_silent_relogin())
                except Exception:
                    ok = False

                if ok:
                    self.call_from_thread(self._refresh_status)

            threading.Thread(target=work, daemon=True).start()

        # Keep the UI informative while background playback/recording tasks run.
        self.set_interval(1.0, self._refresh_status)

        # Background scheduled recording runner (Option A: works while SatStash is running).
        try:
            if not hasattr(self, "_schedule_runner_stop"):
                self._schedule_runner_stop = threading.Event()
        except Exception:
            self._schedule_runner_stop = threading.Event()
        try:
            if not hasattr(self, "_schedule_last_started"):
                self._schedule_last_started = {}
        except Exception:
            self._schedule_last_started = {}

        def sched_loop() -> None:
            while not self._schedule_runner_stop.is_set():
                try:
                    self._schedule_tick()
                except Exception:
                    pass
                self._schedule_runner_stop.wait(1.0)

        try:
            threading.Thread(target=sched_loop, daemon=True).start()
        except Exception:
            pass

    def _schedule_tick(self) -> None:
        # Don't auto-start if a recording is already running.
        try:
            if getattr(self, "_record_handle", None) is not None or bool(getattr(self, "_record_pending", False)):
                return
        except Exception:
            pass

        now_local = datetime.now().astimezone()
        now_utc = now_local.astimezone(timezone.utc)

        items = [it for it in _load_scheduled_recordings() if bool(it.get("enabled", True))]

        def start_key(it: dict) -> float:
            st = _schedule_dt_local(str(it.get("start_time_iso") or ""))
            return float(st.timestamp()) if st else 0.0
        items.sort(key=start_key)

        for it in items:
            sid = str(it.get("id") or "")
            st = _schedule_dt_local(str(it.get("start_time_iso") or ""))
            en = _schedule_dt_local(str(it.get("end_time_iso") or ""))
            if st is None or en is None:
                continue
            if en <= now_local:
                continue
            if not (st <= now_local < en):
                continue

            # Avoid starting if the schedule is about to end. Very short windows often
            # yield an empty/invalid output because ffmpeg may not receive any media.
            try:
                if (en - now_local).total_seconds() < 15.0:
                    continue
            except Exception:
                pass

            last = None
            try:
                last = float(getattr(self, "_schedule_last_started", {}).get(sid) or 0.0)
            except Exception:
                last = 0.0
            if last and (time.time() - last) < 30.0:
                continue
            try:
                self._schedule_last_started[sid] = time.time()
            except Exception:
                pass

            ch_id = str(it.get("channel_id") or "")
            if not ch_id:
                continue
            ch_name = str(it.get("channel_name") or "Scheduled")
            ch_type = str(it.get("channel_type") or "channel-linear")

            def ui_toast() -> None:
                try:
                    self.notify(f"⏰ Scheduled recording started: {ch_name} (ends {en.strftime('%I:%M %p')})")
                except Exception:
                    pass
            try:
                self.call_from_thread(ui_toast)
            except Exception:
                pass

            def work() -> None:
                try:
                    client = self._get_client()
                    base_dir = _output_category_dir(self.settings, "Live")
                    safe_chan = "".join(c if c.isalnum() or c in " _-." else "_" for c in ch_name).strip() or "channel"
                    out_dir = base_dir / safe_chan
                    out_dir.mkdir(parents=True, exist_ok=True)

                    handle = start_recording(
                        client=client,
                        channel_id=ch_id,
                        channel_type=ch_type,
                        preferred_quality=self.settings.preferred_quality or "256k",
                        out_dir=out_dir,
                        title=safe_chan,
                        start_pdt=now_utc,
                        end_pdt=en.astimezone(timezone.utc),
                        debug=bool(getattr(self, "_debug_enabled", False)),
                    )

                    def ui_start() -> None:
                        # Reuse normal recording UI/status plumbing.
                        try:
                            self._record_handle = handle
                            self._record_channel = Channel(id=ch_id, name=ch_name, number=None, description=None, genre=None, logo_url=None, channel_type=ch_type)
                            self._record_started_at = time.time()
                            self._record_pending = False
                            self._record_mode = "single"
                            self._record_stop = None
                        except Exception:
                            pass

                        try:
                            with self._record_single_track_lock:
                                self._record_single_started_wall = datetime.now(timezone.utc)
                                self._record_single_track_index = []
                        except Exception:
                            pass

                        def poll_index() -> None:
                            last_id = None
                            base_hls_pdt: Optional[datetime] = None
                            try:
                                base_hls_pdt = getattr(handle, "effective_start_pdt", None)
                            except Exception:
                                base_hls_pdt = None
                            if base_hls_pdt is None:
                                try:
                                    base_hls_pdt = _get_playlist_pdt_at_or_before(handle.proxy_info.url, datetime.now(timezone.utc))
                                except Exception:
                                    base_hls_pdt = None

                            while True:
                                try:
                                    if handle.process.poll() is not None:
                                        return
                                except Exception:
                                    return

                                try:
                                    data = client.live_update(channel_id=ch_id)
                                    items = data.get("items") or []
                                    cur: dict = items[-1] if items else {}
                                except Exception:
                                    cur = {}
                                    items = []

                                audio_now: Optional[datetime] = None
                                try:
                                    audio_now = handle.proxy.playhead_pdt(behind_segments=3)
                                except Exception:
                                    audio_now = None

                                def parse_item_dt(item: dict) -> Optional[datetime]:
                                    try:
                                        ts = item.get("timestamp")
                                        if isinstance(ts, str) and ts:
                                            return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
                                    except Exception:
                                        return None
                                    return None

                                start_dt: Optional[datetime] = None
                                if items and audio_now is not None:
                                    try:
                                        parsed: list[tuple[datetime, dict]] = []
                                        for it2 in items:
                                            dt2 = parse_item_dt(it2)
                                            if dt2 is not None:
                                                parsed.append((dt2, it2))
                                        parsed.sort(key=lambda x: x[0])
                                        if parsed:
                                            chosen = parsed[0][1]
                                            chosen_dt = parsed[0][0]
                                            for dt2, it2 in parsed:
                                                if dt2 <= audio_now:
                                                    chosen = it2
                                                    chosen_dt = dt2
                                                else:
                                                    break
                                            cur = chosen
                                            start_dt = chosen_dt
                                    except Exception:
                                        pass

                                if start_dt is None:
                                    start_dt = parse_item_dt(cur)
                                if start_dt is None:
                                    start_dt = datetime.now(timezone.utc)

                                tid = cur.get("id")
                                if not tid:
                                    try:
                                        tid = start_dt.isoformat()
                                    except Exception:
                                        tid = None

                                if tid and tid != last_id:
                                    if base_hls_pdt is not None:
                                        offset_s = (start_dt - base_hls_pdt).total_seconds()
                                    else:
                                        offset_s = 0.0
                                    artist = cur.get("artistName") or ""
                                    title = cur.get("name") or ""
                                    album = cur.get("albumName") or ""
                                    display = " - ".join([x for x in [artist, title] if x]) or title or artist
                                    rec_item = {
                                        "id": tid,
                                        "offset_s": max(0.0, float(offset_s)),
                                        "artist": str(artist),
                                        "title": str(title),
                                        "album": str(album),
                                        "display": str(display),
                                    }
                                    try:
                                        with self._record_single_track_lock:
                                            self._record_single_track_index.append(rec_item)
                                    except Exception:
                                        pass
                                    last_id = tid

                                time.sleep(2.0)

                        try:
                            self.run_worker(poll_index, thread=True, exclusive=False)
                        except Exception:
                            threading.Thread(target=poll_index, daemon=True).start()

                    self.call_from_thread(ui_start)
                except Exception as exc:
                    def ui_err() -> None:
                        try:
                            self.notify(f"Scheduled recording failed: {ch_name}: {exc!r}")
                        except Exception:
                            pass
                    try:
                        self.call_from_thread(ui_err)
                    except Exception:
                        pass

            try:
                self.run_worker(work, thread=True)
            except Exception:
                threading.Thread(target=work, daemon=True).start()
            return

    def _home_art_box(self) -> tuple[int, int]:
        # Prefer the actual widget dimensions; rendering larger than the widget gets clipped
        # and can appear as banding/stripes.
        try:
            wdg = self.query_one("#home_player_art", Static)
            sz = getattr(wdg, "size", None)
            if sz is not None:
                ww = int(getattr(sz, "width", 0) or 0)
                wh = int(getattr(sz, "height", 0) or 0)
                if ww > 0 and wh > 0:
                    return (max(8, ww), max(6, wh))
        except Exception:
            pass
        try:
            sz = getattr(self, "size", None)
            if sz is not None:
                tw = int(getattr(sz, "width", 0) or 0)
                th = int(getattr(sz, "height", 0) or 0)
                if tw > 0 and th > 0:
                    # Home is split 50/50; art lives in the left half. Use a large fraction
                    # of the terminal height so cover art is visually dominant.
                    w = max(22, min(120, int((tw * 0.5) - 6)))
                    h = max(10, min(60, int((th * 0.62))))
                    return (w, h)
        except Exception:
            pass
        return (40, 18)

    def _home_live_update_audio_anchor(self, pdt: Optional[datetime]) -> None:
        if pdt is None:
            return
        try:
            if pdt.tzinfo is None:
                pdt = pdt.replace(tzinfo=timezone.utc)
            else:
                pdt = pdt.astimezone(timezone.utc)
        except Exception:
            return
        try:
            prev = getattr(self, "_home_live_audio_anchor_pdt", None)
            if prev is not None:
                try:
                    if pdt <= prev:
                        return
                except Exception:
                    pass
            self._home_live_audio_anchor_pdt = pdt
            self._home_live_audio_anchor_wall = time.time()
        except Exception:
            pass

    def _home_live_audio_now_estimated(self) -> Optional[datetime]:
        try:
            if bool(getattr(self, "_live_paused", False)):
                frozen = getattr(self, "_home_live_paused_audio_now", None)
                if frozen is not None:
                    return frozen
        except Exception:
            pass
        try:
            ap = getattr(self, "_home_live_audio_anchor_pdt", None)
            aw = getattr(self, "_home_live_audio_anchor_wall", None)
            if ap is None or aw is None:
                return None
            dt_s = max(0.0, float(time.time() - float(aw)))
            return ap + timedelta(seconds=dt_s)
        except Exception:
            return None

    def _render_art_url_for_home(self, url: str, *, art_width: int, art_height: int) -> Text:
        u = (url or "").strip()
        if not u:
            return Text("")
        try:
            cache_dir = Path(user_cache_dir("satstash")) / "art"
            cache_dir.mkdir(parents=True, exist_ok=True)
            key = hashlib.sha1(u.encode("utf-8", errors="ignore")).hexdigest()
            img_path = cache_dir / f"{key}.img"
            if not img_path.exists() or img_path.stat().st_size < 100:
                data, _ctype = _fetch_image_bytes(u)
                if data:
                    img_path.write_bytes(data)
            if not img_path.exists() or img_path.stat().st_size < 100:
                return Text("")
            mode = (getattr(getattr(self, "settings", None), "art_render_mode", None) or "halfblock").strip().lower()
            if mode == "braille":
                return _image_to_rich_braille_fit(img_path, width=art_width, height=art_height)
            return _image_to_rich_blocks_fit(img_path, width=art_width, height=art_height)
        except Exception as exc:
            try:
                dbg_path = Path(user_cache_dir("satstash")) / "last_home_art_error.txt"
                dbg_path.parent.mkdir(parents=True, exist_ok=True)
                dbg_path.write_text(f"url={u}\nerr={exc}\n", encoding="utf-8")
            except Exception:
                pass
            return Text("")

    def _home_live_meta_maybe_refresh(self) -> None:
        # Throttle live_update polling; it can be relatively expensive.
        try:
            if self._live_handle is None or self._live_channel is None:
                return
        except Exception:
            return

        try:
            next_wall = float(getattr(self, "_home_live_meta_next_poll_wall", 0.0) or 0.0)
        except Exception:
            next_wall = 0.0
        now = time.time()
        if now < next_wall:
            return

        # Poll at ~2s cadence; proxy PDT interpolation keeps track changes aligned with audio.
        try:
            self._home_live_meta_next_poll_wall = now + 2.0
        except Exception:
            pass

        def work() -> None:
            try:
                ch = self._live_channel
                if ch is None:
                    return
                client = self._get_client()
                data = client.live_update(channel_id=ch.id)
                items = data.get("items") or []
                if not items:
                    try:
                        dbg_path = Path(user_cache_dir("satstash")) / "last_liveupdate_home_empty.json"
                        dbg_path.parent.mkdir(parents=True, exist_ok=True)
                        dbg_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
                    except Exception:
                        pass
                    try:
                        if not bool(getattr(self, "_home_live_meta_warned", False)):
                            self._home_live_meta_warned = True

                            def ui_warn() -> None:
                                try:
                                    self.notify("Live metadata: live_update returned no items")
                                except Exception:
                                    pass

                            self.call_from_thread(ui_warn)
                    except Exception:
                        pass
                    return

                def pick_str(item: dict, keys: tuple[str, ...]) -> str:
                    for k in keys:
                        try:
                            v = item.get(k)
                        except Exception:
                            v = None
                        if isinstance(v, str) and v.strip():
                            return v.strip()
                    return ""

                def parse_item_dt(item: dict) -> Optional[datetime]:
                    try:
                        ts = item.get("timestamp")
                        if isinstance(ts, str) and ts:
                            return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
                    except Exception:
                        return None
                    return None

                # Choose the item that best matches the actual audio time.
                audio_now: Optional[datetime] = None
                try:
                    lh = getattr(self, "_live_handle", None)
                    if lh is not None and getattr(lh, "proxy", None) is not None:
                        audio_now = lh.proxy.playhead_pdt(behind_segments=3)
                except Exception:
                    audio_now = None

                # Anchor/interpolate so we don't lag by a full HLS segment.
                try:
                    self._home_live_update_audio_anchor(audio_now)
                except Exception:
                    pass
                # Use the proxy segment PDT for selecting metadata (conservative).
                # An estimated/interpolated time can drift ahead and cause early track flips.
                audio_now_select = audio_now
                if audio_now_select is None:
                    audio_now_select = self._home_live_audio_now_estimated()

                parsed: list[tuple[Optional[datetime], dict]] = [(parse_item_dt(it), it) for it in items]
                parsed = [p for p in parsed if p[0] is not None]
                parsed.sort(key=lambda x: x[0] or datetime.min.replace(tzinfo=timezone.utc))
                current: dict = items[-1]
                start_dt: Optional[datetime] = None
                if parsed:
                    if audio_now_select is None:
                        current = parsed[-1][1]
                        start_dt = parsed[-1][0]
                    else:
                        chosen_i = len(parsed) - 1
                        for i, (dt, it) in enumerate(parsed):
                            if dt is None:
                                continue
                            if dt <= audio_now_select:
                                chosen_i = i
                            else:
                                break
                        current = parsed[chosen_i][1]
                        start_dt = parsed[chosen_i][0]
                        next_dt = parsed[chosen_i + 1][0] if (chosen_i + 1) < len(parsed) else None

                artist = pick_str(current, ("artistName", "artist", "artistDisplayName", "performerName", "hostName"))
                title = pick_str(current, ("name", "title", "songName", "episodeName", "segmentName"))
                album = pick_str(current, ("albumName", "album", "showName", "programName", "seriesName"))
                start_key = start_dt.isoformat() if start_dt is not None else str(current.get("timestamp") or "")
                track_key = "|".join([start_key, artist, title]).strip("|")

                # Only update UI when the track actually changes.
                prev_key = str(getattr(self, "_home_live_track_key", "") or "")
                if track_key and track_key == prev_key:
                    return

                try:
                    self._home_live_track_key = track_key
                except Exception:
                    pass

                channel_logo = ""
                try:
                    channel_logo = _normalize_art_url(str(getattr(ch, "logo_url", "") or ""))
                except Exception:
                    channel_logo = ""

                art_url = ""
                try:
                    art_url = _extract_art_url_from_live_item(current, channel_logo_url=channel_logo)
                except Exception:
                    art_url = channel_logo

                # If we couldn't derive meaningful metadata or artwork, persist the payload for debugging.
                if (not artist and not title and not album) or (not art_url):
                    try:
                        dbg_path = Path(user_cache_dir("satstash")) / "last_liveupdate_home.json"
                        dbg_path.parent.mkdir(parents=True, exist_ok=True)
                        dbg_path.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
                    except Exception:
                        pass

                label = " - ".join([x for x in [artist, title] if x]).strip()
                if not label:
                    label = f"{ch.number if ch.number is not None else ''} {ch.name}".strip() or "Live"

                # Persist latest live metadata/art for the 1s UI refresh loop to render.
                def ui_store() -> None:
                    try:
                        if getattr(self, "_live_channel", None) is None:
                            return
                        if str(getattr(self, "_home_live_track_key", "") or "") != track_key:
                            return
                    except Exception:
                        return
                    try:
                        self._home_live_display_track = label
                        self._home_live_display_album = album
                        self._home_live_display_art_url = art_url
                        self._home_live_display_updated_wall = time.time()
                        self._home_live_display_start_dt = start_dt
                        self._home_live_display_next_dt = next_dt
                        try:
                            dur_ms = current.get("duration") or 0
                        except Exception:
                            dur_ms = 0
                        # Prefer next_dt delta; else API duration.
                        dur_s: Optional[float] = None
                        if start_dt is not None and next_dt is not None:
                            try:
                                dur_s = max(0.0, float((next_dt - start_dt).total_seconds()))
                            except Exception:
                                dur_s = None
                        if dur_s is None:
                            try:
                                dur_s = float(dur_ms) / 1000.0 if dur_ms else None
                            except Exception:
                                dur_s = None
                        self._home_live_display_duration_s = dur_s
                    except Exception:
                        pass

                self.call_from_thread(ui_store)
            except Exception as exc:
                try:
                    dbg_path = Path(user_cache_dir("satstash")) / "last_liveupdate_home_error.txt"
                    dbg_path.parent.mkdir(parents=True, exist_ok=True)
                    dbg_path.write_text(str(exc), encoding="utf-8")
                except Exception:
                    pass
                try:
                    if not bool(getattr(self, "_home_live_meta_warned", False)):
                        self._home_live_meta_warned = True

                        def ui_warn() -> None:
                            try:
                                self.notify(f"Live metadata: failed ({exc})")
                            except Exception:
                                pass

                        self.call_from_thread(ui_warn)
                except Exception:
                    pass
                return

        try:
            # Do not mark this as exclusive; other long-running background work (downloads, etc)
            # can otherwise starve metadata updates.
            self.run_worker(work, thread=True)
        except Exception:
            pass

    def _refresh_status(self) -> None:
        sess = load_session()
        try:
            who = ""
            try:
                who = (getattr(self.settings, "auth_username", "") or "").strip()
            except Exception:
                who = ""

            pq = ""
            try:
                pq = str(getattr(self.settings, "preferred_quality", "") or "").strip()
            except Exception:
                pq = ""

            pl = ""
            try:
                pl = str(getattr(self.settings, "player_preference", "") or "").strip()
            except Exception:
                pl = ""

            if sess and sess.is_valid():
                sess_s = "OK"
            elif sess:
                sess_s = "expired"
            else:
                sess_s = "none"

            parts: list[str] = [f"Session: {sess_s}"]
            if who:
                parts.append(f"User: {who}")
            if pq:
                parts.append(f"Quality: {pq}")
            if pl:
                parts.append(f"Player: {pl}")
            pb = self._playback_summary()
            if pb:
                parts.append(pb)
            self.query_one("#status", Static).update("   ".join(parts))
        except Exception:
            try:
                self.query_one("#status", Static).update("")
            except Exception:
                pass

        # Update the home-screen mini-player if present. Guarded so other screens are unaffected.
        try:
            src = ""
            track = ""
            album = ""
            timing = ""
            total = 1.0
            pos = 0.0
            art_obj: object = ""
            art_url: str = ""
            hint = ""

            if self._dvr_player_proc is not None:
                aw, ah = self._home_art_box()
                info = self.get_dvr_now_playing(art_width=aw, art_height=ah)
                src = str(info.get("source") or "DVR").strip() or "DVR"
                track = str(info.get("track") or "").strip()
                album = str(info.get("album") or "").strip()
                art_obj = info.get("art") or ""
                hint = "Keys: space Play/Pause   n Next   b Prev   [ ] Seek   s Stop   p Playlists   h Home   S Schedule"
                try:
                    total = float(info.get("duration_s") or 0.0) or 1.0
                except Exception:
                    total = 1.0
                try:
                    pos = float(info.get("position_s") or 0.0)
                except Exception:
                    pos = 0.0
                pos = max(0.0, min(pos, total))
                timing = f"{_fmt_time(pos)} / {_fmt_time(total)}"
            elif self._live_handle is not None and self._live_channel is not None:
                ch = self._live_channel
                label = f"{ch.number if ch.number is not None else ''} {ch.name}".strip() or "Live"
                src = "Live"
                hint = "Keys: space Play/Pause   [ ] Seek   s Stop   p Playlists   h Home   S Schedule"
                # Kick a background refresh of live track metadata/art.
                try:
                    self._home_live_meta_maybe_refresh()
                except Exception:
                    pass

                # Prefer stored live_update metadata (kept stable across 1s refresh ticks).
                track = label
                album = "paused" if bool(self._live_paused) else "playing"
                try:
                    stored_track = str(getattr(self, "_home_live_display_track", "") or "").strip()
                    stored_album = str(getattr(self, "_home_live_display_album", "") or "").strip()
                    stored_art = str(getattr(self, "_home_live_display_art_url", "") or "").strip()
                except Exception:
                    stored_track = ""
                    stored_album = ""
                    stored_art = ""

                if stored_track:
                    track = stored_track
                if stored_album:
                    album = stored_album
                if stored_art:
                    art_url = stored_art
                else:
                    try:
                        art_url = _normalize_art_url(str(getattr(ch, "logo_url", "") or ""))
                    except Exception:
                        art_url = ""
                # Progress/timing based on audio-time and stored track boundaries.
                total = 1.0
                pos = 0.0
                timing = ""

                try:
                    lh = getattr(self, "_live_handle", None)
                    pdt = None
                    if lh is not None and getattr(lh, "proxy", None) is not None:
                        pdt = lh.proxy.playhead_pdt(behind_segments=3)
                    self._home_live_update_audio_anchor(pdt)
                    audio_now = self._home_live_audio_now_estimated() or pdt
                except Exception:
                    audio_now = None

                try:
                    start_dt = getattr(self, "_home_live_display_start_dt", None)
                    dur_s = getattr(self, "_home_live_display_duration_s", None)
                    if audio_now is not None and start_dt is not None and dur_s is not None and float(dur_s) > 0:
                        try:
                            if start_dt.tzinfo is None:
                                start_dt = start_dt.replace(tzinfo=timezone.utc)
                            else:
                                start_dt = start_dt.astimezone(timezone.utc)
                        except Exception:
                            pass
                        try:
                            if audio_now.tzinfo is None:
                                audio_now = audio_now.replace(tzinfo=timezone.utc)
                            else:
                                audio_now = audio_now.astimezone(timezone.utc)
                        except Exception:
                            pass

                        total = max(1.0, float(dur_s))
                        try:
                            pos = float((audio_now - start_dt).total_seconds())
                        except Exception:
                            pos = 0.0
                        pos = max(0.0, min(pos, total))
                        timing = f"{_fmt_time(pos)} / {_fmt_time(total)}"
                except Exception:
                    pass
            else:
                src = ""
                track = "(nothing playing)"
                album = ""
                timing = ""
                total = 1.0
                pos = 0.0
                art_obj = ""
                art_url = ""
                hint = "Keys: h Home   p Playlists   S Schedule   Q Quit"

            try:
                self.query_one("#home_player_source", Static).update(src)
                self.query_one("#home_player_track", Static).update(track)
                self.query_one("#home_player_album", Static).update(album)
                try:
                    self.query_one("#home_player_hint", Static).update(hint)
                except Exception:
                    pass
            except Exception:
                pass

            try:
                art_w = self.query_one("#home_player_art", Static)
                if art_obj:
                    art_w.update(art_obj)
                    try:
                        self._home_art_url = ""
                        self._home_art_box_last = None
                    except Exception:
                        pass
                elif art_url:
                    prev = str(getattr(self, "_home_art_url", "") or "")
                    try:
                        aw, ah = self._home_art_box()
                    except Exception:
                        aw, ah = (0, 0)
                    try:
                        last_box = getattr(self, "_home_art_box_last", None)
                    except Exception:
                        last_box = None
                    box_changed = False
                    try:
                        if isinstance(last_box, tuple) and len(last_box) == 2:
                            box_changed = (int(last_box[0]) != int(aw) or int(last_box[1]) != int(ah))
                        else:
                            box_changed = True
                    except Exception:
                        box_changed = True
                    if art_url != prev:
                        try:
                            self._home_art_url = art_url
                            self._home_art_box_last = (int(aw), int(ah))
                        except Exception:
                            pass
                        art_w.update("Loading artwork...")

                        def work() -> None:
                            rendered = self._render_art_url_for_home(art_url, art_width=aw, art_height=ah)

                            def ui() -> None:
                                try:
                                    if str(getattr(self, "_home_art_url", "") or "") != art_url:
                                        return
                                    self.query_one("#home_player_art", Static).update(rendered or "")
                                except Exception:
                                    pass

                            self.call_from_thread(ui)

                        self.run_worker(work, thread=True)
                    elif box_changed and prev:
                        # Same art URL, but the widget resized; re-render to the new dimensions.
                        try:
                            self._home_art_box_last = (int(aw), int(ah))
                        except Exception:
                            pass

                        def work2() -> None:
                            rendered = self._render_art_url_for_home(art_url, art_width=aw, art_height=ah)

                            def ui2() -> None:
                                try:
                                    if str(getattr(self, "_home_art_url", "") or "") != art_url:
                                        return
                                    self.query_one("#home_player_art", Static).update(rendered or "")
                                except Exception:
                                    pass

                            self.call_from_thread(ui2)

                        self.run_worker(work2, thread=True)
                else:
                    art_w.update("")
                    try:
                        self._home_art_url = ""
                        self._home_art_box_last = None
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                self.query_one("#home_player_progress", ProgressBar).update(total=total, progress=pos)
                self.query_one("#home_player_timing", Static).update(timing)
            except Exception:
                pass
        except Exception:
            pass

        try:
            rec = self._record_handle
            ch = self._record_channel
            started = self._record_started_at
            if ch and started:
                # Detect unexpected ffmpeg exit.
                try:
                    if rec is not None and rec.process.poll() is not None:
                        rc = rec.process.returncode

                        # If ffmpeg exited cleanly, treat it as a normal completion.
                        if rc == 0:
                            handle = rec
                            channel = ch

                            track_index: list[dict]
                            with self._record_single_track_lock:
                                track_index = list(self._record_single_track_index)
                                self._record_single_track_index = []
                                self._record_single_started_wall = None

                            self._record_handle = None
                            self._record_channel = None
                            self._record_started_at = None
                            self._record_pending = False
                            self.query_one("#recording_status", Static).update("")

                            def work_done() -> None:
                                out_path = None
                                cue_path = None
                                ffmeta_path = None
                                dur_s: Optional[float] = None
                                try:
                                    out_path = stop_recording_with_options(handle, finalize_on_stop=False)
                                except Exception as exc:
                                    def ui_fail() -> None:
                                        try:
                                            self.notify(f"Recording finalize failed: {exc}")
                                            self.notify(f"ffmpeg log: {handle.log_path}")
                                        except Exception:
                                            pass
                                    try:
                                        self.call_from_thread(ui_fail)
                                    except Exception:
                                        pass
                                    return

                                try:
                                    if out_path and out_path.exists() and out_path.suffix.lower() == ".m4a" and track_index:
                                        dur_s = _probe_duration_s(out_path)
                                        safe_tracks = _sanitize_track_index(tracks=track_index, duration_s=dur_s)
                                        cue_path = _write_cue(audio_path=out_path, tracks=safe_tracks, duration_s=dur_s)
                                        ffmeta_path = _write_ffmetadata(audio_path=out_path, tracks=safe_tracks, duration_s=dur_s)
                                        try:
                                            if ffmeta_path and ffmeta_path.exists() and shutil.which("ffmpeg"):
                                                tmp_ch = _unique_path(out_path.with_name(out_path.stem + ".__chapters.m4a"))
                                                _ffmpeg_mux_chapters_from_ffmeta(src=out_path, ffmeta=ffmeta_path, dst=tmp_ch)
                                                try:
                                                    if tmp_ch.exists() and tmp_ch.stat().st_size > 0:
                                                        tmp_ch.replace(out_path)
                                                except Exception:
                                                    pass
                                        except Exception:
                                            pass
                                except Exception:
                                    cue_path = None
                                    ffmeta_path = None

                                def ui_ok() -> None:
                                    try:
                                        if out_path is None:
                                            self.notify("Recording finished")
                                            self.notify(f"ffmpeg log: {handle.log_path}")
                                            return
                                        try:
                                            if out_path.exists() and out_path.stat().st_size <= 0:
                                                self.notify(f"Recording finished but output was empty: {out_path}")
                                                self.notify(f"ffmpeg log: {handle.log_path}")
                                                return
                                        except Exception:
                                            pass

                                        self.notify(f"Recording saved: {out_path}")
                                        try:
                                            if cue_path and cue_path.exists() and cue_path.stat().st_size > 0:
                                                self.notify(f"CUE: {cue_path}")
                                        except Exception:
                                            pass
                                        try:
                                            if ffmeta_path and ffmeta_path.exists() and ffmeta_path.stat().st_size > 0:
                                                self.notify(f"Chapters: {ffmeta_path}")
                                        except Exception:
                                            pass
                                    except Exception:
                                        pass

                                try:
                                    self.call_from_thread(ui_ok)
                                except Exception:
                                    pass

                            try:
                                self.run_worker(work_done, thread=True, exclusive=False)
                            except Exception:
                                threading.Thread(target=work_done, daemon=True).start()
                            return

                        self._record_handle = None
                        self._record_channel = None
                        self._record_started_at = None
                        self._record_pending = False
                        self.query_one("#recording_status", Static).update("")
                        self.notify(
                            f"Recording stopped unexpectedly (code {rc}). Log: {rec.log_path}"
                        )
                        return
                except Exception:
                    pass

                elapsed = max(0, int(time.time() - started))
                m, s = divmod(elapsed, 60)
                h, m = divmod(m, 60)
                t = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
                blink_on = (int(time.time()) % 2) == 0
                label = f"{ch.number if ch.number is not None else ''} {ch.name}".strip()
                if rec is not None:
                    try:
                        tt = Text()
                        tt.append("● ", style="red" if blink_on else "")
                        tt.append(f"Recording: {label} ({t})\n{rec.final_path}")
                        self.query_one("#recording_status", Static).update(tt)
                    except Exception:
                        self.query_one("#recording_status", Static).update(
                            f"{'●' if blink_on else ' '} Recording: {label} ({t})\n{rec.final_path}"
                        )
                else:
                    try:
                        tt = Text()
                        tt.append("● ", style="red" if blink_on else "")
                        tt.append(f"Recording: {label} ({t})\n(starting...)")
                        self.query_one("#recording_status", Static).update(tt)
                    except Exception:
                        self.query_one("#recording_status", Static).update(
                            f"{'●' if blink_on else ' '} Recording: {label} ({t})\n(starting...)"
                        )
            else:
                self.query_one("#recording_status", Static).update("")
        except Exception:
            pass

    def _prompt_login_and_retry(self, *, after_login: Callable[[], None]) -> None:
        # Prefer a silent re-login if credentials are configured.
        try:
            if self._try_silent_relogin():
                self._client = None
                self._refresh_status()
                self.notify("Logged in, retrying...")
                self.run_worker(after_login, thread=True, exclusive=True)
                return
        except Exception:
            pass

        def done(ok: bool) -> None:
            if ok:
                self._client = None
                self._refresh_status()
                self.notify("Logged in, retrying...")
                self.run_worker(after_login, thread=True, exclusive=True)

        self.push_screen(LoginScreen(), done)

    def _try_silent_relogin(self) -> bool:
        try:
            settings = self.settings
        except Exception:
            settings = load_settings()
        username = (getattr(settings, "auth_username", "") or "").strip()
        password = getattr(settings, "auth_password", "") or ""
        if not username or not password:
            return False
        try:
            result = SiriusXMDirectAuth().authenticate(username, password)
            save_session(result.bearer_token, result.cookies, lifetime_hours=12)
            return True
        except Exception:
            return False

    def _refresh_session_from_cookies(self) -> None:
        sess = load_session()
        if not sess:
            raise NotLoggedInError("Not logged in")
        with self._auth_lock:
            latest = load_session() or sess
            refreshed = SiriusXMDirectAuth().refresh_with_cookies((latest or sess).cookies)
            if not refreshed:
                if self._try_silent_relogin():
                    return
                raise SessionExpiredError("Session expired")
            save_session(refreshed.bearer_token, refreshed.cookies, lifetime_hours=12)

    def _client_on_unauthorized(self) -> Optional[tuple[str, dict[str, str]]]:
        self._refresh_session_from_cookies()
        sess = load_session()
        if not sess:
            return None
        return sess.bearer_token, sess.cookies

    def _get_client(self, *, force_refresh: bool = False) -> SxmClient:
        sess = load_session()
        if not sess:
            # If we have saved credentials, prefer a silent re-login over prompting.
            try:
                if self._try_silent_relogin():
                    sess = load_session()
            except Exception:
                pass
            if not sess:
                raise NotLoggedInError("Not logged in")

        # Refresh the bearer token if expired or nearing expiry.
        if force_refresh or (not sess.is_valid()) or sess.is_expiring_soon(threshold_seconds=5 * 60):
            with self._auth_lock:
                latest = load_session() or sess
                if latest and latest.is_valid() and not latest.is_expiring_soon(threshold_seconds=5 * 60):
                    sess = latest
                else:
                    refreshed = SiriusXMDirectAuth().refresh_with_cookies((latest or sess).cookies)
                    if not refreshed:
                        # If cookie-based refresh fails (expired cookies), fall back to
                        # saved credentials (if configured) before forcing interactive login.
                        try:
                            if self._try_silent_relogin():
                                sess = load_session() or sess
                                refreshed = None
                            else:
                                raise SessionExpiredError("Session expired")
                        except SessionExpiredError:
                            raise
                        except Exception:
                            raise SessionExpiredError("Session expired")
                    # Keep the existing 12h cache lifetime; Session.is_valid() prefers JWT exp when present.
                    if refreshed is not None:
                        save_session(refreshed.bearer_token, refreshed.cookies, lifetime_hours=12)
                        sess = load_session() or sess

        if not sess.is_valid():
            # Session might be missing/invalid after refresh; try saved-credentials login.
            try:
                if self._try_silent_relogin():
                    sess = load_session() or sess
            except Exception:
                pass
            if not sess.is_valid():
                raise NotLoggedInError("Not logged in")

        if not self._client:
            self._client = SxmClient(
                bearer_token=sess.bearer_token,
                cookies=sess.cookies,
                on_unauthorized=self._client_on_unauthorized,
            )
        else:
            self._client.set_bearer(sess.bearer_token)
        return self._client

    def start_live(self, *, channel: Channel, push_screen: bool = True) -> None:
        # Prevent multiple player instances if the user double-clicks play.
        try:
            # Live and DVR should be mutually exclusive.
            self.stop_dvr_playback()
        except Exception:
            pass

        try:
            self.stop_live()
        except Exception:
            pass

        self._record_mode = "single"
        self._record_stop = None
        self._record_split_proxy = None

        client = self._get_client()
        chan_type = channel.channel_type or "channel-linear"

        def progress(msg: str) -> None:
            # start_live_playback emits a lot of step-by-step progress messages
            # (variant/proxy/preflight/player). In the split-pane UI this can spam
            # over controls, so only show them when debug is enabled.
            if not bool(self._debug_enabled):
                return
            self.call_from_thread(lambda: self.notify(msg))

        try:
            label = f"{channel.number if channel.number is not None else ''} {channel.name}".strip() or channel.name
            self.notify(f"Loading: {label}")
        except Exception:
            pass

        try:
            handle = start_live_playback(
                client=client,
                channel_id=channel.id,
                channel_type=chan_type,
                preferred_quality=self.settings.preferred_quality or "256k",
                progress=progress,
                player_preference=self.settings.player_preference or "auto",
                debug=bool(self._debug_enabled),
            )
        except HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 401:
                # Client already attempted refresh+retry once.
                raise SessionExpiredError("Session expired")
            raise

        self._live_handle = handle
        self._live_channel = channel

        if push_screen:
            def show_player() -> None:
                self.push_screen(NowPlayingScreen(channel=channel))

            self.call_from_thread(show_player)

    def stop_live(self) -> None:
        handle = self._live_handle
        if not handle:
            return
        try:
            handle.process.terminate()
        except Exception:
            pass
        try:
            handle.proxy.stop(handle.proxy_info)
        except Exception:
            pass
        self._live_handle = None
        self._live_channel = None
        self._live_paused = False
        self.notify("Live playback stopped")

    def start_record_now(self, *, channel: Channel) -> None:
        self.start_record_single(channel=channel)

    def start_record_single(self, *, channel: Channel) -> None:
        if self._record_handle is not None:
            self.notify("A recording is already running")
            return

        # Set pending state immediately so UI shows activity even while ffmpeg/proxy spin up.
        self._record_channel = channel
        self._record_started_at = time.time()
        self._record_pending = True

        self._record_mode = "single"
        stop_ev = threading.Event()
        self._record_stop = stop_ev
        self._record_split_proxy = None

        client = self._get_client()
        chan_type = channel.channel_type or "channel-linear"

        def progress(msg: str) -> None:
            self.call_from_thread(lambda: self.notify(msg))

        def work():
            try:
                if stop_ev.is_set():
                    return
                base_dir = _output_category_dir(self.settings, "Live")
                chan_label = f"{channel.number if channel.number is not None else ''} {channel.name}".strip() or "channel"
                safe_chan = "".join(c if c.isalnum() or c in " _-." else "_" for c in chan_label).strip() or "channel"
                out_dir = base_dir / safe_chan
                label = f"{channel.number if channel.number is not None else ''} {channel.name}".strip()
                self.call_from_thread(lambda: self.notify("Starting recording..."))

                start_pdt: Optional[datetime] = None
                if bool(getattr(self.settings, "start_record_from_track_start", True)):
                    try:
                        lu = client.live_update(channel_id=channel.id)
                        items = (lu.get("items") or [])
                        cur = items[-1] if items else {}
                        ts = cur.get("timestamp")
                        if isinstance(ts, str) and ts:
                            start_pdt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
                    except Exception:
                        start_pdt = None
                handle = start_recording(
                    client=client,
                    channel_id=channel.id,
                    channel_type=chan_type,
                    preferred_quality=self.settings.preferred_quality or "256k",
                    out_dir=out_dir,
                    title=label or "satstash",
                    progress=progress,
                    debug=bool(self._debug_enabled),
                    start_pdt=start_pdt,
                )
            except Exception as exc:
                def fail() -> None:
                    self._record_handle = None
                    self._record_channel = None
                    self._record_started_at = None
                    self._record_pending = False
                    self.notify(f"Recording failed: {exc}")

                self.call_from_thread(fail)
                return

            # NOTE: cue offsets must be anchored to when audio starts getting written.
            # Recording startup (proxy + ffmpeg) can take a few seconds, which would
            # make all offsets early and cause "tail of previous track" when jumping.
            started_wall: Optional[datetime] = None

            def wait_for_audio_start() -> datetime:
                # Wait briefly until ffmpeg writes some bytes.
                deadline = time.time() + 12.0
                while time.time() < deadline:
                    if stop_ev.is_set():
                        return datetime.now(timezone.utc)
                    try:
                        if handle.tmp_path.exists() and handle.tmp_path.stat().st_size > 128 * 1024:
                            return datetime.now(timezone.utc)
                    except Exception:
                        pass
                    time.sleep(0.25)
                return datetime.now(timezone.utc)

            started_wall = wait_for_audio_start()
            # Anchor recording time-zero.
            # Use the effective_start_pdt that the recorder actually used (it may have
            # fallen back to live edge). This keeps chapters/cues consistent with the
            # audio we truly captured.
            base_hls_pdt_start: Optional[datetime] = None
            eff_start = getattr(handle, "effective_start_pdt", None)
            if eff_start is not None:
                base_hls_pdt_start = eff_start
            else:
                base_hls_pdt_start = _get_playlist_pdt_at_or_before(handle.proxy_info.url, started_wall)

            def poll_index() -> None:
                last_id = None
                base_track_ts: Optional[datetime] = None
                base_hls_pdt: Optional[datetime] = base_hls_pdt_start
                last_playlist_poll: float = 0.0
                last_timeline: list[tuple[datetime, float, float]] = []
                while True:
                    try:
                        if stop_ev.is_set():
                            return
                        if handle.process.poll() is not None:
                            return
                    except Exception:
                        return

                    # Poll the local proxy playlist periodically to anchor offsets to HLS PDT.
                    try:
                        now = time.time()
                        if now - last_playlist_poll >= 1.0:
                            last_playlist_poll = now
                            r = requests.get(handle.proxy_info.url, timeout=5)
                            if r.status_code == 200:
                                tl = _parse_hls_timeline(r.text or "")
                                if tl:
                                    last_timeline = tl
                                    # Keep base_hls_pdt stable once set; it represents the start of the file.
                    except Exception:
                        pass

                    try:
                        data = client.live_update(channel_id=channel.id)
                        items = data.get("items") or []
                        cur: dict = items[-1] if items else {}

                        # Choose item based on the actual audio time (proxy segment PDT).
                        audio_now: Optional[datetime] = None
                        try:
                            audio_now = handle.proxy.playhead_pdt(behind_segments=3)
                        except Exception:
                            audio_now = None

                        def parse_item_dt(item: dict) -> Optional[datetime]:
                            try:
                                ts = item.get("timestamp")
                                if isinstance(ts, str) and ts:
                                    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
                            except Exception:
                                return None
                            return None

                        start_dt: Optional[datetime] = None
                        if items and audio_now is not None:
                            try:
                                parsed: list[tuple[datetime, dict]] = []
                                for it in items:
                                    dt = parse_item_dt(it)
                                    if dt is not None:
                                        parsed.append((dt, it))
                                parsed.sort(key=lambda x: x[0])
                                if parsed:
                                    chosen = parsed[0][1]
                                    chosen_dt = parsed[0][0]
                                    for dt, it in parsed:
                                        if dt <= audio_now:
                                            chosen = it
                                            chosen_dt = dt
                                        else:
                                            break
                                    cur = chosen
                                    start_dt = chosen_dt
                            except Exception:
                                pass

                        if start_dt is None:
                            start_dt = parse_item_dt(cur)
                        if start_dt is None:
                            start_dt = datetime.now(timezone.utc)

                        tid = cur.get("id")
                        if not tid:
                            try:
                                tid = start_dt.isoformat()
                            except Exception:
                                tid = None

                        if tid and tid != last_id:

                            # Prefer HLS playlist timeline offsets (EXT-X-PROGRAM-DATE-TIME), fall back
                            # to a track-timestamp baseline.
                            offset_s: float
                            if base_hls_pdt is not None:
                                # Fast path: use absolute PDT delta against file start segment.
                                offset_s = (start_dt - base_hls_pdt).total_seconds()
                            else:
                                if base_track_ts is None:
                                    base_track_ts = start_dt
                                offset_s = (start_dt - base_track_ts).total_seconds()
                            artist = cur.get("artistName") or ""
                            title = cur.get("name") or ""
                            album = cur.get("albumName") or ""
                            display = " - ".join([x for x in [artist, title] if x]) or title or artist
                            rec = {
                                "id": tid,
                                "offset_s": max(0.0, float(offset_s)),
                                "artist": str(artist),
                                "title": str(title),
                                "album": str(album),
                                "display": str(display),
                            }
                            with self._record_single_track_lock:
                                self._record_single_started_wall = started_wall
                                self._record_single_track_index.append(rec)
                            last_id = tid
                    except Exception:
                        pass

                    time.sleep(2.0)

            def ui():
                if stop_ev.is_set():
                    try:
                        stop_recording_with_options(handle, finalize_on_stop=False)
                    except Exception:
                        pass
                    self._record_handle = None
                    self._record_channel = None
                    self._record_started_at = None
                    self._record_pending = False
                    return
                self._record_handle = handle
                self._record_channel = channel
                # Keep the original start timestamp so the timer includes startup.
                if not self._record_started_at:
                    self._record_started_at = time.time()
                self._record_pending = False
                with self._record_single_track_lock:
                    self._record_single_started_wall = started_wall
                    self._record_single_track_index = []
                self.notify(f"Recording started: {handle.final_path}")
                try:
                    self.notify(f"ffmpeg log: {handle.log_path}")
                except Exception:
                    pass

                # Start track index polling (non-exclusive) so it doesn't cancel other workers.
                self.run_worker(poll_index, thread=True, exclusive=False)

            self.call_from_thread(ui)

        self.run_worker(work, thread=True, exclusive=True)

    def start_record_tracks(self, *, channel: Channel) -> None:
        if self._record_handle is not None:
            self.notify("A recording is already running")
            return

        # UI shows pending immediately.
        self._record_channel = channel
        self._record_started_at = time.time()
        self._record_pending = True
        self._record_mode = "tracks"
        stop_ev = threading.Event()
        self._record_stop = stop_ev

        client = self._get_client()
        chan_type = channel.channel_type or "channel-linear"

        def progress(msg: str) -> None:
            self.call_from_thread(lambda: self.notify(msg))

        def work() -> None:
            proxy: Optional[HlsProxy] = None
            proxy_info = None
            current_ff: Optional[FfmpegRecordHandle] = None
            try:
                base_dir = _output_category_dir(self.settings, "Live")
                chan_label = f"{channel.number if channel.number is not None else ''} {channel.name}".strip() or "channel"
                safe_chan = "".join(c if c.isalnum() or c in " _-." else "_" for c in chan_label).strip() or "channel"
                out_dir = base_dir / safe_chan
                out_dir.mkdir(parents=True, exist_ok=True)

                # Tune + proxy (same as record_now but kept alive across tracks)
                manifest_variant = "WEB" if chan_type == "channel-linear" else "FULL"
                progress("tuneSource...")
                tune = client.tune_source(entity_id=channel.id, entity_type=chan_type, manifest_variant=manifest_variant)
                master = tune.master_url()
                if not master:
                    raise RuntimeError("tuneSource returned no master URL")

                progress("selecting stream variant...")
                sel = select_variant(master, prefer=self.settings.preferred_quality or "256k", headers=client.session.headers, client=client)
                progress(f"variant: {sel.variant_url}")

                progress("starting local proxy...")
                proxy = HlsProxy(client, sel.variant_url)
                proxy_info = proxy.start()

                # Option 1: wait for next track boundary
                progress("Waiting for next track boundary...")
                last_id: Optional[str] = None
                while not stop_ev.is_set():
                    data = client.live_update(channel_id=channel.id)
                    items = data.get("items") or []
                    current = items[-1] if items else {}
                    tid = current.get("id")
                    if last_id is None:
                        last_id = tid
                    elif tid and tid != last_id:
                        last_id = tid
                        break
                    time.sleep(1.0)

                if stop_ev.is_set():
                    return

                # Now record each track until stop.
                while not stop_ev.is_set():
                    data = client.live_update(channel_id=channel.id)
                    items = data.get("items") or []
                    current = items[-1] if items else {}
                    artist = (current.get("artistName") or "").strip()
                    title = (current.get("name") or "").strip()
                    album = (current.get("albumName") or "").strip()
                    tid = current.get("id")
                    if not tid:
                        time.sleep(1.0)
                        continue

                    ts_local = datetime.now().strftime("%Y-%m-%d %H%M")
                    base_name = " - ".join([x for x in [artist, title] if x])
                    if not base_name:
                        base_name = album or "Unknown"
                    safe_name = "".join(c if c.isalnum() or c in " _-." else "_" for c in base_name).strip() or "track"
                    final_path = out_dir / f"{ts_local} - {safe_name}.m4a"
                    tmp_dir = out_dir / ".satstash_tmp"
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    tmp_path = tmp_dir / (final_path.name + ".part")
                    log_path = Path(user_cache_dir("satstash")) / f"ffmpeg-track-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

                    progress(f"Recording track: {base_name}")
                    current_ff = start_ffmpeg_recording(
                        input_url=str(getattr(proxy_info, "url", proxy_info)),
                        tmp_path=tmp_path,
                        final_path=final_path,
                        log_path=log_path,
                        container="mp4-aac",
                        debug=bool(self._debug_enabled),
                    )

                    def ui_start() -> None:
                        self._record_handle = _SplitRecordShim(proxy=proxy, proxy_info=proxy_info, ff=current_ff)
                        self._record_pending = False
                        if not self._record_started_at:
                            self._record_started_at = time.time()

                    self.call_from_thread(ui_start)

                    # Wait until track changes.
                    while not stop_ev.is_set():
                        d2 = client.live_update(channel_id=channel.id)
                        items2 = d2.get("items") or []
                        cur2 = items2[-1] if items2 else {}
                        tid2 = cur2.get("id")
                        if tid2 and tid2 != tid:
                            break
                        time.sleep(1.0)

                    out_path = stop_ffmpeg_recording(current_ff, finalize=True)
                    current_ff = None

                    # Tag + embed artwork (best experience with ffplay and other players).
                    try:
                        if out_path.exists() and out_path.suffix.lower() == ".m4a":
                            logo_url = None
                            try:
                                if getattr(channel, "logo_url", None):
                                    logo_url = str(channel.logo_url)
                            except Exception:
                                logo_url = None
                            art_url = _extract_art_url_from_live_item(current, channel_logo_url=logo_url)
                            cover_bytes, cover_type = (b"", "")
                            if art_url:
                                try:
                                    cover_bytes, cover_type = _fetch_image_bytes(art_url)
                                except Exception:
                                    cover_bytes, cover_type = (b"", "")
                            _tag_m4a(
                                path=out_path,
                                title=title,
                                artist=artist,
                                album=album,
                                cover_bytes=cover_bytes,
                                cover_content_type=cover_type,
                            )
                    except Exception as exc:
                        self.call_from_thread(lambda: self.notify(f"Tagging failed (non-fatal): {exc}"))

                    def ui_done_one() -> None:
                        if out_path.exists() and out_path.suffix.lower() == ".m4a":
                            self.notify(f"Saved track: {out_path.name}")
                        try:
                            top = self.screen
                            if isinstance(top, BrowseDvrScreen):
                                top.reload()
                        except Exception:
                            pass

                    self.call_from_thread(ui_done_one)

            except Exception as exc:
                def fail() -> None:
                    self._record_handle = None
                    self._record_channel = None
                    self._record_started_at = None
                    self._record_pending = False
                    self._record_mode = "single"
                    self.notify(f"Track recording failed: {exc}")

                self.call_from_thread(fail)
            finally:
                try:
                    if current_ff is not None:
                        stop_ffmpeg_recording(current_ff, finalize=True)
                except Exception:
                    pass
                try:
                    if proxy is not None and proxy_info is not None:
                        proxy.stop(proxy_info)
                except Exception:
                    pass

        self.run_worker(work, thread=True, exclusive=True)

    def stop_record_now(self, *, silent: bool = False) -> None:
        handle = self._record_handle
        pending = bool(getattr(self, "_record_pending", False))
        mode = (getattr(self, "_record_mode", "single") or "single")

        if handle is None and not pending:
            if not silent:
                self.notify("No active recording")
            return

        if mode == "tracks":
            try:
                if self._record_stop is not None:
                    self._record_stop.set()
            except Exception:
                pass

            if not silent:
                self.notify("Stopping track recording...")

            # Clear UI state immediately.
            self._record_handle = None
            self._record_channel = None
            self._record_started_at = None
            self._record_pending = False
            self._record_mode = "single"
            return

        # Single-file mode: allow stop during pending startup.
        try:
            if self._record_stop is not None:
                self._record_stop.set()
        except Exception:
            pass

        if handle is None:
            if not silent:
                self.notify("Stopping recording...")
            self._record_handle = None
            self._record_channel = None
            self._record_started_at = None
            self._record_pending = False
            return

        if not silent:
            self.notify("Stopping recording...")

        # Clear UI state immediately so it doesn't look stuck if ffmpeg takes a moment to exit.
        self._record_handle = None
        self._record_channel = None
        self._record_started_at = None
        self._record_pending = False

        track_index: list[dict]
        with self._record_single_track_lock:
            track_index = list(self._record_single_track_index)
            self._record_single_track_index = []
            self._record_single_started_wall = None

        def work():
            try:
                progress_msgs: list[str] = []

                def progress(msg: str) -> None:
                    try:
                        try:
                            progress_msgs.append(str(msg))
                        except Exception:
                            pass
                        self.call_from_thread(lambda: self.notify(str(msg)))
                    except Exception:
                        pass

                out_path = stop_recording_with_options(handle, finalize_on_stop=True, progress=progress)
            except Exception as exc:
                if not silent:
                    self.call_from_thread(lambda: self.notify(f"Stop recording failed: {exc}"))
                return

            cue_path = None
            ffmeta_path = None
            dur_s: Optional[float] = None
            try:
                if out_path.exists() and out_path.suffix.lower() == ".m4a" and track_index:
                    dur_s = _probe_duration_s(out_path)
                    safe_tracks = _sanitize_track_index(tracks=track_index, duration_s=dur_s)
                    cue_path = _write_cue(audio_path=out_path, tracks=safe_tracks, duration_s=dur_s)
                    ffmeta_path = _write_ffmetadata(audio_path=out_path, tracks=safe_tracks, duration_s=dur_s)
                    # Embed chapters into the m4a for better player compatibility (VLC, etc.).
                    try:
                        if ffmeta_path and ffmeta_path.exists() and shutil.which("ffmpeg"):
                            tmp_ch = _unique_path(out_path.with_name(out_path.stem + ".__chapters.m4a"))
                            _ffmpeg_mux_chapters_from_ffmeta(src=out_path, ffmeta=ffmeta_path, dst=tmp_ch)
                            try:
                                if tmp_ch.exists() and tmp_ch.stat().st_size > 0:
                                    tmp_ch.replace(out_path)
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                cue_path = None
                ffmeta_path = None

            def ui():
                if not silent:
                    dur_suffix = ""
                    try:
                        if isinstance(dur_s, (int, float)) and dur_s and dur_s > 0:
                            dur_suffix = f" ({_fmt_time(float(dur_s))})"
                    except Exception:
                        dur_suffix = ""

                    repaired = False
                    try:
                        repaired = any("repair" in (m or "").lower() for m in progress_msgs)
                    except Exception:
                        repaired = False

                    if out_path == handle.final_path:
                        if repaired:
                            self.notify(f"Recording saved (repaired): {out_path}{dur_suffix}")
                            self.notify(f"ffmpeg log: {handle.log_path}")
                        else:
                            self.notify(f"Recording saved: {out_path}{dur_suffix}")
                    else:
                        self.notify(f"Recording stopped (partial): {out_path}{dur_suffix}")
                        self.notify(f"ffmpeg log: {handle.log_path}")

                    try:
                        if cue_path and cue_path.exists() and cue_path.stat().st_size > 0:
                            self.notify(f"CUE: {cue_path}")
                    except Exception:
                        pass
                    try:
                        if ffmeta_path and ffmeta_path.exists() and ffmeta_path.stat().st_size > 0:
                            self.notify(f"Chapters: {ffmeta_path}")
                    except Exception:
                        pass

                # If the user is browsing DVR, refresh so the new file appears right away.
                try:
                    top = self.screen
                    if isinstance(top, BrowseDvrScreen):
                        top.reload()
                except Exception:
                    pass

            self.call_from_thread(ui)

        self.run_worker(work, thread=True, exclusive=True)

    def fetch_channels(self, *, force_refresh: bool = False) -> List[Channel]:
        if not force_refresh:
            cached = load_cached_channels(max_age_hours=24)
            if cached:
                return cached

        client = self._get_client()
        try:
            raw = client.channel_list()
        except HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 401:
                # Client already performed refresh+retry once; treat as expired session.
                raise SessionExpiredError("Session expired")
            raise
        channels: List[Channel] = []
        for ch in raw:
            cid = ch.get("id")
            name = ch.get("name")
            if not cid or not name:
                continue
            num = ch.get("number")
            try:
                num_i = int(num) if num is not None else None
            except Exception:
                num_i = None
            channels.append(
                Channel(
                    id=str(cid),
                    name=str(name),
                    number=num_i,
                    description=ch.get("description"),
                    genre=ch.get("genre"),
                    logo_url=ch.get("logo_url"),
                    channel_type=ch.get("channel_type") or "channel-linear",
                )
            )
        try:
            save_cached_channels(channels)
        except Exception:
            pass
        return channels

    def on_button_pressed(self, event: Button.Pressed) -> None:
        # App-level routing for main menu / global controls. Stop propagation so
        # nested widgets or default handlers don't swallow clicks/enter presses.
        try:
            event.stop()
        except Exception:
            pass

        # If a screen is active that knows how to handle play/back, route those here.
        screen = self.screen
        if event.button.id in {"play", "back"} and hasattr(screen, "play_selected") and hasattr(screen, "go_back"):
            if event.button.id == "back":
                screen.go_back()
            else:
                screen.play_selected()
            return

        if event.button.id == "quit":
            self.exit()
            return

        if event.button.id == "logout":
            clear_session()
            self._client = None
            self._refresh_status()
            self.notify("Logged out")
            return

        if event.button.id == "login":
            def done(ok: bool) -> None:
                if ok:
                    self._client = None
                    self._refresh_status()
                    self.notify("Logged in")
            self.push_screen(LoginScreen(), done)
            return

        if event.button.id == "settings":
            self.push_screen(SettingsScreen(self.settings.player_preference))
            return

        if event.button.id == "schedule":
            try:
                self.action_schedule()
            except Exception as exc:
                try:
                    self.notify(f"Failed to open Schedule: {exc}")
                except Exception:
                    pass
            return

        if event.button.id == "listen_live":
            # Non-blocking: update the right pane immediately, then fetch channels in background.
            # This avoids UI freezes when auth refresh or network calls take time.
            try:
                self.notify("Listen Live: loading channels...")
            except Exception:
                pass
            try:
                self._set_right_pane(Static("Loading channels...", id="right_loading"))
            except Exception:
                # Right pane not available; fall back to legacy screen flow.
                try:
                    self.notify("Listen Live: right pane not available; using legacy screen")
                except Exception:
                    pass
                try:
                    self._open_live_select(client=None)
                except Exception as exc:
                    self.notify(f"Failed to load channels: {exc}")
                return

            def work() -> None:
                try:
                    # Ensure we have a valid client/session (may refresh).
                    self._get_client()
                    channels = self.fetch_channels(force_refresh=False)
                except (NotLoggedInError, SessionExpiredError):
                    def ui_login() -> None:
                        self.notify("Session expired. Please login again.")

                        def after_login() -> None:
                            try:
                                self._open_live_select_in_right_pane_async(client=None)
                            except Exception:
                                pass

                        self._prompt_login_and_retry(after_login=after_login)

                    self.call_from_thread(ui_login)
                    return
                except Exception as exc:
                    def ui_err() -> None:
                        try:
                            self.notify(f"Failed to load channels: {exc}")
                        except Exception:
                            pass
                        try:
                            self._set_right_pane(Static("Failed to load channels.", id="right_error"))
                        except Exception:
                            pass

                    self.call_from_thread(ui_err)
                    return

                def ui_ok() -> None:
                    try:
                        self._set_right_pane(LiveSelectPane(channels))
                    except Exception as exc2:
                        try:
                            self.notify(f"Failed to open Live UI: {exc2}")
                        except Exception:
                            pass

                self.call_from_thread(ui_ok)

            self.run_worker(work, thread=True, exclusive=True)
            return

        if event.button.id == "vod":
            try:
                self._set_right_pane(VodPane())
            except Exception as exc:
                try:
                    self.notify(f"Failed to open VOD: {exc}")
                except Exception:
                    pass
            return

        if event.button.id == "catch_up":
            try:
                self.notify("Catch Up: loading channels...")
            except Exception:
                pass
            try:
                self._set_right_pane(Static("Loading channels...", id="right_loading"))
            except Exception:
                # Right pane not available; fall back to legacy screen flow.
                try:
                    self.notify("Catch Up: right pane not available; using legacy screen")
                except Exception:
                    pass
                try:
                    channels = self.fetch_channels(force_refresh=False)
                    self.push_screen(CatchUpSelectScreen(channels))
                except Exception as exc:
                    self.notify(f"Failed to load channels: {exc}")
                return

            def work() -> None:
                try:
                    self._get_client()
                    channels = self.fetch_channels(force_refresh=False)
                except (NotLoggedInError, SessionExpiredError):
                    def ui_login() -> None:
                        self.notify("Session expired. Please login again.")

                        def after_login() -> None:
                            try:
                                self._set_right_pane(CatchUpSelectPane(self.fetch_channels(force_refresh=False)))
                            except Exception:
                                pass

                        self._prompt_login_and_retry(after_login=after_login)

                    self.call_from_thread(ui_login)
                    return
                except Exception as exc:
                    def ui_err() -> None:
                        try:
                            self.notify(f"Failed to load channels: {exc}")
                        except Exception:
                            pass
                        try:
                            self._set_right_pane(Static("Failed to load channels.", id="right_error"))
                        except Exception:
                            pass

                    self.call_from_thread(ui_err)
                    return

                def ui_ok() -> None:
                    try:
                        self._set_right_pane(CatchUpSelectPane(channels))
                    except Exception as exc2:
                        try:
                            self.notify(f"Failed to open Catch Up UI: {exc2}")
                        except Exception:
                            pass

                self.call_from_thread(ui_ok)

            self.run_worker(work, thread=True, exclusive=True)
            return

        if event.button.id == "record_now":
            try:
                self.notify("Record Now: loading channels...")
            except Exception:
                pass
            try:
                self._set_right_pane(Static("Loading channels...", id="right_loading"))
            except Exception:
                # Right pane not available; fall back to legacy screen flow.
                try:
                    self.notify("Record Now: right pane not available; using legacy screen")
                except Exception:
                    pass
                try:
                    channels = self.fetch_channels(force_refresh=False)
                    self.push_screen(RecordSelectScreen(channels))
                except Exception as exc:
                    self.notify(f"Failed to load channels: {exc}")
                return

            def work() -> None:
                try:
                    self._get_client()
                    channels = self.fetch_channels(force_refresh=False)
                except (NotLoggedInError, SessionExpiredError):
                    def ui_login() -> None:
                        self.notify("Session expired. Please login again.")

                        def after_login() -> None:
                            try:
                                self._set_right_pane(RecordSelectPane(self.fetch_channels(force_refresh=False)))
                            except Exception:
                                pass

                        self._prompt_login_and_retry(after_login=after_login)

                    self.call_from_thread(ui_login)
                    return
                except Exception as exc:
                    def ui_err() -> None:
                        try:
                            self.notify(f"Failed to load channels: {exc}")
                        except Exception:
                            pass
                        try:
                            self._set_right_pane(Static("Failed to load channels.", id="right_error"))
                        except Exception:
                            pass

                    self.call_from_thread(ui_err)
                    return

                def ui_ok() -> None:
                    try:
                        self._set_right_pane(RecordSelectPane(channels))
                    except Exception as exc2:
                        try:
                            self.notify(f"Failed to open Record Now UI: {exc2}")
                        except Exception:
                            pass

                self.call_from_thread(ui_ok)

            self.run_worker(work, thread=True, exclusive=True)
            return

        if event.button.id == "browse_dvr":
            try:
                d = Path(str(getattr(self.settings, "output_dir", "") or getattr(self.settings, "recordings_dir", "~/Music/SiriusXM"))).expanduser()
                self._set_right_pane(BrowseDvrPane(d))
            except Exception as exc:
                self.notify(f"Failed to open recordings: {exc}")
            return

        self.notify(f"Not implemented yet: {event.button.label}")

    def action_schedule(self) -> None:
        try:
            self._set_right_pane(ScheduledRecordingsPane())
        except Exception:
            # If right pane isn't available, fall back to a toast.
            self.notify("Schedule: right pane not available")
            return

    def _clear_right_pane(self) -> None:
        try:
            host = self.query_one("#right_content", Container)
        except Exception:
            return
        try:
            host.remove_children()
        except Exception:
            # Older Textual may not support remove_children
            try:
                for c in list(getattr(host, "children", []) or []):
                    try:
                        c.remove()
                    except Exception:
                        pass
            except Exception:
                pass
        try:
            host.mount(RightIdleLogoPane())
        except Exception:
            pass

    def _show_right_idle(self) -> None:
        try:
            self._set_right_pane(RightIdleLogoPane())
        except Exception:
            pass
        try:
            # Ensure the search input no longer captures spacebar/etc.
            try:
                self.set_focus(None)
            except Exception:
                pass
            try:
                self.query_one("#home_player", Container).focus()
            except Exception:
                pass
        except Exception:
            pass

    def _set_right_pane(self, w: Widget) -> None:
        try:
            host = self.query_one("#right_content", Container)
        except Exception:
            raise RuntimeError("right pane not available")
        try:
            host.remove_children()
        except Exception:
            try:
                for c in list(getattr(host, "children", []) or []):
                    try:
                        c.remove()
                    except Exception:
                        pass
            except Exception:
                pass
        host.mount(w)

    def _open_live_select_in_right_pane(self, *, client: Optional[SxmClient] = None) -> None:
        # client is accepted for symmetry with legacy path; we fetch channels via app method.
        try:
            channels = self.fetch_channels(force_refresh=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load channels: {exc}")
        self._set_right_pane(LiveSelectPane(channels))

    def _open_live_select_in_right_pane_async(self, *, client: Optional[SxmClient] = None) -> None:
        # client is accepted for symmetry; the worker will use fetch_channels.
        try:
            self._set_right_pane(Static("Loading channels...", id="right_loading"))
        except Exception:
            raise

        def work() -> None:
            try:
                channels = self.fetch_channels(force_refresh=False)
            except Exception as exc:
                def ui_err() -> None:
                    try:
                        self.notify(f"Failed to load channels: {exc}")
                    except Exception:
                        pass
                    try:
                        self._set_right_pane(Static("Failed to load channels.", id="right_error"))
                    except Exception:
                        pass

                self.call_from_thread(ui_err)
                return

            def ui_ok() -> None:
                try:
                    self._set_right_pane(LiveSelectPane(channels))
                except Exception as exc2:
                    try:
                        self.notify(f"Failed to open Live UI: {exc2}")
                    except Exception:
                        pass

            self.call_from_thread(ui_ok)

        self.run_worker(work, thread=True, exclusive=True)

    def on_exit(self) -> None:
        try:
            scr = self.screen
            if isinstance(scr, CatchUpScreen):
                scr.cancel_export()
        except Exception:
            pass
        try:
            self.stop_live()
        except Exception:
            pass
        try:
            self.stop_record_now(silent=True)
        except Exception:
            pass

    def _open_live_select(self, client: Optional[SxmClient] = None) -> None:
        try:
            channels = self.fetch_channels(force_refresh=False)
            self.push_screen(LiveSelectScreen(channels))
        except (NotLoggedInError, SessionExpiredError):
            self.notify("Session expired. Please login again.")

            def done(ok: bool) -> None:
                if ok:
                    self._client = None
                    self._refresh_status()
                    try:
                        channels2 = self.fetch_channels(force_refresh=True)
                        self.push_screen(LiveSelectScreen(channels2))
                    except Exception as exc:
                        self.notify(f"Failed to load channels: {exc}")

            self.push_screen(LoginScreen(), done)


class NowPlayingScreen(Screen[None]):
    DEFAULT_CSS = """
    NowPlayingScreen {
        layout: vertical;
    }

    /* Now Playing layout CSS (art + metadata) */
    #art {
        width: 7fr;
        height: 1fr;
        overflow: hidden;
    }
    #main {
        layout: horizontal;
        height: 1fr;
    }
    #right {
        layout: vertical;
        height: 1fr;
        width: 3fr;
    }
    #meta {
        layout: vertical;
        height: 1fr;
    }
    #controls {
        height: auto;
    }
    #progress {
        width: 1fr;
    }
    """
    def __init__(self, *, channel: Channel):
        super().__init__()
        self._channel = channel
        self._poll_meta = None
        self._poll_progress = None
        self._track_id: Optional[str] = None
        self._track_start: Optional[datetime] = None
        self._track_duration_s: Optional[float] = None
        self._art_url: Optional[str] = None
        self._login_prompted: bool = False
        self._audio_anchor_pdt: Optional[datetime] = None
        self._audio_anchor_wall: Optional[float] = None

    def _update_audio_anchor(self, pdt: Optional[datetime]) -> None:
        if pdt is None:
            return
        try:
            if pdt.tzinfo is None:
                pdt = pdt.replace(tzinfo=timezone.utc)
            else:
                pdt = pdt.astimezone(timezone.utc)
        except Exception:
            return
        try:
            # Only reset the wall-clock anchor when the segment PDT actually advances.
            # The proxy PDT updates roughly once per segment (~10s). If we reset the
            # wall clock every second while PDT is unchanged, interpolation will never
            # advance and the progress bar will appear to update only every 10s.
            if self._audio_anchor_pdt is not None:
                try:
                    if pdt <= self._audio_anchor_pdt:
                        return
                except Exception:
                    pass
            self._audio_anchor_pdt = pdt
            self._audio_anchor_wall = time.time()
        except Exception:
            pass

    def _audio_now_estimated(self) -> Optional[datetime]:
        """Estimate current audio PDT using last observed segment PDT + wall-clock delta.

        The proxy's segment PDT only advances per-segment (~10s). Interpolating gives
        smooth 1s progress updates.
        """
        try:
            ap = self._audio_anchor_pdt
            aw = self._audio_anchor_wall
            if ap is None or aw is None:
                return None
            dt_s = max(0.0, float(time.time() - float(aw)))
            return ap + timedelta(seconds=dt_s)
        except Exception:
            return None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container():
            title = f"Now Playing: {self._channel.number if self._channel.number is not None else ''} {self._channel.name}".strip()
            yield Static(title, id="title")
            yield Static("", id="record_status")
            with Horizontal(id="main"):
                yield Static("", id="art")
                with Vertical(id="right"):
                    with Vertical(id="meta"):
                        yield Static("", id="track")
                        yield Static("", id="album")
                        yield ProgressBar(total=1, show_eta=False, id="progress")
                        yield Static("", id="timing")
                    with Horizontal(id="controls"):
                        yield Button("Stop", id="stop", variant="primary")
                        yield Button("Start Recording", id="stop_record")
                        yield Button("Back", id="back")
        yield Footer()

    def on_mount(self) -> None:
        self._closing = False
        self._refresh()
        self._poll_meta = self.set_interval(10.0, self._refresh)
        self._poll_progress = self.set_interval(1.0, self._update_progress)
        self.set_interval(1.0, self._refresh_record_status)

    def on_unmount(self) -> None:
        # Ensure background callbacks cannot fire after the screen is closed.
        self._closing = True
        try:
            if getattr(self, "_poll_meta", None) is not None:
                self._poll_meta.stop()
        except Exception:
            pass
        try:
            if getattr(self, "_poll_progress", None) is not None:
                self._poll_progress.stop()
        except Exception:
            pass

    def _refresh_record_status(self) -> None:
        try:
            rec = getattr(self.app, "_record_handle", None)
            ch = getattr(self.app, "_record_channel", None)
            started = getattr(self.app, "_record_started_at", None)
            pending = bool(getattr(self.app, "_record_pending", False))

            is_active = (rec is not None) or pending

            try:
                btn = self.query_one("#stop_record", Button)
                btn.label = "Stop Recording" if is_active else "Start Recording"
            except Exception:
                pass

            if not is_active or ch is None or started is None:
                self.query_one("#record_status", Static).update("")
                return

            elapsed = max(0, int(time.time() - float(started)))
            m, s = divmod(elapsed, 60)
            h, m = divmod(m, 60)
            t = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
            label = f"{ch.number if ch.number is not None else ''} {ch.name}".strip()
            suffix = " (starting...)" if rec is None else ""
            self.query_one("#record_status", Static).update(f"Recording: {label} ({t}){suffix}")
        except Exception:
            pass

    def on_resize(self, event) -> None:
        if self._art_url:
            art_w, art_h = self._current_art_box()
            self.app.run_worker(
                lambda: self._fetch_and_render_art(self._art_url or "", art_width=art_w, art_height=art_h),
                thread=True,
            )

    def _refresh(self) -> None:
        if bool(getattr(self, "_closing", False)):
            return
        try:
            app = self.app
        except NoActiveAppError:
            return
        app.run_worker(self._refresh_worker, thread=True, exclusive=True)

    def _refresh_worker(self) -> None:
        if bool(getattr(self, "_closing", False)):
            return
        try:
            client = self.app._get_client()
            audio_now: Optional[datetime] = None
            try:
                lh = getattr(self.app, "_live_handle", None)
                if lh is not None and getattr(lh, "proxy", None) is not None:
                    audio_now = lh.proxy.playhead_pdt(behind_segments=3)
            except Exception:
                audio_now = None

            # Anchor/interpolate audio time so progress can tick every second.
            try:
                self._update_audio_anchor(audio_now)
            except Exception:
                pass
            # Use the proxy segment PDT for selecting metadata (conservative).
            # Interpolated time can drift ahead and cause early track/art flips.
            audio_now_select = audio_now
            if audio_now_select is None:
                audio_now_select = self._audio_now_estimated()

            # Use interpolated time for progress ticking.
            audio_now_est = self._audio_now_estimated() or audio_now

            data = client.live_update(channel_id=self._channel.id)
            items = data.get("items") or []

            def parse_item_dt(item: dict) -> Optional[datetime]:
                try:
                    ts = item.get("timestamp")
                    if isinstance(ts, str) and ts:
                        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
                except Exception:
                    return None
                return None

            # Choose the metadata item that corresponds to the actual audio time.
            current: dict = {}
            start_dt: Optional[datetime] = None
            next_dt: Optional[datetime] = None
            if items:
                # liveUpdate typically returns items oldest->newest.
                parsed: list[tuple[Optional[datetime], dict]] = [(parse_item_dt(it), it) for it in items]
                parsed = [p for p in parsed if p[0] is not None]
                parsed.sort(key=lambda x: x[0] or datetime.min.replace(tzinfo=timezone.utc))

                if audio_now_select is not None and parsed:
                    chosen_i = len(parsed) - 1
                    for i, (dt, it) in enumerate(parsed):
                        if dt is None:
                            continue
                        if dt <= audio_now_select:
                            chosen_i = i
                        else:
                            break
                    current = parsed[chosen_i][1]
                    start_dt = parsed[chosen_i][0]
                    if chosen_i + 1 < len(parsed):
                        next_dt = parsed[chosen_i + 1][0]
                else:
                    current = items[-1]
                    start_dt = parse_item_dt(current)
                    next_dt = None

            artist = current.get("artistName") or ""
            title = current.get("name") or ""
            album = current.get("albumName") or ""
            dur_ms = current.get("duration") or 0
            ts = current.get("timestamp")
            track_id = current.get("id")

            def pick_image_url(root: dict) -> Optional[str]:
                try:
                    tile = (root.get("tile") or {})
                    a11 = (tile.get("aspect_1x1") or {})
                    pref = (a11.get("preferredImage") or {}).get("url")
                    if isinstance(pref, str) and pref:
                        return pref
                    default = (a11.get("defaultImage") or {}).get("url")
                    if isinstance(default, str) and default:
                        return default
                    a169 = (tile.get("aspect_16x9") or {})
                    pref2 = (a169.get("preferredImage") or {}).get("url")
                    if isinstance(pref2, str) and pref2:
                        return pref2
                    default2 = (a169.get("defaultImage") or {}).get("url")
                    if isinstance(default2, str) and default2:
                        return default2
                except Exception:
                    return None
                return None

            def find_first_url(node: object) -> Optional[str]:
                try:
                    if isinstance(node, str):
                        s = node.strip()
                        if s.startswith("http://") or s.startswith("https://"):
                            return s
                        # Protocol-relative
                        if s.startswith("//"):
                            return "https:" + s
                        # Some payloads carry image keys/paths; try turning those into imgsrv URLs.
                        if "/" in s and not s.startswith("{") and not s.startswith("["):
                            # Common SiriusXM keys look like: entity-management/.../image.jpg
                            if any(ext in s.lower() for ext in (".jpg", ".jpeg", ".png", ".webp")):
                                key = s.lstrip("/")
                                return _imgsrv_url_from_key(key, width=300, height=300)
                        return None
                    if isinstance(node, dict):
                        for k in (
                            "url",
                            "imageUrl",
                            "imageURL",
                            "uri",
                            "imageKey",
                            "key",
                            "assetKey",
                            "path",
                        ):
                            v = node.get(k)
                            if isinstance(v, str):
                                u = find_first_url(v)
                                if u:
                                    return u
                        for v in node.values():
                            u = find_first_url(v)
                            if u:
                                return u
                        return None
                    if isinstance(node, list):
                        for v in node:
                            u = find_first_url(v)
                            if u:
                                return u
                        return None
                except Exception:
                    return None
                return None

            img_url = pick_image_url(current.get("images") or {})
            if not img_url:
                img_url = pick_image_url(current.get("artistImages") or {})
            if not img_url:
                img_url = find_first_url(current.get("images") or {})
            if not img_url:
                img_url = find_first_url(current.get("artistImages") or {})
            if not img_url:
                img_url = find_first_url(current)

            # Some items (e.g., cut-linear interstitials) have no images at all.
            # Fall back to channel art if available.
            if not img_url:
                try:
                    if getattr(self._channel, "logo_url", None):
                        img_url = str(self._channel.logo_url)
                except Exception:
                    pass

            if not img_url:
                try:
                    dbg_path = Path(user_cache_dir("satstash")) / "last_liveupdate_item.json"
                    dbg_path.parent.mkdir(parents=True, exist_ok=True)
                    dbg_path.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
                except Exception:
                    pass

            if isinstance(img_url, str) and img_url:
                img_url = _normalize_art_url(img_url)

            # Prefer deriving duration from the next item start when available. This keeps
            # progress accurate even when API durationMillis is wrong.
            duration_s: Optional[float] = None
            if start_dt is not None and next_dt is not None:
                try:
                    duration_s = max(0.0, float((next_dt - start_dt).total_seconds()))
                except Exception:
                    duration_s = None
            if duration_s is None:
                try:
                    duration_s = float(dur_ms) / 1000.0 if dur_ms else None
                except Exception:
                    duration_s = None

            # Build a stable key even when API does not provide an id (or id is unstable).
            try:
                start_key = start_dt.isoformat() if start_dt is not None else ""
                track_key = "|".join([start_key, str(artist or ""), str(title or "")])
            except Exception:
                track_key = track_id or ""

            def apply():
                self.query_one("#track", Static).update(" - ".join([x for x in [artist, title] if x]) or "(no metadata)")
                self.query_one("#album", Static).update(album)

                # Track-change detection for progress
                changed = False
                if track_key and track_key != (self._track_id or ""):
                    changed = True
                if start_dt is not None and self._track_start is not None:
                    try:
                        if abs((start_dt - self._track_start).total_seconds()) > 0.5:
                            changed = True
                    except Exception:
                        pass
                if self._track_start is None and start_dt is not None:
                    changed = True

                if changed:
                    self._track_id = track_key or track_id
                    self._track_start = start_dt
                    self._track_duration_s = duration_s

                    pb = self.query_one("#progress", ProgressBar)
                    pb.update(total=duration_s or 1, progress=0)

                if img_url and img_url != self._art_url:
                    self._art_url = img_url
                    self.query_one("#art", Static).update("Loading artwork...")
                    # Don't make this exclusive; metadata polling should not cancel artwork fetch.
                    art_w, art_h = self._current_art_box()
                    self.app.run_worker(lambda: self._fetch_and_render_art(img_url, art_width=art_w, art_height=art_h), thread=True)
                elif not img_url and not self._art_url:
                    msg = "(no art)"
                    try:
                        dbg_path = Path(user_cache_dir("satstash")) / "last_liveupdate_item.json"
                        msg = msg + "\n" + f"debug: {dbg_path}"
                    except Exception:
                        pass
                    self.query_one("#art", Static).update(msg)

                self._update_progress()

            try:
                if bool(getattr(self, "_closing", False)):
                    return
                app = self.app
            except NoActiveAppError:
                return
            try:
                app.call_from_thread(apply)
            except NoActiveAppError:
                return
            self._login_prompted = False
        except HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 401:
                # The client already attempted a refresh+retry once.
                if not self._login_prompted:
                    self._login_prompted = True

                    def ui():
                        self.app.notify("Session expired. Please login again.")
                        self.app._prompt_login_and_retry(after_login=lambda: self._refresh())

                    self.app.call_from_thread(ui)
                return

            try:
                if not bool(getattr(self, "_closing", False)):
                    self.app.call_from_thread(lambda: self.app.notify(f"Metadata update failed: {exc}"))
            except NoActiveAppError:
                return
        except (NotLoggedInError, SessionExpiredError):
            if not self._login_prompted:
                self._login_prompted = True

                def ui():
                    self.app.notify("Session expired. Please login again.")
                    self.app._prompt_login_and_retry(after_login=lambda: self._refresh())

                try:
                    if not bool(getattr(self, "_closing", False)):
                        self.app.call_from_thread(ui)
                except NoActiveAppError:
                    return
            else:
                try:
                    if not bool(getattr(self, "_closing", False)):
                        self.app.call_from_thread(lambda: None)
                except NoActiveAppError:
                    return
        except Exception as exc:
            try:
                if not bool(getattr(self, "_closing", False)):
                    self.app.call_from_thread(lambda: self.app.notify(f"Metadata update failed: {exc}"))
            except NoActiveAppError:
                return

    def _current_art_box(self) -> tuple[int, int]:
        try:
            art = self.query_one("#art", Static)
            w = int(getattr(art.size, "width", 0) or 0)
            h = int(getattr(art.size, "height", 0) or 0)
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass
        # Fallback if size isn't available yet.
        try:
            total_w = int(getattr(self.size, "width", 0) or 0)
            total_h = int(getattr(self.size, "height", 0) or 0)
        except Exception:
            total_w, total_h = 80, 24
        return max(24, int(total_w * 0.70)), max(10, total_h - 8)

    def _update_progress(self) -> None:
        try:
            if not self._track_start or not self._track_duration_s:
                return
            try:
                lh = getattr(self.app, "_live_handle", None)
                if lh is not None and getattr(lh, "proxy", None) is not None:
                    pdt = lh.proxy.playhead_pdt(behind_segments=3)
                    self._update_audio_anchor(pdt)
            except Exception:
                pass
            now = self._audio_now_estimated()
            if now is None:
                now = datetime.now(timezone.utc)
            elapsed = (now - self._track_start).total_seconds()
            elapsed = max(0.0, min(elapsed, self._track_duration_s))
            remaining = max(0.0, self._track_duration_s - elapsed)

            pb = self.query_one("#progress", ProgressBar)
            pb.update(total=self._track_duration_s, progress=elapsed)
            self.query_one("#timing", Static).update(
                f"{_fmt_time(elapsed)} / {_fmt_time(self._track_duration_s)}  (-{_fmt_time(remaining)})"
            )
        except Exception:
            return

    def _fetch_and_render_art(self, url: str, *, art_width: int = 30, art_height: int = 20) -> None:
        try:
            cache_dir = Path(user_cache_dir("satstash")) / "art"
            cache_dir.mkdir(parents=True, exist_ok=True)
            name = url.split("/")[-1].split("?")[0] or "art"
            path = cache_dir / name
            def fetch() -> None:
                from urllib.parse import urlparse

                headers = {
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "image/*,*/*;q=0.8",
                }

                parsed = urlparse(url)
                rel_path = parsed.path.lstrip("/")
                # Try a couple of known SiriusXM imgix hosts. Some paths (notably
                # entity-management) return 410 on siriusxm.imgix.net.
                candidates = [url]
                if rel_path:
                    # SiriusXM web apps sometimes use an image transformer service
                    # (imgsrv-*) with a base64 JSON payload. This can work when the
                    # raw imgix hosts return 410 for a given key.
                    candidates.append(_imgsrv_url_from_key(rel_path, width=300, height=300))
                    candidates.append(f"https://siriusxm-prd.imgix.net/{rel_path}")
                    candidates.append(f"https://siriusxm.imgix.net/{rel_path}")

                last_exc: Optional[Exception] = None
                for u in candidates:
                    try:
                        r = requests.get(u, timeout=15, headers=headers)
                        # Special-case 410: try the next host.
                        if r.status_code == 410:
                            last_exc = HTTPError("410 Client Error", response=r)
                            continue
                        r.raise_for_status()
                        ctype = (r.headers.get("content-type") or "").lower()
                        if "image/" not in ctype:
                            raise RuntimeError(f"Artwork fetch returned {ctype or 'non-image'}")
                        path.write_bytes(r.content)
                        return
                    except Exception as exc:
                        last_exc = exc
                        continue

                if last_exc:
                    raise last_exc

            # Fetch if missing/empty, or refetch if cached file isn't a readable image.
            if not path.exists() or path.stat().st_size == 0:
                fetch()
            else:
                try:
                    PILImage.open(path).verify()
                except Exception:
                    try:
                        path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    fetch()

            # Prefer high-res color rendering; fall back to ASCII.
            mode = "halfblock"
            try:
                mode = (getattr(getattr(self, "app", None), "settings", None).art_render_mode or "halfblock").strip().lower()
            except Exception:
                mode = "halfblock"

            if mode == "braille":
                art: object = _image_to_rich_braille_fit(path, width=max(16, int(art_width)), height=max(8, int(art_height)))
            else:
                art = _image_to_rich_blocks_fit(path, width=max(16, int(art_width)), height=max(8, int(art_height)))

            if isinstance(art, Text) and art.plain.strip():
                pass
            else:
                art = _image_to_ascii(path, width=max(16, min(80, int(art_width))))
                if not isinstance(art, str) or not art.strip():
                    raise RuntimeError("empty render")

            def apply():
                self.query_one("#art", Static).update(art)  # Text or str

            self.app.call_from_thread(apply)
        except Exception as exc:
            def apply():
                # Keep layout stable; show a short status rather than a giant URL.
                err = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
                msg = "(art unavailable)\n" + _wrap_text(err, width=28)
                if self._art_url:
                    msg = msg + "\n" + _wrap_text(self._art_url, width=28)
                self.query_one("#art", Static).update(msg)

            self.app.call_from_thread(apply)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "stop":
            self.app.stop_live()
            self.app.pop_screen()
            return
        if event.button.id == "stop_record":
            rec = getattr(self.app, "_record_handle", None)
            pending = bool(getattr(self.app, "_record_pending", False))
            is_active = (rec is not None) or pending
            if is_active:
                try:
                    self.app.stop_record_now()
                except Exception as exc:
                    self.app.notify(f"Stop recording failed: {exc}")
                return
            try:
                self.app.start_record_single(channel=self._channel)
            except Exception as exc:
                self.app.notify(f"Start recording failed: {exc}")
            return
        if event.button.id == "back":
            self.app.pop_screen()
            return


class SettingsScreen(Screen[None]):
    def __init__(self, current_player: str):
        super().__init__()
        self._player = (current_player or "auto").lower()
        try:
            self._start_from_track = bool(getattr(load_settings(), "start_record_from_track_start", True))
        except Exception:
            self._start_from_track = True
        try:
            self._art_mode = (getattr(load_settings(), "art_render_mode", "halfblock") or "halfblock").strip().lower()
        except Exception:
            self._art_mode = "halfblock"

    def compose(self) -> ComposeResult:
        settings = load_settings()
        yield Header(show_clock=True)
        with Container():
            yield Static("Settings", id="title")
            yield Static(f"Player: {self._player}", id="player_value")
            yield Static(
                f"Album art render mode: {getattr(settings, 'art_render_mode', 'halfblock')}",
                id="art_mode_value",
            )
            yield Static(
                f"Start recordings from track start: {'ON' if getattr(settings, 'start_record_from_track_start', True) else 'OFF'}",
                id="start_from_track_value",
            )
            yield Static("Output folder:", id="output_label")
            yield Input(value=getattr(settings, "output_dir", "~/Music/SiriusXM"), id="output_dir")
            yield Static("Recordings folder (legacy):", id="recordings_label")
            yield Input(value=settings.recordings_dir, id="recordings_dir")
            yield Static("Auto-login username (optional):", id="auth_user_label")
            yield Input(value=getattr(settings, "auth_username", "") or "", id="auth_username")
            yield Static("Auto-login password (optional):", id="auth_pass_label")
            yield Input(value=getattr(settings, "auth_password", "") or "", password=True, id="auth_password")
            with Horizontal():
                yield Button("Auto", id="player_auto", variant="primary")
                yield Button("mpv", id="player_mpv")
                yield Button("ffplay", id="player_ffplay")
            with Horizontal():
                yield Button("Toggle Track-Start", id="toggle_track_start")
                yield Button("Toggle Art Mode", id="toggle_art_mode")
            with Horizontal():
                yield Button("Save", id="save", variant="primary")
                yield Button("Back", id="back")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "back":
            self.app.pop_screen()
            return
        if event.button.id == "toggle_track_start":
            self._start_from_track = not bool(self._start_from_track)
            self.query_one("#start_from_track_value", Static).update(
                f"Start recordings from track start: {'ON' if self._start_from_track else 'OFF'}"
            )
            return
        if event.button.id == "toggle_art_mode":
            self._art_mode = "braille" if (self._art_mode or "halfblock") == "halfblock" else "halfblock"
            self.query_one("#art_mode_value", Static).update(f"Album art render mode: {self._art_mode}")
            return
        if event.button.id == "save":
            settings = load_settings()
            settings.player_preference = self._player
            settings.art_render_mode = self._art_mode
            settings.start_record_from_track_start = bool(self._start_from_track)
            try:
                settings.art_render_mode = (self._art_mode or "halfblock").strip().lower()
            except Exception:
                pass
            try:
                settings.start_record_from_track_start = bool(self._start_from_track)
            except Exception:
                pass
            settings.output_dir = (
                (self.query_one("#output_dir", Input).value or getattr(settings, "output_dir", "")).strip()
                or getattr(settings, "output_dir", "~/Music/SiriusXM")
            )
            settings.recordings_dir = (
                (self.query_one("#recordings_dir", Input).value or settings.recordings_dir).strip()
                or settings.recordings_dir
            )
            try:
                settings.auth_username = (self.query_one("#auth_username", Input).value or "").strip()
            except Exception:
                settings.auth_username = getattr(settings, "auth_username", "") or ""
            try:
                settings.auth_password = self.query_one("#auth_password", Input).value or ""
            except Exception:
                settings.auth_password = getattr(settings, "auth_password", "") or ""
            save_settings(settings)
            self.app.settings = settings
            try:
                self.app.query_one("#subtitle_player", Static).update(f"Player: {settings.player_preference}")
            except Exception:
                pass
            self.app.notify("Saved settings")
            self.app.pop_screen()
            return
        if event.button.id == "player_auto":
            self._player = "auto"
        elif event.button.id == "player_mpv":
            self._player = "mpv"
        elif event.button.id == "player_ffplay":
            self._player = "ffplay"
        self.query_one("#player_value", Static).update(f"Player: {self._player}")

