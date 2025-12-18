from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Callable, Optional
from urllib.parse import quote

import requests

from platformdirs import user_cache_dir

from satstash.api.client import SxmClient
from satstash.hls.proxy import HlsProxy, ProxyInfo
from satstash.hls.variants import select_variant


@dataclass
class RecordHandle:
    proxy: HlsProxy
    proxy_info: ProxyInfo
    process: subprocess.Popen
    tmp_path: Path
    final_path: Path
    log_path: Path
    effective_start_pdt: Optional[datetime]


def _ffprobe_duration_s(path: Path) -> Optional[float]:
    try:
        if not shutil.which("ffprobe"):
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
            return None
        return float(out)
    except Exception:
        return None


def _validate_mp4(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 64:
        return False
    if not _mp4_has_moov(path):
        return False
    # ffprobe validation (best-effort). If ffprobe is missing, fall back to moov heuristic.
    dur = _ffprobe_duration_s(path)
    if dur is None:
        return True
    try:
        return float(dur) > 0.1
    except Exception:
        return False


def _attempt_remux_mp4(*, src: Path, dst: Path) -> bool:
    """Best-effort repair: remux to produce a non-fragmented MP4/M4A with a proper moov."""
    try:
        if not shutil.which("ffmpeg"):
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp sibling so we can atomically promote if it succeeds.
        tmp = dst.with_suffix(dst.suffix + ".remux.tmp")
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass

        # Remux only; if src is unreadable, ffmpeg will fail.
        argv = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(tmp),
        ]
        rc = subprocess.call(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if rc != 0:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            return False
        if not _validate_mp4(tmp):
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            return False
        tmp.replace(dst)
        return True
    except Exception:
        return False


def _mp4_has_moov(path: Path) -> bool:
    """Best-effort check that an MP4/M4A file contains a moov atom.

    If ffmpeg is terminated too abruptly, MP4 may be missing the trailer/moov and
    will fail to play ('moov atom not found'). This check is used to avoid
    presenting broken files as finished recordings.
    """
    try:
        if not path.exists() or path.stat().st_size < 64:
            return False
        # moov is usually near the end (unless faststart); check head+tail.
        size = path.stat().st_size
        with open(path, "rb") as f:
            head = f.read(min(256 * 1024, size))
            if b"moov" in head:
                return True
            if size > 0:
                tail_n = min(1024 * 1024, size)
                f.seek(max(0, size - tail_n))
                tail = f.read(tail_n)
                if b"moov" in tail:
                    return True
    except Exception:
        return False
    return False


@dataclass
class FfmpegRecordHandle:
    process: subprocess.Popen
    tmp_path: Path
    final_path: Path
    log_path: Path


def start_ffmpeg_recording(
    *,
    input_url: str,
    tmp_path: Path,
    final_path: Path,
    log_path: Path,
    container: str,
    debug: bool = False,
    preroll_s: Optional[float] = None,
    duration_s: Optional[float] = None,
    headers: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> FfmpegRecordHandle:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found (required for recording)")

    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    argv = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel",
        "info" if debug else "warning",
    ]

    # Allow callers to provide auth context (needed when segments are fetched directly
    # from SiriusXM CDN URLs instead of being proxied through localhost).
    if user_agent:
        argv += ["-user_agent", str(user_agent)]
    if headers:
        argv += ["-headers", str(headers)]

    # When we provide a timeshifted/DVR playlist (start_pdt/end_pdt), we want to start at the
    # *first* segment in the playlist (not near the live edge). ffmpeg defaults to starting
    # close to the end for live HLS streams, which makes finite downloads behave like realtime.
    if preroll_s is not None:
        argv += ["-live_start_index", "0"]
    else:
        try:
            if "start_pdt=" in (input_url or "") or "end_pdt=" in (input_url or ""):
                argv += ["-live_start_index", "0"]
        except Exception:
            pass

    argv += [
        "-i",
        input_url,
    ]

    # If we have a preroll to trim (e.g. proxy playlist starts slightly before requested start_pdt),
    # apply an accurate seek AFTER input so timestamps align with the intended track start.
    # This keeps the resulting file time-zero close to start_pdt.
    if preroll_s is not None:
        try:
            ss = float(preroll_s)
        except Exception:
            ss = 0.0
        if ss > 0.01:
            argv += ["-ss", f"{ss:.3f}"]

    # If we have an intended duration (e.g. per-track exports), enforce an exact cut.
    # This prevents overlap at boundaries when the HLS playlist includes a segment that
    # starts before end_pdt but extends past it.
    if duration_s is not None:
        try:
            tt = float(duration_s)
        except Exception:
            tt = 0.0
        if tt > 0.01:
            argv += ["-t", f"{tt:.3f}"]

    if container == "mpegts":
        argv += ["-c", "copy", "-f", "mpegts", str(tmp_path)]
    elif container == "mp4-aac":
        # SiriusXM HLS audio is AAC in ADTS. Write M4A/MP4 without re-encode.
        argv += [
            "-vn",
            "-c:a",
            "copy",
            "-bsf:a",
            "aac_adtstoasc",
            # Make output resilient if ffmpeg is interrupted (write initial moov + fragments).
            "-movflags",
            "+frag_keyframe+empty_moov+default_base_moof",
            "-f",
            "mp4",
            str(tmp_path),
        ]
    else:
        raise ValueError(f"Unsupported container: {container}")

    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
        )

    return FfmpegRecordHandle(process=proc, tmp_path=tmp_path, final_path=final_path, log_path=log_path)


def stop_ffmpeg_recording(
    handle: FfmpegRecordHandle,
    *,
    finalize: bool,
    progress: Optional[Callable[[str], None]] = None,
) -> Path:
    def say(msg: str) -> None:
        try:
            if progress:
                progress(msg)
        except Exception:
            pass

    # For MP4/M4A, prefer a graceful stop so ffmpeg can write the moov atom.
    say("Stopping ffmpeg...")
    try:
        handle.process.send_signal(signal.SIGINT)
    except Exception:
        try:
            handle.process.terminate()
        except Exception:
            pass

    rc: Optional[int]
    try:
        say("Finalizing stream...")
        handle.process.wait(timeout=15)
        rc = handle.process.returncode
    except Exception:
        try:
            handle.process.kill()
        except Exception:
            pass
        try:
            handle.process.wait(timeout=5)
        except Exception:
            pass
        rc = handle.process.returncode

    if handle.tmp_path.exists() and handle.tmp_path.stat().st_size > 0:
        should_finalize = finalize or (rc == 0)
        if should_finalize:
            if handle.final_path.suffix.lower() in {".m4a", ".mp4"}:
                say("Validating output file...")
                if not _validate_mp4(handle.tmp_path):
                    # Try to repair via remux. If repair fails, keep as partial.
                    say("Output appears incomplete; attempting repair (remux)...")
                    if not _attempt_remux_mp4(src=handle.tmp_path, dst=handle.final_path):
                        say("Repair failed; keeping partial recording")
                        return handle.tmp_path
                    say("Repair succeeded")
                    return handle.final_path
            try:
                handle.tmp_path.replace(handle.final_path)
            except Exception:
                return handle.tmp_path
            return handle.final_path

    return handle.tmp_path


def start_recording(
    *,
    client: SxmClient,
    channel_id: str,
    channel_type: str = "channel-linear",
    preferred_quality: str = "256k",
    out_dir: Path,
    title: str,
    progress: Optional[Callable[[str], None]] = None,
    debug: bool = False,
    start_pdt: Optional[datetime] = None,
    end_pdt: Optional[datetime] = None,
) -> RecordHandle:
    def say(msg: str) -> None:
        if progress:
            progress(msg)

    # If we're attempting DVR rewind, make logs more verbose by default so we can diagnose
    # start_pdt/timeshift issues from the field.
    if start_pdt is not None:
        debug = True

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found (required for recording)")

    out_dir.mkdir(parents=True, exist_ok=True)

    safe_title = "".join(c if c.isalnum() or c in " _-." else "_" for c in (title or "recording"))
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    final_path = out_dir / f"{safe_title}-{ts}.m4a"

    # NP1: keep in-progress partials out of the recordings directory. We still
    # write a .part file, but into a hidden temp folder within out_dir so:
    # - file browsers typically hide it
    # - we remain on the same filesystem for atomic rename/replace
    tmp_dir = out_dir / ".satstash_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / (final_path.name + ".part")

    # For channel-linear, WEB manifests can be live-edge limited. When the user requests
    # "start from track start" we must use FULL so SiriusXM returns a DVR/timeshifted
    # manifest that actually contains older segments.
    if start_pdt is not None:
        manifest_variant = "FULL"
    else:
        manifest_variant = "WEB" if channel_type == "channel-linear" else "FULL"
    say("tuneSource...")

    ts_arg: Optional[str] = None
    if start_pdt is not None:
        try:
            sp = start_pdt
            if sp.tzinfo is None:
                sp = sp.replace(tzinfo=timezone.utc)
            else:
                sp = sp.astimezone(timezone.utc)
            ts_arg = sp.isoformat().replace("+00:00", "Z")
        except Exception:
            ts_arg = None

    tune = client.tune_source(
        entity_id=channel_id,
        entity_type=channel_type,
        manifest_variant=manifest_variant,
        start_timestamp=ts_arg,
    )
    master = tune.master_url()
    if not master:
        raise RuntimeError("tuneSource returned no master URL")

    say("selecting stream variant...")
    sel = select_variant(master, prefer=preferred_quality, headers=client.session.headers, client=client)
    say(f"variant: {sel.variant_url}")

    say("starting local proxy...")
    proxy = HlsProxy(client, sel.variant_url)
    proxy_info = proxy.start()

    playlist_url = proxy_info.url
    requested_start_pdt = start_pdt
    if start_pdt is not None:
        playlist_url = playlist_url + "?start_pdt=" + quote(start_pdt.isoformat())

    if end_pdt is not None:
        # IMPORTANT: Only use end_pdt to slice the proxy playlist when the end bound is
        # in the past (finite download). If end_pdt is in the future (scheduled/live
        # recording), the upstream playlist cannot contain future segments; slicing would
        # yield a tiny finite playlist and ffmpeg would exit immediately.
        try:
            now_utc = datetime.now(timezone.utc)
        except Exception:
            now_utc = None
        should_slice_end = False
        try:
            if now_utc is not None:
                # Small tolerance for clock skew.
                should_slice_end = end_pdt <= (now_utc + timedelta(seconds=2))
        except Exception:
            should_slice_end = False

        if should_slice_end:
            try:
                sep = "&" if "?" in playlist_url else "?"
                playlist_url = playlist_url + sep + "end_pdt=" + quote(end_pdt.isoformat())
            except Exception:
                pass

    preroll_s: Optional[float] = None
    duration_s: Optional[float] = None

    if start_pdt is not None and end_pdt is not None:
        try:
            duration_s = float((end_pdt - start_pdt).total_seconds())
        except Exception:
            duration_s = None
        try:
            if duration_s is not None and duration_s <= 0.01:
                duration_s = None
        except Exception:
            duration_s = None

    try:
        def preflight(url: str) -> tuple[str, int]:
            say("preflight proxy playlist...")
            rr = requests.get(url, timeout=10)
            rr.raise_for_status()
            txt = rr.text or ""
            count = 0
            for ln in txt.splitlines():
                ss = ln.strip()
                if not ss or ss.startswith("#"):
                    continue
                count += 1
            if count == 0:
                raise RuntimeError("playlist contains no media segments")
            return txt, count

        playlist_text, seg_count = preflight(playlist_url)

        # Save diagnostics to disk so the user can inspect what the proxy served.
        try:
            diag_dir = Path(user_cache_dir("satstash"))
            diag_dir.mkdir(parents=True, exist_ok=True)
            if requested_start_pdt is not None:
                (diag_dir / "last_record_proxy_playlist.m3u8").write_text(playlist_text, encoding="utf-8")
                import json as _json

                ctx = {
                    "channel_id": channel_id,
                    "channel_type": channel_type,
                    "preferred_quality": preferred_quality,
                    "proxy_url": proxy_info.url,
                    "playlist_url": playlist_url,
                    "variant_url": sel.variant_url,
                    "requested_start_pdt": requested_start_pdt.isoformat() if requested_start_pdt else None,
                }
                (diag_dir / "last_record_context.json").write_text(_json.dumps(ctx, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

        # If we requested a start_pdt, the proxy should have sliced the playlist.
        # The first PDT should be very close (<= segment duration) to start_pdt.
        # If not, treat slicing as failed and fall back to live-edge recording.
        if start_pdt is not None:
            first_pdt: Optional[datetime] = None
            for line in playlist_text.splitlines():
                s = line.strip()
                if s.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
                    try:
                        first_pdt = datetime.fromisoformat(s.split(":", 1)[1].strip().replace("Z", "+00:00"))
                        if first_pdt.tzinfo is None:
                            first_pdt = first_pdt.replace(tzinfo=timezone.utc)
                        else:
                            first_pdt = first_pdt.astimezone(timezone.utc)
                    except Exception:
                        first_pdt = None
                    break

            if first_pdt is not None:
                try:
                    delta = float((start_pdt - first_pdt).total_seconds())
                except Exception:
                    delta = 0.0

                # Delta should be within a single segment (~10s). If it's huge, we did not slice.
                # Before falling all the way back to live edge (which guarantees a truncated first track),
                # retry WITHOUT local slicing but KEEP the server-side timeshift (tuneSource startTimestamp).
                if delta < -0.25 or delta > 30.0:
                    say(f"start_pdt slice mismatch (delta={delta:.3f}s); retrying without local slice")
                    preroll_s = None
                    playlist_url = proxy_info.url
                    playlist_text, seg_count = preflight(playlist_url)

                    # Recompute delta relative to the unsliced playlist.
                    first_pdt2: Optional[datetime] = None
                    for line in playlist_text.splitlines():
                        s = line.strip()
                        if s.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
                            try:
                                first_pdt2 = datetime.fromisoformat(s.split(":", 1)[1].strip().replace("Z", "+00:00"))
                                if first_pdt2.tzinfo is None:
                                    first_pdt2 = first_pdt2.replace(tzinfo=timezone.utc)
                                else:
                                    first_pdt2 = first_pdt2.astimezone(timezone.utc)
                            except Exception:
                                first_pdt2 = None
                            break
                    if first_pdt2 is not None:
                        try:
                            delta2 = float((start_pdt - first_pdt2).total_seconds())
                        except Exception:
                            delta2 = 0.0
                        if -0.25 <= delta2 <= 30.0:
                            if delta2 > 0.05:
                                preroll_s = delta2
                        else:
                            say(f"start_pdt still mismatched after retry (delta={delta2:.3f}s); falling back to live edge")
                            preroll_s = None
                            playlist_url = proxy_info.url
                            start_pdt = None
                            playlist_text, seg_count = preflight(playlist_url)
                    else:
                        say("start_pdt retry failed (no PDT in playlist); falling back to live edge")
                        preroll_s = None
                        playlist_url = proxy_info.url
                        start_pdt = None
                        playlist_text, seg_count = preflight(playlist_url)
                else:
                    if delta > 0.05:
                        preroll_s = delta

        if "#EXT-X-KEY" in playlist_text:
            key_url = proxy_info.url.rsplit("/", 1)[0] + "/key"
            say("preflight proxy key...")
            rk = requests.get(key_url, timeout=10)
            rk.raise_for_status()
            kb = rk.content
            if not kb:
                raise RuntimeError("key endpoint returned empty body")
            if len(kb) not in (16, 24, 32):
                raise RuntimeError(f"invalid key length: {len(kb)}")
    except Exception as exc:
        proxy.stop(proxy_info)
        raise RuntimeError(f"Record preflight failed: {exc}")

    log_dir = Path(user_cache_dir("satstash"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"ffmpeg-record-{ts}.log"

    if preroll_s is not None and preroll_s > 0.01:
        try:
            say(f"applying preroll trim: {preroll_s:.3f}s")
        except Exception:
            pass

    proc = start_ffmpeg_recording(
        input_url=playlist_url,
        tmp_path=tmp_path,
        final_path=final_path,
        log_path=log_path,
        container="mp4-aac",
        debug=debug,
        preroll_s=preroll_s,
        duration_s=duration_s,
    )

    return RecordHandle(
        proxy=proxy,
        proxy_info=proxy_info,
        process=proc.process,
        tmp_path=proc.tmp_path,
        final_path=proc.final_path,
        log_path=proc.log_path,
        effective_start_pdt=start_pdt,
    )


def stop_recording(handle: RecordHandle) -> Path:
    """Stop ffmpeg and finalize output.
    We only rename the .part file to the final name when ffmpeg exits cleanly.
    """
    return stop_recording_with_options(handle, finalize_on_stop=True)


def probe_dvr_buffer_start_pdt(
    *,
    client: SxmClient,
    channel_id: str,
    channel_type: str = "channel-linear",
    preferred_quality: str = "256k",
    lookback_hours: int = 5,
) -> Optional[datetime]:
    try:
        ts_arg: Optional[str] = None
        try:
            hrs = int(lookback_hours or 0)
            if hrs > 0:
                ts_arg = (datetime.now(timezone.utc) - timedelta(hours=hrs)).isoformat().replace("+00:00", "Z")
        except Exception:
            ts_arg = None

        tune = client.tune_source(
            entity_id=channel_id,
            entity_type=channel_type,
            manifest_variant="FULL",
            start_timestamp=ts_arg,
        )
        master = tune.master_url()
        if not master:
            return None
        sel = select_variant(master, prefer=preferred_quality, headers=client.session.headers, client=client)
        rr = client._request("GET", sel.variant_url)
        rr.raise_for_status()
        text = rr.text or ""
        earliest: Optional[datetime] = None
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
                try:
                    dt = datetime.fromisoformat(s.split(":", 1)[1].strip().replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                    if earliest is None or dt < earliest:
                        earliest = dt
                except Exception:
                    continue
        if earliest is not None:
            return earliest
        return None
    except Exception:
        return None


def stop_recording_with_options(
    handle: RecordHandle,
    *,
    finalize_on_stop: bool,
    progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """Stop ffmpeg and finalize output.

    - When finalize_on_stop=True, we treat this as a user-initiated stop and
      finalize the file (atomic rename) if the temp file is non-empty, even if
      ffmpeg exits with a non-zero code due to terminate/kill.
    - When finalize_on_stop=False, we only finalize when ffmpeg exits cleanly.
    """
    def say(msg: str) -> None:
        try:
            if progress:
                progress(msg)
        except Exception:
            pass

    rc: Optional[int] = None

    # If ffmpeg already exited (e.g. because we used -t), don't send signals.
    already_done = False
    try:
        already_done = handle.process.poll() is not None
    except Exception:
        already_done = False

    if already_done:
        try:
            rc = handle.process.returncode
        except Exception:
            rc = None
    else:
        # For MP4/M4A, prefer a graceful stop so ffmpeg can write the moov atom.
        # SIGINT is equivalent to Ctrl-C and triggers a clean trailer write.
        say("Stopping ffmpeg...")
        try:
            handle.process.send_signal(signal.SIGINT)
        except Exception:
            try:
                handle.process.terminate()
            except Exception:
                pass

        try:
            say("Finalizing stream...")
            handle.process.wait(timeout=15)
            rc = handle.process.returncode
        except Exception:
            try:
                handle.process.terminate()
            except Exception:
                pass
            try:
                handle.process.wait(timeout=5)
                rc = handle.process.returncode
            except Exception:
                rc = None
            try:
                handle.process.kill()
            except Exception:
                pass
            try:
                handle.process.wait(timeout=5)
            except Exception:
                pass
            try:
                rc = handle.process.returncode
            except Exception:
                rc = None

    try:
        say("Stopping local proxy...")
        handle.proxy.stop(handle.proxy_info)
    except Exception:
        pass

    if handle.tmp_path.exists() and handle.tmp_path.stat().st_size > 0:
        should_finalize = finalize_on_stop or (rc == 0)
        if should_finalize:
            if handle.final_path.suffix.lower() in {".m4a", ".mp4"}:
                say("Validating output file...")
                if not _validate_mp4(handle.tmp_path):
                    say("Output appears incomplete; attempting repair (remux)...")
                    if not _attempt_remux_mp4(src=handle.tmp_path, dst=handle.final_path):
                        say("Repair failed; keeping partial recording")
                        return handle.tmp_path
                    say("Repair succeeded")
                    return handle.final_path
            try:
                handle.tmp_path.replace(handle.final_path)
            except Exception:
                return handle.tmp_path
            return handle.final_path

    return handle.tmp_path
