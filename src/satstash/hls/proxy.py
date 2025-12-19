from __future__ import annotations

import re
import threading
from collections import deque
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, quote, urlparse

import requests

from satstash.api.client import SxmClient


@dataclass
class ProxyInfo:
    url: str
    server: HTTPServer
    thread: threading.Thread


class HlsProxy:
    """Local proxy that:

    - Fetches the upstream variant playlist (with auth headers)
    - Rewrites relative segment URLs to absolute
    - Rewrites the AES key URI to /key (serves raw key bytes, not JSON/base64)

    This makes ffplay/mpv able to play SiriusXM streams.
    """

    def __init__(self, client: SxmClient, variant_url: str, *, proxy_segments: bool = True):
        self.client = client
        self.variant_url = variant_url
        self.proxy_segments = bool(proxy_segments)
        self._key_url: Optional[str] = None
        self._last_seg_pdt: Optional[datetime] = None
        self._last_seg_lock = threading.Lock()
        self._seg_pdts: deque[datetime] = deque(maxlen=32)

    def last_segment_pdt(self) -> Optional[datetime]:
        try:
            with self._last_seg_lock:
                return self._last_seg_pdt
        except Exception:
            return None

    def playhead_pdt(self, *, behind_segments: int = 3) -> Optional[datetime]:
        """Estimate the PDT of audio currently being heard.

        Media players often prefetch future segments. Using the newest-requested segment
        PDT can run the UI ahead of actual audio (early art/title flips, skipped bumpers).
        We keep a rolling window of recently requested segment PDTs and return one a few
        segments behind the newest.
        """
        try:
            with self._last_seg_lock:
                if not self._seg_pdts:
                    return self._last_seg_pdt
                pdts = list(self._seg_pdts)
        except Exception:
            return self._last_seg_pdt

        try:
            pdts.sort()
            idx = max(0, len(pdts) - 1 - max(0, int(behind_segments)))
            return pdts[idx]
        except Exception:
            return self._last_seg_pdt

    def start(self, host: str = "127.0.0.1", port: int = 0) -> ProxyInfo:
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            # Use HTTP/1.0 so clients can reliably detect end-of-response without
            # chunked encoding. (Some consumers time out if HTTP/1.1 is used without
            # explicit Content-Length / close semantics.)
            protocol_version = "HTTP/1.0"
            def log_message(self, format: str, *args) -> None:  # noqa: A002
                return

            def do_GET(self):  # noqa: N802
                parsed = urlparse(self.path)
                path = parsed.path
                qs = parse_qs(parsed.query or "")

                if path == "/listen.m3u8":
                    start_pdt: Optional[datetime] = None
                    end_pdt: Optional[datetime] = None
                    try:
                        raw = (qs.get("start_pdt") or [None])[0]
                        if raw:
                            # Accept Z or +00:00
                            start_pdt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                            if start_pdt.tzinfo is None:
                                start_pdt = start_pdt.replace(tzinfo=timezone.utc)
                            else:
                                start_pdt = start_pdt.astimezone(timezone.utc)
                    except Exception:
                        start_pdt = None

                    try:
                        raw2 = (qs.get("end_pdt") or [None])[0]
                        if raw2:
                            end_pdt = datetime.fromisoformat(raw2.replace("Z", "+00:00"))
                            if end_pdt.tzinfo is None:
                                end_pdt = end_pdt.replace(tzinfo=timezone.utc)
                            else:
                                end_pdt = end_pdt.astimezone(timezone.utc)
                    except Exception:
                        end_pdt = None
                    try:
                        playlist = proxy._build_playlist(start_pdt=start_pdt, end_pdt=end_pdt)
                    except Exception:
                        self.send_response(500)
                        self.end_headers()
                        return

                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-mpegURL")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    try:
                        body = playlist.encode("utf-8")
                        self.send_header("Content-Length", str(len(body)))
                    except Exception:
                        body = playlist.encode("utf-8")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if path == "/key":
                    if not proxy._key_url:
                        self.send_response(404)
                        self.end_headers()
                        return
                    try:
                        key_bytes = proxy.client.fetch_key_bytes(proxy._key_url)
                    except Exception:
                        self.send_response(500)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    try:
                        self.send_header("Content-Length", str(len(key_bytes)))
                    except Exception:
                        pass
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(key_bytes)
                    return

                if path == "/seg":
                    # Proxy a media segment so we can track which PDT is actually being played.
                    raw_u = (qs.get("u") or [None])[0]
                    raw_pdt = (qs.get("pdt") or [None])[0]
                    if not raw_u:
                        self.send_response(400)
                        self.end_headers()
                        return

                    seg_pdt: Optional[datetime] = None
                    if raw_pdt:
                        try:
                            seg_pdt = datetime.fromisoformat(raw_pdt.replace("Z", "+00:00"))
                            if seg_pdt.tzinfo is None:
                                seg_pdt = seg_pdt.replace(tzinfo=timezone.utc)
                            else:
                                seg_pdt = seg_pdt.astimezone(timezone.utc)
                        except Exception:
                            seg_pdt = None

                    if seg_pdt is not None:
                        try:
                            with proxy._last_seg_lock:
                                proxy._last_seg_pdt = seg_pdt
                                try:
                                    proxy._seg_pdts.append(seg_pdt)
                                except Exception:
                                    pass
                        except Exception:
                            pass

                    try:
                        # Use the client's request wrapper so any required headers/cookies are applied.
                        rr = proxy.client._request("GET", raw_u)
                        rr.raise_for_status()
                        body = rr.content
                    except Exception:
                        self.send_response(502)
                        self.end_headers()
                        return

                    self.send_response(200)
                    try:
                        ctype = rr.headers.get("Content-Type") or "application/octet-stream"
                    except Exception:
                        ctype = "application/octet-stream"
                    self.send_header("Content-Type", ctype)
                    try:
                        self.send_header("Content-Length", str(len(body)))
                    except Exception:
                        pass
                    self.send_header("Connection", "close")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(body)
                    return

                self.send_response(404)
                self.end_headers()

        # Threaded server is required because players often fetch many media segments
        # concurrently while also polling the playlist; a single-threaded server can
        # deadlock and cause preflight timeouts.
        server = ThreadingHTTPServer((host, port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        addr, port = server.server_address
        return ProxyInfo(url=f"http://{addr}:{port}/listen.m3u8", server=server, thread=thread)

    def stop(self, info: ProxyInfo) -> None:
        try:
            info.server.shutdown()
        except Exception:
            pass

    def _build_playlist(self, *, start_pdt: Optional[datetime] = None, end_pdt: Optional[datetime] = None) -> str:
        # Use the client's request wrapper so 401 triggers auth refresh + retry.
        r = self.client._request("GET", self.variant_url)
        r.raise_for_status()
        text = r.text

        base = self.variant_url.rsplit("/", 1)[0] + "/"

        def abs_seg(line: str) -> str:
            s = line.strip()
            if not s or s.startswith("#"):
                return line
            if s.startswith("http"):
                return s
            return base + s

        def proxied_seg(abs_url: str, pdt: Optional[datetime]) -> str:
            # Use a relative path so the player fetches via this same proxy.
            # This allows us to observe which segment is actually being consumed.
            q_u = quote(abs_url, safe="")
            q_pdt = ""
            if pdt is not None:
                try:
                    q_pdt = quote(pdt.astimezone(timezone.utc).isoformat(), safe="")
                except Exception:
                    q_pdt = ""
            if q_pdt:
                return f"/seg?u={q_u}&pdt={q_pdt}"
            return f"/seg?u={q_u}"

        raw_lines = text.splitlines()

        # Parse media sequence so we can adjust it when slicing the playlist.
        media_seq: Optional[int] = None
        for line in raw_lines:
            if line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
                try:
                    media_seq = int(line.split(":", 1)[1].strip())
                except Exception:
                    media_seq = None
                break

        # Normalize requested start_pdt.
        if start_pdt is not None:
            try:
                if start_pdt.tzinfo is None:
                    start_pdt = start_pdt.replace(tzinfo=timezone.utc)
                else:
                    start_pdt = start_pdt.astimezone(timezone.utc)
            except Exception:
                start_pdt = None

        # Normalize requested end_pdt.
        if end_pdt is not None:
            try:
                if end_pdt.tzinfo is None:
                    end_pdt = end_pdt.replace(tzinfo=timezone.utc)
                else:
                    end_pdt = end_pdt.astimezone(timezone.utc)
            except Exception:
                end_pdt = None

        # Parse segment blocks with PDT + duration + URI.
        segments: list[dict] = []
        cur_pdt: Optional[datetime] = None
        cur_inf: Optional[str] = None
        cur_dur: Optional[float] = None

        for line in raw_lines:
            s = line.strip()
            if s.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
                try:
                    cur_pdt = datetime.fromisoformat(s.split(":", 1)[1].strip().replace("Z", "+00:00"))
                    if cur_pdt.tzinfo is None:
                        cur_pdt = cur_pdt.replace(tzinfo=timezone.utc)
                    else:
                        cur_pdt = cur_pdt.astimezone(timezone.utc)
                except Exception:
                    cur_pdt = None
                continue
            if s.startswith("#EXTINF:"):
                cur_inf = line
                try:
                    cur_dur = float(s.split(":", 1)[1].split(",", 1)[0].strip())
                except Exception:
                    cur_dur = None
                continue
            if s and (not s.startswith("#")):
                segments.append({"pdt": cur_pdt, "dur": cur_dur, "uri": line, "inf": cur_inf})
                # Some playlists only provide PROGRAM-DATE-TIME occasionally.
                # For deterministic slicing, advance PDT by EXTINF duration when possible.
                try:
                    if cur_pdt is not None and cur_dur is not None and cur_dur > 0:
                        cur_pdt = cur_pdt + timedelta(seconds=float(cur_dur))
                except Exception:
                    pass
                cur_inf = None
                cur_dur = None

        # If the upstream playlist doesn't include PROGRAM-DATE-TIME at all, we cannot
        # reliably slice by absolute PDT. However, when the caller provides BOTH start_pdt
        # and end_pdt, we can still bound the playlist by summing EXTINF durations.
        any_pdt = False
        try:
            any_pdt = any(bool(seg.get("pdt")) for seg in segments)
        except Exception:
            any_pdt = False

        # Choose slice start index based on requested PDT.
        start_index = 0
        if start_pdt is not None and segments:
            chosen: Optional[int] = None
            for i, seg in enumerate(segments):
                pdt = seg.get("pdt")
                dur = seg.get("dur")
                if pdt is None:
                    continue
                try:
                    if dur is not None and dur > 0:
                        seg_end = pdt + timedelta(seconds=float(dur))
                        if pdt <= start_pdt < seg_end:
                            chosen = i
                            break
                    # Fallback: newest PDT not after start_pdt.
                    if pdt <= start_pdt:
                        chosen = i
                    else:
                        break
                except Exception:
                    # Keep scanning; worst case we fall back to start_index=0
                    continue
            if chosen is not None:
                start_index = max(0, chosen)

        sliced_segments = segments[start_index:]
        if end_pdt is not None and sliced_segments:
            bounded: list[dict] = []
            # If PDT exists, bound by end_pdt. Otherwise, fall back to bounding by duration.
            if any_pdt:
                for seg in sliced_segments:
                    pdt = seg.get("pdt")
                    if pdt is None:
                        bounded.append(seg)
                        continue
                    try:
                        if pdt < end_pdt:
                            bounded.append(seg)
                        else:
                            break
                    except Exception:
                        bounded.append(seg)
            else:
                target_s: Optional[float] = None
                if start_pdt is not None:
                    try:
                        target_s = max(0.0, float((end_pdt - start_pdt).total_seconds()))
                    except Exception:
                        target_s = None
                acc = 0.0
                for seg in sliced_segments:
                    bounded.append(seg)
                    try:
                        d = float(seg.get("dur") or 0.0)
                    except Exception:
                        d = 0.0
                    if d > 0:
                        acc += d
                    if target_s is not None and target_s > 0 and acc >= target_s:
                        break
            sliced_segments = bounded
        new_media_seq = None
        if media_seq is not None:
            new_media_seq = media_seq + start_index

        out_lines = []
        # Rebuild header + tags, rewriting key URI and media sequence.
        for line in raw_lines:
            if line.startswith("#EXT-X-KEY"):
                m = re.search(r'URI="([^"]+)"', line)
                if m:
                    key_url = m.group(1)
                    if key_url and not key_url.startswith("http"):
                        key_url = base + key_url.lstrip("/")
                    self._key_url = key_url
                    line = re.sub(r'URI="[^"]+"', 'URI="/key"', line)

            if line.startswith("#EXT-X-MEDIA-SEQUENCE:") and new_media_seq is not None:
                out_lines.append(f"#EXT-X-MEDIA-SEQUENCE:{new_media_seq}")
                continue

            # Stop header copy once we reach the first segment markers; we'll append sliced segments.
            # (We must NOT include the original playlist's first segment PDT/EXTINF, otherwise
            # consumers will think slicing failed.)
            s = line.strip()
            if s.startswith("#EXT-X-PROGRAM-DATE-TIME:") or s.startswith("#EXTINF:"):
                break
            if s and (not s.startswith("#")):
                break
            out_lines.append(line)

        # Append sliced segments (PDT, EXTINF, URI) with absolute URIs.
        for seg in sliced_segments:
            pdt = seg.get("pdt")
            inf = seg.get("inf")
            uri = seg.get("uri")
            if pdt is not None:
                out_lines.append(f"#EXT-X-PROGRAM-DATE-TIME:{pdt.isoformat()}")
            if inf:
                out_lines.append(inf)
            if uri:
                abs_url = abs_seg(uri)
                if self.proxy_segments:
                    out_lines.append(proxied_seg(abs_url, pdt))
                else:
                    out_lines.append(abs_url)

        # If we were asked for an end bound, make this a finite playlist so ffmpeg can
        # download as fast as possible and exit naturally.
        if end_pdt is not None:
            out_lines.append("#EXT-X-ENDLIST")

        return "\n".join(out_lines) + "\n"
