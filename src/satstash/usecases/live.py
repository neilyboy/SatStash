from __future__ import annotations

from dataclasses import dataclass
import subprocess
from pathlib import Path
from typing import Callable, Dict, Optional

import requests

from platformdirs import user_cache_dir

from satstash.api.client import SxmClient
from satstash.hls.proxy import HlsProxy, ProxyInfo
from satstash.hls.variants import select_variant
from satstash.player import build_player_command_with_preference_and_debug, run_player


@dataclass
class LivePlaybackHandle:
    proxy: HlsProxy
    proxy_info: ProxyInfo
    process: subprocess.Popen


def start_live_playback(
    *,
    client: SxmClient,
    channel_id: str,
    channel_type: str = "channel-linear",
    preferred_quality: str = "256k",
    progress: Optional[Callable[[str], None]] = None,
    player_preference: str = "auto",
    debug: bool = False,
) -> LivePlaybackHandle:
    def say(msg: str) -> None:
        if progress:
            progress(msg)

    manifest_variant = "WEB" if channel_type == "channel-linear" else "FULL"
    say("tuneSource...")
    tune = client.tune_source(
        entity_id=channel_id,
        entity_type=channel_type,
        manifest_variant=manifest_variant,
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

    # Preflight: ensure the proxy can serve a playlist before launching a player.
    try:
        say("preflight proxy playlist...")
        r = requests.get(proxy_info.url, timeout=10)
        r.raise_for_status()

        playlist_text = r.text or ""
        seg_count = 0
        for line in playlist_text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            seg_count += 1
        if seg_count == 0:
            raise RuntimeError("playlist contains no media segments")
        needs_key = "#EXT-X-KEY" in playlist_text

        # If the playlist references a key, ensure /key returns sane bytes.
        # (This avoids the common "starts but no audio" failure mode.)
        if needs_key:
            key_url = proxy_info.url.rsplit("/", 1)[0] + "/key"
            say("preflight proxy key...")
            rk = requests.get(key_url, timeout=10)
            rk.raise_for_status()
            key_bytes = rk.content
            if not key_bytes:
                raise RuntimeError("key endpoint returned empty body")
            if len(key_bytes) not in (16, 24, 32):
                # Some providers return JSON/base64; our proxy should serve raw bytes.
                # Treat unexpected sizes as a hard failure.
                raise RuntimeError(f"invalid key length: {len(key_bytes)}")
    except Exception as exc:
        diag_dir = None
        playlist_path = None
        key_path = None
        ctx_path = None
        try:
            diag_dir = Path(user_cache_dir("satstash"))
            diag_dir.mkdir(parents=True, exist_ok=True)
            playlist_path = diag_dir / "last_proxy_playlist.m3u8"
            playlist_path.write_text(locals().get("playlist_text", ""), encoding="utf-8")

            try:
                import json as _json

                ctx = {
                    "channel_id": channel_id,
                    "channel_type": channel_type,
                    "preferred_quality": preferred_quality,
                    "master_url": master,
                    "variant_url": sel.variant_url,
                    "proxy_url": proxy_info.url,
                }
                ctx_path = diag_dir / "last_playback_context.json"
                ctx_path.write_text(_json.dumps(ctx, indent=2) + "\n", encoding="utf-8")
            except Exception:
                pass

            if "rk" in locals():
                try:
                    import json as _json
                    import hashlib

                    kb = locals().get("key_bytes", b"")
                    sha256 = None
                    try:
                        if isinstance(kb, (bytes, bytearray)):
                            sha256 = hashlib.sha256(kb).hexdigest()
                    except Exception:
                        sha256 = None
                    info = {
                        "status": getattr(locals().get("rk"), "status_code", None),
                        "length": len(kb) if isinstance(kb, (bytes, bytearray)) else None,
                        "sha256": sha256,
                    }
                    key_path = diag_dir / "last_proxy_key.json"
                    key_path.write_text(_json.dumps(info, indent=2) + "\n", encoding="utf-8")
                except Exception:
                    pass
        except Exception:
            pass
        proxy.stop(proxy_info)

        extra = []
        if playlist_path:
            extra.append(f"playlist={playlist_path}")
        if key_path:
            extra.append(f"key={key_path}")
        if ctx_path:
            extra.append(f"ctx={ctx_path}")
        suffix = f" ({', '.join(extra)})" if extra else ""
        raise RuntimeError(f"Proxy preflight failed: {exc}{suffix}")

    say("launching player...")
    player_cmd = build_player_command_with_preference_and_debug(
        proxy_info.url,
        preference=player_preference,
        debug=debug,
    )
    if not player_cmd:
        proxy.stop(proxy_info)
        raise RuntimeError("No supported player found (install mpv or ffplay)")

    say(f"player: {' '.join(player_cmd.argv)}")

    if debug:
        try:
            for a in player_cmd.argv:
                if a.startswith("--log-file="):
                    say(f"player log: {a.split('=', 1)[1]}")
                    break
        except Exception:
            pass

    proc = run_player(player_cmd)
    return LivePlaybackHandle(proxy=proxy, proxy_info=proxy_info, process=proc)
