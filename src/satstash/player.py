from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
import time
import os
import re
from typing import List, Optional

from platformdirs import user_cache_dir


@dataclass
class PlayerCommand:
    argv: List[str]


def build_player_command(url: str) -> Optional[PlayerCommand]:
    """Return a command that plays the given URL.

    Keep this simple: play the local proxy m3u8.
    """

    if shutil.which("mpv"):
        return PlayerCommand(argv=["mpv", "--no-terminal", "--msg-level=all=fatal", url])

    if shutil.which("ffplay"):
        return PlayerCommand(argv=["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", url])

    return None


def run_player(cmd: PlayerCommand) -> subprocess.Popen:
    env = os.environ.copy()

    # On some systems (notably Chrome Remote Desktop / PipeWire setups),
    # the Pulse server socket isn't the default /run/user/.../pulse/native.
    # If PULSE_SERVER isn't set, try to discover it from pactl.
    if cmd.argv and cmd.argv[0].endswith("mpv") and not env.get("PULSE_SERVER"):
        try:
            out = subprocess.check_output(["pactl", "info"], text=True, stderr=subprocess.DEVNULL)
            m = re.search(r"^Server String:\s*(.+)$", out, flags=re.MULTILINE)
            if m:
                env["PULSE_SERVER"] = m.group(1).strip()
        except Exception:
            pass

    proc = subprocess.Popen(
        cmd.argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    # If the player dies immediately, surface the reason.
    time.sleep(0.6)
    rc = proc.poll()
    if rc is not None and rc != 0:
        out, err = proc.communicate(timeout=2)
        msg = (err or out or "").strip()
        msg = msg[-1200:] if len(msg) > 1200 else msg
        # If mpv is logging to a file, include the tail for better diagnostics.
        log_path = None
        try:
            for a in (cmd.argv or []):
                if isinstance(a, str) and a.startswith("--log-file="):
                    log_path = a.split("=", 1)[1]
                    break
        except Exception:
            log_path = None

        if log_path:
            try:
                if os.path.exists(log_path):
                    tail = ""
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        data = f.read()
                    lines = data.splitlines()
                    tail_lines = lines[-80:] if len(lines) > 80 else lines
                    tail = "\n".join(tail_lines).strip()
                    if tail:
                        msg = (msg + "\n\nmpv log tail:\n" + tail).strip()
            except Exception:
                pass

        raise RuntimeError(f"Player exited immediately (code {rc}). {msg}")

    return proc


def build_player_command_with_preference(url: str, *, preference: str = "auto") -> Optional[PlayerCommand]:
    pref = (preference or "auto").strip().lower()
    if pref == "mpv":
        if shutil.which("mpv"):
            return PlayerCommand(argv=["mpv", "--no-terminal", "--msg-level=all=fatal", url])
        # If mpv isn't installed, treat this like auto so playback still works.
        return build_player_command(url)
    if pref == "ffplay":
        if shutil.which("ffplay"):
            return PlayerCommand(argv=["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", url])
        # If ffplay isn't installed, treat this like auto so playback still works.
        return build_player_command(url)
    return build_player_command(url)


def build_player_command_with_preference_and_debug(
    url: str, *, preference: str = "auto", debug: bool = False
) -> Optional[PlayerCommand]:
    cmd = build_player_command_with_preference(url, preference=preference)
    if not cmd:
        return None

    if not debug:
        return cmd

    # Write a per-launch log file for troubleshooting.
    try:
        log_dir = user_cache_dir("satstash")
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        log_path = os.path.join(log_dir, f"player-{ts}.log")
    except Exception:
        log_path = f"player-{int(time.time())}.log"

    argv = list(cmd.argv)
    if argv and argv[0].endswith("mpv"):
        # Increase verbosity and log to file.
        argv = ["mpv", "--no-terminal", "--msg-level=all=info", f"--log-file={log_path}"] + argv[3:]
    elif argv and argv[0].endswith("ffplay"):
        argv = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "info", url]
    return PlayerCommand(argv=argv)
