# SatStash V1.0

Minimal, working snapshot of the SatStash SiriusXM CLI recorder / player.

This tree is intended to be pushed to GitHub and cloned onto a fresh server.
It contains only the files needed to install and run SatStash, including the
background scheduler daemon.

## Features

- Listen Live playback via local HLS proxy and `ffplay`
- DVR-backed history with quick track downloads
- Simple TUI with single-key controls
- Background scheduler daemon for timed recordings

## Requirements

- Linux (tested on Ubuntu/Debian-like systems)
- `python3` (3.10+ recommended)
- `python3-venv` package installed
- `ffmpeg` (for `ffplay`)

On Debian/Ubuntu you can install base deps with:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv ffmpeg
```

## Quick start

Clone your GitHub repo, then:

```bash
cd SatStash_V1.0
./install.sh       # sets up venv, installs Python deps, ensures Pillow & ffplay
./sxm_app          # launches the SatStash CLI/TUI
```

After `install.sh` finishes, you can also use:

- `./run.sh` – simple wrapper around `./sxm_app`
- `satstash` – launcher installed into `~/.local/bin` (if on your `PATH`)

The `sxm_app` script will:

- Create a local `venv/` if it does not exist
- Install required Python packages (mutagen, requests, playwright, rich, Pillow, etc.)
- Install Playwright Chromium on first run
- Start the SatStash CLI (`sxm_cli.py`)

## Running the scheduler daemon

The scheduler runs as a simple long-lived Python process. Use the helper script:

```bash
cd SatStash_V1.0
./scripts/run_scheduler.sh &
```

This will:

- Use `venv/bin/python` from this directory
- Start `scheduler_daemon.py`
- Log output to `~/.satstash/logs/scheduler.log`

You can integrate it with `cron` or `systemd` if you prefer. For example,
from `cron` you might run:

```cron
@reboot /path/to/SatStash_V1.0/scripts/run_scheduler.sh
```

### Running the scheduler as a systemd user service

`install.sh` can also configure a systemd *user* service that runs the
scheduler in the background:

```bash
cd SatStash_V1.0
./install.sh --install-service
```

This will create and enable a `satstash-scheduler.service` under
`~/.config/systemd/user/`. You can check its status with:

```bash
systemctl --user status satstash-scheduler
```

## Configuration & data

SatStash stores user config and runtime data under your home directory:

- `~/.seriouslyxm/` – config, auth/session data, scheduled recordings JSON
- `~/.satstash/logs/` – scheduler logs
- `~/Music/SiriusXM/` – recorded audio output

These directories are **not** part of the Git repo and will be created on demand.

## Updating

To deploy a new version:

1. Update code in your main development repo.
2. Refresh the `SatStash_V1.0` snapshot (copy updated files into it).
3. Commit and push `SatStash_V1.0` to GitHub.
4. On the server, `git pull` and restart `./sxm_app` and the scheduler.
