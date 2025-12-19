# SatStash (Initial Release)

SatStash is a terminal UI (TUI) helper for SiriusXM that supports:

- Live playback
- DVR browsing + queue
- Recording (single file or per-track)
- Catch Up export
- Scheduling
- VOD / AOD episode downloads (paste episode URL)

This folder is a curated, GitHub-ready snapshot.

## Install

### pipx (recommended)

```bash
pipx install -e .
```

### venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

```bash
satstash
```

## Settings

Open **Settings** in the UI to configure:

- **Output folder** (default: `~/Music/SiriusXM`)
- Preferred quality (e.g. `256k`)

## Where files are stored

All media outputs are written under **Settings → Output folder** using these subfolders:

- `Live/` — Record Now + scheduled recordings
- `CatchUp/` — catch-up exports
- `VOD/` — episode URL downloads

## VOD / AOD

VOD downloads do **not** require browser automation.

Workflow:

1. Open **VOD**
2. Paste an episode URL like `https://www.siriusxm.com/player/episode-audio/entity/<id>`
3. **Inspect** to load metadata and variants
4. **Download** (single file) or **Download Split** (per-track files)

## Docs

- [Cheat Sheet](./docs/CHEAT_SHEET.md)
- [DVR Browse + Queue User Guide](./docs/DVR_USAGE.md)
- [Project README / deeper notes](./docs/README.md)
