# SatStash — Cheat Sheet (Keys by Screen)

This is the quick-reference key map for SatStash.

## Global keys (work everywhere)

- **`Q`**: Quit
- **`h`**: Home
- **`?`**: Help (context keys overlay)
- **`p`**: Playlists (fullscreen)
- **`space`**: Play/Pause
- **`s`**: Stop playback
- **`n`**: Next track
- **`b`**: Previous track
- **`[` / `]`**: Seek -10s / +10s
- **`S`**: Schedule

## Home screen

- **Home menu buttons**: use mouse or `Tab` then `Enter`
- **Playback keys**: see Global keys

## DVR Browse (right pane)

Table-focused:

- **`Enter`**: Open folder / Play file
- **`Backspace` / `Esc`**: Back (up one folder / back to root)
- **`q`**: Toggle Queue view
- **`e`**: Enqueue selected (file/folder/playlist)
- **`E`**: Enqueue folder (folder only)
- **`f`**: Play folder now (folder only)

## DVR Queue (right pane, after `q`)

- **`Enter`**: Jump/play selected queue item
- **`x`**: Shuffle ON/OFF
- **`w`**: Save queue to playlist (prompt)
- **`l`**: Load playlist into queue (prompt)
- **`d`**: Remove selected item from queue
- **`c`**: Clear queue (confirm)
- **`J` / `K`**: Move selected item down / up
- **`q`**: Back to Browse

## DVR Now Playing screen

- Uses Global keys
- (Also has on-screen buttons)

## Playlist Manager (fullscreen)

- **`Esc`**: Back
- **`Enter`**: Load selected playlist into queue (replace)
- **`a`**: Append selected playlist into queue
- **`d`**: Delete selected playlist (confirm)
- **`n`**: New playlist from current queue
- **`r`**: Rename selected playlist
- **`/`**: Focus search box
- **`R`**: Refresh playlist list

## Where files are stored

- **Current DVR queue** (auto-saved): `~/.config/satstash/dvr_queue.json`
- **Saved playlists**: `~/.config/satstash/playlists/<name>.json`
- **Scheduled recordings**: `~/.config/satstash/scheduled_recordings.json`

Media outputs are written under **Settings → Output folder** (default: `~/Music/SiriusXM`):

- `Live/` (Record Now + scheduled recordings)
- `CatchUp/` (catch-up exports)
- `VOD/` (episode URL downloads)
