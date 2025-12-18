# SatStash v3 — DVR Browse + Queue (User Guide)

This document describes how to use the **Browse DVR** pane and the **DVR Queue** from the SatStash v3 TUI.

## Concepts

### DVR files vs queue

- **Browse DVR** is a file/folder browser rooted at your **Settings → Output folder**.
- **DVR Queue** is an ordered list of local files SatStash will play through.
- The queue behaves like a playlist with a **current index**:
  - Items do **not** disappear after playback.
  - SatStash advances the internal queue index to the next item.

By default, recordings and exports are written under:

- `Live/`
- `CatchUp/`
- `VOD/`

### What “Play” does

- Playing a single file sets the DVR queue to just that file (queue length becomes 1).
- Playing a folder builds a queue from that folder’s playable files.

## Global keys (always available)

- **`Q`**: Quit
- **`h`**: Home
- **`space`**: Play/Pause
- **`n`**: Next track
- **`b`**: Previous track
- **`p`**: Playlists
- **`s`**: Stop playback

## Browse DVR pane

The right pane has two main views:

- **Browse DVR** (folders/files)
- **DVR Queue** (queue list)

### Navigation

- **Arrow keys**: Move the highlight in the table
- **`Enter`** (table focused): Open/Play the highlighted item
- **`Backspace` / `Esc`**:
  - If you are inside nested folders: go **up one directory**
  - If you are at the recordings root: return to the top folder list

### Open/Play (`Enter`)

When the table is focused:

- If the highlighted row is a **folder**: enter that folder
- If the highlighted row is a **file**: play it immediately

### Queue view (`q`)

When the table is focused:

- **`q`**: Toggle between Browse DVR and DVR Queue

In **DVR Queue** view:

- The table lists queued files.
- The **Info** column shows `playing` for the current queue item.
- **`Enter`**: Jump/play the highlighted queued item.

#### Queue management (queue view only)

- **`x`**: Toggle shuffle (ON/OFF)
- **`w`**: Save the current queue to a named playlist (prompts for playlist name)
- **`l`**: Load a named playlist into the queue (prompts for playlist name)
- **`d`**: Remove the selected item from the queue
- **`c`**: Clear the entire queue (confirm)
- **`J` / `K`**: Move the selected queue item down / up

Playlists are stored under:

- `~/.config/satstash/playlists/<name>.json`

The current DVR queue is persisted automatically across restarts at:

- `~/.config/satstash/dvr_queue.json`

### Enqueue / play-folder shortcuts

These are table-focused shortcuts intended for keyboard-only operation:

- **`e`**: Enqueue the highlighted selection
  - File: enqueue that file
  - Folder: enqueue playable files under that folder
  - Playlist file (`.m3u` / `.m3u8`): enqueue playlist entries
- **`E`**: Enqueue folder (only meaningful when the selection is a folder)
- **`F`**: Play folder now (immediately builds a queue from that folder and starts playback)

Enqueue is designed to be **non-destructive**:

- It should not stop your current playback.
- If the item is already present in the queue, SatStash will report it as an existing/duplicate entry.

## Notes / expected behavior

- **Progress bar**:
  - With `mpv`, SatStash can read exact playback timing via IPC.
  - With `ffplay`/`vlc`, SatStash estimates progress based on start time and probed duration.
- **Auto-next**:
  - SatStash will advance to the next queued item when a track ends.
  - For `ffplay`/`vlc`, auto-next uses a conservative “near end” heuristic.

## Troubleshooting

- If keys don’t seem to work, make sure the **table** is focused (click it once, then use keyboard).
- If you want to inspect the queue at any time, press **`q`** in Browse DVR.
