#!/usr/bin/env python3
"""
Recording Library Screen

Browse and play saved recordings
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, DataTable, Button, Label
from textual.containers import Container, Horizontal, Vertical
from textual.binding import Binding
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime


class RecordingLibrary(Screen):
    """Browse saved recordings"""
    
    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back", priority=True),
        Binding("r", "refresh", "Refresh", priority=True),
        Binding("d", "delete_selected", "Delete", priority=True),
        Binding("o", "open_folder", "Open Folder", priority=True),
    ]
    
    CSS = """
    RecordingLibrary {
        layout: vertical;
        background: $surface;
    }
    
    #title_panel {
        height: 6;
        border: solid $primary;
        padding: 1;
        margin: 1;
    }
    
    #stats_panel {
        height: 5;
        padding: 1;
        background: $surface-darken-1;
        margin: 1;
    }
    
    #recordings_table {
        height: 1fr;
        border: solid $accent;
    }
    
    #controls {
        height: auto;
        padding: 1;
        align: center middle;
    }
    
    #status {
        height: 3;
        padding: 1;
        background: $surface-darken-1;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.recordings = []
        self.output_dir = None
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(id="title_panel"):
            yield Label("📚 Recording Library")
            yield Label("Browse and manage your recordings")
        
        with Container(id="stats_panel"):
            yield Label("", id="stats_text")
        
        yield DataTable(id="recordings_table")
        
        with Horizontal(id="controls"):
            yield Button("🔄 Refresh", id="btn_refresh", variant="primary")
            yield Button("📂 Open Folder", id="btn_open", variant="success")
            yield Button("🗑️  Delete", id="btn_delete", variant="error")
            yield Button("🔙 Back", id="btn_back")
        
        with Container(id="status"):
            yield Label("", id="status_text")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize library"""
        # Get output directory from config
        from sxm.utils.config import Config
        config = Config()
        self.output_dir = Path(config.get('output_directory', '~/Music/SiriusXM')).expanduser()
        
        # Setup table
        table = self.query_one("#recordings_table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Date", "Channel", "Duration", "Tracks", "Size", "Path")
        
        # Load recordings
        self.action_refresh()
    
    def action_refresh(self) -> None:
        """Refresh recording list"""
        self.update_status("📡 Scanning for recordings...")
        
        try:
            self.recordings = self.scan_recordings()
            
            # Update stats
            total_size = sum(r['size_mb'] for r in self.recordings)
            total_tracks = sum(r['track_count'] for r in self.recordings)
            
            stats = self.query_one("#stats_text", Label)
            stats.update(
                f"📊 {len(self.recordings)} sessions | "
                f"🎵 {total_tracks} tracks | "
                f"💾 {total_size:.1f} MB"
            )
            
            # Update table
            table = self.query_one("#recordings_table", DataTable)
            table.clear()
            
            for rec in self.recordings:
                table.add_row(
                    rec['date'],
                    rec['channel'],
                    rec['duration'],
                    str(rec['track_count']),
                    f"{rec['size_mb']:.1f} MB",
                    rec['path']
                )
            
            self.update_status(f"✅ Found {len(self.recordings)} recordings")
            
        except Exception as e:
            self.update_status(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def scan_recordings(self) -> List[Dict]:
        """Scan output directory for recordings"""
        recordings = []
        
        if not self.output_dir.exists():
            return recordings
        
        # Iterate through channel folders
        for channel_dir in self.output_dir.iterdir():
            if not channel_dir.is_dir():
                continue
            
            channel_name = channel_dir.name
            
            # Iterate through session folders
            for session_dir in channel_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                
                try:
                    # Look for recording_info.json
                    info_file = session_dir / 'recording_info.json'
                    if not info_file.exists():
                        continue
                    
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    
                    # Calculate size
                    size_bytes = sum(
                        f.stat().st_size 
                        for f in session_dir.rglob('*') 
                        if f.is_file()
                    )
                    size_mb = size_bytes / (1024 * 1024)
                    
                    # Count tracks
                    tracks_dir = session_dir / 'tracks'
                    track_count = len(list(tracks_dir.glob('*.m4a'))) if tracks_dir.exists() else 0
                    
                    # Parse date from session name or info
                    session_name = session_dir.name
                    try:
                        date_str = datetime.strptime(session_name, '%Y-%m-%d_%H-%M').strftime('%Y-%m-%d %H:%M')
                    except:
                        date_str = info.get('recorded_at', session_name)[:16]
                    
                    # Calculate duration
                    duration_sec = info.get('actual_duration_seconds', 0)
                    duration_str = f"{int(duration_sec // 60)}m"
                    
                    recordings.append({
                        'date': date_str,
                        'channel': channel_name,
                        'duration': duration_str,
                        'track_count': track_count,
                        'size_mb': size_mb,
                        'path': str(session_dir),
                        'info': info
                    })
                    
                except Exception as e:
                    # Skip invalid recordings
                    continue
        
        # Sort by date (newest first)
        recordings.sort(key=lambda x: x['date'], reverse=True)
        
        return recordings
    
    def action_open_folder(self) -> None:
        """Open selected recording folder"""
        table = self.query_one("#recordings_table", DataTable)
        
        if table.cursor_row is None or table.cursor_row >= len(self.recordings):
            self.update_status("⚠️  No recording selected")
            return
        
        recording = self.recordings[table.cursor_row]
        path = recording['path']
        
        try:
            import subprocess
            subprocess.run(['xdg-open', path], check=True)
            self.update_status(f"📂 Opened: {path}")
        except Exception as e:
            self.update_status(f"❌ Could not open folder: {e}")
    
    def action_delete_selected(self) -> None:
        """Delete selected recording"""
        table = self.query_one("#recordings_table", DataTable)
        
        if table.cursor_row is None or table.cursor_row >= len(self.recordings):
            self.update_status("⚠️  No recording selected")
            return
        
        recording = self.recordings[table.cursor_row]
        
        # TODO: Add confirmation dialog
        self.update_status(f"⚠️  Delete functionality requires confirmation dialog")
    
    def update_status(self, message: str) -> None:
        """Update status message"""
        status_label = self.query_one("#status_text", Label)
        status_label.update(message)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id
        
        if button_id == "btn_refresh":
            self.action_refresh()
        elif button_id == "btn_open":
            self.action_open_folder()
        elif button_id == "btn_delete":
            self.action_delete_selected()
        elif button_id == "btn_back":
            self.app.pop_screen()
