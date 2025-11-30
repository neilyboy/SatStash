#!/usr/bin/env python3
"""
DVR Browser Screen

Browse and download past content from 5-hour DVR buffer
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, DataTable, Button, Input, Label
from textual.containers import Container, Horizontal, Vertical
from textual.binding import Binding
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional


class DVRBrowser(Screen):
    """Browse DVR buffer and download past tracks"""
    
    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back", priority=True),
        Binding("r", "refresh", "Refresh", priority=True),
        Binding("d", "download_selected", "Download", priority=True),
        Binding("a", "download_all", "Download All", priority=True),
    ]
    
    CSS = """
    DVRBrowser {
        layout: vertical;
    }
    
    #info_panel {
        height: 7;
        border: solid $primary;
        padding: 1;
        margin: 1;
    }
    
    #controls {
        height: auto;
        padding: 1;
    }
    
    #dvr_table {
        height: 1fr;
        border: solid $accent;
    }
    
    #status {
        height: 3;
        padding: 1;
        background: $surface;
    }
    
    .button_row {
        height: auto;
        align: center middle;
    }
    """
    
    def __init__(self, channel: Dict):
        super().__init__()
        self.channel = channel
        self.dvr_tracks = []
        self.api = None
        self.bearer_token = None
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(id="info_panel"):
            yield Label(f"📼 DVR Browser - {self.channel['name']} (Ch {self.channel.get('number', '?')})")
            yield Label("Browse last 5+ hours of DVR content")
            yield Label("↑↓: Select track | D: Download | A: Download All | R: Refresh")
        
        with Horizontal(id="controls", classes="button_row"):
            yield Button("🔄 Refresh", id="btn_refresh", variant="primary")
            yield Button("💾 Download Selected", id="btn_download", variant="success")
            yield Button("📥 Download All", id="btn_download_all", variant="warning")
            yield Button("🔙 Back", id="btn_back")
        
        yield DataTable(id="dvr_table")
        
        with Container(id="status"):
            yield Label("", id="status_text")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize DVR browser"""
        table = self.query_one("#dvr_table", DataTable)
        table.cursor_type = "row"
        
        # Add columns with specific widths
        table.add_column("Time", width=30)
        table.add_column("Artist", width=25)
        table.add_column("Title", width=30)
        table.add_column("Album", width=25)
        table.add_column("Duration", width=10)
        
        # Load DVR content
        self.action_refresh()
    
    def action_refresh(self) -> None:
        """Refresh DVR content"""
        self.update_status("📡 Loading DVR buffer...")
        
        try:
            # Get session and bearer token
            from sxm.utils.session_manager import SessionManager
            sm = SessionManager()
            
            session_info = sm.get_session_info()
            if not session_info or not session_info['is_valid']:
                self.update_status("❌ Session expired - please login")
                return
            
            # Get bearer token directly from session manager
            self.bearer_token = sm.get_bearer_token()
            
            if not self.bearer_token:
                self.update_status("❌ No bearer token found - try logging in again")
                return
            
            # Initialize API
            from sxm.core.api import SiriusXMAPI
            self.api = SiriusXMAPI(self.bearer_token)
            
            # Get DVR tracks (last 5 hours)
            self.update_status("📡 Fetching DVR tracks (last 5 hours)...")
            tracks = self.api.get_dvr_tracks(self.channel['id'], hours_back=5)
            
            if not tracks:
                self.update_status("❌ No DVR tracks found (try refreshing or check your session)")
                return
            
            self.dvr_tracks = tracks
            
            # Update table
            table = self.query_one("#dvr_table", DataTable)
            table.clear()
            
            # Sort tracks by time (oldest first for better display)
            sorted_tracks = sorted(tracks, key=lambda t: t['timestamp_utc'])
            
            now = datetime.now(timezone.utc)
            oldest_time = None
            newest_time = None
            
            for track in sorted_tracks:
                # Parse timestamp
                track_time = datetime.fromisoformat(
                    track['timestamp_utc'].replace('Z', '+00:00')
                )
                
                # Track oldest and newest
                if oldest_time is None or track_time < oldest_time:
                    oldest_time = track_time
                if newest_time is None or track_time > newest_time:
                    newest_time = track_time
                
                # Format time with "ago" indicator
                time_ago = now - track_time
                hours_ago = int(time_ago.total_seconds() // 3600)
                mins_ago = int((time_ago.total_seconds() % 3600) // 60)
                
                if hours_ago > 0:
                    time_str = f"{track_time.strftime('%I:%M %p')} ({hours_ago}h {mins_ago}m ago)"
                else:
                    time_str = f"{track_time.strftime('%I:%M %p')} ({mins_ago}m ago)"
                
                # Format duration
                duration_sec = track['duration_ms'] / 1000
                duration_str = f"{int(duration_sec // 60)}:{int(duration_sec % 60):02d}"
                
                table.add_row(
                    time_str,
                    track['artist'],
                    track['title'],
                    track.get('album', 'Unknown'),
                    duration_str
                )
            
            # Show time range in status
            if oldest_time and newest_time:
                time_span = (newest_time - oldest_time).total_seconds() / 3600
                self.update_status(f"✅ Loaded {len(tracks)} tracks ({time_span:.1f} hours of content)")
            
        except Exception as e:
            self.update_status(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def action_download_selected(self) -> None:
        """Download selected track"""
        table = self.query_one("#dvr_table", DataTable)
        
        if table.cursor_row is None or table.cursor_row >= len(self.dvr_tracks):
            self.update_status("⚠️  No track selected")
            return
        
        track = self.dvr_tracks[table.cursor_row]
        self.download_track(track)
    
    def action_download_all(self) -> None:
        """Download all tracks in DVR buffer"""
        if not self.dvr_tracks:
            self.update_status("⚠️  No tracks to download")
            return
        
        self.update_status(f"📥 Downloading {len(self.dvr_tracks)} tracks...")
        
        # TODO: Implement bulk download
        self.update_status("⚠️  Bulk download not yet implemented")
    
    def download_track(self, track: Dict) -> None:
        """Download a single track from DVR"""
        self.update_status(f"📥 Downloading: {track['artist']} - {track['title']}...")
        
        # TODO: Implement actual download
        # This would use the dvr_downloader module
        self.update_status("⚠️  Download not yet implemented")
    
    def update_status(self, message: str) -> None:
        """Update status message"""
        status_label = self.query_one("#status_text", Label)
        status_label.update(message)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id
        
        if button_id == "btn_refresh":
            self.action_refresh()
        elif button_id == "btn_download":
            self.action_download_selected()
        elif button_id == "btn_download_all":
            self.action_download_all()
        elif button_id == "btn_back":
            self.app.pop_screen()
