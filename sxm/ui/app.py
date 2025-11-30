"""Main SatStash application"""
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Header, Footer, Static, Button
from textual.binding import Binding
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sxm.utils.config import Config
from sxm.utils.session_manager import SessionManager
from sxm.ui.channel_browser import ChannelBrowser
from sxm.ui.login_screen import LoginScreen
from sxm.ui.recording_library import RecordingLibrary

class SiriusXMPro(App):
    """The Ultimate SiriusXM Terminal App"""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    #title {
        width: 100%;
        height: 3;
        content-align: center middle;
        background: $primary;
        color: $text;
        text-style: bold;
    }
    
    #session_status {
        width: 100%;
        height: 1;
        content-align: center middle;
        background: $surface-darken-1;
        color: $success;
    }
    
    .menu-button {
        width: 100%;
        height: 3;
        margin: 1 2;
    }
    
    Container {
        height: auto;
        align: center middle;
    }
    """
    
    TITLE = "SatStash"
    SUB_TITLE = "The Ultimate Terminal Experience"
    
    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("?", "help", "Help"),
    ]
    
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.session_mgr = SessionManager()
        self.current_username = None
        self.session_valid = False
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header(show_clock=True)
        yield Container(
            Static("🎵 SatStash", id="title"),
            Static("Checking session...", id="session_status"),
            Vertical(
                Button("🔐 Login", id="login", variant="success", classes="menu-button"),
                Button("📻 Browse Channels", id="browse", variant="primary", classes="menu-button"),
                Button("🔴 Record Now", id="record", variant="success", classes="menu-button"),
                Button("📼 DVR Mode", id="dvr", variant="warning", classes="menu-button"),
                Button("⏰ Scheduled Recordings", id="schedule", classes="menu-button"),
                Button("📚 Recording Library", id="library", classes="menu-button"),
                Button("⚙️  Settings", id="settings", classes="menu-button"),
                Button("🚪 Quit", id="quit", variant="error", classes="menu-button"),
                id="menu"
            )
        )
        yield Footer()
    
    async def on_mount(self) -> None:
        """Check session on mount"""
        await self.check_session()
    
    async def check_session(self):
        """Check if we have a valid session"""
        session_info = self.session_mgr.get_session_info()
        status_widget = self.query_one("#session_status", Static)
        
        if session_info and session_info['is_valid']:
            self.current_username = session_info['username']
            self.session_valid = True
            
            time_remaining = session_info['time_remaining']
            status_widget.update(f"✅ Logged in as: {self.current_username} (expires in {time_remaining})")
        else:
            self.session_valid = False
            status_widget.update("⚠️  No valid session - Click 🔐 Login button to authenticate")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events"""
        button_id = event.button.id
        
        if button_id == "quit":
            self.exit()
        elif button_id == "login":
            self.push_screen(LoginScreen())
        elif button_id == "browse":
            if self.session_valid:
                self.push_screen(ChannelBrowser(dvr_mode=False))
            else:
                self.notify("Please login first!", severity="error")
        elif button_id == "record":
            if self.session_valid:
                # Open channel browser first to select channel
                self.push_screen(ChannelBrowser(dvr_mode=False))
            else:
                self.notify("Please login first!", severity="error")
        elif button_id == "dvr":
            if self.session_valid:
                self.push_screen(ChannelBrowser(dvr_mode=True))
            else:
                self.notify("Please login first!", severity="error")
        elif button_id == "schedule":
            self.notify("Scheduled Recordings - Coming soon!", severity="warning")
        elif button_id == "library":
            self.push_screen(RecordingLibrary())
        elif button_id == "settings":
            from sxm.ui.settings_screen import SettingsScreen
            self.push_screen(SettingsScreen())
    
    def action_help(self) -> None:
        """Show help"""
        self.notify("Help: Use arrow keys and Enter to navigate. Press Q to quit.", title="Help")


if __name__ == "__main__":
    app = SiriusXMPro()
    app.run()
