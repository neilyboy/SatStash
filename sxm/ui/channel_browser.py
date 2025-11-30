"""Channel browser screen with search"""
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, Input, DataTable, Static
from textual.containers import Container, Vertical
from textual.binding import Binding
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sxm.utils.search import ChannelSearch
from sxm.utils.channel_manager import ChannelManager
from sxm.utils.config import Config
from sxm.utils.session_manager import SessionManager


class ChannelBrowser(Screen):
    """Browse and search SiriusXM channels"""
    
    def __init__(self, dvr_mode: bool = False):
        """Initialize channel browser
        
        Args:
            dvr_mode: If True, selecting a channel opens DVR browser instead of recording
        """
        super().__init__()
        self.dvr_mode = dvr_mode
        self.config = Config()
        self.session_mgr = SessionManager()
        # Get bearer token from session for dynamic channel fetching
        bearer_token = self.session_mgr.get_bearer_token()
        self.channel_manager = ChannelManager(bearer_token=bearer_token)
        self.channels = []
        self.search_engine = None
        self.selected_channel = None
    
    CSS = """
    ChannelBrowser {
        background: $surface;
    }
    
    #search_container {
        height: 5;
        width: 100%;
        padding: 1;
        background: $surface-darken-1;
    }
    
    #search_input {
        width: 100%;
    }
    
    #results_container {
        height: 1fr;
        width: 100%;
    }
    
    DataTable {
        height: 1fr;
    }
    
    #status_bar {
        height: 3;
        width: 100%;
        background: $surface-darken-1;
        padding: 1;
        text-align: center;
    }
    """
    
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("enter", "select", "Select"),
        Binding("r", "record", "Record"),
        Binding("/", "focus_search", "Search"),
        Binding("f", "toggle_favorite", "Favorite"),
    ]
    
    def compose(self) -> ComposeResult:
        """Create child widgets"""
        yield Header()
        yield Container(
            Static("🔍 Search Channels", id="search_label"),
            Input(
                placeholder="Type to search (name, genre, description)...",
                id="search_input"
            ),
            id="search_container"
        )
        yield Container(
            DataTable(id="channel_table"),
            id="results_container"
        )
        yield Static("↑↓: Navigate | R: Record | F: Favorite | /: Search | Esc: Back", id="status_bar")
        yield Footer()
    
    async def on_mount(self) -> None:
        """Load channels on mount"""
        # Load channels
        await self.load_channels()
        
        # Setup table
        table = self.query_one("#channel_table", DataTable)
        table.cursor_type = "row"
        table.add_column("⭐", width=3)
        table.add_column("Channel", width=25)
        table.add_column("Ch #", width=8)
        table.add_column("Description", width=50)
        
        # Display all channels initially
        await self.update_results(self.channels)
        
        # Focus table so keyboard shortcuts work!
        table.focus()
    
    async def load_channels(self):
        """Load channel list"""
        try:
            self.channels = self.channel_manager.list_channels()
            self.search_engine = ChannelSearch(self.channels)
            self.notify(f"Loaded {len(self.channels)} channels", severity="information")
        except Exception as e:
            self.notify(f"Error loading channels: {e}", severity="error")
            self.channels = []
    
    async def update_results(self, channels):
        """Update table with search results"""
        table = self.query_one("#channel_table", DataTable)
        table.clear()
        
        favorites = self.config.get('favorites', [])
        
        # Sort: favorites first, then by name
        sorted_channels = sorted(channels, key=lambda ch: (
            0 if ch.get('id') in favorites else 1,  # Favorites first
            ch.get('name', 'ZZZ')  # Then alphabetically
        ))
        
        for channel in sorted_channels:
            is_fav = channel.get('id') in favorites
            fav_icon = "⭐" if is_fav else ""
            
            table.add_row(
                fav_icon,
                channel.get('name', 'Unknown'),
                str(channel.get('number', '')),
                channel.get('description', '')[:50],
                key=channel.get('id')
            )
    
    async def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes"""
        if event.input.id == "search_input":
            query = event.value.strip()
            
            if self.search_engine:
                results = self.search_engine.search(query)
                await self.update_results(results)
                
                # Update status
                status = self.query_one("#status_bar", Static)
                if query:
                    status.update(f"Found {len(results)} channels | Tab: Navigate results | Esc: Back to table")
                else:
                    status.update("↑↓: Navigate | R: Record | F: Favorite | /: Search | Esc: Back")
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """When user presses Enter in search, focus table"""
        if event.input.id == "search_input":
            table = self.query_one("#channel_table", DataTable)
            table.focus()
    
    async def action_back(self):
        """Go back to main menu"""
        self.app.pop_screen()
    
    async def action_select(self):
        """Select highlighted channel"""
        table = self.query_one("#channel_table", DataTable)
        
        if table.row_count > 0 and table.cursor_coordinate is not None:
            try:
                # Get row key from cursor coordinate
                row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
                channel_id = row_key.value if hasattr(row_key, 'value') else str(row_key)
                
                # Find full channel data
                for channel in self.channels:
                    if channel.get('id') == channel_id:
                        self.selected_channel = channel
                        self.notify(f"Selected: {channel.get('name')}", severity="success")
                        break
            except Exception as e:
                self.notify(f"Error selecting channel: {e}", severity="error")
    
    async def action_focus_search(self):
        """Focus the search input"""
        self.query_one("#search_input", Input).focus()
    
    async def action_record(self):
        """Open recording screen or DVR browser with selected channel"""
        try:
            table = self.query_one("#channel_table", DataTable)
            
            if table.row_count > 0 and table.cursor_coordinate is not None:
                # Get row key from cursor coordinate
                row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
                channel_id = row_key.value if hasattr(row_key, 'value') else str(row_key)
                
                # Find full channel data
                for channel in self.channels:
                    if channel.get('id') == channel_id:
                        # Check if in DVR mode (use getattr for safety)
                        if getattr(self, 'dvr_mode', False):
                            # DVR Mode - Open DVR browser
                            from sxm.ui.dvr_browser import DVRBrowser
                            self.notify(f"Opening DVR for {channel.get('name')}", severity="information")
                            self.app.push_screen(DVRBrowser(channel=channel))
                        else:
                            # Regular Mode - Open recording screen
                            from sxm.ui.recording_screen import RecordingScreen
                            self.notify(f"Opening recording for {channel.get('name')}", severity="information")
                            # 🎯 REAL RECORDING ENABLED!
                            self.app.push_screen(RecordingScreen(channel=channel, use_real_recorder=True))
                        return
                
                self.notify(f"Channel not found: {channel_id}", severity="error")
        except Exception as e:
            self.notify(f"Error opening: {e}", severity="error")
    
    async def action_toggle_favorite(self):
        """Toggle favorite status of current channel"""
        try:
            table = self.query_one("#channel_table", DataTable)
            
            if table.row_count > 0 and table.cursor_coordinate is not None:
                # Get row key from cursor coordinate
                row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
                channel_id = row_key.value if hasattr(row_key, 'value') else str(row_key)
                
                # Find channel name for better message
                channel_name = "Channel"
                for channel in self.channels:
                    if channel.get('id') == channel_id:
                        channel_name = channel.get('name')
                        break
                
                # Toggle favorite
                if self.config.is_favorite(channel_id):
                    self.config.remove_favorite(channel_id)
                    self.notify(f"⭐ Removed {channel_name} from favorites", severity="information")
                else:
                    self.config.add_favorite(channel_id)
                    self.notify(f"⭐ Added {channel_name} to favorites!", severity="success")
                
                # Refresh display
                query = self.query_one("#search_input", Input).value
                if self.search_engine:
                    results = self.search_engine.search(query) if query else self.channels
                    await self.update_results(results)
        except Exception as e:
            self.notify(f"Error toggling favorite: {e}", severity="error")
