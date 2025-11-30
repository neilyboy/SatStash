#!/usr/bin/env python3
"""
Settings Screen

Configure application preferences
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Button, Input, Label, Select
from textual.containers import Container, Vertical, Horizontal
from textual.binding import Binding
from pathlib import Path
from sxm.utils.config import Config


class SettingsScreen(Screen):
    """Configure application settings"""
    
    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back", priority=True),
        Binding("s", "save", "Save", priority=True),
    ]
    
    CSS = """
    SettingsScreen {
        layout: vertical;
        background: $surface;
    }
    
    #title_panel {
        height: 5;
        border: solid $primary;
        padding: 1;
        margin: 1;
    }
    
    #settings_container {
        height: 1fr;
        padding: 2;
        overflow-y: auto;
    }
    
    .setting_row {
        height: auto;
        padding: 1;
        margin-bottom: 1;
    }
    
    .setting_label {
        width: 30;
        padding-right: 2;
    }
    
    .setting_input {
        width: 1fr;
    }
    
    #button_row {
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
        self.config = Config()
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(id="title_panel"):
            yield Label("⚙️  Settings")
            yield Label("Configure application preferences")
        
        with Vertical(id="settings_container"):
            # Output Directory
            with Horizontal(classes="setting_row"):
                yield Label("📁 Output Directory:", classes="setting_label")
                yield Input(
                    value=str(self.config.get('output_directory', '~/Music/SiriusXM')),
                    id="output_dir",
                    classes="setting_input"
                )
            
            # Audio Quality
            with Horizontal(classes="setting_row"):
                yield Label("🎵 Audio Quality:", classes="setting_label")
                yield Select(
                    [(line, line) for line in ["256k", "128k", "64k", "32k"]],
                    value=self.config.get('audio_quality', '256k'),
                    id="audio_quality",
                    classes="setting_input"
                )
            
            # Output Format
            with Horizontal(classes="setting_row"):
                yield Label("📦 Output Format:", classes="setting_label")
                yield Select(
                    [(line, line) for line in ["m4a", "mp3"]],
                    value=self.config.get('output_format', 'm4a'),
                    id="output_format",
                    classes="setting_input"
                )
            
            # Download Cover Art
            with Horizontal(classes="setting_row"):
                yield Label("🎨 Download Cover Art:", classes="setting_label")
                yield Select(
                    [("Yes", "true"), ("No", "false")],
                    value="true" if self.config.get('download_cover_art', True) else "false",
                    id="download_cover_art",
                    classes="setting_input"
                )
            
            # Tag Metadata
            with Horizontal(classes="setting_row"):
                yield Label("🏷️  Tag Metadata:", classes="setting_label")
                yield Select(
                    [("Yes", "true"), ("No", "false")],
                    value="true" if self.config.get('tag_metadata', True) else "false",
                    id="tag_metadata",
                    classes="setting_input"
                )
        
        with Horizontal(id="button_row"):
            yield Button("💾 Save Settings", id="btn_save", variant="primary")
            yield Button("🔄 Reset to Defaults", id="btn_reset", variant="warning")
            yield Button("🔙 Back", id="btn_back")
        
        with Container(id="status"):
            yield Label("", id="status_text")
        
        yield Footer()
    
    def action_save(self) -> None:
        """Save settings"""
        self.save_settings()
    
    def save_settings(self) -> None:
        """Save current settings to config"""
        try:
            # Get values from inputs
            output_dir = self.query_one("#output_dir", Input).value
            audio_quality = self.query_one("#audio_quality", Select).value
            output_format = self.query_one("#output_format", Select).value
            download_cover_art = self.query_one("#download_cover_art", Select).value == "true"
            tag_metadata = self.query_one("#tag_metadata", Select).value == "true"
            
            # Validate output directory
            output_path = Path(output_dir).expanduser()
            if not output_path.exists():
                try:
                    output_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    self.update_status(f"❌ Invalid output directory: {e}")
                    return
            
            # Update config
            self.config.set('output_directory', str(output_path))
            self.config.set('audio_quality', audio_quality)
            self.config.set('output_format', output_format)
            self.config.set('download_cover_art', download_cover_art)
            self.config.set('tag_metadata', tag_metadata)
            
            self.config.save()
            
            self.update_status("✅ Settings saved successfully!")
            self.app.notify("Settings saved!", severity="success")
            
        except Exception as e:
            self.update_status(f"❌ Error saving settings: {e}")
            self.app.notify(f"Error: {e}", severity="error")
    
    def reset_settings(self) -> None:
        """Reset settings to defaults"""
        try:
            # Reset to defaults
            defaults = {
                'output_directory': '~/Music/SiriusXM',
                'audio_quality': '256k',
                'output_format': 'm4a',
                'download_cover_art': True,
                'tag_metadata': True
            }
            
            for key, value in defaults.items():
                self.config.set(key, value)
            
            self.config.save()
            
            # Update UI
            self.query_one("#output_dir", Input).value = defaults['output_directory']
            self.query_one("#audio_quality", Select).value = defaults['audio_quality']
            self.query_one("#output_format", Select).value = defaults['output_format']
            self.query_one("#download_cover_art", Select).value = "true"
            self.query_one("#tag_metadata", Select).value = "true"
            
            self.update_status("✅ Settings reset to defaults!")
            self.app.notify("Settings reset!", severity="success")
            
        except Exception as e:
            self.update_status(f"❌ Error resetting settings: {e}")
            self.app.notify(f"Error: {e}", severity="error")
    
    def update_status(self, message: str) -> None:
        """Update status message"""
        status_label = self.query_one("#status_text", Label)
        status_label.update(message)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id
        
        if button_id == "btn_save":
            self.save_settings()
        elif button_id == "btn_reset":
            self.reset_settings()
        elif button_id == "btn_back":
            self.app.pop_screen()
