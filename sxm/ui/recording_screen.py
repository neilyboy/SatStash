"""Recording screen with real-time progress"""
import sys
import asyncio
import threading
from pathlib import Path
from datetime import datetime

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, Button, Static, ProgressBar, Input
from textual.containers import Container, Horizontal, Vertical
from textual.binding import Binding
from textual import work

from sxm.utils.config import Config
from sxm.utils.session_manager import SessionManager
from sxm.utils.recorder_integration import RecorderIntegration


class RecordingScreen(Screen):
    """Live recording screen with progress"""
    
    CSS = """
    RecordingScreen {
        background: $surface;
    }
    
    #recording_header {
        height: 3;
        background: $error;
        color: $text;
        text-align: center;
    }
    
    #main_content {
        padding: 1 2;
    }
    
    #duration_input {
        width: 20;
    }
    
    #time_unit_selector {
        width: auto;
    }
    
    Button {
        margin: 0 1;
    }
    """
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("s", "start", "Start"),
    ]
    
    def __init__(self, channel=None, use_real_recorder=False):
        super().__init__()
        self.channel = channel
        self.config = Config()
        self.session_mgr = SessionManager()
        self.recorder_integration = RecorderIntegration(self.config)
        self.recording = False
        self.tracks_collected = []
        self.use_real_recorder = use_real_recorder
        self.recording_thread = None
        self.time_unit = "minutes"  # Default to minutes
    
    def compose(self) -> ComposeResult:
        """Create child widgets"""
        yield Header()
        
        channel_name = self.channel.get('name', 'Unknown') if self.channel else 'Select Channel'
        channel_num = f"Ch {self.channel.get('number', '')}" if self.channel else ''
        
        yield Static(
            f"🔴 RECORDING - {channel_name} {channel_num}",
            id="recording_header"
        )
        
        yield Container(
            Static("Duration:"),
            Horizontal(
                Input(value="5", placeholder="5", id="duration_input"),
                Button("Minutes", id="unit_minutes", variant="primary"),
                Button("Hours", id="unit_hours"),
                id="time_unit_selector"
            ),
            Static("Quality:"),
            Horizontal(
                Button("32k", id="quality_32k", classes="quality-btn"),
                Button("64k", id="quality_64k", classes="quality-btn"),
                Button("128k", id="quality_128k", classes="quality-btn"),
                Button("256k", id="quality_256k", variant="primary", classes="quality-btn"),
            ),
            Static(""),
            Static("Ready to record...", id="status_text"),
            ProgressBar(total=100, show_eta=False, id="progress_bar"),
            Static("", id="time_elapsed"),
            Static(""),
            Static("🎵 Current Track:", id="current_track_label"),
            Static("Waiting to start...", id="current_track"),
            Static(""),
            Static("📝 Tracks Recorded:", id="tracks_label"),
            Static("None yet", id="tracks_list"),
            Static(""),
            Horizontal(
                Button("▶ Start Recording", id="start_btn", variant="success"),
                Button("⏹ Stop", id="stop_btn", variant="error", disabled=True),
                Button("🚪 Cancel", id="cancel_btn"),
            ),
            id="main_content"
        )
        
        yield Footer()
    
    async def on_mount(self) -> None:
        """Setup on mount"""
        if not self.channel:
            self.notify("No channel selected!", severity="error")
            await asyncio.sleep(2)
            self.app.pop_screen()
        else:
            # Focus the duration input and select all text
            duration_input = self.query_one("#duration_input", Input)
            duration_input.focus()
            # Select all so user can just type to replace
            duration_input.action_select_all()
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id
        
        if button_id == "start_btn":
            await self.start_recording()
        elif button_id == "stop_btn":
            await self.stop_recording()
        elif button_id == "cancel_btn":
            if self.recording:
                self.notify("Stop recording first!", severity="warning")
            else:
                self.app.pop_screen()
        elif button_id == "unit_minutes":
            self.time_unit = "minutes"
            self.query_one("#unit_minutes", Button).variant = "primary"
            self.query_one("#unit_hours", Button).variant = "default"
        elif button_id == "unit_hours":
            self.time_unit = "hours"
            self.query_one("#unit_hours", Button).variant = "primary"
            self.query_one("#unit_minutes", Button).variant = "default"
        elif button_id.startswith("quality_"):
            # Update selected quality
            quality = button_id.replace("quality_", "")
            self.config.set('audio_quality', quality)
            self.notify(f"Quality set to {quality}", severity="information")
            # Update button variants
            for btn in self.query(".quality-btn"):
                if btn.id == button_id:
                    btn.variant = "primary"
                else:
                    btn.variant = "default"
    
    async def start_recording(self):
        """Start the recording"""
        if self.recording:
            self.notify("Already recording!", severity="warning")
            return
        
        # Get duration
        duration_input = self.query_one("#duration_input", Input)
        try:
            duration_value = int(duration_input.value)
            if duration_value < 1:
                self.notify("Duration must be at least 1", severity="error")
                return
            
            # Convert to minutes if hours selected
            if self.time_unit == "hours":
                duration = duration_value * 60
                if duration > 480:  # 8 hours max
                    self.notify("Maximum duration is 8 hours", severity="error")
                    return
            else:
                duration = duration_value
                if duration > 480:
                    self.notify("Maximum duration is 480 minutes (8 hours)", severity="error")
                    return
        except ValueError:
            self.notify("Invalid duration", severity="error")
            return
        
        # Disable start, enable stop
        self.query_one("#start_btn", Button).disabled = True
        self.query_one("#stop_btn", Button).disabled = False
        
        self.recording = True
        self.notify(f"Starting {duration} minute recording...", severity="success")
        
        # Update status
        self.query_one("#status_text", Static).update(
            f"🔴 Recording {self.channel['name']} for {duration} minutes..."
        )
        
        # Start recording in background task (non-blocking)
        asyncio.create_task(self.run_recording(duration))
    
    async def run_recording(self, duration):
        """Run the actual recording (background task)"""
        try:
            if self.use_real_recorder:
                # Use real recorder in background thread (status shown in run_real_recording)
                await self.run_real_recording(duration)
            else:
                self.notify("WARNING: Using placeholder mode!", severity="warning")
                # Use simulation for testing
                await self.run_simulated_recording(duration)
        except Exception as e:
            self.notify(f"Recording error: {e}", severity="error")
        finally:
            self.recording = False
            # Safely re-enable buttons (screen might be closed)
            try:
                self.query_one("#start_btn", Button).disabled = False
                self.query_one("#stop_btn", Button).disabled = True
            except:
                pass  # Screen was closed, ignore
    
    async def run_real_recording(self, duration):
        """Run the real recorder in a background thread"""
        # Show initial status (don't use notify to avoid duplicates)
        status_text = self.query_one("#status_text", Static)
        status_text.update("⏳ Initializing recorder...")
        
        recording_error = None
        recorded_tracks = []
        
        def record_thread():
            nonlocal recording_error, recorded_tracks
            try:
                
                # Define callbacks for real-time updates
                def track_update(track_info):
                    """Called by recorder to show current track or tracks list"""
                    # Update current track OR tracks list display in main thread
                    def update_display():
                        try:
                            # If track_info starts with number+period, it's a tracks list
                            if track_info and (track_info.strip().startswith('1.') or track_info.strip().startswith('1 ')):
                                # This is a tracks list - update tracks_list widget
                                tracks_widget = self.query_one("#tracks_list", Static)
                                tracks_widget.update(f"📋 Planned:\n{track_info}")
                            else:
                                # This is current track info - update current_track widget
                                current_track_widget = self.query_one("#current_track", Static)
                                if track_info:
                                    current_track_widget.update(track_info)
                                else:
                                    current_track_widget.update("Waiting to start...")
                        except Exception as e:
                            print(f"⚠️  track_update error: {e}")
                    
                    try:
                        self.app.call_from_thread(update_display)
                    except:
                        pass  # App might be closing, ignore
                
                def progress_update(msg):
                    """Called by recorder to show progress"""
                    # Update status text in main thread
                    def update_status():
                        try:
                            status_widget = self.query_one("#status_text", Static)
                            status_widget.update(msg)
                        except:
                            pass
                    
                    try:
                        self.app.call_from_thread(update_status)
                    except:
                        pass  # App might be closing, ignore
                
                tracks = self.recorder_integration.record_channel(
                    self.channel,
                    duration_minutes=duration,
                    quality='256k',
                    progress_callback=progress_update,
                    track_callback=track_update
                )
                print(f"\n{'='*60}")
                print(f"🔍 DEBUG recording_screen THREAD: Received tracks from recorder_integration")
                print(f"🔍 DEBUG recording_screen THREAD: tracks type = {type(tracks)}")
                print(f"🔍 DEBUG recording_screen THREAD: tracks value = {tracks}")
                print(f"🔍 DEBUG recording_screen THREAD: tracks length = {len(tracks) if tracks else 0}")
                print(f"🔍 DEBUG recording_screen THREAD: tracks is truthy? {bool(tracks)}")
                recorded_tracks = tracks or []
                print(f"🔍 DEBUG recording_screen THREAD: recorded_tracks = {recorded_tracks}")
                print(f"🔍 DEBUG recording_screen THREAD: recorded_tracks is truthy? {bool(recorded_tracks)}")
                print(f"🔍 DEBUG recording_screen THREAD: About to exit thread with recorded_tracks={recorded_tracks}")
                print(f"{'='*60}\n")
                msg = f" Recording complete! Got {len(recorded_tracks)} tracks"
                print(msg)
                
                # Show in UI
                try:
                    self.app.call_from_thread(
                        self.notify,
                        f"Recording done: {len(recorded_tracks)} tracks",
                        severity="success" if recorded_tracks else "warning"
                    )
                except:
                    pass  # App might be closing
            except Exception as e:
                msg = f" Recording error: {e}"
                print(msg)
                import traceback
                traceback.print_exc()
                recording_error = str(e)
                
                # Show error in UI
                try:
                    self.app.call_from_thread(
                        self.notify,
                        f"Recording error: {e}",
                        severity="error"
                    )
                except:
                    pass  # App might be closing
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=record_thread, daemon=True)
        self.recording_thread.start()
        
        # Show progress (estimate based on duration)
        progress_bar = self.query_one("#progress_bar", ProgressBar)
        status_text = self.query_one("#status_text", Static)
        total_seconds = duration * 60
        
        for elapsed in range(total_seconds + 1):
            if not self.recording:
                break
            
            percent = int((elapsed / total_seconds) * 100)
            progress_bar.update(progress=percent)
            status_text.update(f"🔴 Recording... {elapsed}s / {total_seconds}s")
            
            await asyncio.sleep(1)
        
        # Wait for thread to finish (longer timeout for splitting/tagging)
        status_text.update("⏳ Finalizing recording...")
        self.query_one("#tracks_list", Static).update("⏳ Processing tracks...")
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=120)  # 2 minutes for splitting/tagging
        
        # Reset recording state
        self.recording = False
        
        print(f"\n{'='*60}")
        print(f"🔍 DEBUG async part: Thread finished")
        print(f"🔍 DEBUG async part: recording_error = {recording_error}")
        print(f"🔍 DEBUG async part: recorded_tracks type = {type(recorded_tracks)}")
        print(f"🔍 DEBUG async part: recorded_tracks value = {recorded_tracks}")
        print(f"🔍 DEBUG async part: recorded_tracks length = {len(recorded_tracks) if recorded_tracks else 0}")
        print(f"🔍 DEBUG async part: recorded_tracks is truthy? {bool(recorded_tracks)}")
        print(f"🔍 DEBUG async part: recorded_tracks == []? {recorded_tracks == []}")
        print(f"{'='*60}\n")
        
        # Show results
        if recording_error:
            print(f"🔍 DEBUG: Taking recording_error path")
            self.notify(f"Recording failed: {recording_error}", severity="error")
            status_text.update(f"❌ Recording failed")
            await asyncio.sleep(3)
            self.app.pop_screen()
        elif recorded_tracks is not None and len(recorded_tracks) > 0:
            print(f"🔍 DEBUG: Taking recorded_tracks path (success)")
            self.tracks_collected = recorded_tracks
            # Update tracks display
            tracks_text = ""
            for track in recorded_tracks:
                artist = track.get('artist', 'Unknown')
                title = track.get('title', 'Unknown')
                tracks_text += f"  ✅ {artist} - {title}\n"
            self.query_one("#tracks_list", Static).update(tracks_text)
            self.notify(f"Recording complete! Got {len(recorded_tracks)} tracks!", severity="success")
            status_text.update(f"✅ Complete! {len(recorded_tracks)} tracks recorded")
            await asyncio.sleep(3)
            self.app.pop_screen()
        else:
            print(f"🔍 DEBUG: Taking else path (no tracks)")
            self.notify("Recording complete but no tracks returned", severity="warning")
            status_text.update("⚠️  Complete but no tracks found")
            await asyncio.sleep(3)
            self.app.pop_screen()
    
    async def run_simulated_recording(self, duration):
        """Run simulated recording for testing"""
        progress_bar = self.query_one("#progress_bar", ProgressBar)
        status_text = self.query_one("#status_text", Static)
        
        total_seconds = duration * 60
        for i in range(total_seconds):
            if not self.recording:
                break
            
            # Update progress
            progress = (i / total_seconds) * 100
            progress_bar.update(progress=progress)
            
            elapsed_min = i // 60
            elapsed_sec = i % 60
            status_text.update(
                f"🔴 Recording... {elapsed_min:02d}:{elapsed_sec:02d} / {duration:02d}:00"
            )
            
            # Simulate track updates every 30 seconds
            if i > 0 and i % 30 == 0:
                await self.add_fake_track(i)
            
            await asyncio.sleep(1)
        
        # Complete
        if self.recording:
            self.notify("Recording complete!", severity="success")
            status_text.update("✅ Recording complete!")
            await asyncio.sleep(2)
            self.app.pop_screen()
    
    async def add_fake_track(self, elapsed):
        """Add a simulated track (for testing)"""
        track_num = len(self.tracks_collected) + 1
        
        fake_track = f"Track {track_num} - Artist {track_num} - Song {track_num}"
        self.tracks_collected.append(fake_track)
        
        # Build full track list
        tracks_text = ""
        for track in self.tracks_collected:
            tracks_text += f"  ✅ {track}\n"
        
        self.query_one("#tracks_list", Static).update(tracks_text)
        
        # Update current track
        self.update_current_track(fake_track)
    
    async def run_simulated_recording(self, duration):
        """Run simulated recording for testing"""
        progress_bar = self.query_one("#progress_bar", ProgressBar)
        status_text = self.query_one("#status_text", Static)
        
        total_seconds = duration * 60
        for i in range(total_seconds):
            if not self.recording:
                break
            
            # Update progress
            progress = (i / total_seconds) * 100
            progress_bar.update(progress=progress)
            
            elapsed_min = i // 60
            elapsed_sec = i % 60
            status_text.update(
                f"🔴 Recording... {elapsed_min:02d}:{elapsed_sec:02d} / {duration:02d}:00"
            )
            
            # Simulate track updates every 30 seconds
            if i > 0 and i % 30 == 0:
                await self.add_fake_track(i)
            
            await asyncio.sleep(1)
        
        # Complete
        if self.recording:
            self.notify("Recording complete!", severity="success")
            status_text.update("✅ Recording complete!")
            await asyncio.sleep(2)
            self.app.pop_screen()
    
    def update_current_track(self, track_info: str):
        """Update current track display"""
        try:
            current_track_widget = self.query_one("#current_track", Static)
            if track_info:
                current_track_widget.update(track_info)  # Don't duplicate the label!
            else:
                current_track_widget.update("Waiting to start...")
        except Exception as e:
            # Widget might not exist yet, that's okay
            pass
    
    async def stop_recording(self):
        """Stop recording early"""
        if not self.recording:
            return
        
        self.recording = False
        self.notify("Stopping recording...", severity="information")
    
    async def action_cancel(self):
        """Cancel and go back"""
        if self.recording:
            self.notify("Stop recording first!", severity="warning")
        else:
            self.app.pop_screen()
    
    async def action_start(self):
        """Keyboard shortcut to start"""
        await self.start_recording()
