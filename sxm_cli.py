#!/usr/bin/env python3
"""
SatStash - Interactive CLI
Simple, powerful, working interface for recording and DVR
"""

import sys
import os
import subprocess
import shutil
import signal
import threading
import time
import io
import re
import tempfile
import select
import termios
import tty
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List
from http.server import HTTPServer, BaseHTTPRequestHandler
from rich.live import Live
from rich.console import Console
from rich.panel import Panel

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from sxm.utils.recorder_integration import RecorderIntegration
from sxm.utils.config import Config
from sxm.utils.session_manager import SessionManager
from sxm.utils.channel_manager import ChannelManager
from sxm.core.api import SiriusXMAPI
from sxm.core.browser_session import BrowserSession
from sxm.core.hls_downloader import HLSDownloader

try:
    from PIL import Image
except ImportError:
    Image = None

import requests


class SXMCli:
    """Interactive CLI for SatStash"""
    QUALITY_LEVELS = ['32k', '64k', '128k', '256k']
    STREAM_COOKIE_WHITELIST = {
        'AUTH_TOKEN',
        'SXMAUTHORIZATION',
        'DEVICE_GRANT',
        'sxm-refresh-token',
        'sxm_geo',
        'SESSION',
        'connect.sid',
        'AWSALB',
        'AWSALBTG',
        'AWSALBTGCORS'
    }

    def __init__(self):
        self.config = Config()
        self.session_mgr = SessionManager()
        self.bearer_token = None
        self.channel_manager = None
        self.api = None
        self.recorder_integration = None
        self.console = Console()
        initial_cookies = self._select_stream_cookies(
            self.session_mgr.get_current_session_cookies()
        )
        self.hls_downloader = HLSDownloader(
            bearer_token=None,
            session_cookies=initial_cookies
        )
        self._player_process = None
        self._player_control_mode = None
        self._metadata_thread = None
        self._metadata_stop = None
        self._status_thread = None
        self._status_stop = None
        self._player_paused = False
        self._listen_history: List[Dict] = []
        self._listen_start_ts = None
        self._listen_quality = None
        self._listen_channel = None
        self._channel_art_cache: Dict[str, str] = {}
        self._live_display: Live | None = None
        self._listen_panel_state: Dict[str, any] = {}
        self._listen_panel_lock = threading.Lock()
        self._listen_proxy_server: HTTPServer | None = None
        self._listen_proxy_thread: threading.Thread | None = None
        self._player_log_fp = None
        self._history_select_mode = False
        self._history_page = 0
        self._history_input = ""
        
    def _refresh_bearer_token(self, channel: Dict) -> bool:
        """Refresh bearer token via browser session when API reports 401."""
        try:
            cookies = self.session_mgr.get_current_session_cookies()
            if not cookies:
                print("⚠️  Cannot refresh authentication: no valid session cookies")
                return False
            browser = BrowserSession(cookies=cookies, headless=True)
            channel_url = f"https://www.siriusxm.com/player/channel-linear/entity/{channel['id']}"
            print("🔄 Session expired – refreshing authentication...")
            bearer_token, _ = browser.get_stream_info(channel_url, channel, art_dir=None)
            if not bearer_token:
                print("❌ Failed to capture new authentication token")
                return False
            self.bearer_token = bearer_token
            self.api.bearer_token = bearer_token
            self.api.headers['Authorization'] = f'Bearer {bearer_token}'
            if self.hls_downloader:
                self.hls_downloader.bearer_token = bearer_token
            print("✅ Authentication refreshed")
            return True
        except Exception as exc:
            print(f"❌ Auth refresh error: {exc}")
            return False

    def initialize(self):
        """Initialize connections"""
        print("\n🔧 Initializing SatStash...")
        
        # Get session
        self.bearer_token = self.session_mgr.get_bearer_token()
        
        if not self.bearer_token:
            print("\n🔑 No active session found. Let's log in!")
            
            # Get credentials
            username = input("\n📧 Email: ").strip()
            password = input("🔒 Password: ").strip()
            
            if not username or not password:
                print("❌ Email and password required")
                return False
            
            print("\n🌐 Logging in via browser...")
            print("(This will open a browser window briefly - 10-15 seconds)")
            
            try:
                # Use SiriusXMAuth for login
                from sxm.core.auth import SiriusXMAuth
                
                auth = SiriusXMAuth(headless=True, debug=False)
                cookies = auth.login(username, password)
                
                if not cookies:
                    print("❌ Login failed - check your credentials")
                    return False
                
                # Save session
                self.session_mgr.save_session(cookies, username)
                
                # Get bearer token
                self.bearer_token = self.session_mgr.get_bearer_token()
                
                if not self.bearer_token:
                    print("❌ Failed to get bearer token from session")
                    return False
                
                print("✅ Logged in successfully!")
                
            except Exception as e:
                print(f"❌ Login error: {e}")
                print("\n💡 Make sure Playwright is installed:")
                print("   source venv/bin/activate")
                print("   playwright install chromium")
                import traceback
                traceback.print_exc()
                return False
        
        # Initialize managers
        print("📡 Loading channel list...")
        self.channel_manager = ChannelManager(bearer_token=self.bearer_token)
        
        # Force refresh if channels seem stale or missing
        channels = self.channel_manager.list_channels()
        if len(channels) < 500:  # Should have 700+
            print(f"⚠️  Channel list seems incomplete ({len(channels)} channels), refreshing...")
            self.channel_manager.refresh_channels()
        
        self.api = SiriusXMAPI(self.bearer_token)
        self.hls_downloader = HLSDownloader(
            bearer_token=self.bearer_token,
            session_cookies=self._select_stream_cookies(
                self.session_mgr.get_current_session_cookies()
            )
        )
        self.recorder_integration = RecorderIntegration(self.config)
        
        print("✅ Ready!\n")
        return True
    
    def show_main_menu(self):
        """Display main menu"""
        print("\n" + "="*60)
        print("📻 SatStash - Main Menu")
        print("="*60)
        print("\n1. 🔴 Record Live Now")
        print("2. ▶️  Listen Live (Terminal Player)")
        print("3. 📺 Browse DVR History")
        print("4. ⏰ Schedule Recording")
        print("5. 📋 View Scheduled Recordings")
        print("6. 📻 Browse Channels")
        print("7. ⚙️  Settings")
        print("8. 🚪 Exit")
        print()
        
    def select_channel(self, prompt="Select a channel"):
        """Interactive channel selector"""
        print(f"\n{prompt}")
        print("-" * 60)
        
        # Get channels
        channels = self.channel_manager.list_channels()
        
        # Show search prompt
        print("\nType channel name or number to search (or 'list' to see all):")
        query = input("> ").strip()
        
        if query.lower() == 'list':
            # Show all channels paginated
            return self._browse_all_channels(channels)
        
        # Search for channel
        matches = []
        query_lower = query.lower()
        
        for ch in channels:
            if query_lower in ch['name'].lower() or query == str(ch.get('number', '')):
                matches.append(ch)
        
        if not matches:
            print(f"❌ No channels found matching '{query}'")
            return None
        
        if len(matches) == 1:
            channel = matches[0]
            print(f"\n✅ Selected: {channel['name']} (Ch {channel.get('number', '?')})")
            return channel
        
        # Multiple matches - let user choose
        print(f"\nFound {len(matches)} channels:")
        for i, ch in enumerate(matches[:10], 1):
            print(f"  {i}. {ch['name']} (Ch {ch.get('number', '?')}) - {ch.get('genre', 'Music')}")
        
        if len(matches) > 10:
            print(f"  ... and {len(matches) - 10} more")

        choice = input("\nSelect number (1-10) or press Enter to cancel: ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= min(10, len(matches)):
            channel = matches[int(choice) - 1]
            print(f"\n✅ Selected: {channel['name']}")
            return channel

        return None

    def _select_quality(self, allow_default: bool = True) -> str:
        """Prompt for audio quality, defaulting to config setting"""
        default_quality = self.config.get('audio_quality', '256k')
        print("\nSelect audio quality:")
        for idx, level in enumerate(self.QUALITY_LEVELS, 1):
            marker = " (default)" if level == default_quality else ""
            print(f"  {idx}. {level}{marker}")

        prompt = f"Choice (1-{len(self.QUALITY_LEVELS)})"
        if allow_default:
            prompt += f" [Enter = {default_quality}]"
        prompt += ": "

        choice = input(prompt).strip()
        if allow_default and choice == '':
            return default_quality

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(self.QUALITY_LEVELS):
                return self.QUALITY_LEVELS[idx]
        elif choice in self.QUALITY_LEVELS:
            return choice

        print(f"⚠️  Invalid selection, using {default_quality}")
        return default_quality

    def record_live_now(self):
        """Record live from a channel"""
        print("\n" + "="*60)
        print("🔴 Record Live Now")
        print("="*60)
        
        # Select channel
        channel = self.select_channel("Which channel to record?")
        if not channel:
            return
        
        # Get duration
        print("\nHow long to record?")
        print("  1. 5 minutes")
        print("  2. 10 minutes")
        print("  3. 15 minutes")
        print("  4. 30 minutes")
        print("  5. 1 hour")
        print("  6. 2 hours")
        print("  7. Custom")
        
        choice = input("\nSelect (1-7): ").strip()
        
        duration_map = {'1': 5, '2': 10, '3': 15, '4': 30, '5': 60, '6': 120}
        
        if choice in duration_map:
            duration = duration_map[choice]
        elif choice == '7':
            duration = int(input("Enter duration in minutes: "))
        else:
            print("❌ Invalid choice")
            return
        
        # Quality (default from settings, allow override)
        quality = self._select_quality()

        # Confirm and record
        print(f"\n{'='*60}")
        print(f"📻 Channel: {channel['name']} (Ch {channel.get('number', '?')})")
        print(f"⏱️  Duration: {duration} minutes")
        print(f"🎚️  Quality: {quality}")
        base_output = Path(self.config.get('output_directory', str(Path.home() / "Music" / "SiriusXM")))
        print(f"📁 Output: {base_output / channel['name']}/")
        print(f"{'='*60}")
        
        confirm = input("\nStart recording? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ Cancelled")
            return
        
        # Record
        print(f"\n🔴 STARTING RECORDING...")
        print("-" * 60 + "\n")
        
        def progress(msg):
            print(f"  {msg}")
        
        def track_update(msg):
            print(f"\n{msg}")
        
        tracks = self.recorder_integration.record_channel(
            channel,
            duration_minutes=duration,
            quality=quality,
            progress_callback=progress,
            track_callback=track_update
        )
        
        # Show results
        print(f"\n{'='*60}")
        print(f"✅ RECORDING COMPLETE!")
        print(f"{'='*60}\n")
        
        if tracks:
            print(f"📋 Recorded {len(tracks)} tracks:\n")
            for i, track in enumerate(tracks, 1):
                artist = track.get('artist', 'Unknown')
                title = track.get('title', 'Unknown')
                size_mb = track.get('file_size', 0) / (1024 * 1024)
                print(f"  {i}. {artist} - {title} ({size_mb:.1f} MB)")
        else:
            print("⚠️  Check output directory for files")
        
        print(f"\n📁 Location: ~/Music/SiriusXM/{channel['name']}/")
        input("\nPress Enter to continue...")
    
    def listen_live_terminal(self):
        """Stream a channel live inside the terminal"""
        print("\n" + "="*60)
        print("▶️  Listen Live - Terminal Player")
        print("="*60)
        channel = self.select_channel("Which channel do you want to listen to?")
        if not channel:
            return

        quality = self._select_quality()
        try:
            self._play_live_stream(channel, quality)
        except KeyboardInterrupt:
            print("\n🛑 Playback interrupted")
        except Exception as exc:
            print(f"\n❌ Player error: {exc}")
        finally:
            self._stop_player()
            input("\nPress Enter to return to the menu...")

    def _play_live_stream(self, channel: Dict, quality: str):
        """Launch ffplay/mpv for the requested channel"""
        cookies = self.session_mgr.get_current_session_cookies()
        if not cookies:
            print("❌ No valid session. Please log in via the main menu first.")
            return

        # Ensure we have a bearer token and API client
        if not self.bearer_token:
            self.bearer_token = self.session_mgr.get_bearer_token()
            if not self.bearer_token:
                print("❌ No authentication token. Please log in via the main menu first.")
                return

        if not self.api:
            self.api = SiriusXMAPI(self.bearer_token)
        else:
            self.api.bearer_token = self.bearer_token
            self.api.headers['Authorization'] = f'Bearer {self.bearer_token}'

        # Update downloader headers (Authorization + cookies)
        self.hls_downloader.bearer_token = self.bearer_token
        self.hls_downloader.update_session_cookies(
            self._select_stream_cookies(cookies)
        )

        # Get live HLS master playlist from pure API (same path DVR uses)
        print("\n🌐 Getting live stream URL (API)...")
        master_url = self.api.get_stream_url(channel['id'])
        if not master_url and self._refresh_bearer_token(channel):
            master_url = self.api.get_stream_url(channel['id'])

        if not master_url:
            print("❌ Failed to get live stream URL")
            return

        variant_url = self.hls_downloader.get_variant_url(master_url, quality)
        if not variant_url:
            print(f"❌ Could not find a {quality} variant for this channel")
            return
        print(f"🎚️  Using variant: {variant_url}")

        # Start a tiny local HTTP proxy that serves a patched playlist and raw AES key.
        proxy_url = self._start_listen_live_proxy(variant_url)
        if not proxy_url:
            print("❌ Failed to start local stream proxy for playback")
            return
        print(f"🔗 Local stream proxy: {proxy_url}")

        header_lines = self._build_stream_header_lines()
        player_cmd, control_mode = self._build_player_command(proxy_url, header_lines)
        if not player_cmd:
            print("❌ No supported player found. Install mpv or ffplay and try again.")
            return

        start_ts = time.time()
        self._listen_history = []
        self._listen_start_ts = start_ts
        self._listen_quality = quality
        self._listen_channel = channel
        self._player_paused = False
        self._history_page = 0
        self._history_select_mode = False
        self._history_input = ""
        # Always give the player a private stdin pipe so it does not
        # compete with the CLI for terminal input.
        stdin = subprocess.PIPE

        with Live("", refresh_per_second=4, console=self.console, screen=True) as live:
            self._live_display = live
            self._print_listen_live_header(channel, quality, control_mode)
            log_path = self._prepare_mpv_logfile()
            stdout_target = None
            stderr_target = None
            if log_path:
                try:
                    self._player_log_fp = open(log_path, "w", encoding="utf-8")
                    # Capture both stdout and stderr from the player
                    stdout_target = self._player_log_fp
                    stderr_target = self._player_log_fp
                except Exception:
                    self._player_log_fp = None
                    stdout_target = None
                    stderr_target = None

            self._player_process = subprocess.Popen(
                player_cmd,
                stdin=stdin,
                stdout=stdout_target,
                stderr=stderr_target,
            )
            self._player_control_mode = control_mode
            self._start_metadata_loop(channel, quality, start_ts)
            self._start_status_loop(channel, quality, start_ts)
            # Show a stable LIVE/quality status; the timer itself is handled
            # by the status worker via the Listening line.
            self._update_panel_state(status_line=f"LIVE │ {quality}")
            self._listen_for_key_commands()

        self._live_display = None
        self._stop_player()

    def _listen_for_key_commands(self):
        """Non-blocking single-key listener for Listen Live controls.

        Keys:
        - q / s : stop playback and return to menu
        - h     : show history
        - d     : download most recent track
        """
        # Use low-level select + cbreak mode so single keypresses are read
        # immediately (no Enter required) without breaking the Live layout.
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                if self._player_process is None or self._player_process.poll() is not None:
                    break

                # Wait up to 0.5s for a key; continue if none
                rlist, _, _ = select.select([fd], [], [], 0.5)
                if not rlist:
                    continue

                try:
                    ch = os.read(fd, 1)
                except Exception:
                    continue

                if not ch:
                    continue

                key = ch.decode(errors='ignore').lower()
                # q/s always take precedence (stop playback)
                if key in ('q', 's'):
                    # Graceful stop
                    if self._player_control_mode == 'mpv':
                        self._send_player_command('quit')
                    else:
                        try:
                            if self._player_process and self._player_process.poll() is None:
                                self._player_process.terminate()
                        except Exception:
                            pass
                    break
                elif key == 'h':
                    # Toggle history selection mode and print the current
                    # listen history so the user can pick a track.
                    if not self._history_select_mode:
                        # Refresh DVR-backed history so the user gets up to
                        # the last hour of tracks, not just those seen in
                        # this session.
                        self._load_dvr_history_for_channel(self._listen_channel, hours_back=1)
                        self._print_history_summary(show_details=True)
                        self._history_select_mode = True
                        self._history_input = ""
                        self._update_panel_state(
                            status_line="History: type track number, Enter to download, h to cancel",
                            controls=self._compute_listen_controls(self._player_control_mode),
                        )
                    else:
                        self._history_select_mode = False
                        self._history_input = ""
                        self._update_panel_state(
                            status_line="History selection cancelled",
                            controls=self._compute_listen_controls(self._player_control_mode),
                        )
                elif key == 'd':
                    # Download the most recent track in the listen history.
                    try:
                        if self._listen_history and self._listen_channel:
                            # Index 1 = most recent entry; run DVR work in a
                            # background thread so playback/UI stay responsive.
                            latest_track = self._listen_history[0]
                            latest_artist = latest_track.get('artist', 'Unknown')
                            latest_title = latest_track.get('title', 'Unknown')
                            threading.Thread(
                                target=self._download_history_track,
                                args=(1, self._listen_channel, self._listen_quality or '256k'),
                                daemon=True,
                            ).start()
                            self._update_panel_state(
                                status_line=f"Downloading: {latest_artist} - {latest_title}"
                            )
                    except Exception:
                        self._update_panel_state(status_line="Download failed; see logs")
                elif key == 'p' and self._history_select_mode:
                    # Older history page (further back in time)
                    if self._listen_history and len(self._listen_history) > 5:
                        page_size = 5
                        max_page = max(0, (len(self._listen_history) - 1) // page_size)
                        if self._history_page < max_page:
                            self._history_page += 1
                            self._refresh_history_page()
                            self._update_panel_state(
                                status_line=f"History page {self._history_page + 1}/{max_page + 1}",
                                controls=self._compute_listen_controls(self._player_control_mode),
                            )
                        else:
                            self._update_panel_state(status_line="No older tracks")
                elif key == 'n' and self._history_select_mode:
                    # Newer history page (towards current track)
                    if self._listen_history and self._history_page > 0 and len(self._listen_history) > 5:
                        page_size = 5
                        max_page = max(0, (len(self._listen_history) - 1) // page_size)
                        self._history_page -= 1
                        self._refresh_history_page()
                        self._update_panel_state(
                            status_line=f"History page {self._history_page + 1}/{max_page + 1}",
                            controls=self._compute_listen_controls(self._player_control_mode),
                        )
                    else:
                        self._update_panel_state(status_line="Already at newest tracks")
                elif self._history_select_mode and key.isdigit():
                    # Build up a multi-digit history index (e.g. 01, 12, 21).
                    if len(self._history_input) < 2:
                        self._history_input += key
                    else:
                        # If the buffer is already full, start a new number
                        # with the latest digit.
                        self._history_input = key
                    self._update_panel_state(
                        status_line=f"History: selected {self._history_input} (Enter to download, h to cancel)")
                elif self._history_select_mode and key in ('\n', '\r'):
                    # Enter confirms the current numeric selection, if any.
                    if not self._history_input:
                        self._history_select_mode = False
                        self._update_panel_state(
                            status_line="History selection cancelled",
                            controls=self._compute_listen_controls(self._player_control_mode),
                        )
                        continue

                    try:
                        index = int(self._history_input)
                        page_size = 5
                        base = self._history_page * page_size
                        max_index_on_page = min(page_size, max(0, len(self._listen_history) - base))
                        if not self._listen_channel or index < 1 or index > max_index_on_page:
                            self._update_panel_state(status_line=f"No track {index} on this page")
                        else:
                            # Translate page-relative index to absolute index
                            global_index = base + index
                            sel_track = self._listen_history[global_index - 1]
                            sel_artist = sel_track.get('artist', 'Unknown')
                            sel_title = sel_track.get('title', 'Unknown')
                            threading.Thread(
                                target=self._download_history_track,
                                args=(global_index, self._listen_channel, self._listen_quality or '256k'),
                                daemon=True,
                            ).start()
                            self._update_panel_state(
                                status_line=f"Downloading: {sel_artist} - {sel_title}"
                            )
                    except Exception:
                        self._update_panel_state(status_line="Invalid history index")
                    finally:
                        self._history_input = ""
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass

    def _start_listen_live_proxy(self, variant_url: str) -> str | None:
        """Start a lightweight HTTP proxy that serves patched HLS for Listen Live.

        The proxy:
        - Downloads the 256k WEB/V3 variant playlist with auth headers
        - Fetches & decodes the AES key via HLSDownloader.get_decryption_key
        - Serves the playlist at /listen.m3u8 with:
            * EXT-X-KEY URI rewritten to http://127.0.0.1:<port>/key
            * Relative segment URLs rewritten to absolute HTTPS URLs
        - Serves the raw AES key bytes at /key
        """
        try:
            if not self.hls_downloader:
                return None

            # Stop any previous proxy
            self._stop_listen_live_proxy()

            headers = self.hls_downloader.get_http_headers()
            resp = requests.get(variant_url, headers=headers, timeout=10)
            if resp.status_code != 200:
                print(f"❌ Failed to fetch variant playlist (HTTP {resp.status_code})")
                return None

            # Use the initial playlist only to derive the AES key; the
            # playlist itself will be refreshed on each proxy request so
            # that players keep getting new segments.
            initial_playlist_text = resp.text
            base_url = variant_url.rsplit('/', 1)[0] + '/'

            # Get AES key as hex, then raw bytes
            key_hex = self.hls_downloader.get_decryption_key(initial_playlist_text)
            if not key_hex:
                print("❌ Failed to obtain AES decryption key")
                return None

            try:
                key_bytes = bytes.fromhex(key_hex)
            except ValueError:
                print("❌ Invalid AES key format")
                return None

            # Define handler class capturing playlist and key
            outer_self = self

            class ListenLiveProxyHandler(BaseHTTPRequestHandler):
                def log_message(self, format, *args):  # Silence HTTP logs
                    return

                def do_GET(self):
                    try:
                        if self.path.startswith('/listen.m3u8'):
                            # Fetch a fresh copy of the upstream variant playlist on
                            # each request so we continue to get new segments instead
                            # of a static snapshot.
                            upstream_headers = outer_self.hls_downloader.get_http_headers()
                            upstream_resp = requests.get(variant_url, headers=upstream_headers, timeout=5)
                            if upstream_resp.status_code != 200:
                                self.send_response(502)
                                self.end_headers()
                                return

                            playlist_text = upstream_resp.text

                            # Patch playlist on the fly with local /key URL and absolute segments
                            patched_lines: List[str] = []
                            for line in playlist_text.split('\n'):
                                if line.startswith('#EXT-X-KEY:'):
                                    key_url = f"http://127.0.0.1:{self.server.server_port}/key"
                                    line = re.sub(r'URI="[^"]+"', f'URI="{key_url}"', line)
                                    patched_lines.append(line)
                                elif line.strip() and not line.startswith('#') and not line.startswith('http'):
                                    patched_lines.append(base_url + line.strip())
                                else:
                                    patched_lines.append(line)
                            body = '\n'.join(patched_lines).encode('utf-8')
                            self.send_response(200)
                            self.send_header('Content-Type', 'application/x-mpegURL')
                            self.send_header('Content-Length', str(len(body)))
                            self.end_headers()
                            self.wfile.write(body)
                        elif self.path.startswith('/key'):
                            body = key_bytes
                            self.send_response(200)
                            self.send_header('Content-Type', 'application/octet-stream')
                            self.send_header('Content-Length', str(len(body)))
                            self.end_headers()
                            self.wfile.write(body)
                        else:
                            self.send_response(404)
                            self.end_headers()
                    except Exception:
                        self.send_response(500)
                        self.end_headers()

            # Bind to an ephemeral port on localhost
            server = HTTPServer(('127.0.0.1', 0), ListenLiveProxyHandler)
            self._listen_proxy_server = server
            self._listen_proxy_thread = threading.Thread(
                target=server.serve_forever,
                daemon=True,
            )
            self._listen_proxy_thread.start()

            return f"http://127.0.0.1:{server.server_port}/listen.m3u8"

        except Exception as exc:
            print(f"❌ Error starting local stream proxy: {exc}")
            return None

    def _stop_listen_live_proxy(self):
        """Stop the Listen Live local HTTP proxy if it's running."""
        try:
            if self._listen_proxy_server:
                try:
                    self._listen_proxy_server.shutdown()
                except Exception:
                    pass
        finally:
            if self._listen_proxy_thread and self._listen_proxy_thread.is_alive():
                try:
                    self._listen_proxy_thread.join(timeout=2)
                except Exception:
                    pass
            self._listen_proxy_server = None
            self._listen_proxy_thread = None

    def _prepare_local_variant_playlist(self, variant_url: str) -> str | None:
        """Fetch and patch variant playlist so FFmpeg sees a raw AES key.

        This mirrors the m3u8XM trick:
        - Download the 256k WEB/V3 variant playlist with auth headers
        - Fetch & decode the JSON AES key via HLSDownloader.get_decryption_key
        - Write the raw key bytes to a local file
        - Rewrite the EXT-X-KEY URI to that local key file
        - Rewrite relative segment URLs to absolute HTTPS URLs so a local playlist path works
        Returns the filesystem path to the patched playlist, or None on error.
        """
        try:
            if not self.hls_downloader:
                return None

            headers = self.hls_downloader.get_http_headers()
            resp = requests.get(variant_url, headers=headers, timeout=10)
            if resp.status_code != 200:
                print(f"❌ Failed to fetch variant playlist ({resp.status_code})")
                return None

            playlist_text = resp.text
            base_url = variant_url.rsplit('/', 1)[0] + '/'

            # Get AES key as hex, then store raw bytes to a local key file
            key_hex = self.hls_downloader.get_decryption_key(playlist_text)
            if not key_hex:
                print("❌ Failed to obtain AES decryption key")
                return None

            try:
                key_bytes = bytes.fromhex(key_hex)
            except ValueError:
                print("❌ Invalid AES key format")
                return None

            base_runtime = Path(tempfile.gettempdir()) / f"satstash_{os.getuid()}"
            key_dir = base_runtime / 'keys'
            key_dir.mkdir(parents=True, exist_ok=True)
            key_file = key_dir / 'listen_live_aes.key'
            key_file.write_bytes(key_bytes)

            key_uri = key_file.as_uri()  # file:///... path

            # Patch playlist: point EXT-X-KEY at local key file and
            # make all relative segment URLs absolute HTTPS URLs.
            patched_lines: List[str] = []
            for line in playlist_text.split('\n'):
                if line.startswith('#EXT-X-KEY:'):
                    # Replace only the URI="..." part
                    line = re.sub(r'URI="[^"]+"', f'URI="{key_uri}"', line)
                    patched_lines.append(line)
                elif line.strip() and not line.startswith('#') and not line.startswith('http'):
                    # Segment or sub-playlist path; make absolute against CDN base
                    patched_lines.append(base_url + line.strip())
                else:
                    patched_lines.append(line)

            patched_playlist = '\n'.join(patched_lines)

            playlist_dir = base_runtime / 'hls'
            playlist_dir.mkdir(parents=True, exist_ok=True)
            playlist_file = playlist_dir / 'listen_live_256k.m3u8'
            playlist_file.write_text(patched_playlist, encoding='utf-8')

            return str(playlist_file)

        except Exception as exc:
            print(f"❌ Error preparing local playlist: {exc}")
            return None

    def _build_stream_header_lines(self) -> List[str]:
        if not self.hls_downloader:
            return []
        headers = self.hls_downloader.get_http_headers()
        header_lines = []
        for key, value in headers.items():
            if value:
                header_lines.append(f"{key}: {value}")
        return header_lines

    def _prepare_mpv_logfile(self) -> str | None:
        runtime_dir = Path(tempfile.gettempdir()) / f"satstash_{os.getuid()}" / 'logs'
        try:
            runtime_dir.mkdir(parents=True, exist_ok=True)
            log_path = runtime_dir / 'listen_live_mpv.log'
            # Truncate previous log to keep output current
            log_path.write_text('', encoding='utf-8')
            return str(log_path)
        except Exception:
            return None

    def _build_player_command(self, variant_url: str, header_lines: List[str]):
        """Determine the best available player and command"""
        mpv_path = shutil.which('mpv')
        header_str_mpv = "\r\n".join(header_lines)
        if header_str_mpv:
            header_str_mpv += "\r\n"
        header_str_ffplay = "\r\n".join(header_lines)
        if header_str_ffplay:
            header_str_ffplay += "\r\n"

        # Prefer ffplay first, since mpv is silent on some setups even though
        # the stream and proxy are working (VLC plays fine). ffplay gives us
        # a simpler, known-good audio path; mpv remains a fallback.
        ffplay_path = shutil.which('ffplay')
        if ffplay_path:
            return ([
                ffplay_path,
                '-autoexit',
                '-nodisp',
                '-loglevel', 'error',
                '-headers', header_str_ffplay,
                '-i', variant_url
            ], 'ffplay')

        if mpv_path:
            cmd = [
                mpv_path,
                '--no-config',
                '--no-video',
                '--volume=100',
                '--msg-level=all=info',
                '--cache=yes',
                '--cache-secs=5',
                '--input-terminal=no'
            ]
            if header_str_mpv:
                cmd.append(f"--http-header-fields={header_str_mpv}")
            cmd.append(variant_url)
            return (cmd, 'mpv')

        return None, None

    def _select_stream_cookies(self, cookies: Dict[str, str] | None) -> Dict[str, str]:
        """Reduce cookie payload to essentials for streaming headers."""
        if not cookies:
            return {}
        filtered = {
            key: value
            for key, value in cookies.items()
            if key in self.STREAM_COOKIE_WHITELIST
        }
        return filtered or {}

    def _send_player_command(self, command: str):
        """Send a control command to the active player"""
        if not self._player_process:
            return
        if self._player_control_mode == 'mpv' and self._player_process.stdin:
            try:
                self._player_process.stdin.write((command + '\n').encode('utf-8'))
                self._player_process.stdin.flush()
            except Exception:
                pass
        else:
            try:
                if command == 'q':
                    self._player_process.terminate()
            except Exception:
                pass

    def _compute_listen_controls(self, control_mode: str | None) -> str:
        """Build context-aware controls string for the Listen Live panel.

        - Always show (s), (d), (h).
        - Only show (n)/(p) history paging hints when history mode is active
          and there are additional pages available.
        """
        mode = (control_mode or self._player_control_mode or '').lower()
        if mode == 'mpv':
            base = "(s)top · (p)ause/resume · (+/-) volume · (</>) seek · (h)istory · (d)ownload current track"
        else:
            # ffplay and any other player share the same simpler controls
            base = "(s)top · (d)ownload current track · (h)istory"

        # Add compact paging hints only when we're in history selection mode
        # and have more than one page of history.
        if self._history_select_mode and self._listen_history and len(self._listen_history) > 5:
            page_size = 5
            total = len(self._listen_history)
            max_page = max(0, (total - 1) // page_size)
            hints = []
            if self._history_page < max_page:
                hints.append("(p)")
            if self._history_page > 0:
                hints.append("(n)")
            if hints:
                base += " · hist " + "/".join(hints)

        return base

    def _print_listen_live_header(self, channel: Dict, quality: str, control_mode: str | None):
        controls = self._compute_listen_controls(control_mode)
        self._listen_panel_state = {
            'channel_name': channel['name'],
            'channel_number': channel.get('number', '?'),
            'channel_genre': channel.get('genre', 'Live Radio'),
            'quality': quality,
            'player': control_mode or 'unknown',
            'artist': 'Loading...',
            'title': 'Waiting for metadata',
            'started_local': '--:--',
            'session_elapsed': '--:--',
            'track_elapsed': '--:--',
            'track_duration': '--:--',
            'progress_line': '[----------------------------]',
            'waveform': '▁▂▃▄▅▆▇█',
            'history': [],
            'status_line': 'Initializing...',
            'controls': controls
        }
        self._refresh_listen_panel()

    def _get_channel_art(self, channel: Dict) -> str:
        cache_key = channel.get('id') if channel else None
        if not cache_key:
            return self._render_channel_art(channel)
        if cache_key not in self._channel_art_cache:
            self._channel_art_cache[cache_key] = self._render_channel_art(channel)
        return self._channel_art_cache[cache_key]

    def _render_channel_art(self, channel: Dict) -> str:
        if not Image:
            return "[Artwork requires Pillow; run ./install.sh]"
        url = None
        images = channel.get('images') or {}
        url = images.get('large') or images.get('tile') or images.get('thumbnail')
        if isinstance(url, dict):
            # SiriusXM returns nested dicts with "preferred"/"default" entries
            url_candidate = url.get('preferred') or url.get('default')
            if isinstance(url_candidate, dict):
                url = url_candidate.get('url')
            else:
                url = url_candidate
        if not url and images:
            # pick first nested image url
            for value in images.values():
                if isinstance(value, dict):
                    choice = value.get('preferred') or value.get('default')
                    if isinstance(choice, dict):
                        url = choice.get('url')
                    elif isinstance(choice, str):
                        url = choice
                if url:
                    break
        if url and not url.startswith('http'):
            url = self._normalize_channel_image_url(url)
        if not url:
            return "[No channel artwork URL]"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code != 200:
                return f"[Artwork HTTP {resp.status_code}]"
            ascii_art = self._ascii_art_from_image(resp.content)
            return ascii_art or "[Artwork decode failed]"
        except Exception:
            return "[Artwork error]"

    def _normalize_channel_image_url(self, url: str) -> str:
        if not url:
            return url
        base = 'https://siriusxm-prd.cdn.jumpwirecloud.net/'
        clean = url.lstrip('/')
        return base + clean

    def _ascii_art_from_image(self, data: bytes, width: int = 40) -> str:
        try:
            img = Image.open(io.BytesIO(data)).convert('L')
            aspect_ratio = img.height / img.width
            height = max(4, int(width * aspect_ratio * 0.5))
            img = img.resize((width, height))
            chars = "@#%*+=-:. "
            lines = []
            for y in range(img.height):
                row = ''
                for x in range(img.width):
                    pixel = img.getpixel((x, y))
                    row += chars[pixel * (len(chars) - 1) // 255]
                lines.append(row)
            return "\n".join(lines)
        except Exception:
            return ""

    def _start_metadata_loop(self, channel: Dict, quality: str, start_ts: float):
        self._metadata_stop = threading.Event()
        self._metadata_thread = threading.Thread(
            target=self._metadata_worker,
            args=(channel, quality, start_ts),
            daemon=True
        )
        self._metadata_thread.start()

    def _metadata_worker(self, channel: Dict, quality: str, start_ts: float):
        last_track_id = None
        while not self._metadata_stop.wait(2):
            try:
                schedule = self.api.get_schedule(channel['id']) if self.api else None
                if not schedule:
                    continue
                current = schedule[-1]
                track_id = current.get('timestamp_utc')
                is_new_track = track_id != last_track_id
                if is_new_track:
                    last_track_id = track_id
                artist = current.get('artist', 'Unknown')
                title = current.get('title', 'Unknown')
                started_utc = datetime.fromisoformat(track_id.replace('Z', '+00:00'))
                started_local = started_utc.astimezone()
                now_utc = datetime.now(started_utc.tzinfo)
                track_elapsed = max(0, (now_utc - started_utc).total_seconds())
                session_elapsed = max(0, time.time() - start_ts)
                # API uses duration_ms for track length; fall back to duration when needed
                raw_duration_ms = current.get('duration_ms')
                if raw_duration_ms is None:
                    raw_duration_ms = current.get('duration', 0)
                duration_sec = max(0, raw_duration_ms / 1000) if raw_duration_ms else 0
                waveform = self._render_waveform(int(track_elapsed))
                progress_line = self._render_progress_bar(track_elapsed, duration_sec)
                if is_new_track:
                    self._listen_history.insert(0, current)
                    self._listen_history = self._listen_history[:20]
                self._update_panel_state(
                    artist=f"{artist}",
                    title=f"{title}",
                    started_local=started_local.strftime('%I:%M %p'),
                    session_elapsed=self._format_duration(session_elapsed),
                    track_elapsed=self._format_duration(track_elapsed),
                    track_duration=self._format_duration(duration_sec),
                    progress_line=progress_line,
                    waveform=waveform,
                )
                # Keep the visible history page in sync with the backing list.
                self._refresh_history_page()
            except Exception:
                continue

    def _render_waveform(self, elapsed: int) -> str:
        blocks = "▁▂▃▄▅▆▇█"
        length = 28
        wave = []
        # Simple deterministic pseudo-random based on elapsed time so the
        # pattern looks like a bar graph rather than a fixed ramp.
        seed = max(1, int(elapsed))
        for _ in range(length):
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            level = seed % len(blocks)
            wave.append(blocks[level])
        return ''.join(wave)

    def _render_progress_bar(self, elapsed: float, duration: float, width: int = 28) -> str:
        elapsed = max(0, elapsed)
        if not duration or duration <= 0:
            indicator = int(elapsed) % width
            segments = ['─'] * width
            segments[indicator] = '█'
            bar = ''.join(segments)
            return f"[{bar}] {self._format_duration(elapsed)} / --:--"
        ratio = min(1.0, max(0.0, elapsed / duration))
        filled = int(ratio * width)
        bar = '█' * filled + '─' * (width - filled)
        return f"[{bar}] {self._format_duration(elapsed)} / {self._format_duration(duration)}"

    def _format_duration(self, seconds: float | None) -> str:
        if seconds is None or seconds <= 0:
            return "--:--"
        mins, secs = divmod(int(seconds), 60)
        return f"{mins:02d}:{secs:02d}"

    def _print_history_summary(self, show_details: bool = False):
        if not self._listen_history:
            print("📜 History: (empty)")
            return
        print("📜 Recent Tracks:")
        for idx, track in enumerate(self._listen_history, 1):
            artist = track.get('artist', 'Unknown')
            title = track.get('title', 'Unknown')
            ts = track.get('timestamp_utc', '')
            when = ''
            if ts:
                try:
                    when = datetime.fromisoformat(ts.replace('Z', '+00:00')).astimezone().strftime('%I:%M %p')
                except Exception:
                    when = ''
            line = f"  {idx}. {artist} - {title}"
            if when:
                line += f" ({when})"
            print(line)
            if show_details:
                dur = track.get('duration', 0) / 1000
                print(f"     Duration: {int(dur // 60)}m {int(dur % 60)}s")

    def _load_dvr_history_for_channel(self, channel: Dict, hours_back: int = 1):
        """Refresh _listen_history from DVR API for quick-access history.

        Uses the same schedule data as the DVR recorder, but only fetches the
        last `hours_back` hours and keeps newest tracks first.
        """
        if not self.api or not channel:
            return
        try:
            # Reuse the existing API helper if present; otherwise fall back to
            # a raw schedule fetch and simple time filter.
            if hasattr(self.api, "get_dvr_tracks"):
                tracks = self.api.get_dvr_tracks(channel["id"], hours_back=hours_back)
            else:
                schedule = self.api.get_schedule(channel["id"]) or []
                if not schedule:
                    self._listen_history = []
                    self._update_panel_state(history=[])
                    return
                cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
                tracks = []
                for item in schedule:
                    ts = item.get("timestamp_utc")
                    if not ts:
                        continue
                    try:
                        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        if t >= cutoff:
                            tracks.append(item)
                    except Exception:
                        continue

            if not tracks:
                self._listen_history = []
                self._history_page = 0
                self._refresh_history_page()
                return

            # Ensure newest-first ordering for selection convenience
            tracks_sorted = sorted(
                tracks,
                key=lambda t: t.get("timestamp_utc", ""),
                reverse=True,
            )

            # Deduplicate by timestamp so the currently playing track (and
            # others) only appear once.
            seen = set()
            unique: List[Dict] = []
            for item in tracks_sorted:
                ts = item.get("timestamp_utc")
                if not ts or ts in seen:
                    continue
                seen.add(ts)
                unique.append(item)

            self._listen_history = unique
            self._history_page = 0
            self._refresh_history_page()
        except Exception:
            # Keep existing in-memory history if DVR fetch fails
            return

    def _download_history_track(self, index: int, channel: Dict, quality: str):
        if index < 1 or index > len(self._listen_history):
            print("❌ Invalid history index")
            return
        track = self._listen_history[index - 1]
        print(f"\n⬇️  Downloading '{track.get('artist', 'Unknown')} - {track.get('title', 'Unknown')}' from DVR...")
        try:
            tracks = self.recorder_integration.record_dvr_tracks(
                channel,
                [track],
                quality=quality,
                progress_callback=lambda msg: print(f"  {msg}"),
                track_callback=lambda msg: print(f"  {msg}")
            )
            if tracks:
                print(f"✅ Saved track to {tracks[0].get('file_path', 'output directory')}")
                # Surface completion in the Listen Live panel as well.
                artist = track.get('artist', 'Unknown')
                title = track.get('title', 'Unknown')
                self._update_panel_state(status_line=f"Downloaded: {artist} - {title}")
            else:
                print("⚠️  No files returned (check output folder)")
                self._update_panel_state(status_line="No files returned; check output folder")
        except Exception as exc:
            print(f"❌ DVR download failed: {exc}")
            self._update_panel_state(status_line="DVR download failed; see log")

    def _refresh_history_page(self):
        """Update panel history slice based on current page index.

        History is stored newest-first in _listen_history; we show up to 5
        items per page and clamp the page index to the available range.
        """
        page_size = 5
        total = len(self._listen_history)
        if total <= 0:
            self._history_page = 0
            self._update_panel_state(history=[])
            return

        max_page = max(0, (total - 1) // page_size)
        if self._history_page < 0:
            self._history_page = 0
        if self._history_page > max_page:
            self._history_page = max_page

        start = self._history_page * page_size
        end = start + page_size
        self._update_panel_state(history=self._listen_history[start:end])

    def _start_status_loop(self, channel: Dict, quality: str, start_ts: float):
        self._status_stop = threading.Event()
        self._status_thread = threading.Thread(
            target=self._status_worker,
            args=(channel, quality, start_ts),
            daemon=True
        )
        self._status_thread.start()

    def _status_worker(self, channel: Dict, quality: str, start_ts: float):
        while not self._status_stop.wait(1):
            elapsed = int(time.time() - start_ts)
            mins, secs = divmod(max(elapsed, 0), 60)
            # Keep status_line free for one-off messages (history/download);
            # only refresh the session elapsed timer here.
            self._update_panel_state(session_elapsed=f"{mins:02d}:{secs:02d}")

    def _stop_player(self):
        """Ensure any running player process is terminated"""
        if self._metadata_stop:
            self._metadata_stop.set()
        if self._metadata_thread and self._metadata_thread.is_alive():
            self._metadata_thread.join(timeout=2)
        self._metadata_thread = None
        self._metadata_stop = None
        if self._status_stop:
            self._status_stop.set()
        if self._status_thread and self._status_thread.is_alive():
            self._status_thread.join(timeout=2)
        self._status_thread = None
        self._status_stop = None

        if self._player_process and self._player_process.poll() is None:
            try:
                self._player_process.terminate()
                self._player_process.wait(timeout=5)
            except Exception:
                self._player_process.kill()
        self._player_process = None
        self._player_control_mode = None
        # Stop local Listen Live proxy if running
        self._stop_listen_live_proxy()
        # Close any active player log file
        if self._player_log_fp:
            try:
                self._player_log_fp.close()
            except Exception:
                pass
            self._player_log_fp = None
    
    def _update_panel_state(self, **updates):
        if not updates:
            return
        with self._listen_panel_lock:
            self._listen_panel_state.update({k: v for k, v in updates.items() if v is not None})
        self._refresh_listen_panel()

    def _refresh_listen_panel(self):
        if not self._live_display or not self._listen_panel_state:
            return
        with self._listen_panel_lock:
            state = dict(self._listen_panel_state)

        history_lines = self._format_history_lines(state.get('history', []))

        panel_text = (
            f"Channel : {state.get('channel_name')} (Ch {state.get('channel_number')})\n"
            f"Genre   : {state.get('channel_genre')}\n"
            f"Quality : {state.get('quality')}    Player: {state.get('player')}\n"
            f"Status  : {state.get('status_line', '')}\n\n"
            f"Now Playing : {state.get('artist')} - {state.get('title')}\n"
            f"Started     : {state.get('started_local')}\n"
            f"Listening   : {state.get('session_elapsed')} elapsed\n"
            f"Track       : {state.get('track_elapsed')} / {state.get('track_duration')}\n"
            f"Progress    : {state.get('progress_line')}\n"
            f"Waveform    : {state.get('waveform')}\n\n"
            "Recent Tracks:\n" + "\n".join(history_lines) + "\n\n"
            f"Controls: {state.get('controls')}"
        )

        panel = Panel(panel_text, title="▶️  LISTEN LIVE", expand=True)
        self._live_display.update(panel)

    def _format_history_lines(self, history: List[Dict]) -> List[str]:
        if not history:
            return ['(history empty)']
        lines = []
        for idx, track in enumerate(history, 1):
            artist = track.get('artist', 'Unknown')
            title = track.get('title', 'Unknown')
            clock = ''
            ts = track.get('timestamp_utc')
            if ts:
                try:
                    clock = datetime.fromisoformat(ts.replace('Z', '+00:00')).astimezone().strftime('%I:%M %p')
                except Exception:
                    clock = ''
            line = f"{idx}. {artist} - {title}"
            if clock:
                line += f" ({clock})"
            lines.append(line)
        return lines

    def browse_dvr_history(self):
        """Browse and download DVR history"""
        print("\n" + "="*60)
        print("📺 DVR History Browser")
        print("="*60)
        
        # Select channel
        channel = self.select_channel("Which channel's DVR history?")
        if not channel:
            return
        
        # Fetch DVR tracks
        print(f"\n📡 Fetching DVR history for {channel['name']}...")
        
        # Get last 5 hours of tracks
        dvr_tracks = self.api.get_dvr_tracks(channel['id'], hours_back=5)
        if not dvr_tracks and self.api.last_schedule_status == 401:
            if self._refresh_bearer_token(channel):
                dvr_tracks = self.api.get_dvr_tracks(channel['id'], hours_back=5)

        if not dvr_tracks:
            print("❌ No DVR tracks found")
            return
        
        # Show time range options
        print(f"\n✅ Found {len(dvr_tracks)} tracks in last 5 hours")
        print("\nWhat would you like to download?")
        print("  1. Last 15 minutes")
        print("  2. Last 30 minutes")
        print("  3. Last 45 minutes")
        print("  4. Last 1 hour")
        print("  5. Last 2 hours")
        print("  6. Last 5 hours (all)")
        print("  7. Select individual tracks")
        print("  8. Custom time range")
        
        choice = input("\nSelect (1-8): ").strip()
        
        now = datetime.now()
        
        if choice == '1':
            self._download_dvr_range(channel, dvr_tracks, minutes=15)
        elif choice == '2':
            self._download_dvr_range(channel, dvr_tracks, minutes=30)
        elif choice == '3':
            self._download_dvr_range(channel, dvr_tracks, minutes=45)
        elif choice == '4':
            self._download_dvr_range(channel, dvr_tracks, minutes=60)
        elif choice == '5':
            self._download_dvr_range(channel, dvr_tracks, minutes=120)
        elif choice == '6':
            self._download_dvr_range(channel, dvr_tracks, minutes=300)
        elif choice == '7':
            self._select_individual_tracks(channel, dvr_tracks)
        elif choice == '8':
            self._custom_time_range(channel, dvr_tracks)
        else:
            print("❌ Invalid choice")
    
    def _download_dvr_range(self, channel, dvr_tracks, minutes):
        """Download tracks from last N minutes"""
        from datetime import datetime, timezone
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        
        # Filter tracks
        selected_tracks = []
        for track in dvr_tracks:
            track_time = datetime.fromisoformat(track['timestamp_utc'].replace('Z', '+00:00'))
            if track_time >= cutoff_time:
                selected_tracks.append(track)
        
        if not selected_tracks:
            print(f"❌ No tracks found in last {minutes} minutes")
            return
        
        print(f"\n📋 Found {len(selected_tracks)} tracks in last {minutes} minutes:")
        for i, track in enumerate(selected_tracks[:5], 1):
            local_time = datetime.fromisoformat(track['timestamp_utc'].replace('Z', '+00:00')).astimezone()
            print(f"  {i}. {track['artist']} - {track['title']}")
            print(f"      {local_time.strftime('%I:%M %p')}")
        
        if len(selected_tracks) > 5:
            print(f"  ... and {len(selected_tracks) - 5} more")
        
        confirm = input(f"\nDownload these {len(selected_tracks)} tracks? (y/n): ").strip().lower()
        if confirm != 'y':
            return

        quality = self._select_quality()
        
        print("\n🚧 DVR download feature coming soon!")
        print("For now, use the record_live feature to capture content.")
        input("\nPress Enter to continue...")
    
    def _select_individual_tracks(self, channel, dvr_tracks):
        """Let user select individual tracks with pagination"""
        from datetime import datetime
        
        page_size = 20
        page = 0
        
        while True:
            start = page * page_size
            end = start + page_size
            page_tracks = dvr_tracks[start:end]
            
            print(f"\n📋 Available tracks (Page {page + 1}/{(len(dvr_tracks) + page_size - 1) // page_size})")
            print("-" * 60)
            
            # Show tracks with absolute numbers
            for i, track in enumerate(page_tracks, start + 1):
                local_time = datetime.fromisoformat(track['timestamp_utc'].replace('Z', '+00:00')).astimezone()
                duration = track.get('duration', 0) // 1000 // 60  # Convert to minutes
                print(f"{i:2d}. {track['artist']:<25} - {track['title']:<30}")
                print(f"    {local_time.strftime('%I:%M %p')} ({duration}min)")
            
            print(f"\nCommands: [numbers] to select | [n]ext | [p]rev | [a]ll | [q]uit")
            print("Examples: 1,3,5 or 1-10 or all")
            cmd = input("> ").strip().lower()
            
            if cmd == 'n' and end < len(dvr_tracks):
                page += 1
                continue
            elif cmd == 'p' and page > 0:
                page -= 1
                continue
            elif cmd == 'q':
                return
            elif cmd == 'a' or cmd == 'all':
                # Download all tracks
                self._download_dvr_tracks(channel, dvr_tracks)
                return
            else:
                # Parse selection
                selected_indices = set()
                try:
                    for part in cmd.split(','):
                        part = part.strip()
                        if '-' in part:
                            # Range like "5-8"
                            s, e = map(int, part.split('-'))
                            selected_indices.update(range(s, e + 1))
                        elif part.isdigit():
                            selected_indices.add(int(part))
                    
                    # Get selected tracks
                    selected_tracks = [dvr_tracks[i-1] for i in sorted(selected_indices) if 1 <= i <= len(dvr_tracks)]
                    
                    if selected_tracks:
                        self._download_dvr_tracks(channel, selected_tracks)
                        return
                except ValueError:
                    print("❌ Invalid selection")
                    input("\nPress Enter to continue...")
    
    def _download_dvr_tracks(self, channel, selected_tracks):
        """Download selected DVR tracks"""
        from datetime import datetime, timezone
        
        print(f"\n{'='*60}")
        print(f"📥 Downloading {len(selected_tracks)} DVR Tracks")
        print(f"{'='*60}\n")
        
        # Show what we're downloading
        print("Selected tracks:")
        for i, track in enumerate(selected_tracks[:5], 1):
            local_time = datetime.fromisoformat(track['timestamp_utc'].replace('Z', '+00:00')).astimezone()
            print(f"  {i}. {track['artist']} - {track['title']}")
            print(f"     {local_time.strftime('%I:%M %p')}")
        
        if len(selected_tracks) > 5:
            print(f"  ... and {len(selected_tracks) - 5} more")
        
        confirm = input(f"\nDownload these {len(selected_tracks)} tracks? (y/n): ").strip().lower()
        if confirm != 'y':
            return
        
        # Calculate time range for recording
        first_track = selected_tracks[0]
        last_track = selected_tracks[-1]
        
        start_time = datetime.fromisoformat(first_track['timestamp_utc'].replace('Z', '+00:00'))
        last_track_duration = last_track.get('duration', 300000) / 1000  # ms to seconds
        end_time = datetime.fromisoformat(last_track['timestamp_utc'].replace('Z', '+00:00'))
        end_time = end_time + timedelta(seconds=last_track_duration + 30)  # Add buffer
        
        duration_minutes = int((end_time - start_time).total_seconds() / 60) + 1
        
        print(f"\n{'='*60}")
        print(f"📻 Recording DVR Content")
        print(f"{'='*60}")
        print(f"Channel: {channel['name']}")
        print(f"Start: {start_time.astimezone().strftime('%I:%M %p')}")
        print(f"End: {end_time.astimezone().strftime('%I:%M %p')}")
        print(f"Duration: ~{duration_minutes} minutes")
        print(f"Quality: {quality}")
        print(f"{'='*60}\n")
        
        def progress(msg):
            print(f"  {msg}")
        
        def track_update(msg):
            print(f"\n{msg}")
        
        # Download the DVR tracks!
        print("🔴 Starting DVR download...")
        print("-" * 60 + "\n")
        
        try:
            tracks = self.recorder_integration.record_dvr_tracks(
                channel,
                selected_tracks,
                quality=quality,
                progress_callback=progress,
                track_callback=track_update
            )
            
            # Show results
            print(f"\n{'='*60}")
            print(f"✅ DVR DOWNLOAD COMPLETE!")
            print(f"{'='*60}\n")
            
            if tracks:
                print(f"📋 Downloaded {len(tracks)} tracks:\n")
                for i, track in enumerate(tracks, 1):
                    artist = track.get('artist', 'Unknown')
                    title = track.get('title', 'Unknown')
                    size_mb = track.get('file_size', 0) / (1024 * 1024)
                    print(f"  {i}. {artist} - {title} ({size_mb:.1f} MB)")
                
                print(f"\n📁 Location: ~/Music/SiriusXM/{channel['name']}/")
            else:
                print("⚠️  No tracks returned (check output directory for files)")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        input("\nPress Enter to continue...")
    
    def _custom_time_range(self, channel, dvr_tracks):
        """Custom time range selection"""
        print("\n🚧 Custom time range feature coming soon!")
        input("\nPress Enter to continue...")
    
    def schedule_recording(self):
        """Schedule a future recording"""
        print("\n" + "="*60)
        print("⏰ Schedule Recording")
        print("="*60)
        
        # Select channel
        channel = self.select_channel("Which channel to record?")
        if not channel:
            return
        
        # Get date/time
        print("\nWhen to start recording?")
        print("Current time:", datetime.now().strftime("%Y-%m-%d %I:%M %p"))
        print("\n  1. Today")
        print("  2. Tomorrow")
        print("  3. Specific date/time")
        
        choice = input("\nSelect (1-3): ").strip()
        
        now = datetime.now()
        
        if choice == '1':
            date_str = now.strftime("%Y-%m-%d")
        elif choice == '2':
            tomorrow = now + timedelta(days=1)
            date_str = tomorrow.strftime("%Y-%m-%d")
        elif choice == '3':
            date_str = input("Enter date (YYYY-MM-DD): ").strip()
        else:
            print("❌ Invalid choice")
            return
        
        time_str = input("Enter time (HH:MM in 24-hour format, e.g., 14:30): ").strip()
        
        try:
            scheduled_time = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        except ValueError:
            print("❌ Invalid date/time format")
            return
        
        if scheduled_time <= now:
            print("❌ Scheduled time must be in the future")
            return
        
        # Get duration
        duration = int(input("\nDuration in minutes: "))
        
        quality = self._select_quality()

        # Confirm
        print(f"\n{'='*60}")
        print(f"⏰ Scheduled Recording")
        print(f"{'='*60}")
        print(f"📻 Channel: {channel['name']}")
        print(f"📅 Start: {scheduled_time.strftime('%Y-%m-%d %I:%M %p')}")
        print(f"⏱️  Duration: {duration} minutes")
        print(f"🎚️  Quality: {quality}")
        print(f"🏁 End: {(scheduled_time + timedelta(minutes=duration)).strftime('%I:%M %p')}")
        print(f"{'='*60}")
        
        confirm = input("\nSchedule this recording? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ Cancelled")
            return
        
        # Save schedule
        self._save_scheduled_recording(channel, scheduled_time, duration, quality)
        
        print("\n✅ Recording scheduled!")
        print("\n💡 To run scheduled recordings, use: ./sxm_cli.py --run-scheduled")
        input("\nPress Enter to continue...")
    
    def _save_scheduled_recording(self, channel, scheduled_time, duration, quality):
        """Save scheduled recording to file"""
        import json
        
        schedule_file = Path.home() / ".seriouslyxm" / "scheduled_recordings.json"
        schedule_file.parent.mkdir(exist_ok=True)
        
        # Load existing schedules
        if schedule_file.exists():
            with open(schedule_file) as f:
                schedules = json.load(f)
        else:
            schedules = []
        
        # Add new schedule
        schedules.append({
            'id': len(schedules) + 1,
            'channel': {
                'id': channel['id'],
                'name': channel['name'],
                'number': channel.get('number')
            },
            'start_time': scheduled_time.isoformat(),
            'duration_minutes': duration,
            'audio_quality': quality,
            'status': 'pending',
            'created_at': datetime.now().isoformat()
        })
        
        # Save
        with open(schedule_file, 'w') as f:
            json.dump(schedules, f, indent=2)
    
    def view_scheduled_recordings(self):
        """View and manage scheduled recordings"""
        import json
        
        schedule_file = Path.home() / ".seriouslyxm" / "scheduled_recordings.json"
        
        if not schedule_file.exists():
            print("\n📋 No scheduled recordings")
            input("\nPress Enter to continue...")
            return
        
        with open(schedule_file) as f:
            schedules = json.load(f)
        
        # Filter pending
        pending = [s for s in schedules if s['status'] == 'pending']
        
        if not pending:
            print("\n📋 No pending recordings")
            input("\nPress Enter to continue...")
            return
        
        print("\n" + "="*60)
        print("📋 Scheduled Recordings")
        print("="*60 + "\n")
        
        now = datetime.now()
        
        for i, schedule in enumerate(pending, 1):
            start_time = datetime.fromisoformat(schedule['start_time'])
            time_until = start_time - now
            
            print(f"{i}. {schedule['channel']['name']} (Ch {schedule['channel'].get('number', '?')})")
            print(f"   📅 {start_time.strftime('%Y-%m-%d %I:%M %p')}")
            print(f"   ⏱️  Duration: {schedule['duration_minutes']} minutes")
            print(f"   🎚️  Quality: {schedule.get('audio_quality', self.config.get('audio_quality', '256k'))}")
            
            if time_until.total_seconds() > 0:
                hours = int(time_until.total_seconds() // 3600)
                minutes = int((time_until.total_seconds() % 3600) // 60)
                print(f"   ⏰ Starts in: {hours}h {minutes}m")
            else:
                print(f"   ⚠️  OVERDUE!")
            print()
        
        print("\n1. Delete a recording")
        print("2. Back to main menu")
        
        choice = input("\nSelect (1-2): ").strip()
        
        if choice == '1':
            idx = int(input(f"Which recording to delete (1-{len(pending)})? ")) - 1
            if 0 <= idx < len(pending):
                pending[idx]['status'] = 'cancelled'
                with open(schedule_file, 'w') as f:
                    json.dump(schedules, f, indent=2)
                print("✅ Recording cancelled")
        
        input("\nPress Enter to continue...")
    
    def browse_channels(self):
        """Browse channel list"""
        channels = self.channel_manager.list_channels()
        self._browse_all_channels(channels)
        input("\nPress Enter to continue...")
    
    def manage_settings(self):
        """Manage SatStash settings"""
        from getpass import getpass

        while True:
            username = self.config.get('username', '')
            password = self.config.get('password', '')
            audio_quality = self.config.get('audio_quality', '256k')
            output_dir = self.config.get('output_directory', str(Path.home() / "Music" / "SiriusXM"))

            print("\n" + "="*60)
            print("⚙️  SatStash Settings")
            print("="*60)
            print(f"1. Username        : {username or 'not set'}")
            print(f"2. Password        : {'********' if password else 'not set'}")
            print(f"3. Audio Quality   : {audio_quality}")
            print(f"4. Output Directory: {output_dir}")
            print("5. Back to main menu")

            choice = input("\nSelect an option (1-5): ").strip()

            if choice == '1':
                new_user = input("Enter SiriusXM username/email (leave blank to clear): ").strip()
                self.config.set('username', new_user)
                print("✅ Username updated")
            elif choice == '2':
                new_pass = getpass("Enter SiriusXM password (leave blank to clear): ")
                self.config.set('password', new_pass)
                print("✅ Password updated")
            elif choice == '3':
                new_quality = self._select_quality(allow_default=False)
                self.config.set('audio_quality', new_quality)
                print(f"✅ Audio quality set to {new_quality}")
            elif choice == '4':
                new_dir = input(f"Enter output directory (current: {output_dir}): ").strip()
                if new_dir:
                    path = Path(new_dir).expanduser()
                    try:
                        path.mkdir(parents=True, exist_ok=True)
                        self.config.set('output_directory', str(path))
                        print(f"✅ Output directory set to {path}")
                    except Exception as exc:
                        print(f"❌ Failed to set directory: {exc}")
                else:
                    print("⚠️  Directory unchanged")
            elif choice == '5':
                break
            else:
                print("❌ Invalid choice")

            input("\nPress Enter to continue...")
    
    def run(self):
        """Main run loop"""
        if not self.initialize():
            return

        while True:
            try:
                self.show_main_menu()
                choice = input("Select an option (1-8): ").strip()

                if choice == '1':
                    self.record_live_now()
                elif choice == '2':
                    self.listen_live_terminal()
                elif choice == '3':
                    self.browse_dvr_history()
                elif choice == '4':
                    self.schedule_recording()
                elif choice == '5':
                    self.view_scheduled_recordings()
                elif choice == '6':
                    self.browse_channels()
                elif choice == '7':
                    self.manage_settings()
                elif choice == '8':
                    print("\n👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
                input("\nPress Enter to continue...")


def main():
    """Entry point"""
    cli = SXMCli()
    cli.run()


if __name__ == '__main__':
    main()
