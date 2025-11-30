#!/usr/bin/env python3
"""
Recorder Integration for SatStash
Connects the CleanLiveRecorder to the TUI
"""

import logging
import sys
import os
import io
from pathlib import Path
from contextlib import contextmanager, redirect_stdout, redirect_stderr

# Setup logging to file only (not console)
log_file = Path.home() / ".seriouslyxm" / "recorder_debug.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file)
        # Removed StreamHandler() to prevent console spam
    ]
)
logger = logging.getLogger(__name__)

# Use our OWN simple recorder - no complex imports
from sxm.core.recorder import ZeroPartialRecorder
from sxm.core.api import SiriusXMAPI
from sxm.core.hls_downloader import HLSDownloader
from sxm.utils.session_manager import SessionManager


@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr output (for TUI mode)"""
    # Redirect to devnull instead of capturing to string
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


class RecorderIntegration:
    """Manages recorder instance with session handling"""
    
    def __init__(self, config):
        self.config = config
        self.session_mgr = SessionManager()
        self.recorder = None
        self._bearer_token = None
    
    def get_recorder(self):
        """Get or create recorder instance - SIMPLE AND CLEAN"""
        try:
            logger.info("=== get_recorder() called ===")
            if self.recorder:
                logger.info("Returning existing recorder")
                return self.recorder
            
            # Get session info
            session = self.session_mgr.get_session_info()
            if not session or not session.get('is_valid'):
                raise Exception("No valid session. Please log in first.")
            
            username = session.get('username')
            cookies = self.session_mgr.load_session(username)
            if not cookies:
                raise Exception("No session cookies found")
            
            # Extract bearer token
            bearer_token = self._extract_bearer_token(cookies)
            if not bearer_token:
                raise Exception("Could not extract bearer token")
            
            # Create components including browser session
            from sxm.core.browser_session import BrowserSession
            
            api = SiriusXMAPI(bearer_token)
            browser = BrowserSession(cookies=cookies, headless=True)  # NEED browser for stream URL!
            downloader = HLSDownloader()
            downloader.bearer_token = bearer_token
            
            # Output directory
            output_dir = self.config.get('output_directory', '~/Music/SiriusXM')
            
            # Create recorder WITH browser (needed to capture stream URL)
            self.recorder = ZeroPartialRecorder(
                api=api,
                browser_session=browser,  # Need this!
                hls_downloader=downloader,
                audio_processor=None,
                output_dir=output_dir
            )
            
            logger.info("✅ Recorder created successfully!")
            return self.recorder
        
        except Exception as e:
            logger.error(f"Error creating recorder: {e}", exc_info=True)
            raise Exception(f"Error creating recorder: {e}")
    
    def _extract_bearer_token(self, cookies):
        """Extract bearer token from cookies"""
        # Try SXMAUTHORIZATION first (raw token)
        if 'SXMAUTHORIZATION' in cookies:
            return cookies['SXMAUTHORIZATION']
        
        # Try AUTH_TOKEN (URL-encoded JSON)
        if 'AUTH_TOKEN' in cookies:
            try:
                import urllib.parse
                import json
                # URL decode the cookie value
                decoded = urllib.parse.unquote(cookies['AUTH_TOKEN'])
                # Parse JSON
                auth_data = json.loads(decoded)
                # Extract access token
                return auth_data.get('session', {}).get('accessToken')
            except Exception as e:
                logger.error(f"Failed to decode AUTH_TOKEN: {e}")
                return None
        
        return None
    
    def record_channel(self, channel, duration_minutes=5, quality: str | None = None, 
                      progress_callback=None, track_callback=None):
        """Start recording a channel using DVR buffer (DOCUMENTED WORKING CODE)"""
        print("\n" + "="*80)
        print("🎙️  RECORDING WITH DVR BUFFER")
        print("="*80)
        logger.info("=== record_channel() called ===")
        logger.info(f"Channel: {channel.get('name')} (ID: {channel.get('id')})")
        logger.info(f"Duration: {duration_minutes} minutes")
        if not quality:
            quality = self.config.get('audio_quality', '256k')
        logger.info(f"Quality: {quality}")
        print(f"Channel: {channel.get('name')}")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Quality: {quality}")
        
        try:
            # Send progress updates to UI
            if progress_callback:
                progress_callback("🔐 Authenticating with browser...")
            
            # Use our own recorder with DVR buffer support!
            logger.info("Getting recorder instance...")
            recorder = self.get_recorder()
            
            if progress_callback:
                progress_callback("✅ Authentication complete!")
            
            logger.info("Calling recorder.record_live()...")
            print("\n🔴 STARTING RECORDING WITH DVR BUFFER...")
            
            # Pass callbacks to recorder
            recorder.progress_callback = progress_callback
            recorder.track_callback = track_callback
            
            # Call our recorder (now with working DVR code!)
            tracks = recorder.record_live(channel, duration_minutes, quality=quality)
            
            logger.info(f"✅ Recording complete! Got {len(tracks) if tracks else 0} tracks")
            return tracks if tracks else []
            
        except Exception as e:
            logger.error(f"❌ Error in record_channel: {e}", exc_info=True)
            raise
    
    def record_dvr_tracks(self, channel, selected_tracks, quality: str | None = None,
                          progress_callback=None, track_callback=None):
        """Record specific tracks from DVR history"""
        from datetime import datetime, timedelta, timezone
        
        logger.info("=== record_dvr_tracks() called ===")
        logger.info(f"Channel: {channel.get('name')}")
        if not quality:
            quality = self.config.get('audio_quality', '256k')
        logger.info(f"Tracks to record: {len(selected_tracks)}")
        logger.info(f"Quality: {quality}")
        
        if not selected_tracks:
            logger.error("No tracks provided")
            return []
        
        # Calculate time range
        first_track = selected_tracks[0]
        last_track = selected_tracks[-1]
        
        start_time = datetime.fromisoformat(first_track['timestamp_utc'].replace('Z', '+00:00'))
        
        # Add last track's duration to get end time
        last_track_duration_seconds = last_track.get('duration', 300000) / 1000
        end_time = datetime.fromisoformat(last_track['timestamp_utc'].replace('Z', '+00:00'))
        end_time = end_time + timedelta(seconds=last_track_duration_seconds + 30)
        
        duration_minutes = int((end_time - start_time).total_seconds() / 60) + 1
        
        logger.info(f"DVR time range: {start_time.isoformat()} to {end_time.isoformat()}")
        logger.info(f"Duration: {duration_minutes} minutes")
        
        try:
            if progress_callback:
                progress_callback("🔐 Authenticating with browser...")
            
            recorder = self.get_recorder()
            
            if progress_callback:
                progress_callback("✅ Authentication complete!")
            
            # Pass callbacks
            recorder.progress_callback = progress_callback
            recorder.track_callback = track_callback
            
            # Record using DVR with specific start time
            logger.info("Starting DVR recording from specific timestamp...")
            tracks = recorder.record_dvr(
                channel,
                start_time,
                duration_minutes,
                selected_tracks=selected_tracks,
                quality=quality
            )
            
            logger.info(f"✅ Recording complete! Got {len(tracks) if tracks else 0} tracks")
            return tracks if tracks else []
            
        except Exception as e:
            logger.error(f"Error during DVR recording: {e}", exc_info=True)
            if progress_callback:
                progress_callback(f"❌ Error: {e}")
            return []
