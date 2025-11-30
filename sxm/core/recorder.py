#!/usr/bin/env python3
"""
Zero-Partial Recorder

Smart recording strategy that eliminates partial tracks!

Strategy:
1. Fetch API schedule
2. Find current track (last in list)
3. Smart start: Begin at current track's start time
4. Smart end: Stop at last complete track before duration expires
5. Result: ZERO partials, all complete tracks!
"""

import time
import subprocess
import requests
import threading
import json
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict
from sxm.utils.audio_recorder import AudioRecorder
from sxm.core.track_splitter import split_and_tag_tracks, create_playlist

# Setup logging
logger = logging.getLogger(__name__)


class ZeroPartialRecorder:
    """Records with zero-partial guarantee using API-driven timing"""
    
    def __init__(self, api, browser_session, hls_downloader, audio_processor, 
                 output_dir: str = "~/Music/SiriusXM"):
        """
        Initialize recorder
        
        Args:
            api: SiriusXMAPI instance
            browser_session: Browser session for stream info
            hls_downloader: HLS downloader instance
            audio_processor: Audio processor instance
            output_dir: Output directory
        """
        self.api = api
        self.browser_session = browser_session
        self.hls_downloader = hls_downloader
        self.audio_processor = audio_processor
        self.output_dir = Path(output_dir).expanduser()
        self.bearer_token = None
        self.master_url = None
        self.variant_url = None
        self.is_recording = False
        self.audio_recorder = AudioRecorder(output_format='m4a')
        self.recording_start_time = None
        self.actual_audio_start_time = None
        self.track_changes = []
        self.variant_url = None
        self.progress_callback = None
        self.track_callback = None
        self.channel_id = None  # Store for schedule refreshes
        self.allow_track_expansion = True  # Allow schedule-driven track expansion (DVR can disable)
        self.channel_info: Optional[Dict] = None
    
    def record_live(self, channel: Dict, duration_minutes: int = 30, quality: str = '256k') -> List[Dict]:
        """
        Record live stream with zero-partial guarantee
        
        Args:
            channel: Channel dict with id, name, etc.
            duration_minutes: Target duration in minutes
            
        Returns:
            List of recorded tracks
        """
        print("="*80)
        print("🎯 ZERO-PARTIAL LIVE RECORDER")
        print("="*80)
        print()
        print(f"📻 Channel: {channel['name']} (Ch {channel.get('number', '?')})")
        print(f"⏱️  Target Duration: {duration_minutes} minutes")
        print(f"📁 Output: {self.output_dir}")
        print(f"🎚️  Quality: {quality}")
        print()
        
        # Remember current channel for token refreshes
        self.channel_info = channel
        
        # Step 1: Create output directories early (for channel art)
        session_name = datetime.now().strftime('%Y-%m-%d_%H-%M')
        channel_dir = self.output_dir / channel['name'] / session_name
        temp_dir = channel_dir / 'temp'
        tracks_dir = channel_dir / 'tracks'
        cover_art_dir = channel_dir / 'cover_art'
        channel_art_dir = channel_dir / 'channel_art'
        
        for d in [channel_dir, temp_dir, tracks_dir, cover_art_dir, channel_art_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Step 2: Get bearer token from browser AND capture channel art!
        print("🌐 Capturing bearer token from browser...")
        
        if not self.browser_session:
            print("❌ No browser session available")
            return []
        
        channel_url = f"https://www.siriusxm.com/player/channel-linear/entity/{channel['id']}"
        bearer_token, _ = self.browser_session.get_stream_info(
            channel_url, 
            channel, 
            channel_art_dir,
            progress_callback=self.progress_callback  # Pass callback through!
        )
        
        if not bearer_token:
            print("❌ Failed to capture bearer token")
            if self.progress_callback:
                self.progress_callback("❌ Failed to capture authentication")
            return []
        
        self.bearer_token = bearer_token
        
        # Update API with fresh bearer token
        self.api.bearer_token = bearer_token
        self.api.headers['Authorization'] = f'Bearer {bearer_token}'
        print(f"✅ Got bearer token")
        if self.progress_callback:
            self.progress_callback("✅ Authentication captured!")
        print()
        
        # Step 3: Fetch schedule
        print("📡 Fetching track schedule from API...")
        if self.progress_callback:
            self.progress_callback("📡 Fetching schedule...")
        
        schedule = self.api.get_schedule(channel['id'])
        if not schedule and self.api.last_schedule_status == 401:
            print("⚠️  Schedule auth expired - refreshing token...")
            if self._refresh_bearer_token() and self.progress_callback:
                self.progress_callback("🔄 Refreshed authentication, retrying schedule fetch...")
            if schedule := self.api.get_schedule(channel['id']):
                print("✅ Schedule fetch succeeded after refresh")
        
        if not schedule:
            print("❌ Could not get schedule")
            return []
        
        print(f"✅ Got {len(schedule)} tracks from schedule")
        if self.progress_callback:
            self.progress_callback(f"✅ Found {len(schedule)} tracks in schedule")
        
        # Step 3: Identify current track
        current_track = self.api.get_current_track(schedule)
        
        if not current_track:
            print("❌ Could not identify current track")
            return []
        
        # Calculate track timing info
        track_start = datetime.fromisoformat(current_track['timestamp_utc'].replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        time_since_start = (now - track_start).total_seconds()
        minutes_ago = int(time_since_start // 60)
        seconds_ago = int(time_since_start % 60)
        
        # Get track duration if available
        track_duration = current_track.get('duration', 0)
        duration_mins = int(track_duration // 60)
        duration_secs = int(track_duration % 60)
        
        print()
        # Convert to local timezone for display
        local_start = track_start.astimezone()
        
        print(f"🎵 Currently Playing:")
        print(f"   {current_track['artist']} - {current_track['title']}")
        print(f"   Started: {local_start.strftime('%I:%M:%S %p %Z')} ({minutes_ago}m {seconds_ago}s ago)")
        if track_duration > 0:
            print(f"   Duration: {duration_mins}:{duration_secs:02d}")
        
        # Update UI with current track
        if self.track_callback:
            track_info = f"{current_track['artist']} - {current_track['title']}\n"
            # Show when it started + duration
            track_info += f"Started: {minutes_ago}m {seconds_ago}s ago at {track_start.strftime('%H:%M:%S')}"
            if track_duration > 0:
                track_info += f"\nDuration: {duration_mins}:{duration_secs:02d}"
                # Calculate where we are in the track
                progress_seconds = time_since_start
                if progress_seconds < track_duration:
                    remaining = track_duration - progress_seconds
                    remaining_mins = int(remaining // 60)
                    remaining_secs = int(remaining % 60)
                    track_info += f" (will be complete in ~{remaining_mins}m {remaining_secs}s)"
            
            print(f"🎵 Calling track_callback with: {track_info}")
            try:
                self.track_callback(track_info)
            except Exception as e:
                print(f"⚠️  Track callback error: {e}")
        
        print()
        
        # Step 4: Calculate zero-partial recording window
        if self.progress_callback:
            self.progress_callback("📋 Planning recording window...")
        
        recording_plan = self._calculate_recording_window(
            schedule, current_track, duration_minutes
        )
        
        if not recording_plan:
            print("❌ Could not calculate recording window")
            return []
        
        # Show track count in UI
        if self.progress_callback:
            track_count = recording_plan['track_count']
            self.progress_callback(f"📋 Will record {track_count} track{'s' if track_count > 1 else ''}")
        
        # Show planned tracks list in UI immediately!
        if self.track_callback and recording_plan['tracks']:
            tracks_preview = ""
            for i, track in enumerate(recording_plan['tracks'][:10], 1):  # Show first 10
                artist = track.get('artist', 'Unknown')
                title = track.get('title', 'Unknown')
                tracks_preview += f"{i}. {artist} - {title}\n"
            if len(recording_plan['tracks']) > 10:
                tracks_preview += f"... and {len(recording_plan['tracks']) - 10} more\n"
            
            try:
                self.track_callback(tracks_preview)  # Send to UI!
                print(f"📋 Sent {len(recording_plan['tracks'])} planned tracks to UI")
            except Exception as e:
                print(f"⚠️  Failed to send tracks list: {e}")
        
        # Step 5: Display plan
        self._display_recording_plan(recording_plan)
        print()
        
        # Step 6: Get DVR stream URL with start timestamp!
        track_start_time = recording_plan['start_time']
        print(f"🕐 Getting DVR stream starting from {track_start_time.strftime('%H:%M:%S')} UTC...")
        if self.progress_callback:
            self.progress_callback("🕐 Getting DVR stream URL...")
        
        # Use API to get DVR stream with startTimestamp parameter
        # Convert datetime to ISO string format for API
        start_timestamp_str = track_start_time.isoformat().replace('+00:00', 'Z')
        dvr_master_url = self.api.get_stream_url(channel['id'], start_timestamp=start_timestamp_str)
        
        if not dvr_master_url:
            print(f"❌ Failed to get DVR stream URL")
            return []
        
        print(f"✅ Got DVR stream starting from {recording_plan['start_time'].strftime('%H:%M:%S')} UTC")
        print(f"✅ Got DVR stream starting from {recording_plan['start_time'].strftime('%H:%M:%S')}")
        
        # Step 7: Update HLS downloader with bearer token  
        self.hls_downloader.bearer_token = bearer_token
        
        # Get variant URL (256k quality) from DVR stream
        print(f"📺 Getting {quality} variant from DVR stream...")
        self.variant_url = self.hls_downloader.get_variant_url(dvr_master_url, quality)
        
        if not self.variant_url:
            print("❌ Failed to get variant URL")
            return []
        
        print(f"✅ Got {quality} DVR stream (has segments from track beginning!)\n")
        
        # Step 8: Download and process
        print("🔴 Starting recording...")
        if self.progress_callback:
            self.progress_callback("🔴 Starting download...")
        print("="*80)
        print()
        
        # Directories already created earlier
        
        # Record using proven method
        continuous_file = temp_dir / 'continuous.aac'
        
        # CRITICAL: Set is_recording flag and store tracking info!
        self.is_recording = True
        self.recording_start_time = recording_plan['start_time']
        self.actual_audio_start_time = None
        self.track_changes = recording_plan['tracks']
        self.current_schedule = schedule  # Store for live monitoring
        self.channel_id = channel['id']  # Store for schedule refresh
        
        success = self._record_timerange(
            recording_plan['start_time'],
            recording_plan['end_time'],
            continuous_file,
            temp_dir,
            schedule  # Pass schedule for live monitoring
        )
        
        self.is_recording = False
        
        if not success:
            print("❌ Recording failed")
            if self.progress_callback:
                self.progress_callback("❌ Recording failed - no segments downloaded")
            return []
        
        print(f"✅ Recording complete!\n")
        
        # Split into individual tracks with metadata and cover art
        if continuous_file.exists():
            file_size = continuous_file.stat().st_size / (1024 * 1024)
            print(f"📁 Continuous file created: {file_size:.1f} MB")
            print(f"🔪 Splitting into {len(self.track_changes)} tracks...")
            
            if self.progress_callback:
                self.progress_callback(f"🔪 Splitting {len(self.track_changes)} tracks...")
            
            split_and_tag_tracks(
                continuous_file, 
                tracks_dir, 
                cover_art_dir, 
                channel, 
                self.track_changes,
                self.actual_audio_start_time or self.recording_start_time,
                self.audio_recorder
            )
            
            # Create playlist
            create_playlist(tracks_dir, channel, session_name, self.track_changes)
            
            # Count actual track files created
            track_files = sorted(list(tracks_dir.glob('*.m4a')))
            print(f"✅ Created {len(track_files)} track files!")
            print(f"📂 Files: {[f.name for f in track_files]}")
            
            if self.progress_callback:
                self.progress_callback(f"✅ Created {len(track_files)} tracks!")
            
            # Return track info with file paths - match by index
            result_tracks = []
            for i, (track, track_file) in enumerate(zip(self.track_changes, track_files), 1):
                result_tracks.append({
                    **track,
                    'file_path': str(track_file),
                    'file_size': track_file.stat().st_size,
                    'track_number': i
                })
            
            # Fallback: if no files but we have track metadata, return metadata only
            if not result_tracks and self.track_changes:
                print(f"⚠️  No track files found, but returning {len(self.track_changes)} track metadata entries")
                for i, track in enumerate(self.track_changes, 1):
                    result_tracks.append({
                        **track,
                        'track_number': i
                    })
            
            print(f"📋 Returning {len(result_tracks)} track metadata entries")
            print(f"🔍 DEBUG: self.track_changes had {len(self.track_changes)} tracks")
            print(f"🔍 DEBUG: track_files had {len(track_files)} files")
            print(f"🔍 DEBUG: result_tracks = {result_tracks}")
            print(f"🔍 DEBUG: Returning from record_live() with {len(result_tracks)} tracks")
            return result_tracks
        else:
            print(f"❌ Continuous file not found: {continuous_file}")
            if self.progress_callback:
                self.progress_callback("❌ Recording failed - no audio file created")
            print(f"🔍 DEBUG: Returning empty list from record_live()")
            return []
    
    def record_dvr(self, channel: Dict, start_time: datetime, duration_minutes: int,
                   selected_tracks: Optional[List[Dict]] = None,
                   quality: str = '256k') -> List[Dict]:
        """
        Record from DVR buffer starting at a specific time
        
        Args:
            channel: Channel dict with id, name, etc.
            start_time: When to start recording (datetime object with timezone)
            duration_minutes: How long to record
            
        Returns:
            List of recorded tracks
        """
        logger.info("="*80)
        logger.info("📺 DVR RECORDER")
        logger.info("="*80)
        logger.info(f"📻 Channel: {channel['name']}")
        logger.info(f"⏰ Start Time: {start_time.isoformat()}")
        logger.info(f"⏱️  Duration: {duration_minutes} minutes")
        
        print("="*80)
        print("📺 DVR RECORDER")
        print("="*80)
        print(f"📻 Channel: {channel['name']}")
        print(f"⏰ Start: {start_time.astimezone().strftime('%I:%M %p')}")
        print(f"⏱️  Duration: {duration_minutes} minutes")
        print(f"🎚️  Quality: {quality}")
        print()
        
        # Create output directories
        session_name = datetime.now().strftime('%Y-%m-%d_%H-%M')
        channel_dir = self.output_dir / channel['name'] / f"DVR-{session_name}"
        temp_dir = channel_dir / 'temp'
        tracks_dir = channel_dir / 'tracks'
        cover_art_dir = channel_dir / 'cover_art'
        channel_art_dir = channel_dir / 'channel_art'
        
        for d in [channel_dir, temp_dir, tracks_dir, cover_art_dir, channel_art_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Get authentication
        print("🌐 Getting authentication...")
        if self.progress_callback:
            self.progress_callback("🌐 Getting authentication...")
        
        if not self.browser_session:
            print("❌ No browser session available")
            return []
        
        # Remember channel for token refreshes
        self.channel_info = channel
        
        channel_url = f"https://www.siriusxm.com/player/channel-linear/entity/{channel['id']}"
        bearer_token, _ = self.browser_session.get_stream_info(
            channel_url, 
            channel, 
            channel_art_dir,
            progress_callback=self.progress_callback
        )
        
        if not bearer_token:
            print("❌ Failed to get bearer token")
            return []
        
        self.bearer_token = bearer_token
        self.api.bearer_token = bearer_token
        self.api.headers['Authorization'] = f'Bearer {bearer_token}'
        self.hls_downloader.bearer_token = bearer_token
        
        print("✅ Authenticated")
        if self.progress_callback:
            self.progress_callback("✅ Authenticated!")
        
        # Get DVR stream URL for the specific start time
        print(f"🕐 Getting DVR stream starting from {start_time.astimezone().strftime('%I:%M %p')}...")
        if self.progress_callback:
            self.progress_callback(f"🕐 Getting DVR stream...")
        
        # Format timestamp for API
        start_timestamp = start_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        
        stream_url = self.api.get_stream_url(channel['id'], start_timestamp=start_timestamp)
        
        if not stream_url:
            print("❌ Could not get DVR stream URL")
            return []
        
        print(f"✅ Got DVR stream starting from {start_time.astimezone().strftime('%I:%M:%S %p')}")
        if self.progress_callback:
            self.progress_callback(f"✅ Got DVR stream starting from {start_time.astimezone().strftime('%I:%M %p')}")
        
        # Get variant URL from DVR stream
        print(f"📺 Getting {quality} variant from DVR stream...")
        if self.progress_callback:
            self.progress_callback("📺 Getting variant...")
        
        self.variant_url = self.hls_downloader.get_variant_url(stream_url, quality)
        
        if not self.variant_url:
            print("❌ Failed to get variant URL")
            return []
        
        print(f"✅ Got {quality} DVR stream\n")
        if self.progress_callback:
            self.progress_callback("✅ Got stream variant")
        
        # Get schedule for metadata
        logger.info("📡 Fetching schedule for metadata...")
        if self.progress_callback:
            self.progress_callback("📡 Fetching schedule...")
        
        schedule = self.api.get_schedule(channel['id'])
        
        if not schedule:
            logger.warning("⚠️  Could not get schedule, will record without track splitting")
            print("⚠️  Could not get schedule")
            schedule = []
        else:
            logger.info(f"✅ Got schedule with {len(schedule)} tracks")
            print(f"✅ Got {len(schedule)} tracks from schedule")
        
        # Record using the same continuous download method
        logger.info("🔴 Starting DVR download...")
        print("\n🔴 Starting DVR download...")
        print("="*80)
        
        if self.progress_callback:
            self.progress_callback("🔴 Starting download...")
        
        # Store channel ID for schedule refresh
        self.channel_id = channel['id']
        
        # Reset track tracking
        # DVR downloads should only include the tracks requested – do NOT expand beyond selection
        self.track_changes = list(selected_tracks or [])
        self.recording_start_time = start_time
        self.actual_audio_start_time = None
        self.is_recording = True
        self.current_schedule = schedule  # Store for live monitoring/display
        self.allow_track_expansion = False
        
        # Calculate end time
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        # Download from DVR stream
        continuous_file = channel_dir / 'continuous.aac'
        
        success = self._record_timerange(
            start_time,
            end_time,
            continuous_file,
            temp_dir,
            schedule,
            hard_end_time=end_time
        )

        self.is_recording = False
        
        if not success or not continuous_file.exists():
            print("❌ DVR download failed")
            return []
        
        print(f"\n✅ Downloaded {continuous_file.stat().st_size / (1024*1024):.1f} MB")
        logger.info(f"✅ Downloaded {continuous_file.stat().st_size / (1024*1024):.1f} MB")
        
        # Split into tracks if we have track metadata
        if self.track_changes:
            logger.info(f"🎵 Splitting into {len(self.track_changes)} tracks...")
            print(f"🎵 Splitting into {len(self.track_changes)} tracks...")
            
            if self.progress_callback:
                self.progress_callback(f"🎵 Splitting into {len(self.track_changes)} tracks...")
            
            split_and_tag_tracks(
                continuous_file,
                tracks_dir,
                cover_art_dir,
                channel,
                self.track_changes,
                self.actual_audio_start_time or self.recording_start_time,
                self.audio_recorder
            )
            
            # Return track info with file paths
            track_files = sorted(list(tracks_dir.glob('*.m4a')))
            logger.info(f"✅ Created {len(track_files)} track files!")
            
            if self.progress_callback:
                self.progress_callback(f"✅ Created {len(track_files)} tracks!")
            
            result_tracks = []
            for i, (track, track_file) in enumerate(zip(self.track_changes, track_files), 1):
                result_tracks.append({
                    **track,
                    'file_path': str(track_file),
                    'file_size': track_file.stat().st_size,
                    'track_number': i
                })
            
            logger.info(f"📋 Returning {len(result_tracks)} track metadata entries")
            return result_tracks
        else:
            logger.warning("⚠️  No track metadata, continuous file only")
            print("⚠️  No track metadata found")
            return []
    
    def _calculate_recording_window(self, schedule: List[Dict], 
                                    current_track: Dict, 
                                    duration_minutes: int) -> Optional[Dict]:
        """
        Calculate recording window - SIMPLE WORKING LOGIC
        
        Just record from current track beginning for the duration!
        DVR buffer has the segments, we'll get them.
        """
        # Start from current track beginning (DVR buffer has it!)
        start_time = datetime.fromisoformat(current_track['timestamp_utc'].replace('Z', '+00:00'))
        
        # Check if we joined midway
        now = datetime.now(timezone.utc)
        time_since_start = (now - start_time).total_seconds()
        
        if time_since_start > 10:
            print(f"   ⏪ Joined {int(time_since_start)}s into track - will get from DVR buffer")
        else:
            print(f"   ✅ Track just started ({int(time_since_start)}s ago)")
        
        # Initial end time: duration from NOW
        target_end_time = now + timedelta(minutes=duration_minutes)
        
        # Collect all tracks in this window
        # ALWAYS include current track first!
        tracks_to_record = [current_track]
        
        # Find any tracks that START during our recording window (NOW to NOW+duration)
        for track in schedule:
            track_start = datetime.fromisoformat(track['timestamp_utc'].replace('Z', '+00:00'))
            
            # Skip current track (already added)
            if track['timestamp_utc'] == current_track['timestamp_utc']:
                continue
            
            # Include track if it starts during our recording window (from NOW onwards)
            if now <= track_start <= target_end_time:
                tracks_to_record.append(track)
                print(f"   ➕ Found track starting during recording: {track['artist']} - {track['title']}")
                print(f"      Starts at: {track_start.strftime('%H:%M:%S')}")
        
        # CRITICAL: Extend to end of last track for zero partials!
        last_track = tracks_to_record[-1]
        last_track_start = datetime.fromisoformat(last_track['timestamp_utc'].replace('Z', '+00:00'))
        last_track_duration = last_track.get('duration', 0)
        last_track_end = last_track_start + timedelta(seconds=last_track_duration)
        
        # Extend recording end time to complete the last track
        if last_track_end > target_end_time and last_track_duration > 0:
            extension_seconds = (last_track_end - target_end_time).total_seconds()
            print(f"   ⏱️  Extending recording by {int(extension_seconds)}s to complete last track")
            end_time = last_track_end
        else:
            end_time = target_end_time
        
        print(f"   📋 Will record {len(tracks_to_record)} tracks")
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'target_duration_minutes': duration_minutes,
            'actual_duration_seconds': (end_time - start_time).total_seconds(),
            'tracks': tracks_to_record,
            'track_count': len(tracks_to_record)
        }
    
    def _display_recording_plan(self, plan: Dict):
        """Display recording plan to user"""
        print("📋 RECORDING PLAN:")
        print("="*80)
        print()
        
        print(f"🎯 Recording Window:")
        print(f"   Start: {plan['start_time'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"   End:   {plan['end_time'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"   Duration: {plan['actual_duration_seconds'] / 60:.1f} minutes")
        print()
        
        print(f"🎵 Tracks to Record: {plan['track_count']} tracks")
        if plan['tracks']:
            print(f"   First: {plan['tracks'][0]['artist']} - {plan['tracks'][0]['title']}")
            if len(plan['tracks']) > 1:
                print(f"   Last:  {plan['tracks'][-1]['artist']} - {plan['tracks'][-1]['title']}")
        print()
        
        print(f"✅ Zero Partials Guarantee:")
        print(f"   ✓ All tracks will be COMPLETE")
        print(f"   ✓ No partial first track (starting from track beginning!)")
        print(f"   ✓ No partial last track (ending at track end!)")
        print()

    def _refresh_bearer_token(self) -> bool:
        """Refresh bearer token using browser session and existing cookies."""
        if not self.browser_session:
            logger.error("Cannot refresh bearer token: no browser session available")
            return False
        if not self.channel_info:
            logger.error("Cannot refresh bearer token: channel info unknown")
            return False
        try:
            channel = self.channel_info
            channel_url = f"https://www.siriusxm.com/player/channel-linear/entity/{channel['id']}"
            print("🔄 Refreshing authentication via browser...")
            new_bearer, _ = self.browser_session.get_stream_info(
                channel_url,
                channel,
                art_dir=None,
                progress_callback=None
            )
            if not new_bearer:
                logger.error("Failed to capture new bearer token during refresh")
                return False
            self.bearer_token = new_bearer
            self.api.bearer_token = new_bearer
            self.api.headers['Authorization'] = f'Bearer {new_bearer}'
            self.hls_downloader.bearer_token = new_bearer
            print("✅ Bearer token refreshed")
            if self.progress_callback:
                try:
                    self.progress_callback("✅ Authentication refreshed")
                except Exception:
                    pass
            return True
        except Exception as exc:
            logger.error(f"Error refreshing bearer token: {exc}", exc_info=True)
            return False
        print("="*80)
        print()
    
    def _record_timerange(self, start_time: datetime, end_time: datetime,
                         output_file: Path, temp_dir: Path,
                         schedule: List[Dict] = None,
                         hard_end_time: Optional[datetime] = None) -> bool:
        """
        Record specific time range using DVR buffer
        
        WORKING CODE from v5_PURE_API - Downloads continuously!
        
        Args:
            start_time: Start time (UTC)
            end_time: End time (UTC)  
            output_file: Output file path
            temp_dir: Temporary directory
            schedule: Optional schedule for live track monitoring
            
        Returns:
            True if successful
        """
        try:
            print(f"🔴 Starting continuous download...")
            print(f"   Duration: {(end_time - start_time).total_seconds() / 60:.1f} minutes\n")
            requested_start_time = start_time if start_time.tzinfo else start_time.replace(tzinfo=timezone.utc)
            buffered_start_time = requested_start_time - timedelta(seconds=5)
            effective_end_time = hard_end_time or end_time
            if effective_end_time.tzinfo is None:
                effective_end_time = effective_end_time.replace(tzinfo=timezone.utc)
            buffered_end_time = effective_end_time + timedelta(seconds=5)
            strict_window = hard_end_time is not None
            
            # Get decryption key first
            response = requests.get(self.variant_url, timeout=10)
            if response.status_code != 200:
                print(f"❌ Failed to get playlist")
                return False
            
            variant_playlist = response.text
            
            print(f"🔐 Getting decryption key...")
            key_hex = self.hls_downloader.get_decryption_key(variant_playlist)
            
            if not key_hex:
                print("❌ Failed to get decryption key")
                return False
            
            # Download and decrypt segments (FIXED CONTINUOUS LOOP!)
            decrypted_files = []
            duration = int((effective_end_time - requested_start_time).total_seconds())
            start = time.time()
            end = start + duration
            segment_index = 0
            processed_urls = set()
            first_download = True
            last_seen_time = None  # Track latest segment time seen
            downloaded_duration = 0.0
            
            print(f"📥 Downloading segments...")
            
            last_track_check = time.time()
            last_schedule_refresh = time.time()
            current_displayed_track = None
            initial_track_count = len(self.track_changes)
            
            import logging
            logger = logging.getLogger(__name__)
            
            logger.info(f"\n🔍 DEBUG: Starting recording loop:")
            logger.info(f"   channel_id = {self.channel_id}")
            logger.info(f"   initial_track_count = {initial_track_count}")
            logger.info(f"   duration = {duration}s")
            logger.info(f"   end time = {datetime.fromtimestamp(end, tz=timezone.utc).strftime('%H:%M:%S')}")
            
            if not self.channel_id:
                logger.warning(f"\n⚠️  WARNING: self.channel_id is None - schedule refresh will NOT work!")
            else:
                logger.info(f"\n✅ channel_id is set - schedule refresh enabled every 20 seconds")
            
            # Add 30 second buffer to ensure last track is fully captured
            end_with_buffer = end + 30
            
            stop_download = False
            while time.time() < end_with_buffer and self.is_recording and not stop_download:
                # Refresh schedule every 20 seconds to detect new tracks
                if self.allow_track_expansion and self.channel_id and (time.time() - last_schedule_refresh) >= 20:
                    last_schedule_refresh = time.time()
                    elapsed = time.time() - start
                    logger.info(f"\n⏰ {elapsed:.0f}s elapsed - Time to refresh schedule!")
                    logger.info(f"   self.channel_id = {self.channel_id}")
                    try:
                        # Fetch fresh schedule with new tracks
                        logger.info(f"🔄 Refreshing schedule (currently have {len(schedule)} tracks, detected {len(self.track_changes)} for recording)...")
                        fresh_schedule = self.api.get_schedule(self.channel_id)
                        if not fresh_schedule and self.api.last_schedule_status == 401:
                            logger.warning("   ⚠️  Schedule refresh returned 401 - refreshing bearer token...")
                            if self._refresh_bearer_token():
                                fresh_schedule = self.api.get_schedule(self.channel_id)
                            else:
                                logger.error("   ❌ Failed to refresh bearer token; skipping schedule refresh this cycle")
                                continue
                        logger.info(f"   📊 Fresh schedule has {len(fresh_schedule)} tracks")
                        
                        if fresh_schedule:
                            if len(fresh_schedule) != len(schedule):
                                logger.info(f"   📈 Schedule changed from {len(schedule)} to {len(fresh_schedule)} tracks")
                            else:
                                logger.info(f"   ℹ️  Schedule size unchanged ({len(schedule)} tracks) - still checking for new tracks...")
                            
                            schedule = fresh_schedule
                            
                            # ALWAYS check for NEW tracks that started during recording (schedule content may change even if size doesn't)
                            now_utc = datetime.now(timezone.utc)
                            logger.info(f"   🔍 Checking for new tracks since recording start: {self.recording_start_time.strftime('%H:%M:%S')}")
                            tracks_checked = 0
                            tracks_in_window = 0
                            for track in schedule:
                                track_start = datetime.fromisoformat(track['timestamp_utc'].replace('Z', '+00:00'))
                                tracks_checked += 1
                                # If track started after we began recording AND is not in our list yet
                                if track_start >= self.recording_start_time:
                                    tracks_in_window += 1
                                    track_id = f"{track['artist']}-{track['title']}"
                                    existing_ids = [f"{t['artist']}-{t['title']}" for t in self.track_changes]
                                    if track_id not in existing_ids:
                                        self.track_changes.append(track)
                                        logger.info(f"\n   ➥ NEW TRACK DETECTED: {track['artist']} - {track['title']}")
                                        logger.info(f"      Started at: {track_start.strftime('%H:%M:%S')}")
                                        logger.info(f"      🎵 Total tracks now: {len(self.track_changes)}")
                                        
                                        # Update UI with new track list
                                        if self.track_callback:
                                            try:
                                                tracks_preview = f"📋 Recording {len(self.track_changes)} tracks:\n"
                                                for i, t in enumerate(self.track_changes[-5:], len(self.track_changes)-4):  # Show last 5
                                                    if i > 0:
                                                        tracks_preview += f"{i}. {t.get('artist', 'Unknown')} - {t.get('title', 'Unknown')}\n"
                                                self.track_callback(tracks_preview)
                                            except:
                                                pass
                                        
                                        # Extend end time if needed to complete this new track
                                        track_duration = track.get('duration', 0)
                                        if track_duration > 0:
                                            track_end = track_start + timedelta(seconds=track_duration)
                                            if track_end > datetime.fromtimestamp(end, tz=timezone.utc):
                                                extension = (track_end - datetime.fromtimestamp(end, tz=timezone.utc)).total_seconds()
                                                logger.info(f"      ⏱️  EXTENDING recording by {int(extension)}s to complete track")
                                                end = track_end.timestamp()
                                                duration = int(end - start)  # Update duration
                                                logger.info(f"      🕒 New end time: {datetime.fromtimestamp(end, tz=timezone.utc).strftime('%H:%M:%S')}")
                                                
                                                # Update UI about extension
                                                if self.progress_callback:
                                                    try:
                                                        self.progress_callback(f"⏱️  Extended by {int(extension)}s to complete new track!")
                                                    except:
                                                        pass
                            logger.info(f"   📋 Checked {tracks_checked} tracks, {tracks_in_window} in recording window, {len(self.track_changes)} total for recording")
                        else:
                            logger.warning(f"   ⚠️  Failed to fetch fresh schedule!")
                    except Exception as e:
                        logger.error(f"   ⚠️  Schedule refresh failed: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                # Live track monitoring every 15 seconds
                if schedule and self.track_callback and (time.time() - last_track_check) >= 15:
                    last_track_check = time.time()
                    now_utc = datetime.now(timezone.utc)
                    
                    # Find current track in schedule
                    for track in schedule:
                        track_start = datetime.fromisoformat(track['timestamp_utc'].replace('Z', '+00:00'))
                        track_duration = track.get('duration', 300)  # Default 5 min
                        track_end = track_start + timedelta(seconds=track_duration)
                        
                        if track_start <= now_utc < track_end:
                            track_id = f"{track['artist']}-{track['title']}"
                            if track_id != current_displayed_track:
                                current_displayed_track = track_id
                                # New track! Update UI
                                elapsed = (now_utc - track_start).total_seconds()
                                remaining = track_duration - elapsed
                                mins_elapsed = int(elapsed // 60)
                                secs_elapsed = int(elapsed % 60)
                                mins_remaining = int(remaining // 60)
                                secs_remaining = int(remaining % 60)
                                dur_mins = int(track_duration // 60)
                                dur_secs = int(track_duration % 60)
                                
                                track_info = f"{track['artist']} - {track['title']}\n"
                                track_info += f"Currently playing ({mins_elapsed}m {secs_elapsed}s elapsed)\n"
                                track_info += f"Duration: {dur_mins}:{dur_secs:02d} | ~{mins_remaining}m {secs_remaining}s remaining"
                                
                                # Show how many tracks collected so far
                                if len(self.track_changes) > initial_track_count:
                                    track_info += f"\n\n📋 Tracks captured: {len(self.track_changes)} (including {len(self.track_changes) - initial_track_count} new)"
                                
                                try:
                                    self.track_callback(track_info)
                                except Exception as e:
                                    # Callback failed (UI might be closed), just log it
                                    pass
                            break
                
                # Refresh playlist
                response = requests.get(self.variant_url, timeout=10)
                if response.status_code != 200:
                    time.sleep(5)
                    continue
                
                variant_playlist = response.text
                segment_base = self.variant_url.rsplit('/', 1)[0] + '/'
                
                # Parse segments
                segment_data = []
                current_time = None
                lines = variant_playlist.split('\n')
                current_duration = None
                
                for line in lines:
                    # Parse timestamp
                    if line.startswith('#EXT-X-PROGRAM-DATE-TIME:'):
                        time_str = line.split(':', 1)[1].strip()
                        try:
                            current_time = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                        except:
                            pass
                    elif line.startswith('#EXTINF:'):
                        try:
                            duration_str = line.split(':', 1)[1].split(',')[0]
                            current_duration = float(duration_str)
                        except:
                            current_duration = 10.0
                    
                    # Get segment URL
                    elif line.strip() and not line.startswith('#') and '.aac' in line:
                        segment_url = segment_base + line.strip()
                        segment_data.append({
                            'url': segment_url,
                            'time': current_time,
                            'duration': current_duration or 10.0
                        })
                        current_time = None
                        current_duration = None
                
                # On first download, start from recording start time (with small buffer)
                if first_download and len(segment_data) > 0:
                    print(f"   Found {len(segment_data)} segments (DVR buffer)")
                    
                    # Make start_time timezone-aware if needed
                    if start_time.tzinfo is None:
                        start_time = start_time.replace(tzinfo=timezone.utc)
                        requested_start_time = start_time
                        buffered_start_time = requested_start_time - timedelta(seconds=5)
                    
                    # Keep segments within our recording window + small preroll buffer
                    filtered_segments = [seg for seg in segment_data 
                                         if seg['time'] and seg['time'] >= buffered_start_time]
                    if strict_window:
                        bounded_segments = [seg for seg in filtered_segments
                                            if seg['time'] and seg['time'] <= buffered_end_time]
                        if bounded_segments:
                            filtered_segments = bounded_segments
                    if filtered_segments:
                        segment_data = filtered_segments
                    
                    if segment_data:
                        print(f"   Starting from {len(segment_data)} segments at target time")
                        print(f"   Target start: {requested_start_time}")
                        # Track the latest time we've seen
                        for seg in segment_data:
                            if seg['time'] and (last_seen_time is None or seg['time'] > last_seen_time):
                                last_seen_time = seg['time']
                    else:
                        # Fallback: last 10 segments
                        segment_data = segment_data[-10:] if len(segment_data) > 10 else segment_data
                        print(f"   No timestamps, starting from last segments")
                    
                    first_download = False
                else:
                    # On subsequent downloads, only get NEW segments (after last_seen_time)
                    if last_seen_time:
                        new_segments = []
                        for seg in segment_data:
                            seg_time = seg.get('time')
                            if seg_time and seg_time <= last_seen_time:
                                continue
                            if strict_window and seg_time and seg_time > buffered_end_time:
                                continue
                            new_segments.append(seg)
                        
                        if new_segments:
                            print(f"   📊 Found {len(new_segments)} new segments...")
                        
                        segment_data = new_segments
                        
                        # Update last_seen_time
                        for seg in segment_data:
                            if seg['time'] and seg['time'] > last_seen_time:
                                last_seen_time = seg['time']
                
                # Download new segments
                for seg in segment_data:
                    segment_url = seg['url']
                    segment_time = seg.get('time')
                    seg_duration = seg.get('duration') or 10.0
                    if not self.is_recording:
                        break
                    if strict_window and segment_time and segment_time > buffered_end_time:
                        stop_download = True
                        break
                    
                    # Skip if already processed
                    if segment_url in processed_urls:
                        continue
                    
                    processed_urls.add(segment_url)
                    
                    enc_file = temp_dir / f"seg_{segment_index:04d}_enc.aac"
                    dec_file = temp_dir / f"seg_{segment_index:04d}_dec.aac"
                    
                    try:
                        # Download encrypted
                        seg_response = requests.get(segment_url, timeout=10)
                        if seg_response.status_code == 200:
                            with open(enc_file, 'wb') as f:
                                f.write(seg_response.content)
                            
                            # Decrypt
                            iv = f"{segment_index:032x}"
                            
                            cmd = [
                                'openssl', 'aes-128-cbc', '-d',
                                '-in', str(enc_file),
                                '-out', str(dec_file),
                                '-K', key_hex,
                                '-iv', iv
                            ]
                            
                            result = subprocess.run(cmd, capture_output=True)
                            
                            if result.returncode == 0 and dec_file.exists():
                                decrypted_files.append(str(dec_file))
                                enc_file.unlink()
                                segment_index += 1
                                if self.actual_audio_start_time is None and segment_time:
                                    self.actual_audio_start_time = segment_time
                                downloaded_duration += seg_duration
                                
                                # Progress - show every 10 segments
                                if segment_index % 10 == 0:
                                    elapsed = time.time() - start
                                    remaining = max(0, end - time.time())  # Don't show negative remaining
                                    actual_duration = end - start  # Use actual duration (may have been extended)
                                    percent = min(100, (elapsed / actual_duration) * 100)  # Cap at 100%
                                    print(f"   📊 {segment_index} segments | {elapsed:.0f}s / {actual_duration:.0f}s ({percent:.0f}%) | {remaining:.0f}s remaining")
                                if strict_window:
                                    reached_end = False
                                    if segment_time and segment_time >= buffered_end_time:
                                        reached_end = True
                                    elif downloaded_duration >= duration + 5:
                                        reached_end = True
                                    if reached_end:
                                        stop_download = True
                                        break
                            else:
                                # Decryption failed - log it
                                print(f"   ⚠️  Decrypt failed for seg_{segment_index:04d}")
                                if result.stderr:
                                    print(f"       Error: {result.stderr.decode()[:100]}")
                                if enc_file.exists():
                                    enc_file.unlink()
                    except Exception as e:
                        print(f"   ⚠️  Download failed for seg_{segment_index:04d}: {e}")
                        if enc_file.exists():
                            enc_file.unlink()
                if stop_download:
                    print("   ✅ Reached DVR target window; stopping download")
                    break
                
                time.sleep(5)  # Wait before refreshing playlist
            
            # Recording loop ended
            logger.info(f"\n📍 Recording loop ended:")
            logger.info(f"   Total segments: {len(decrypted_files)}")
            logger.info(f"   Tracks detected: {len(self.track_changes)}")
            if len(self.track_changes) > initial_track_count:
                logger.info(f"   📈 Found {len(self.track_changes) - initial_track_count} NEW tracks during recording!")
            
            # Combine segments
            # Fallback: If decrypted_files list is empty, scan temp directory
            if not decrypted_files:
                print("\n⚠️  decrypted_files list empty, scanning temp directory...")
                decrypted_files = sorted([str(f) for f in temp_dir.glob('seg_*_dec.aac')])
                if decrypted_files:
                    print(f"   Found {len(decrypted_files)} segments on disk!")
            
            if decrypted_files:
                print(f"\n✅ Downloaded {len(decrypted_files)} segments")
                print(f"🔗 Combining segments...")
                
                with open(output_file, 'wb') as outfile:
                    for dec_file in sorted(decrypted_files):
                        with open(dec_file, 'rb') as infile:
                            outfile.write(infile.read())
                        Path(dec_file).unlink()
                
                file_size = output_file.stat().st_size / (1024 * 1024)
                print(f"✅ Created continuous file ({file_size:.1f} MB)\n")
                if self.actual_audio_start_time is None:
                    self.actual_audio_start_time = requested_start_time
                return True
            else:
                print("❌ No segments downloaded")
                return False
        
        except Exception as e:
            print(f"❌ Recording error: {e}")
            import traceback
            traceback.print_exc()
            return False
