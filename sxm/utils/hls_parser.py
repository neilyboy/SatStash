#!/usr/bin/env python3
"""
HLS Playlist Parser

Parses SiriusXM HLS variant playlists to extract segment information
including exact UTC timestamps from #EXT-X-PROGRAM-DATE-TIME tags.

This enables:
- Bit-perfect DVR downloads
- Bulk time-range operations
- Zero partial tracks
- Perfect track splitting
"""

import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional


def parse_playlist_segments(playlist_text: str) -> List[Dict]:
    """
    Parse HLS variant playlist into segments with timestamps
    
    Args:
        playlist_text: HLS m3u8 playlist content
    
    Returns:
        List of segment dictionaries:
        [
            {
                'timestamp': datetime (UTC),
                'duration': float (seconds),
                'url': str (segment URL)
            },
            ...
        ]
    
    Example:
        >>> playlist = Path('variant.m3u8').read_text()
        >>> segments = parse_playlist_segments(playlist)
        >>> print(f"Found {len(segments)} segments")
        >>> print(f"First: {segments[0]['timestamp']}")
    """
    segments = []
    current_timestamp = None
    current_duration = None
    
    for line in playlist_text.split('\n'):
        line = line.strip()
        
        # Parse timestamp: #EXT-X-PROGRAM-DATE-TIME:2025-11-22T07:03:39.932+00:00
        if line.startswith('#EXT-X-PROGRAM-DATE-TIME:'):
            timestamp_str = line.replace('#EXT-X-PROGRAM-DATE-TIME:', '')
            # Handle both Z and +00:00 timezone formats
            timestamp_str = timestamp_str.replace('Z', '+00:00')
            try:
                current_timestamp = datetime.fromisoformat(timestamp_str)
            except ValueError:
                # Fallback: try parsing without timezone
                try:
                    current_timestamp = datetime.fromisoformat(timestamp_str.rstrip('Z'))
                    current_timestamp = current_timestamp.replace(tzinfo=timezone.utc)
                except:
                    continue
        
        # Parse duration: #EXTINF:9.752381,
        elif line.startswith('#EXTINF:'):
            match = re.search(r'#EXTINF:([0-9.]+)', line)
            if match:
                current_duration = float(match.group(1))
        
        # Segment URL (non-comment line)
        elif line and not line.startswith('#'):
            if current_timestamp and current_duration:
                segments.append({
                    'timestamp': current_timestamp,
                    'duration': current_duration,
                    'url': line
                })
    
    return segments


def filter_segments_for_track(segments: List[Dict], track: Dict) -> List[Dict]:
    """
    Filter segments that belong to a specific track
    
    Args:
        segments: List from parse_playlist_segments()
        track: Track dictionary with keys:
            - timestamp_utc: ISO format timestamp string
            - duration_ms: Duration in milliseconds
    
    Returns:
        Filtered list of segments for this track's time window
    
    Example:
        >>> track = {
        ...     'timestamp_utc': '2025-11-22T07:05:00.000Z',
        ...     'duration_ms': 240000  # 4 minutes
        ... }
        >>> track_segments = filter_segments_for_track(all_segments, track)
        >>> print(f"Track needs {len(track_segments)} segments")
    """
    # Parse track times
    start_time_str = track['timestamp_utc'].replace('Z', '+00:00')
    start_time = datetime.fromisoformat(start_time_str)
    end_time = start_time + timedelta(milliseconds=track['duration_ms'])
    
    # Filter segments in time window
    track_segments = [
        seg for seg in segments
        if start_time <= seg['timestamp'] < end_time
    ]
    
    return track_segments


def filter_segments_for_time_range(segments: List[Dict], hours_back: int) -> List[Dict]:
    """
    Filter segments for a time range (e.g., past 3 hours)
    
    Args:
        segments: List from parse_playlist_segments()
        hours_back: Number of hours back from now
    
    Returns:
        Filtered list of segments within time range
    
    Example:
        >>> past_3_hours = filter_segments_for_time_range(all_segments, 3)
        >>> print(f"Found {len(past_3_hours)} segments in past 3 hours")
    """
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    
    range_segments = [
        seg for seg in segments
        if seg['timestamp'] >= cutoff_time
    ]
    
    return range_segments


def get_segment_at_time(segments: List[Dict], target_time: datetime) -> Optional[Dict]:
    """
    Get the segment playing at a specific time
    
    Args:
        segments: List from parse_playlist_segments()
        target_time: datetime (UTC) to search for
    
    Returns:
        Segment dict or None if not found
    
    Example:
        >>> target = datetime.now(timezone.utc) - timedelta(hours=2)
        >>> segment = get_segment_at_time(all_segments, target)
        >>> if segment:
        ...     print(f"At that time: {segment['url']}")
    """
    for seg in segments:
        seg_end = seg['timestamp'] + timedelta(seconds=seg['duration'])
        if seg['timestamp'] <= target_time < seg_end:
            return seg
    return None


def calculate_track_segment_count(track_duration_ms: int, avg_segment_duration: float = 9.75) -> int:
    """
    Calculate expected number of segments for a track
    
    Args:
        track_duration_ms: Track duration in milliseconds
        avg_segment_duration: Average segment duration in seconds (default 9.75)
    
    Returns:
        Expected segment count
    
    Example:
        >>> count = calculate_track_segment_count(245000)  # 4:05 track
        >>> print(f"Expected ~{count} segments")  # ~25
    """
    track_duration_sec = track_duration_ms / 1000
    return int(track_duration_sec / avg_segment_duration) + 1


def get_dvr_time_range(segments: List[Dict]) -> tuple:
    """
    Get the start and end times of the DVR buffer
    
    Args:
        segments: List from parse_playlist_segments()
    
    Returns:
        Tuple of (start_time, end_time, duration_hours)
    
    Example:
        >>> start, end, hours = get_dvr_time_range(segments)
        >>> print(f"DVR buffer: {hours:.1f} hours")
    """
    if not segments:
        return None, None, 0
    
    start_time = segments[0]['timestamp']
    last_seg = segments[-1]
    end_time = last_seg['timestamp'] + timedelta(seconds=last_seg['duration'])
    
    duration = end_time - start_time
    duration_hours = duration.total_seconds() / 3600
    
    return start_time, end_time, duration_hours


def extract_key_url(playlist_text: str) -> Optional[str]:
    """
    Extract AES-128 key URL from playlist
    
    Args:
        playlist_text: HLS m3u8 playlist content
    
    Returns:
        Key URL or None
    
    Example:
        >>> key_url = extract_key_url(playlist)
        >>> print(f"Key URL: {key_url}")
    """
    for line in playlist_text.split('\n'):
        if '#EXT-X-KEY:' in line:
            match = re.search(r'URI="([^"]+)"', line)
            if match:
                return match.group(1)
    return None


def validate_playlist(playlist_text: str) -> Dict:
    """
    Validate HLS playlist and return stats
    
    Args:
        playlist_text: HLS m3u8 playlist content
    
    Returns:
        Dictionary with validation results:
        {
            'valid': bool,
            'segment_count': int,
            'has_timestamps': bool,
            'has_key': bool,
            'dvr_hours': float,
            'errors': List[str]
        }
    """
    errors = []
    
    # Check basic structure
    if not playlist_text.startswith('#EXTM3U'):
        errors.append("Missing #EXTM3U header")
    
    segments = parse_playlist_segments(playlist_text)
    
    # Check for timestamps
    has_timestamps = all('timestamp' in seg for seg in segments)
    if not has_timestamps:
        errors.append("Missing timestamps on some segments")
    
    # Check for encryption key
    has_key = extract_key_url(playlist_text) is not None
    if not has_key:
        errors.append("Missing encryption key URL")
    
    # Calculate DVR duration
    if segments:
        _, _, dvr_hours = get_dvr_time_range(segments)
    else:
        dvr_hours = 0
        errors.append("No segments found")
    
    return {
        'valid': len(errors) == 0,
        'segment_count': len(segments),
        'has_timestamps': has_timestamps,
        'has_key': has_key,
        'dvr_hours': dvr_hours,
        'errors': errors
    }
