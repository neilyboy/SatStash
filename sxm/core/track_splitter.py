#!/usr/bin/env python3
"""
Track Splitting Module for Zero-Partial Recorder
Splits continuous recordings into individual tracks with metadata and cover art
"""

import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from sxm.utils.audio_recorder import AudioRecorder


def split_and_tag_tracks(continuous_file: Path, tracks_dir: Path, cover_art_dir: Path,
                         channel: Dict, track_changes: List[Dict], 
                         recording_start_time: datetime, audio_recorder: AudioRecorder):
    """
    Split continuous recording at exact timestamps and tag with metadata
    
    Args:
        continuous_file: Path to continuous .aac file
        tracks_dir: Directory to save individual tracks
        cover_art_dir: Directory for cover art
        channel: Channel information dict
        track_changes: List of track dicts with timing info
        recording_start_time: When recording started (UTC)
        audio_recorder: AudioRecorder instance for tagging
    """
    
    if not track_changes:
        print("   ⚠️  No tracks to split")
        return
    
    print()
    print("✂️  Splitting tracks with EXACT API timing...")
    print()
    
    for i, track in enumerate(track_changes):
        track_num = i + 1
        artist = track.get('artist', 'Unknown')
        title = track.get('title', 'Unknown')
        
        # Calculate start offset from recording beginning
        track_start_time = datetime.fromisoformat(track['timestamp_utc'].replace('Z', '+00:00'))
        start_offset = (track_start_time - recording_start_time).total_seconds()
        start_offset = max(0, start_offset)  # Don't go negative
        
        # Calculate duration - PRIORITY: Use API duration first!
        if track.get('duration_ms'):
            # Use EXACT duration from API (most accurate!)
            duration = track['duration_ms'] / 1000
        elif i < len(track_changes) - 1:
            # Calculate from next track start time
            next_track_time = datetime.fromisoformat(track_changes[i + 1]['timestamp_utc'].replace('Z', '+00:00'))
            duration = (next_track_time - track_start_time).total_seconds()
        else:
            # Last track fallback - this shouldn't happen with zero-partials
            duration = None
        
        # Safe filename
        safe_artist = "".join(c for c in artist if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"{track_num:02d} - {safe_artist} - {safe_title}.m4a"
        output_file = tracks_dir / filename
        
        print(f"   🎵 Track {track_num}/{len(track_changes)}: {artist} - {title}")
        
        # Extract segment with ffmpeg
        cmd = ['ffmpeg', '-i', str(continuous_file), '-ss', str(start_offset)]
        
        if duration:
            cmd.extend(['-t', str(duration)])
        
        cmd.extend(['-c', 'copy', '-y', str(output_file)])
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            
            if result.returncode == 0 and output_file.exists():
                size = output_file.stat().st_size / (1024*1024)
                print(f"      ✅ Extracted ({size:.1f} MB)")
                
                # Download cover art
                cover_file = cover_art_dir / f"{track_num:02d}-{safe_artist.lower().replace(' ', '-')}.jpg"
                cover_art_downloaded = False
                if audio_recorder.download_cover_art(artist, title, cover_file):
                    print(f"      🎨 Downloaded cover art")
                    cover_art_downloaded = True
                
                # Tag file with metadata
                metadata = {
                    'artist': artist,
                    'title': title,
                    'album': f"SiriusXM {channel['name']}",
                    'year': datetime.now().year,
                    'genre': channel.get('genre', 'Radio'),
                    'comment': f"Recorded from SiriusXM {channel['name']} on {datetime.now().strftime('%Y-%m-%d')}"
                }
                
                if audio_recorder.tag_audio_file(output_file, metadata, cover_file if cover_file.exists() else None):
                    print(f"      🏷️  Tagged with embedded cover art")
                
                # Save enhanced track metadata
                track['filename'] = filename
                track['file_size_mb'] = round(size, 2)
                track['duration_seconds'] = round(duration if duration else 0, 1)
                track['cover_art_embedded'] = cover_art_downloaded
            else:
                print(f"      ❌ Extraction failed")
                if result.stderr:
                    print(f"         Error: {result.stderr.decode()[:100]}")
                    
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    print()
    print(f"✅ Split {len(track_changes)} tracks with EXACT timing!")


def create_playlist(tracks_dir: Path, channel: Dict, session_name: str, track_changes: List[Dict]):
    """
    Create M3U playlist and track info file
    
    Args:
        tracks_dir: Directory containing track files
        channel: Channel information
        session_name: Recording session name
        track_changes: List of track dicts
    """
    
    print()
    print("📝 Creating playlist and metadata...")
    
    # Create M3U playlist
    playlist_file = tracks_dir.parent / f"{channel['name']} - {session_name}.m3u"
    
    with open(playlist_file, 'w', encoding='utf-8') as f:
        f.write("#EXTM3U\n")
        f.write(f"# SiriusXM {channel['name']} - Recorded {session_name}\n")
        f.write(f"# Channel: {channel.get('number', '?')} - {channel['name']}\n")
        f.write(f"# Genre: {channel.get('genre', 'Unknown')}\n")
        f.write("\n")
        
        for track in track_changes:
            if 'filename' in track:
                duration = track.get('duration_seconds', 0)
                artist = track.get('artist', 'Unknown')
                title = track.get('title', 'Unknown')
                f.write(f"#EXTINF:{int(duration)},{artist} - {title}\n")
                f.write(f"tracks/{track['filename']}\n")
    
    print(f"   ✅ Playlist: {playlist_file.name}")
    
    # Create detailed track info JSON
    info_file = tracks_dir.parent / f"recording_info.json"
    
    import json
    info = {
        'channel': {
            'name': channel['name'],
            'number': channel.get('number'),
            'genre': channel.get('genre'),
            'id': channel.get('id')
        },
        'session': session_name,
        'recorded_at': datetime.now().isoformat(),
        'total_tracks': len(track_changes),
        'tracks': track_changes
    }
    
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    
    print(f"   ✅ Info: {info_file.name}")
    print()
