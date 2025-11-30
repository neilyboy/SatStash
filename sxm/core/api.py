#!/usr/bin/env python3
"""
SiriusXM API Client

Handles all API interactions:
- liveUpdate API (track schedule with exact timestamps)
- Channel list API (all 247 channels)
- VOD search API (shows and episodes)
- Stream URLs
"""

import requests
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict


class SiriusXMAPI:
    """SiriusXM API client"""
    
    BASE_URL = 'https://api.edge-gateway.siriusxm.com'
    
    def __init__(self, bearer_token: str):
        """
        Initialize API client
        
        Args:
            bearer_token: Bearer token for authentication
        """
        self.bearer_token = bearer_token
        self.headers = {
            'Authorization': f'Bearer {bearer_token}',
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        }
        self.last_schedule_status: Optional[int] = None
    
    def get_stream_url(self, channel_id: str, start_timestamp: str = None) -> Optional[str]:
        """
        Get HLS stream URL for a channel
        
        Uses EUREKA discovery: can specify startTimestamp to get DVR content!
        
        Args:
            channel_id: Channel UUID
            start_timestamp: Optional ISO timestamp to start from (for DVR)
            
        Returns:
            HLS master playlist URL
        """
        try:
            url = f'{self.BASE_URL}/playback/play/v1/tuneSource'
            
            if start_timestamp:
                current_time = start_timestamp
            else:
                current_time = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            
            payload = {
                'id': channel_id,  # Fixed: should be 'id' not 'channelId'!
                'type': 'channel-linear',
                'hlsVersion': 'V3',
                'manifestVariant': 'WEB',
                'mtcVersion': 'V2',
                'startTimestamp': current_time
            }
            
            headers = {**self.headers, 'Content-Type': 'application/json'}
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                # Extract URL from nested structure
                streams = data.get('streams', [])
                if streams and len(streams) > 0:
                    urls = streams[0].get('urls', [])
                    if urls and len(urls) > 0:
                        return urls[0].get('url')
                return None
            else:
                print(f"Stream URL API error: {response.status_code}")
                print(f"Response: {response.text[:200]}")
            
            return None
            
        except Exception as e:
            print(f"Error getting stream URL: {e}")
            return None
    
    def get_schedule(self, channel_id: str) -> List[Dict]:
        """
        Fetch track schedule from liveUpdate API
        
        CRITICAL DISCOVERY:
        - Returns tracks in CHRONOLOGICAL order
        - LAST track in list = CURRENTLY PLAYING!
        - Includes past tracks (5-hour DVR buffer)
        - Each track has EXACT UTC timestamp (millisecond precision)
        
        Args:
            channel_id: Channel ID
            
        Returns:
            List of tracks with exact timestamps
            Empty list on error
        """
        try:
            url = f'{self.BASE_URL}/playback/play/v1/liveUpdate'
            
            current_time = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            
            payload = {
                'channelId': channel_id,
                'hlsVersion': 'V3',
                'manifestVariant': 'WEB',
                'mtcVersion': 'V2',
                'startTimestamp': current_time
            }
            
            headers = {**self.headers, 'Content-Type': 'application/json'}
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            self.last_schedule_status = response.status_code
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])
                
                # Parse tracks (filter promos)
                tracks = []
                for item in items:
                    # Skip promos/interstitials
                    if item.get('isInterstitial', False):
                        continue
                    
                    track = {
                        'artist': item.get('artistName', 'Unknown'),
                        'title': item.get('name', 'Unknown'),
                        'timestamp_utc': item.get('timestamp'),
                        'duration_ms': item.get('duration', 0),
                        'album': item.get('albumName'),
                        'images': item.get('images', {})
                    }
                    tracks.append(track)
                
                return tracks
            else:
                print(f"Schedule API error: {response.status_code}")
                print(f"Response: {response.text[:200]}")
            
            return []
            
        except Exception as e:
            print(f"API error: {e}")
            self.last_schedule_status = None
            return []
    
    def get_dvr_tracks(self, channel_id: str, hours_back: int = 3) -> List[Dict]:
        """
        Get tracks from DVR buffer for a specific time range
        
        Args:
            channel_id: Channel ID
            hours_back: Number of hours back to retrieve (1-5)
        
        Returns:
            List of tracks within the time range
        """
        all_tracks = self.get_schedule(channel_id)
        
        if not all_tracks:
            return []
        
        # Filter for time range
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        
        dvr_tracks = []
        for track in all_tracks:
            if track.get('timestamp_utc'):
                track_time = datetime.fromisoformat(track['timestamp_utc'].replace('Z', '+00:00'))
                if track_time >= cutoff_time:
                    dvr_tracks.append(track)
        
        return dvr_tracks
    
    def get_current_track(self, schedule: List[Dict]) -> Optional[Dict]:
        """
        Get currently playing track from schedule
        
        EUREKA DISCOVERY: Last track in schedule = current track!
        
        Args:
            schedule: Track schedule from get_schedule()
            
        Returns:
            Current track or None
        """
        return schedule[-1] if schedule else None
    
    def fetch_all_channels(self) -> List[Dict]:
        """
        Fetch complete channel list from API
        
        Returns ALL 247 channels with full metadata!
        
        Returns:
            List of all channels
        """
        try:
            url = f'{self.BASE_URL}/metadata/channellist/v3/web/get'
            
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                print(f"Channels API error: {response.status_code}")
                print(f"Response: {response.text[:200]}")
                return []
            
            data = response.json()
            channels = []
            
            # Parse channel groups
            for group in data.get('channelGroups', []):
                group_name = group.get('name', 'Unknown')
                
                for channel in group.get('channels', []):
                    ch = {
                        'id': channel.get('id'),
                        'name': channel.get('name'),
                        'number': channel.get('number'),
                        'category': group_name,
                        'genre': channel.get('genre', {}).get('name') if channel.get('genre') else None,
                        'description': channel.get('description'),
                        'url': channel.get('url'),
                        'images': {
                            'thumbnail': channel.get('images', {}).get('thumbnail'),
                            'large': channel.get('images', {}).get('large')
                        }
                    }
                    channels.append(ch)
            
            return channels
            
        except Exception as e:
            print(f"Error fetching channels: {e}")
            return []
    
    def search_vod(self, query: str, limit: int = 50) -> Dict[str, List]:
        """
        Search VOD content (shows and episodes)
        
        Args:
            query: Search term (artist, show name, etc.)
            limit: Max results
            
        Returns:
            Dict with 'shows' and 'episodes' keys
        """
        try:
            url = f'{self.BASE_URL}/playback/v1/search'
            
            payload = {
                'query': query,
                'types': ['show', 'episode'],
                'limit': limit
            }
            
            headers = {**self.headers, 'Content-Type': 'application/json'}
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'shows': data.get('shows', []),
                    'episodes': data.get('episodes', [])
                }
            
            return {'shows': [], 'episodes': []}
            
        except Exception as e:
            print(f"VOD search error: {e}")
            return {'shows': [], 'episodes': []}
    
    def get_episode_details(self, episode_id: str) -> Optional[Dict]:
        """
        Get detailed information for an episode
        
        Args:
            episode_id: Episode ID
            
        Returns:
            Episode details or None
        """
        try:
            url = f'{self.BASE_URL}/playback/v1/episode/{episode_id}'
            
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            
            return None
            
        except Exception as e:
            print(f"Error getting episode: {e}")
            return None
    
    def get_stream_url_vod(self, channel_id: str, content_type: str = 'channel-linear') -> Optional[str]:
        """
        Get stream URL for VOD content (renamed to avoid conflict)
        
        Args:
            channel_id: Channel or content ID
            content_type: Type (channel-linear or episode)
            
        Returns:
            Stream URL or None
        """
        try:
            url = f'{self.BASE_URL}/playback/play/v1/tuneSource'
            
            payload = {
                'id': channel_id,
                'type': content_type,
                'hlsVersion': 'V3',
                'manifestVariant': 'WEB' if content_type == 'channel-linear' else 'FULL',
                'mtcVersion': 'V2'
            }
            
            headers = {**self.headers, 'Content-Type': 'application/json'}
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('primaryStreamUrl')
            else:
                print(f"Stream URL API error: {response.status_code}")
                print(f"Response: {response.text[:200]}")
            
            return None
            
        except Exception as e:
            print(f"Error getting stream URL: {e}")
            return None
