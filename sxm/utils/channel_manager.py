#!/usr/bin/env python3
"""
Channel Manager
Manages channel database and provides channel lookup functionality

Now with DYNAMIC channel fetching from API!
"""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict


class ChannelManager:
    """Manages SiriusXM channel information"""
    
    CHANNELS_FILE = Path(__file__).parent / "channels_database.json"
    CACHE_FILE = Path.home() / ".seriouslyxm" / "channels_cache.json"
    
    def __init__(self, bearer_token: Optional[str] = None):
        """
        Initialize channel manager
        
        Args:
            bearer_token: Optional bearer token to fetch channels from API
        """
        self.bearer_token = bearer_token
        self.channels = self._load_channels()
    
    def _load_channels(self) -> Dict:
        """
        Load channels - tries multiple sources:
        1. Cached API response (if recent)
        2. Fetch from API (if bearer token provided)
        3. Cached API response (even if old)
        4. Static database file (fallback)
        """
        cached_data = None
        
        # Try cache first
        if self.CACHE_FILE.exists():
            try:
                with open(self.CACHE_FILE, 'r') as f:
                    data = json.load(f)
                    cached_data = data  # Keep reference for later fallback
                    channels = data.get('channels', [])
                    # Cache format v1 did not include artwork images; if we
                    # detect a cache without any 'images' keys, treat it as
                    # stale so we can refresh from the API and enable
                    # channel artwork in the TUI.
                    has_images = any('images' in ch for ch in channels)

                    # Check if cache is recent (less than 7 days old) AND has
                    # images. Otherwise we will fall through to API fetch.
                    import time
                    cache_age = time.time() - data.get('timestamp', 0)
                    cache_hours = cache_age / 3600
                    if cache_age < (7 * 86400) and has_images:  # 7 days
                        print(f"📡 Using cached channel list ({len(channels)} channels, {cache_hours:.1f}h old)")
                        return {ch['id']: ch for ch in channels}
            except Exception as e:
                print(f"⚠️  Cache read error: {e}")
        
        # Try to fetch from API if we have a bearer token
        if self.bearer_token:
            channels = self._fetch_from_api()
            if channels:
                return channels
            else:
                print("⚠️  API fetch failed, trying fallbacks...")
        
        # If API failed but we have cached data (even if old), use it
        if cached_data:
            channels = cached_data.get('channels', [])
            if channels:
                import time
                cache_age = time.time() - cached_data.get('timestamp', 0)
                cache_days = cache_age / 86400
                print(f"📡 Using cached channel list ({len(channels)} channels, {cache_days:.1f} days old)")
                return {ch['id']: ch for ch in channels}
        
        # Fallback to static database
        try:
            with open(self.CHANNELS_FILE, 'r') as f:
                data = json.load(f)
                print(f"📻 Using static channel list ({len(data.get('channels', []))} channels)")
                return {ch['id']: ch for ch in data.get('channels', [])}
        except FileNotFoundError:
            print("⚠️  No channel database found")
            return {}
    
    def _fetch_from_api(self) -> Dict:
        """Fetch channels from SiriusXM API using browse endpoint"""
        try:
            import requests
            
            print("📡 Fetching channel list from API...")
            
            base_url = 'https://api.edge-gateway.siriusxm.com'
            headers = {
                'Authorization': f'Bearer {self.bearer_token}',
                'User-Agent': 'Mozilla/5.0',
                'Content-Type': 'application/json'
            }
            
            # Initial request to get first 50 channels
            init_data = {
                "filter": {"one": {"filterId": "all"}},
                "sets": {
                    "5mqCLZ21qAwnufKT8puUiM": {
                        "sort": {"sortId": "CHANNEL_NUMBER_ASC"},
                        "pagination": {"offset": {"setItemsLimit": 50}}
                    }
                },
                "pagination": {"offset": {"containerLimit": 3, "setItemsLimit": 50}},
                "deviceCapabilities": {"supportsDownloads": False}
            }
            
            url = f'{base_url}/browse/v1/pages/curated-grouping/403ab6a5-d3c9-4c2a-a722-a94a6a5fd056/view'
            response = requests.post(url, headers=headers, json=init_data, timeout=10)
            
            if response.status_code != 200:
                print(f"   ❌ API error: {response.status_code}")
                return {}
            
            data = response.json()
            channels = []
            
            # Parse first batch
            for channel in data["page"]["containers"][0]["sets"][0]["items"]:
                # Try to get channel number from various places
                num = None
                if "decorations" in channel and "channelNumber" in channel["decorations"]:
                    num = channel["decorations"]["channelNumber"]
                elif "channelNumber" in channel.get("entity", {}):
                    num = channel["entity"]["channelNumber"]
                
                ch = {
                    'id': channel["entity"]["id"],
                    'name': channel["entity"]["texts"]["title"]["default"],
                    'description': channel["entity"]["texts"]["description"]["default"],
                    'genre': channel["decorations"].get("genre", ""),
                    'number': num,
                    'images': channel["entity"]["images"]
                }
                channels.append(ch)
            
            # Get total count and fetch remaining in batches
            total = data["page"]["containers"][0]["sets"][0]["pagination"]["offset"]["size"]
            print(f"   Found {total} total channels, fetching all...")
            
            # Fetch remaining channels in batches of 50
            for offset in range(50, total, 50):
                batch_data = {
                    "filter": {"one": {"filterId": "all"}},
                    "sets": {
                        "5mqCLZ21qAwnufKT8puUiM": {
                            "sort": {"sortId": "CHANNEL_NUMBER_ASC"},
                            "pagination": {
                                "offset": {
                                    "setItemsOffset": offset,
                                    "setItemsLimit": 50
                                }
                            }
                        }
                    },
                    "pagination": {"offset": {"setItemsLimit": 50}}
                }
                
                batch_url = f'{base_url}/browse/v1/pages/curated-grouping/403ab6a5-d3c9-4c2a-a722-a94a6a5fd056/containers/3JoBfOCIwo6FmTpzM1S2H7/view'
                batch_response = requests.post(batch_url, headers=headers, json=batch_data, timeout=10)
                
                if batch_response.status_code == 200:
                    batch_json = batch_response.json()
                    for channel in batch_json["container"]["sets"][0]["items"]:
                        # Try to get channel number from various places
                        num = None
                        if "decorations" in channel and "channelNumber" in channel["decorations"]:
                            num = channel["decorations"]["channelNumber"]
                        elif "channelNumber" in channel.get("entity", {}):
                            num = channel["entity"]["channelNumber"]
                        
                        ch = {
                            'id': channel["entity"]["id"],
                            'name': channel["entity"]["texts"]["title"]["default"],
                            'description': channel["entity"]["texts"]["description"]["default"],
                            'genre': channel["decorations"].get("genre", ""),
                            'number': num,
                            'images': channel["entity"]["images"]
                        }
                        channels.append(ch)
            
            print(f"✅ Got {len(channels)} channels from API!")
            
            # Cache the results
            self._cache_channels(channels)
            
            return {ch['id']: ch for ch in channels}
            
        except Exception as e:
            print(f"⚠️  Could not fetch channels from API: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _cache_channels(self, channels: List[Dict]):
        """Cache channels to file"""
        import time
        
        # Ensure cache directory exists
        self.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'version': '2.0',
            'timestamp': time.time(),
            'channels': channels
        }
        
        with open(self.CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_channels(self):
        """Save channels to database"""
        data = {
            'version': '1.0',
            'last_updated': '2024-11-21',
            'channels': list(self.channels.values())
        }
        with open(self.CHANNELS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    
    def refresh_channels(self):
        """Force refresh channels from API"""
        if self.bearer_token:
            print("🔄 Refreshing channel list...")
            channels = self._fetch_from_api()
            if channels:
                self.channels = channels
                return True
        return False
    
    def get_channel(self, identifier: str) -> Optional[Dict]:
        """
        Get channel by ID, name, or number
        
        Args:
            identifier: Channel ID, name, or number
            
        Returns:
            Channel dict or None
        """
        identifier_lower = str(identifier).lower()
        
        # Try exact ID match first
        if identifier_lower in self.channels:
            return self.channels[identifier_lower]
        
        # Try name or number match
        for channel in self.channels.values():
            # Match by name (case-insensitive)
            if channel['name'].lower() == identifier_lower:
                return channel
            
            # Match by name contains
            if identifier_lower in channel['name'].lower():
                return channel
            
            # Match by number
            if channel.get('number') and str(channel['number']) == identifier_lower:
                return channel
        
        return None
    
    def search_channels(self, query: str) -> List[Dict]:
        """
        Search channels by name, genre, or description
        
        Args:
            query: Search query
            
        Returns:
            List of matching channels
        """
        query_lower = query.lower()
        results = []
        
        for channel in self.channels.values():
            # Search in name
            if query_lower in channel['name'].lower():
                results.append(channel)
                continue
            
            # Search in genre
            if channel.get('genre') and query_lower in channel['genre'].lower():
                results.append(channel)
                continue
            
            # Search in description
            if channel.get('description') and query_lower in channel['description'].lower():
                results.append(channel)
                continue
        
        return results
    
    def list_channels(self, genre: Optional[str] = None) -> List[Dict]:
        """
        List all channels, optionally filtered by genre
        
        Args:
            genre: Optional genre filter
            
        Returns:
            List of channels
        """
        channels = list(self.channels.values())
        
        if genre:
            genre_lower = genre.lower()
            channels = [
                ch for ch in channels 
                if ch.get('genre') and genre_lower in ch['genre'].lower()
            ]
        
        # Sort by channel number
        return sorted(channels, key=lambda x: (x.get('number') or 999, x['name']))
    
    def get_channel_url(self, identifier: str) -> Optional[str]:
        """
        Get player URL for a channel
        
        Args:
            identifier: Channel ID, name, or number
            
        Returns:
            Player URL or None
        """
        channel = self.get_channel(identifier)
        if channel:
            return f"https://www.siriusxm.com/player/channel-linear/entity/{channel['id']}"
        return None
    
    def add_channel(self, channel_id: str, name: str, number: Optional[int] = None,
                    genre: Optional[str] = None, description: Optional[str] = None):
        """
        Add a new channel to the database
        
        Args:
            channel_id: Channel ID
            name: Channel name
            number: Channel number (optional)
            genre: Genre (optional)
            description: Description (optional)
        """
        channel = {
            'id': channel_id,
            'name': name
        }
        
        if number:
            channel['number'] = number
        if genre:
            channel['genre'] = genre
        if description:
            channel['description'] = description
        
        self.channels[channel_id] = channel
        self._save_channels()
    
    def get_genres(self) -> List[str]:
        """Get list of all unique genres"""
        genres = set()
        for channel in self.channels.values():
            if channel.get('genre'):
                genres.add(channel['genre'])
        return sorted(genres)
    
    def display_channels(self, channels: Optional[List[Dict]] = None):
        """
        Display channels in a formatted list
        
        Args:
            channels: Optional list of channels (uses all if None)
        """
        if channels is None:
            channels = self.list_channels()
        
        if not channels:
            print("No channels found")
            return
        
        print(f"\n{'='*80}")
        print(f"📻 SIRIUSXM CHANNELS ({len(channels)} channels)")
        print(f"{'='*80}\n")
        
        for channel in channels:
            num = str(channel.get('number', '?')).rjust(3)
            name = channel['name'][:30].ljust(30)
            genre = channel.get('genre', '')[:35]
            
            print(f"  {num}. {name}  {genre}")
        
        print()


if __name__ == '__main__':
    # Demo usage
    manager = ChannelManager()
    
    print("="*80)
    print("📻 CHANNEL MANAGER DEMO")
    print("="*80)
    
    # Show all channels
    manager.display_channels()
    
    # Search example
    print("\n🔍 Search for 'rock':")
    print("-"*80)
    results = manager.search_channels('rock')
    for ch in results[:5]:
        print(f"   • {ch['name']} (Ch {ch.get('number', '?')})")
    
    # Get specific channel
    print("\n📡 Get channel 'lithium':")
    print("-"*80)
    channel = manager.get_channel('lithium')
    if channel:
        print(f"   Name: {channel['name']}")
        print(f"   Number: {channel.get('number')}")
        print(f"   Genre: {channel.get('genre')}")
        print(f"   URL: {manager.get_channel_url('lithium')}")
    
    # List genres
    print("\n🎵 Available Genres:")
    print("-"*80)
    for genre in manager.get_genres()[:10]:
        print(f"   • {genre}")
