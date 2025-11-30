"""Configuration management for SatStash"""
from pathlib import Path
import json

class Config:
    DEFAULT_CONFIG = {
        "audio_quality": "256k",
        "output_directory": str(Path.home() / "Music" / "SiriusXM"),
        "zero_partials": True,
        "download_cover_art": True,
        "tag_metadata": True,
        "keep_continuous": False,
        "favorites": [],
        "theme": "dark",
        "username": "",
        "password": ""
    }
    
    def __init__(self):
        self.config_dir = Path.home() / ".siriusxm_session"
        self.config_file = self.config_dir / "settings.json"
        self.load()
    
    def load(self):
        """Load configuration from file"""
        if self.config_file.exists():
            with open(self.config_file) as f:
                loaded = json.load(f)
                self.data = {**self.DEFAULT_CONFIG, **loaded}
        else:
            self.data = self.DEFAULT_CONFIG.copy()
            self.save()  # Create default config file
    
    def save(self):
        """Save configuration to file"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def get(self, key, default=None):
        """Get configuration value"""
        return self.data.get(key, default)
    
    def set(self, key, value):
        """Set configuration value and save"""
        self.data[key] = value
        self.save()
    
    def add_favorite(self, channel_id):
        """Add channel to favorites"""
        favorites = self.data.get('favorites', [])
        if channel_id not in favorites:
            favorites.append(channel_id)
            self.set('favorites', favorites)
    
    def remove_favorite(self, channel_id):
        """Remove channel from favorites"""
        favorites = self.data.get('favorites', [])
        if channel_id in favorites:
            favorites.remove(channel_id)
            self.set('favorites', favorites)
    
    def is_favorite(self, channel_id):
        """Check if channel is a favorite"""
        return channel_id in self.data.get('favorites', [])
