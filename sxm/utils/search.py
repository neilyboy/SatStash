"""Fuzzy search for channels"""
from fuzzywuzzy import fuzz
from typing import List, Dict

class ChannelSearch:
    """Fuzzy search for SiriusXM channels"""
    
    def __init__(self, channels: List[Dict]):
        self.channels = channels
    
    def search(self, query: str) -> List[Dict]:
        """Search channels with fuzzy matching
        
        Args:
            query: Search query string
            
        Returns:
            List of channels sorted by relevance
        """
        if not query:
            return self.channels
        
        results = []
        query = query.lower()
        
        for channel in self.channels:
            # Calculate match scores
            name = channel.get('name', '').lower()
            description = channel.get('description', '').lower()
            genre = channel.get('genre', '').lower()
            
            # Exact substring match gets highest score
            if query in name:
                score = 100
            elif query in description:
                score = 80
            elif query in genre:
                score = 70
            else:
                # Fuzzy match on name only (more precise)
                name_score = fuzz.partial_ratio(query, name)
                score = name_score
            
            # Higher threshold for better results
            if score >= 70:
                results.append({
                    'channel': channel,
                    'score': score
                })
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)
        return [r['channel'] for r in results]
    
    def get_favorites(self, favorite_ids: List[str]) -> List[Dict]:
        """Get favorite channels
        
        Args:
            favorite_ids: List of favorite channel IDs
            
        Returns:
            List of favorite channels
        """
        return [ch for ch in self.channels if ch.get('id') in favorite_ids]
