"""Session management for persistent authentication."""
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict


class SessionManager:
    """Manages authentication session persistence."""
    
    SESSION_DIR = Path.home() / ".seriouslyxm"
    SESSION_FILE = SESSION_DIR / "session.json"
    SESSION_LIFETIME_HOURS = 24  # Session valid for 24 hours
    
    def __init__(self):
        """Initialize session manager."""
        self.SESSION_DIR.mkdir(parents=True, exist_ok=True)
    
    def save_session(self, cookies: Dict[str, str], username: str) -> bool:
        """Save session cookies.
        
        Args:
            cookies: Dictionary of cookies.
            username: Username associated with session.
            
        Returns:
            bool: True if successful.
        """
        try:
            session_data = {
                'username': username,
                'cookies': cookies,
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(hours=self.SESSION_LIFETIME_HOURS)).isoformat()
            }
            
            with open(self.SESSION_FILE, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            # Secure permissions
            os.chmod(self.SESSION_FILE, 0o600)
            
            print(f"💾 Session saved (valid for {self.SESSION_LIFETIME_HOURS} hours)")
            return True
            
        except Exception as e:
            print(f"⚠️  Failed to save session: {e}")
            return False
    
    def load_session(self, username: str) -> Optional[Dict[str, str]]:
        """Load session cookies if valid.
        
        Args:
            username: Username to check session for.
            
        Returns:
            dict: Cookie dictionary if valid session exists, None otherwise.
        """
        if not self.SESSION_FILE.exists():
            return None
        
        try:
            with open(self.SESSION_FILE, 'r') as f:
                session_data = json.load(f)
            
            # Check if session is for same user
            if session_data.get('username') != username:
                return None
            
            # Check if session is expired
            expires_at = datetime.fromisoformat(session_data['expires_at'])
            if datetime.now() > expires_at:
                print("⏰ Session expired, will re-authenticate")
                return None
            
            print("✅ Using saved session")
            return session_data['cookies']
            
        except Exception as e:
            print(f"⚠️  Failed to load session: {e}")
            return None
    
    def clear_session(self) -> bool:
        """Clear saved session.
        
        Returns:
            bool: True if successful.
        """
        try:
            if self.SESSION_FILE.exists():
                self.SESSION_FILE.unlink()
                print("🗑️  Session cleared")
            return True
        except Exception as e:
            print(f"⚠️  Failed to clear session: {e}")
            return False
    
    def get_session_info(self) -> Optional[Dict]:
        """Get information about current session.
        
        Returns:
            dict: Session info or None if no session exists.
        """
        if not self.SESSION_FILE.exists():
            return None
        
        try:
            with open(self.SESSION_FILE, 'r') as f:
                session_data = json.load(f)
            
            expires_at = datetime.fromisoformat(session_data['expires_at'])
            created_at = datetime.fromisoformat(session_data['created_at'])
            is_valid = datetime.now() < expires_at
            age = datetime.now() - created_at
            cookie_count = len(session_data.get('cookies', {}))
            
            return {
                'username': session_data['username'],
                'created_at': created_at,
                'expires_at': expires_at,
                'is_valid': is_valid,
                'age': age,
                'cookie_count': cookie_count,
                'time_remaining': str(expires_at - datetime.now()) if is_valid else 'Expired'
            }
            
        except Exception:
            return None
    
    def get_current_session_cookies(self) -> Optional[Dict[str, str]]:
        """Get cookies from current session if valid.
        
        Returns:
            dict: Cookie dictionary if valid session exists, None otherwise.
        """
        if not self.SESSION_FILE.exists():
            return None
        
        try:
            with open(self.SESSION_FILE, 'r') as f:
                session_data = json.load(f)
            
            # Check if session is expired
            expires_at = datetime.fromisoformat(session_data['expires_at'])
            if datetime.now() > expires_at:
                return None
            
            return session_data.get('cookies', {})
            
        except Exception:
            return None
    
    def get_bearer_token(self) -> Optional[str]:
        """Extract bearer token from current session.
        
        Returns:
            str: Bearer token if available, None otherwise.
        """
        cookies = self.get_current_session_cookies()
        if not cookies:
            return None
        
        # Try SXMAUTHORIZATION first (raw token)
        if 'SXMAUTHORIZATION' in cookies:
            return cookies['SXMAUTHORIZATION']
        
        # Try AUTH_TOKEN (URL-encoded JSON)
        if 'AUTH_TOKEN' in cookies:
            try:
                import urllib.parse
                # URL decode the cookie value
                decoded = urllib.parse.unquote(cookies['AUTH_TOKEN'])
                # Parse JSON
                auth_data = json.loads(decoded)
                # Extract access token
                return auth_data.get('session', {}).get('accessToken')
            except Exception as e:
                print(f"⚠️  Failed to decode AUTH_TOKEN: {e}")
                return None
        
        return None
