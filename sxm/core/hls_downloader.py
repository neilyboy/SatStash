#!/usr/bin/env python3
"""
HLS Downloader & Decryptor

Handles downloading and decrypting HLS streams from SiriusXM.

Key Discoveries:
- SiriusXM uses AES-128 encryption
- Segments are ~10 seconds each
- DVR buffer contains ~5 hours (1800+ segments)
- Need to match timestamps to find start point
"""

import re
import requests
import subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Tuple


class HLSDownloader:
    """Downloads and decrypts HLS streams"""
    QUALITY_BANDWIDTH = {
        '32k': 32_768,
        '64k': 65_536,
        '128k': 131_072,
        '256k': 262_144,
    }
    QUALITY_KEYWORDS = {
        '256k': ['256k', 'BANDWIDTH=281600', 'BANDWIDTH=256000'],
        '128k': ['128k', 'BANDWIDTH=128000', '256k', 'BANDWIDTH=281600'],
        '64k': ['64k', 'BANDWIDTH=70000', 'BANDWIDTH=64000'],
        '32k': ['32k', 'BANDWIDTH=32768', 'BANDWIDTH=32000']
    }

    def __init__(self, bearer_token: str = None, session_cookies: Optional[Dict[str, str]] = None):
        """
        Initialize downloader
        
        Args:
            bearer_token: Bearer token for authenticated requests
        """
        self.processed_segments = set()
        self.bearer_token = bearer_token
        self.session_cookies = session_cookies or {}
    
    def _build_headers(self) -> Dict[str, str]:
        headers = {'User-Agent': 'Mozilla/5.0'}
        if self.bearer_token:
            headers['Authorization'] = f'Bearer {self.bearer_token}'
        cookie_header = self._format_cookie_header()
        if cookie_header:
            headers['Cookie'] = cookie_header
        return headers

    def _format_cookie_header(self) -> str:
        if not self.session_cookies:
            return ''
        return '; '.join(f"{k}={v}" for k, v in self.session_cookies.items())

    def update_session_cookies(self, cookies: Optional[Dict[str, str]]):
        self.session_cookies = cookies or {}

    def get_http_headers(self) -> Dict[str, str]:
        """Public accessor for current HTTP headers used in requests."""
        return self._build_headers()

    def get_variant_url(self, master_url: str, quality: str = '256k') -> Optional[str]:
        """
        Get variant playlist URL for desired quality
        
        Args:
            master_url: Master playlist URL
            quality: Desired quality (256k, 128k, etc.)
            
        Returns:
            Variant playlist URL or None
        """
        try:
            response = requests.get(master_url, headers=self._build_headers(), timeout=10)
            if response.status_code != 200:
                return None

            master_playlist = response.text
            base_url = master_url.rsplit('/', 1)[0] + '/'
            lines = master_playlist.split('\n')

            # Prefer explicit 256k/AAC-LC entries (known-good)
            preferred = self._find_preferred_variant(lines, base_url)
            if preferred:
                return preferred

            # Fallback: first *.m3u8 entry
            for line in lines:
                if line.strip() and not line.startswith('#') and '.m3u8' in line:
                    return line.strip() if line.strip().startswith('http') else base_url + line.strip()

            return None
            
        except Exception as e:
            print(f"Error getting variant URL: {e}")
            return None

    def _find_preferred_variant(self, lines: List[str], base_url: str) -> Optional[str]:
        for i, line in enumerate(lines):
            if '#EXT-X-STREAM-INF:' in line and ('256k' in line or 'BANDWIDTH=281600' in line):
                if i + 1 < len(lines):
                    path = lines[i + 1].strip()
                    if path and not path.startswith('#'):
                        return path if path.startswith('http') else base_url + path
        return None

    def get_decryption_key(self, variant_playlist: str) -> Optional[str]:
        """
        Extract AES-128 decryption key from variant playlist
        
        Args:
            variant_playlist: Variant playlist content
            
        Returns:
            Hex decryption key or None
        """
        try:
            # Find key URL
            key_url = None
            for line in variant_playlist.split('\n'):
                if '#EXT-X-KEY:' in line:
                    match = re.search(r'URI="([^"]+)"', line)
                    if match:
                        key_url = match.group(1)
                        break
            
            if not key_url:
                print("   ⚠️  No key URL found in playlist")
                # Debug: show first few lines
                lines = variant_playlist.split('\n')[:10]
                for line in lines:
                    print(f"      {line}")
                return None
            
            # Get key (with authentication!)
            headers = self._build_headers()
            
            print(f"   📥 Fetching key from: {key_url[:60]}...")
            response = requests.get(key_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                # Check if response is JSON (SiriusXM returns JSON with base64 key)
                try:
                    import json
                    import base64
                    data = json.loads(response.text)
                    if 'key' in data:
                        # Decode base64 key to bytes, then convert to hex
                        key_b64 = data['key']
                        key_bytes = base64.b64decode(key_b64)
                        key_hex = key_bytes.hex()
                        print(f"   ✅ Got key from JSON ({len(key_hex)} chars hex)")
                        return key_hex
                except:
                    pass
                
                # Fallback: treat as raw bytes
                key_bytes = response.content
                key_hex = key_bytes.hex()
                print(f"   ✅ Got key as raw bytes ({len(key_hex)} chars)")
                return key_hex
            else:
                print(f"   ❌ Key fetch failed: {response.status_code}")
            
            return None
            
        except Exception as e:
            print(f"   ❌ Error getting decryption key: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def parse_segments_with_timestamps(self, variant_playlist: str, 
                                       base_url: str) -> List[Dict]:
        """
        Parse HLS playlist and extract segments with timestamps
        
        CRITICAL for DVR support: Segments have program date/time tags
        
        Args:
            variant_playlist: Variant playlist content
            base_url: Base URL for segments
            
        Returns:
            List of segment dicts with URLs and timestamps
        """
        segments = []
        current_timestamp = None
        current_duration = 0
        
        lines = variant_playlist.split('\n')
        
        for i, line in enumerate(lines):
            # Parse program date/time tag
            if line.startswith('#EXT-X-PROGRAM-DATE-TIME:'):
                time_str = line.split(':', 1)[1].strip()
                try:
                    current_timestamp = datetime.fromisoformat(
                        time_str.replace('Z', '+00:00')
                    )
                except:
                    pass
            
            # Parse segment duration
            elif line.startswith('#EXTINF:'):
                try:
                    duration_str = line.split(':', 1)[1].split(',')[0]
                    current_duration = float(duration_str)
                except:
                    current_duration = 10.0  # Default
            
            # Parse segment URL
            elif line.strip() and not line.startswith('#'):
                segment_url = base_url + line.strip()
                
                segment = {
                    'url': segment_url,
                    'timestamp': current_timestamp,
                    'duration': current_duration
                }
                
                segments.append(segment)
                
                # Update timestamp for next segment
                if current_timestamp:
                    current_timestamp += timedelta(seconds=current_duration)
        
        return segments
    
    def find_segment_by_timestamp(self, segments: List[Dict], 
                                   target_time: datetime) -> int:
        """
        Find segment index matching target timestamp
        
        CRITICAL for zero-partial recordings: Start at exact track time
        
        Args:
            segments: List of segments with timestamps
            target_time: Target start time
            
        Returns:
            Segment index or 0
        """
        # Make target_time timezone-aware if it isn't
        if target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=timezone.utc)
        
        for i, segment in enumerate(segments):
            if segment['timestamp']:
                seg_time = segment['timestamp']
                
                # Make seg_time timezone-aware if needed
                if seg_time.tzinfo is None:
                    seg_time = seg_time.replace(tzinfo=timezone.utc)
                
                # If segment is at or after target, this is our start
                if seg_time >= target_time:
                    return max(0, i - 1)  # Go back one for safety
        
        # Fallback: last 5 segments (live edge)
        return max(0, len(segments) - 5)
    
    def download_and_decrypt_segment(self, segment_url: str, 
                                     segment_index: int,
                                     key_hex: str,
                                     temp_dir: Path) -> Optional[Path]:
        """
        Download and decrypt a single segment
        
        Args:
            segment_url: Segment URL
            segment_index: Segment index (for IV calculation)
            key_hex: Decryption key (hex)
            temp_dir: Temporary directory
            
        Returns:
            Path to decrypted file or None
        """
        enc_file = temp_dir / f"seg_{segment_index:04d}_enc.aac"
        dec_file = temp_dir / f"seg_{segment_index:04d}_dec.aac"
        
        try:
            # Download encrypted segment
            response = requests.get(segment_url, headers=self._build_headers(), timeout=10)
            
            if response.status_code != 200:
                return None
            
            with open(enc_file, 'wb') as f:
                f.write(response.content)
            
            # Decrypt using OpenSSL
            # IV is segment index as 32-char hex (16 bytes)
            iv = f"{segment_index:032x}"
            
            cmd = [
                'openssl', 'aes-128-cbc', '-d',
                '-in', str(enc_file),
                '-out', str(dec_file),
                '-K', key_hex,
                '-iv', iv
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            
            # Clean up encrypted file
            if enc_file.exists():
                enc_file.unlink()
            
            if result.returncode == 0 and dec_file.exists():
                return dec_file
            
            return None
            
        except Exception as e:
            print(f"Error downloading segment: {e}")
            # Clean up
            if enc_file.exists():
                enc_file.unlink()
            if dec_file.exists():
                dec_file.unlink()
            return None
    
    def combine_segments(self, segment_files: List[Path], 
                        output_file: Path) -> bool:
        """
        Combine decrypted segments into single file
        
        Args:
            segment_files: List of segment file paths
            output_file: Output file path
            
        Returns:
            True if successful
        """
        try:
            with open(output_file, 'wb') as outfile:
                for seg_file in sorted(segment_files):
                    if seg_file.exists():
                        with open(seg_file, 'rb') as infile:
                            outfile.write(infile.read())
                        seg_file.unlink()  # Clean up
            
            return output_file.exists()
            
        except Exception as e:
            print(f"Error combining segments: {e}")
            return False
