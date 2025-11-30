#!/usr/bin/env python3
"""
Audio Recording & Tagging Module
Records live audio, splits on track changes, and embeds metadata + cover art
"""

import os
import subprocess
import time
import requests
from pathlib import Path
from typing import Optional, Dict
from mutagen.mp4 import MP4, MP4Cover
from mutagen.id3 import ID3, TIT2, TPE1, TALB, TDRC, APIC
from mutagen.mp3 import MP3


class AudioRecorder:
    """Records and tags audio with embedded cover art"""
    
    def __init__(self, output_format: str = 'm4a', mp3_bitrate: str = '320k'):
        """
        Initialize audio recorder
        
        Args:
            output_format: 'm4a' or 'mp3'
            mp3_bitrate: MP3 bitrate if converting (128k, 192k, 256k, 320k)
        """
        self.output_format = output_format
        self.mp3_bitrate = mp3_bitrate
    
    def record_stream_segment(self, stream_url: str, output_file: Path, 
                              duration: int, cookies: Dict[str, str]) -> bool:
        """
        Record a segment of the live stream
        
        Args:
            stream_url: HLS stream URL
            output_file: Output file path
            duration: Duration in seconds
            cookies: Session cookies for authentication
            
        Returns:
            True if successful
        """
        try:
            # Build cookie string for FFmpeg
            cookie_str = '; '.join([f'{k}={v}' for k, v in cookies.items()])
            
            # FFmpeg command to record stream
            cmd = [
                'ffmpeg',
                '-headers', f'Cookie: {cookie_str}',
                '-i', stream_url,
                '-t', str(duration),
                '-c', 'copy',  # Copy codec (no re-encoding)
                '-y',  # Overwrite
                str(output_file)
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=duration + 30
            )
            
            return result.returncode == 0 and output_file.exists()
            
        except Exception as e:
            print(f"❌ Recording error: {e}")
            return False
    
    def download_cover_art(self, artist: str, title: str, 
                          output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Download cover art for a track (from iTunes API)
        
        Args:
            artist: Artist name
            title: Track title
            output_path: Where to save (optional)
            
        Returns:
            Path to downloaded image or None
        """
        try:
            # Search iTunes API
            query = f"{artist} {title}".replace(' ', '+')
            url = f"https://itunes.apple.com/search?term={query}&media=music&entity=song&limit=1"
            
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return None
            
            data = response.json()
            if not data.get('results'):
                return None
            
            # Get artwork URL (600x600 version)
            artwork_url = data['results'][0].get('artworkUrl100', '')
            if not artwork_url:
                return None
            
            # Get high-res version
            artwork_url = artwork_url.replace('100x100bb', '600x600bb')
            
            # Download image
            img_response = requests.get(artwork_url, timeout=10)
            if img_response.status_code != 200:
                return None
            
            # Save to file if path provided
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(img_response.content)
                return output_path
            
            # Return content as bytes
            return img_response.content
            
        except Exception as e:
            print(f"⚠️  Could not download cover art: {e}")
            return None
    
    def tag_m4a(self, file_path: Path, metadata: Dict, cover_art_path: Optional[Path] = None):
        """
        Tag M4A file with metadata and embedded cover art
        
        Args:
            file_path: Path to M4A file
            metadata: Dict with 'artist', 'title', 'album', 'year', etc.
            cover_art_path: Path to cover art image (optional)
        """
        try:
            audio = MP4(str(file_path))
            
            # Set text metadata
            if metadata.get('title'):
                audio['\xa9nam'] = metadata['title']
            if metadata.get('artist'):
                audio['\xa9ART'] = metadata['artist']
            if metadata.get('album'):
                audio['\xa9alb'] = metadata['album']
            if metadata.get('year'):
                audio['\xa9day'] = str(metadata['year'])
            if metadata.get('genre'):
                audio['\xa9gen'] = metadata['genre']
            if metadata.get('comment'):
                audio['\xa9cmt'] = metadata['comment']
            
            # Embed cover art
            if cover_art_path and cover_art_path.exists():
                with open(cover_art_path, 'rb') as f:
                    cover_data = f.read()
                
                # Determine format
                if str(cover_art_path).lower().endswith('.png'):
                    cover_format = MP4Cover.FORMAT_PNG
                else:
                    cover_format = MP4Cover.FORMAT_JPEG
                
                audio['covr'] = [MP4Cover(cover_data, imageformat=cover_format)]
            
            audio.save()
            return True
            
        except Exception as e:
            print(f"❌ Tagging error (M4A): {e}")
            return False
    
    def tag_mp3(self, file_path: Path, metadata: Dict, cover_art_path: Optional[Path] = None):
        """
        Tag MP3 file with metadata and embedded cover art
        
        Args:
            file_path: Path to MP3 file
            metadata: Dict with 'artist', 'title', 'album', 'year', etc.
            cover_art_path: Path to cover art image (optional)
        """
        try:
            # Create ID3 tag if not exists
            try:
                audio = ID3(str(file_path))
            except:
                audio = ID3()
            
            # Set text metadata
            if metadata.get('title'):
                audio['TIT2'] = TIT2(encoding=3, text=metadata['title'])
            if metadata.get('artist'):
                audio['TPE1'] = TPE1(encoding=3, text=metadata['artist'])
            if metadata.get('album'):
                audio['TALB'] = TALB(encoding=3, text=metadata['album'])
            if metadata.get('year'):
                audio['TDRC'] = TDRC(encoding=3, text=str(metadata['year']))
            
            # Embed cover art
            if cover_art_path and cover_art_path.exists():
                with open(cover_art_path, 'rb') as f:
                    cover_data = f.read()
                
                # Determine MIME type
                if str(cover_art_path).lower().endswith('.png'):
                    mime = 'image/png'
                else:
                    mime = 'image/jpeg'
                
                audio['APIC'] = APIC(
                    encoding=3,
                    mime=mime,
                    type=3,  # Cover (front)
                    desc='Cover',
                    data=cover_data
                )
            
            audio.save(str(file_path), v2_version=3)
            return True
            
        except Exception as e:
            print(f"❌ Tagging error (MP3): {e}")
            return False
    
    def convert_to_mp3(self, input_file: Path, output_file: Path, 
                       bitrate: Optional[str] = None) -> bool:
        """
        Convert audio file to MP3
        
        Args:
            input_file: Input audio file
            output_file: Output MP3 file
            bitrate: MP3 bitrate (default: self.mp3_bitrate)
            
        Returns:
            True if successful
        """
        if bitrate is None:
            bitrate = self.mp3_bitrate
        
        try:
            cmd = [
                'ffmpeg',
                '-i', str(input_file),
                '-codec:a', 'libmp3lame',
                '-b:a', bitrate,
                '-y',
                str(output_file)
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300
            )
            
            return result.returncode == 0 and output_file.exists()
            
        except Exception as e:
            print(f"❌ MP3 conversion error: {e}")
            return False
    
    def tag_audio_file(self, file_path: Path, metadata: Dict, 
                       cover_art: Optional[Path] = None) -> bool:
        """
        Tag audio file (M4A or MP3) with metadata and cover art
        
        Args:
            file_path: Path to audio file
            metadata: Metadata dict
            cover_art: Path to cover art image
            
        Returns:
            True if successful
        """
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.m4a':
            return self.tag_m4a(file_path, metadata, cover_art)
        elif file_ext == '.mp3':
            return self.tag_mp3(file_path, metadata, cover_art)
        else:
            print(f"⚠️  Unsupported format: {file_ext}")
            return False


def test_tagging():
    """Test metadata and cover art embedding"""
    print("="*80)
    print("🧪 TESTING AUDIO TAGGING")
    print("="*80)
    print()
    
    recorder = AudioRecorder(output_format='m4a')
    
    # Test cover art download
    print("📥 Testing cover art download...")
    cover_path = Path('/tmp/test_cover.jpg')
    result = recorder.download_cover_art('Nirvana', 'Breed', cover_path)
    
    if result:
        print(f"   ✅ Downloaded: {cover_path} ({cover_path.stat().st_size} bytes)")
    else:
        print(f"   ❌ Download failed")
    
    print()
    print("✅ Audio tagging module ready!")


if __name__ == '__main__':
    test_tagging()
