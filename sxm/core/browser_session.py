#!/usr/bin/env python3
"""
Browser Session Manager

Handles browser automation for:
- Authentication (bearer token capture)
- Stream URL capture (HLS master playlist)
- Channel artwork

Based on proven v4 code - THIS WORKS!
"""

from playwright.sync_api import sync_playwright, Page
from typing import Dict, Optional, Tuple
from pathlib import Path
import time
import re
import shutil


class BrowserSession:
    """Manages browser session for SiriusXM"""
    
    def __init__(self, cookies: Dict[str, str], headless: bool = True):
        """
        Initialize browser session
        
        Args:
            cookies: Session cookies
            headless: Run in headless mode
        """
        self.cookies = cookies
        self.headless = headless
    
    def get_stream_info(self, channel_url: str, channel: Dict = None, art_dir: Path = None, progress_callback=None) -> Tuple[Optional[str], Optional[str]]:
        """
        Get bearer token and stream URL from browser session
        
        Optionally also capture channel artwork if channel and art_dir provided!
        
        Args:
            channel_url: Full channel URL (e.g., https://player.siriusxm.com/channels/...)
            channel: Channel dict (for art capture)
            art_dir: Directory to save channel art (for art capture)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (bearer_token, master_url) or (None, None)
        """
        captured = {'bearer': None, 'stream_url': None, 'art_captured': False}
        
        if progress_callback:
            progress_callback("🌐 Launching browser...")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            
            if progress_callback:
                progress_callback("🍪 Adding session cookies...")
            context = browser.new_context()
            
            # Add cookies
            context.add_cookies([
                {'name': k, 'value': v, 'domain': '.siriusxm.com', 'path': '/'}
                for k, v in self.cookies.items()
            ])
            
            page = context.new_page()
            
            # UPDATED: Use route interception (more reliable!)
            def handle_route(route):
                request = route.request
                
                # Capture bearer token
                auth = request.headers.get('authorization', '')
                if auth.startswith('Bearer ') and not captured['bearer']:
                    captured['bearer'] = auth.replace('Bearer ', '')
                    print(f"🔑 Captured bearer token!")
                
                # Capture stream URL
                url = request.url
                if '.m3u8' in url and 'siriusxm' in url and not captured['stream_url']:
                    captured['stream_url'] = url
                    print(f"🎬 Captured stream URL!")
                
                # Continue the request
                route.continue_()
            
            # Intercept ALL requests
            page.route('**/*', handle_route)
            
            # Navigate and wait for auth
            if progress_callback:
                progress_callback("🔐 Loading channel page...")
            
            print(f"Navigating to: {channel_url}")
            page.goto(channel_url, wait_until='domcontentloaded', timeout=30000)
            
            if progress_callback:
                progress_callback("⏳ Waiting for authentication...")
            page.wait_for_timeout(3000)
            
            # Click play button
            print("🖱️  Starting playback...")
            if not self._try_click_play(page):
                print("⚠️  Auto-click failed - waiting for stream...")
                page.wait_for_timeout(15000)  # Increased from 10s to 15s
            else:
                print("✅ Playing!")
                page.wait_for_timeout(3000)  # Increased from 2s to 3s
            
            # Wait for auth data (from working v5_headless!)
            print("⏳ Waiting for authentication...")
            timeout = 20  # Increased from 15s to 20s
            start = time.time()
            while (time.time() - start) < timeout:
                if captured['bearer']:
                    # Got authentication!
                    if progress_callback:
                        progress_callback("✅ Authentication captured!")
                    
                    # Capture channel art if requested (while browser is still open!)
                    if channel and art_dir:
                        if progress_callback:
                            progress_callback("🎨 Capturing channel artwork...")
                        
                        print("🎨 Capturing channel artwork...")
                        captured['art_captured'] = self.capture_channel_art(page, channel, art_dir)
                        if captured['art_captured']:
                            print("✅ Captured channel artwork")
                            if progress_callback:
                                progress_callback("✅ Channel artwork captured!")
                    
                    break  # Exit loop once we have auth
                
                page.wait_for_timeout(500)  # Check every 500ms
            
            browser.close()
        
        if captured['bearer'] and captured['stream_url']:
            print(f"✅ Got authentication")
            print(f"✅ Got stream URL")
            return captured['bearer'], captured['stream_url']
        else:
            print("❌ Failed to capture authentication")
            return None, None
    
    def _try_click_play(self, page: Page) -> bool:
        """
        Try to click play button
        
        EXACT code from working v5_headless!
        
        Args:
            page: Playwright page
            
        Returns:
            True if clicked successfully
        """
        selectors = [
            'button[aria-label="Play"]',
            'button[aria-label*="Play"]',
            'button[data-test*="play"]'
        ]
        
        for selector in selectors:
            try:
                button = page.wait_for_selector(selector, timeout=2000)
                if button:
                    button.click()
                    page.wait_for_timeout(2000)
                    return True
            except:
                continue
        
        return False
    
    def capture_channel_art(self, page: Page, channel: Dict, art_dir: Path) -> bool:
        """
        Capture channel artwork from page
        
        Args:
            page: Playwright page
            channel: Channel dict
            art_dir: Output directory
            
        Returns:
            True if successful
        """
        try:
            import requests
            import json
            
            art_dir.mkdir(parents=True, exist_ok=True)
            metadata = []
            saved_files = []
            
            # Wait a moment for page to load
            page.wait_for_timeout(2000)
            
            def classify_image(url: str) -> tuple:
                lower = url.lower()
                if any(k in lower for k in ['logo', 'channel', 'station']):
                    return ('channel_logo', 0)
                if any(k in lower for k in ['tile', 'square', '1x1']):
                    return ('channel_tile', 1)
                if any(k in lower for k in ['banner', 'hero', 'background', 'wide']):
                    return ('channel_banner', 2)
                if any(k in lower for k in ['cover', 'art', 'image']):
                    return ('channel_art', 3)
                return ('channel_image', 4)
            
            candidates = []
            def add_candidate(url):
                if not url or not url.startswith('http'):
                    return
                if any(url == c['url'] for c in candidates):
                    return
                label, priority = classify_image(url)
                candidates.append({'url': url, 'label': label, 'priority': priority})
            
            # Strategy 1: targeted selectors
            selectors = [
                'img[class*="channel"]',
                'img[class*="cover"]',
                'img[class*="art"]',
                'img[class*="logo"]',
                'div[class*="channel"] img',
                'div[class*="artwork"] img',
                '.channel-art img',
                '.cover-art img'
            ]
            for selector in selectors:
                try:
                    for elem in page.query_selector_all(selector):
                        src = elem.get_attribute('src')
                        add_candidate(src)
                        srcset = elem.get_attribute('srcset')
                        if srcset:
                            for entry in srcset.split(','):
                                url_part = entry.strip().split(' ')[0]
                                add_candidate(url_part)
                except Exception:
                    continue
            
            # Strategy 2: all <img> tags
            for img in page.query_selector_all('img'):
                src = img.get_attribute('src')
                add_candidate(src)
                srcset = img.get_attribute('srcset')
                if srcset:
                    for entry in srcset.split(','):
                        url_part = entry.strip().split(' ')[0]
                        add_candidate(url_part)
            
            # Strategy 3: background images via computed styles
            try:
                bg_images = page.evaluate("""
                    Array.from(document.querySelectorAll('*'))
                        .map(el => window.getComputedStyle(el).backgroundImage)
                        .filter(bg => bg && bg.startsWith('url('))
                """)
                for bg in bg_images:
                    matches = re.findall(r"url\((?:'|\")?(.*?)(?:'|\")?\)", bg)
                    for match in matches:
                        add_candidate(match)
            except Exception:
                pass
            
            if not candidates:
                return False
            
            # Sort by priority then keep top N
            candidates.sort(key=lambda c: c['priority'])
            max_images = 20
            saved_any = False
            for idx, candidate in enumerate(candidates[:max_images]):
                img_url = candidate['url']
                label = candidate['label']
                try:
                    response = requests.get(img_url, timeout=10)
                    if response.status_code == 200 and len(response.content) > 1000:
                        ext = 'jpg' if any(x in img_url.lower() for x in ['jpeg', 'jpg']) else 'png'
                        filename = f"{label}_{idx}.{ext}"
                        output_file = art_dir / filename
                        with open(output_file, 'wb') as f:
                            f.write(response.content)
                        print(f"   ✅ Saved: {output_file.name} ({len(response.content)} bytes)")
                        metadata.append({
                            'filename': filename,
                            'url': img_url,
                            'bytes': len(response.content),
                            'label': label,
                            'priority': candidate['priority']
                        })
                        saved_files.append({
                            'path': output_file,
                            'filename': filename,
                            'ext': ext,
                            'url': img_url,
                            'label': label,
                            'priority': candidate['priority']
                        })
                        saved_any = True
                except Exception as e:
                    print(f"   ⚠️  Failed to download {img_url[:60]}: {e}")
                    continue
            
            if metadata:
                with open(art_dir / 'channel_art_index.json', 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
            
            if saved_files:
                cover_target = None
                channel_name = (channel.get('name') if channel else '') or ''
                sanitized = re.sub(r'\s+', '', channel_name.lower())
                for item in saved_files:
                    url_lower = item['url'].lower()
                    if sanitized and sanitized in url_lower and item['filename'].startswith('channel_image'):
                        cover_target = item
                        break
                if not cover_target:
                    cover_target = saved_files[0]
                cover_path = art_dir / f"cover.{cover_target['ext']}"
                shutil.copyfile(cover_target['path'], cover_path)
            
            return saved_any
                    
        except Exception as e:
            print(f"⚠️  Channel art capture error: {e}")
        
        return False
