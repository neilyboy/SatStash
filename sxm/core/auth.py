"""SiriusXM authentication using Playwright."""
from playwright.sync_api import sync_playwright, Page, Browser, TimeoutError as PlaywrightTimeoutError
import time
from typing import Optional, Dict
from pathlib import Path


class SiriusXMAuth:
    """Handles SiriusXM authentication via headless browser."""
    
    def __init__(self, headless: bool = True, debug: bool = False, capture_network: bool = False):
        """Initialize authenticator.
        
        Args:
            headless: Whether to run browser in headless mode.
            debug: Enable debug mode with screenshots and detailed logging.
            capture_network: Capture and log all network traffic.
        """
        self.headless = headless
        self.debug = debug
        self.capture_network = capture_network
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        
        # Create debug directory if needed
        if debug:
            self.debug_dir = Path.home() / ".seriouslyxm" / "debug"
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            print(f"🐛 Debug mode enabled. Files will be saved to: {self.debug_dir}")
    
    def login(self, username: str, password: str) -> Dict[str, str]:
        """Login to SiriusXM and retrieve session cookies.
        
        Args:
            username: SiriusXM username or email.
            password: SiriusXM password.
            
        Returns:
            dict: Dictionary of cookies for authenticated session.
            
        Raises:
            Exception: If login fails.
        """
        print("🔐 Starting authentication process...")
        
        try:
            self.playwright = sync_playwright().start()
            
            # Launch browser with additional args for debugging
            launch_args = {
                'headless': self.headless,
            }
            
            if self.debug:
                # Keep browser open longer, enable devtools
                launch_args['devtools'] = False  # Set to True to open devtools
                launch_args['slow_mo'] = 1000  # Slow down operations by 1 second
            
            self.browser = self.playwright.chromium.launch(**launch_args)
            
            # Create context with viewport
            context = self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            
            self.page = context.new_page()
            
            # Setup network logging if requested
            # Network logging removed in production version
            if self.capture_network and self.debug:
                print("📡 Network capture available in debug mode")
            
            # Step 1: Navigate to player page which should prompt for login
            print("📡 Navigating to player page...")
            self.page.goto("https://player.siriusxm.com/", wait_until="domcontentloaded", timeout=60000)
            time.sleep(5)  # Wait for page to settle
            
            if self.debug:
                self.page.screenshot(path=str(self.debug_dir / "01_player_page.png"))
                print(f"📸 Screenshot saved: 01_player_page.png")
            
            # Step 2: Click Sign In / Login button
            print("🖱️  Looking for Sign In button...")
            try:
                # Try various sign in button selectors
                sign_in_selectors = [
                    'button:has-text("Sign In")',
                    'button:has-text("Log In")',
                    'button:has-text("Login")',
                    'a:has-text("Sign In")',
                    'a:has-text("Log In")',
                    '[data-qa*="sign"]',
                    '[data-qa*="login"]'
                ]
                
                clicked = False
                for selector in sign_in_selectors:
                    try:
                        button = self.page.locator(selector).first
                        if button.is_visible(timeout=2000):
                            print(f"✅ Found Sign In button, clicking...")
                            button.click()
                            time.sleep(3)
                            clicked = True
                            break
                    except:
                        continue
                
                if not clicked:
                    print("⚠️  No sign in button found, login form may already be visible")
                
                if self.debug:
                    self.page.screenshot(path=str(self.debug_dir / "02_after_signin_click.png"))
                    print(f"📸 Screenshot saved: 02_after_signin_click.png")
                    
            except Exception as e:
                print(f"⚠️  Error finding sign in button: {e}")
            
            # Step 3: Enter username/email
            print(f"✏️  Entering username: {username}")
            email_input = self.page.locator('input[data-qa="email-field"]')
            email_input.wait_for(timeout=10000)
            email_input.fill(username)
            time.sleep(1)
            
            if self.debug:
                self.page.screenshot(path=str(self.debug_dir / "03_username_entered.png"))
                print(f"📸 Screenshot saved: 03_username_entered.png")
            
            # Step 4: Submit username / click continue
            print("🖱️  Looking for Continue/Next button after username...")
            try:
                continue_selectors = [
                    'button[data-qa="submit-auth-email"]',
                    'button:has-text("Continue")',
                    'button:has-text("Next")',
                    'button[type="submit"]'
                ]
                
                submitted = False
                for selector in continue_selectors:
                    try:
                        button = self.page.locator(selector).first
                        if button.is_visible(timeout=2000):
                            print(f"✅ Submitting username...")
                            button.click()
                            time.sleep(3)  # Wait for next screen
                            submitted = True
                            break
                    except:
                        continue
                
                if not submitted:
                    print("⚠️  No continue button found, trying to press Enter...")
                    email_input.press("Enter")
                    time.sleep(3)
                
                if self.debug:
                    self.page.screenshot(path=str(self.debug_dir / "04_after_username_submit.png"))
                    print(f"📸 Screenshot saved: 04_after_username_submit.png")
                    
            except Exception as e:
                print(f"⚠️  Error submitting username: {e}")
            
            # Step 5: Wait for authentication options and select password
            print("🔑 Waiting for authentication options...")
            time.sleep(3)  # Give page time to show auth options
            
            if self.debug:
                self.page.screenshot(path=str(self.debug_dir / "05_auth_options.png"))
                print(f"📸 Screenshot saved: 05_auth_options.png")
            
            try:
                # Click the password authentication option
                password_option = self.page.locator('[data-qa="password-auth-option"]').first
                if password_option.is_visible(timeout=5000):
                    print("🔑 Clicking password authentication option...")
                    password_option.click()
                    time.sleep(2)
                    
                    # Click continue/submit button
                    print("🖱️  Clicking continue...")
                    continue_btn = self.page.locator('button[data-qa="submit-auth-option"]')
                    continue_btn.wait_for(timeout=5000)
                    continue_btn.click()
                    time.sleep(3)  # Wait for password field to appear
                    
                    if self.debug:
                        self.page.screenshot(path=str(self.debug_dir / "06_password_selected.png"))
                        print(f"📸 Screenshot saved: 06_password_selected.png")
                else:
                    print("⚠️  Password option not visible, checking if password field is already shown...")
            except Exception as e:
                print(f"⚠️  Could not select password option: {e}")
                if self.debug:
                    self.page.screenshot(path=str(self.debug_dir / "06_password_option_error.png"))
            
            # Step 6: Enter password
            print("🔐 Looking for password field...")
            
            # Try multiple selectors for password field
            password_input = None
            selectors_to_try = [
                'input[data-qa="password-field"]',
                'input[type="password"]',
                'input[name="password"]',
                'input[placeholder*="assword" i]'
            ]
            
            for selector in selectors_to_try:
                try:
                    temp_input = self.page.locator(selector).first
                    if temp_input.is_visible(timeout=3000):
                        password_input = temp_input
                        print(f"✅ Found password field with selector: {selector}")
                        break
                except:
                    continue
            
            if not password_input:
                if self.debug:
                    self.page.screenshot(path=str(self.debug_dir / "07_password_field_not_found.png"))
                raise Exception("Could not find password field with any known selector")
            
            print("🔐 Entering password...")
            password_input.fill(password)
            time.sleep(1)
            
            if self.debug:
                self.page.screenshot(path=str(self.debug_dir / "07_password_entered.png"))
                print(f"📸 Screenshot saved: 07_password_entered.png")
            
            # Step 7: Click final Continue button
            print("✅ Submitting credentials...")
            final_continue = self.page.locator('button:has-text("Continue")').last
            final_continue.click()
            
            # Wait for successful login
            print("⏳ Waiting for login to complete...")
            time.sleep(5)
            
            if self.debug:
                self.page.screenshot(path=str(self.debug_dir / "08_after_login_submit.png"))
                print(f"📸 Screenshot saved: 08_after_login_submit.png")
            
            # Check if we're logged in by looking for player or account elements
            try:
                # Wait for navigation away from login page
                self.page.wait_for_url("**/player/**", timeout=15000)
                print("✅ Login successful!")
                
                if self.debug:
                    self.page.screenshot(path=str(self.debug_dir / "09_login_success.png"))
                    print(f"📸 Screenshot saved: 09_login_success.png")
                    
            except PlaywrightTimeoutError:
                # Check if we're on any authenticated page
                current_url = self.page.url
                print(f"🔍 Current URL: {current_url}")
                
                if self.debug:
                    self.page.screenshot(path=str(self.debug_dir / "09_login_uncertain.png"))
                    print(f"📸 Screenshot saved: 09_login_uncertain.png")
                
                if "login" not in current_url.lower():
                    print("✅ Login successful!")
                else:
                    # Check for error messages
                    error_elements = self.page.locator('text=/error|invalid|incorrect/i')
                    if error_elements.count() > 0:
                        if self.debug:
                            self.page.screenshot(path=str(self.debug_dir / "09_login_error.png"))
                        raise Exception("❌ Login failed: Invalid credentials")
                    print("⚠️  Login state uncertain, proceeding...")
            
            # Extract cookies
            cookies = self.page.context.cookies()
            cookie_dict = {cookie['name']: cookie['value'] for cookie in cookies}
            
            print(f"🍪 Retrieved {len(cookie_dict)} cookies")
            
            if self.debug:
                print("🍪 Cookie names:", list(cookie_dict.keys()))
            
            return cookie_dict
            
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            
            if self.debug and self.page:
                try:
                    self.page.screenshot(path=str(self.debug_dir / "99_error.png"))
                    print(f"📸 Error screenshot saved: 99_error.png")
                    
                    # Save page HTML for inspection
                    html_content = self.page.content()
                    with open(self.debug_dir / "99_error_page.html", 'w') as f:
                        f.write(html_content)
                    print(f"📄 Page HTML saved: 99_error_page.html")
                except:
                    pass
            
            raise
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up browser resources."""
        if self.page:
            self.page.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()


# Alias for backward compatibility
SiriusXMAuthenticator = SiriusXMAuth
