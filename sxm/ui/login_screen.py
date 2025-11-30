#!/usr/bin/env python3
"""
Login Screen

Authenticate with SiriusXM
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Button, Input, Label
from textual.containers import Container, Vertical, Horizontal
from textual.binding import Binding
import threading


class LoginScreen(Screen):
    """Login to SiriusXM"""
    
    BINDINGS = [
        Binding("escape", "app.pop_screen", "Cancel", priority=True),
        Binding("enter", "login", "Login", priority=True),
    ]
    
    CSS = """
    LoginScreen {
        layout: vertical;
        align: center middle;
        background: $surface;
    }
    
    #login_container {
        width: 60;
        height: auto;
        border: solid $primary;
        padding: 2;
        background: $surface;
    }
    
    #title {
        text-align: center;
        color: $text;
        text-style: bold;
        margin-bottom: 1;
        background: $primary;
    }
    
    #subtitle {
        text-align: center;
        margin-bottom: 2;
        color: $text;
    }
    
    .input_row {
        height: auto;
        margin-bottom: 1;
    }
    
    .input_label {
        width: 15;
        padding-right: 1;
        color: $text;
    }
    
    .input_field {
        width: 1fr;
    }
    
    #button_row {
        height: auto;
        align: center middle;
        margin-top: 1;
    }
    
    #status {
        height: 6;
        padding: 1;
        margin-top: 1;
        border: solid $accent;
        background: $surface-darken-1;
    }
    
    #status_text {
        text-align: center;
        color: $text;
        text-style: bold;
    }
    
    .error {
        color: $error;
    }
    
    .success {
        color: $success;
    }
    
    .info {
        color: $primary;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.login_in_progress = False
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(id="login_container"):
            yield Label("🔐 SiriusXM Login", id="title")
            yield Label("Enter your SiriusXM credentials", id="subtitle")
            
            with Horizontal(classes="input_row"):
                yield Label("Username:", classes="input_label")
                yield Input(
                    placeholder="Email or username",
                    id="username_input",
                    classes="input_field"
                )
            
            with Horizontal(classes="input_row"):
                yield Label("Password:", classes="input_label")
                yield Input(
                    placeholder="Password",
                    password=True,
                    id="password_input",
                    classes="input_field"
                )
            
            with Horizontal(id="button_row"):
                yield Button("🔐 Login", id="btn_login", variant="primary")
                yield Button("🚪 Cancel", id="btn_cancel")
            
            with Container(id="status"):
                yield Label("", id="status_text")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Focus username input on mount"""
        self.query_one("#username_input", Input).focus()
    
    def action_login(self) -> None:
        """Login action"""
        if not self.login_in_progress:
            self.perform_login()
    
    def perform_login(self) -> None:
        """Perform the login"""
        if self.login_in_progress:
            self.update_status("Login already in progress...", "info")
            return
        
        # Get credentials
        username = self.query_one("#username_input", Input).value.strip()
        password = self.query_one("#password_input", Input).value
        
        # Validate
        if not username:
            self.update_status("❌ Username is required", "error")
            return
        
        if not password:
            self.update_status("❌ Password is required", "error")
            return
        
        # Start login in background thread
        self.login_in_progress = True
        self.update_status("🔄 Logging in... This may take 10-15 seconds, please wait...", "info")
        
        # Disable inputs
        self.query_one("#username_input", Input).disabled = True
        self.query_one("#password_input", Input).disabled = True
        self.query_one("#btn_login", Button).disabled = True
        
        # Run login in thread to avoid blocking UI
        thread = threading.Thread(target=self._login_thread, args=(username, password))
        thread.daemon = True
        thread.start()
    
    def _login_thread(self, username: str, password: str) -> None:
        """Login thread (runs in background)"""
        try:
            from sxm.core.auth import SiriusXMAuth
            from sxm.utils.session_manager import SessionManager
            
            # Update status - opening browser
            self.app.call_from_thread(
                self.update_status,
                "🌐 Opening browser...",
                "info"
            )
            
            # Create authenticator
            auth = SiriusXMAuth(headless=True, debug=False)
            
            # Update status - logging in
            self.app.call_from_thread(
                self.update_status,
                "🔐 Authenticating with SiriusXM...",
                "info"
            )
            
            # Perform login
            cookies = auth.login(username, password)
            
            if cookies:
                # Update status - saving session
                self.app.call_from_thread(
                    self.update_status,
                    "💾 Saving session...",
                    "info"
                )
                
                # Save session (IMPORTANT: cookies first, then username!)
                session_mgr = SessionManager()
                session_mgr.save_session(cookies, username)
                
                # Success!
                self.app.call_from_thread(
                    self.login_complete,
                    True,
                    "✅ Login successful!"
                )
            else:
                self.app.call_from_thread(
                    self.login_complete,
                    False,
                    "❌ Login failed - no cookies received"
                )
        
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                error_msg = "❌ Login timeout - please try again"
            elif "password" in error_msg.lower() or "credentials" in error_msg.lower():
                error_msg = "❌ Invalid username or password"
            else:
                error_msg = f"❌ Login failed: {error_msg}"
            
            self.app.call_from_thread(
                self.login_complete,
                False,
                error_msg
            )
    
    def login_complete(self, success: bool, message: str) -> None:
        """Called when login completes"""
        self.login_in_progress = False
        
        # Re-enable inputs
        self.query_one("#username_input", Input).disabled = False
        self.query_one("#password_input", Input).disabled = False
        self.query_one("#btn_login", Button).disabled = False
        
        if success:
            self.update_status(message, "success")
            self.app.notify("Login successful!", severity="success")
            
            # Refresh main app session status
            if hasattr(self.app, 'check_session'):
                self.app.call_later(self.app.check_session)
            
            # Close login screen after a moment
            self.set_timer(1.5, self.close_screen)
        else:
            self.update_status(message, "error")
            self.app.notify("Login failed", severity="error")
    
    def close_screen(self) -> None:
        """Close the login screen"""
        self.app.pop_screen()
    
    def update_status(self, message: str, status_type: str = "info") -> None:
        """Update status message"""
        status_label = self.query_one("#status_text", Label)
        status_label.update(message)
        
        # Update styling based on type
        if status_type == "error":
            status_label.add_class("error")
            status_label.remove_class("success")
            status_label.remove_class("info")
        elif status_type == "success":
            status_label.add_class("success")
            status_label.remove_class("error")
            status_label.remove_class("info")
        else:
            status_label.add_class("info")
            status_label.remove_class("error")
            status_label.remove_class("success")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id
        
        if button_id == "btn_login":
            self.perform_login()
        elif button_id == "btn_cancel":
            self.app.pop_screen()
