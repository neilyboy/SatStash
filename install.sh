#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
SatStash installer

Usage: ./install.sh [options]

Options:
  --install-service    Configure and enable the SatStash scheduler as a
                       user-level systemd service (requires systemd --user)
  --skip-playwright    Skip Playwright browser installation (if already done)
  -h, --help           Show this help message
EOF
}

INSTALL_SERVICE=false
SKIP_PLAYWRIGHT=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-service)
            INSTALL_SERVICE=true
            shift
            ;;
        --skip-playwright)
            SKIP_PLAYWRIGHT=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/venv"
PYTHON=${PYTHON:-python3}
REQUIREMENTS_FILE="$ROOT_DIR/requirements.txt"
BOOTSTRAP_INFO_FILE="$VENV_DIR/.satstash_bootstrap"
CURRENT_HOST="$(hostname -f 2>/dev/null || hostname || echo unknown)"
VENV_PYTHON="$VENV_DIR/bin/python"

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    echo "requirements.txt not found in $ROOT_DIR" >&2
    exit 1
fi

ensure_ffplay() {
    if command -v ffplay >/dev/null 2>&1; then
        return
    fi

    echo "ffplay (from the ffmpeg package) is required for Listen Live terminal playback."

    if command -v apt-get >/dev/null 2>&1; then
        echo "Attempting to install ffmpeg via apt-get (may prompt for sudo password)..."
        if sudo apt-get update && sudo apt-get install -y ffmpeg; then
            echo "ffplay/ffmpeg installed successfully."
            return
        else
            echo "Could not install ffmpeg automatically. Please install it manually (e.g. 'sudo apt-get install ffmpeg')." >&2
        fi
    else
        echo "Please install ffmpeg (which provides ffplay) using your distribution's package manager." >&2
    fi
}

bootstrap_virtualenv() {
    local needs_bootstrap=false

    if [[ ! -d "$VENV_DIR" ]]; then
        needs_bootstrap=true
    else
        if [[ ! -x "$VENV_PYTHON" ]]; then
            needs_bootstrap=true
        else
            if [[ -f "$BOOTSTRAP_INFO_FILE" ]]; then
                # shellcheck disable=SC1090
                source "$BOOTSTRAP_INFO_FILE"
            else
                needs_bootstrap=true
            fi

            if [[ "${SATSTASH_ROOT_DIR:-}" != "$ROOT_DIR" ]] || [[ "${SATSTASH_HOST:-}" != "$CURRENT_HOST" ]]; then
                needs_bootstrap=true
            fi
        fi
    fi

    if [[ "$needs_bootstrap" == true ]]; then
        if [[ -d "$VENV_DIR" ]]; then
            echo "Existing virtual environment is incompatible; recreating..."
            rm -rf "$VENV_DIR"
        else
            echo "Creating virtual environment..."
        fi

        if ! "$PYTHON" -m venv "$VENV_DIR"; then
            echo "Failed to create virtual environment. Ensure python3-venv is installed." >&2
            exit 1
        fi
    fi
}

bootstrap_virtualenv

VENV_PYTHON="$VENV_DIR/bin/python"

echo "Upgrading pip..."
"$VENV_PYTHON" -m pip install --upgrade pip >/dev/null

echo "Installing Python dependencies..."
"$VENV_PYTHON" -m pip install -r "$REQUIREMENTS_FILE"

echo "Ensuring Pillow (for ASCII artwork) is installed..."
"$VENV_PYTHON" -m pip install --upgrade Pillow >/dev/null

if [[ "$SKIP_PLAYWRIGHT" = false ]]; then
    echo "Installing Playwright browser (chromium)..."
    "$VENV_PYTHON" -m playwright install chromium
fi

echo "Checking for ffplay (ffmpeg) for terminal audio playback..."
ensure_ffplay

record_bootstrap_metadata() {
    local py_version
    py_version="$("$VENV_PYTHON" -c 'import platform; print(platform.python_version())')"

    {
        printf 'SATSTASH_ROOT_DIR=%q\n' "$ROOT_DIR"
        printf 'SATSTASH_HOST=%q\n' "$CURRENT_HOST"
        printf 'SATSTASH_PY_VERSION=%q\n' "$py_version"
    } > "$BOOTSTRAP_INFO_FILE"
}

record_bootstrap_metadata

chmod +x "$ROOT_DIR"/sxm_app "$ROOT_DIR"/run.sh "$ROOT_DIR"/record "$ROOT_DIR"/record_now.py "$ROOT_DIR"/scripts/run_scheduler.sh 2>/dev/null || true

install_launcher() {
    local bin_dir="$HOME/.local/bin"
    mkdir -p "$bin_dir"
    local launcher="$bin_dir/satstash"

    cat > "$launcher" <<EOF
#!/bin/bash
ROOT_DIR="$ROOT_DIR"

if [[ "${1:-}" == "--install-service" ]]; then
    shift
    exec "$ROOT_DIR/install.sh" --install-service "$@"
fi

cd "$ROOT_DIR"
exec "$ROOT_DIR/sxm_app" "$@"
EOF

    chmod +x "$launcher"
    echo "✅ Installed satstash launcher at $launcher"
    echo "   (Make sure \"$HOME/.local/bin\" is in your PATH)"
}

install_launcher

configure_service() {
    if ! command -v systemctl >/dev/null; then
        echo "systemctl not found; skipping scheduler service setup."
        echo "You can run the scheduler manually via scripts/run_scheduler.sh"
        return
    fi

    if ! systemctl --user status >/dev/null 2>&1; then
        echo "systemd user services are not available in this session."
        echo "Log in via a graphical session or use scripts/run_scheduler.sh manually."
        return
    fi

    local systemd_dir="$HOME/.config/systemd/user"
    mkdir -p "$systemd_dir"
    local service_file="$systemd_dir/satstash-scheduler.service"

    cat > "$service_file" <<EOF
[Unit]
Description=SatStash Scheduler Daemon
After=network.target

[Service]
Type=simple
WorkingDirectory=$ROOT_DIR
ExecStart=$ROOT_DIR/scripts/run_scheduler.sh
Restart=on-failure
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
EOF

    echo "Reloading user systemd daemon..."
    systemctl --user daemon-reload
    echo "Enabling and starting satstash-scheduler.service..."
    systemctl --user enable --now satstash-scheduler.service
    echo "Scheduler service is running. Check status with: systemctl --user status satstash-scheduler"
}

install_launcher

if [[ "$INSTALL_SERVICE" = true ]]; then
    configure_service
else
    cat <<'EOF'
SatStash installation complete!

To run the CLI:      ./sxm_app
To start scheduler:  ./scripts/run_scheduler.sh (or rerun install.sh --install-service)
EOF
fi
