import json
from dataclasses import dataclass, asdict
from pathlib import Path

from platformdirs import user_config_dir


@dataclass
class Settings:
    preferred_quality: str = "256k"
    output_dir: str = "~/Music/SiriusXM"
    recordings_dir: str = "~/Music/SatStash"
    player_preference: str = "mpv"

    art_render_mode: str = "halfblock"

    # When enabled, recordings start from the current track start (DVR buffer)
    # instead of the live edge.
    start_record_from_track_start: bool = True

    # Optional credentials for automatic, non-interactive re-login.
    # NOTE: Stored in plaintext in config.json.
    auth_username: str = ""
    auth_password: str = ""


def config_path() -> Path:
    cfg_dir = Path(user_config_dir("satstash"))
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir / "config.json"


def load_settings() -> Settings:
    path = config_path()
    if not path.exists():
        return Settings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return Settings(**{k: v for k, v in raw.items() if k in Settings.__annotations__})
    except Exception:
        return Settings()


def save_settings(settings: Settings) -> None:
    path = config_path()
    path.write_text(json.dumps(asdict(settings), indent=2) + "\n", encoding="utf-8")
