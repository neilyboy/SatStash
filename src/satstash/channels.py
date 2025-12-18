import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from platformdirs import user_cache_dir


@dataclass
class Channel:
    id: str
    name: str
    number: Optional[int] = None
    description: Optional[str] = None
    genre: Optional[str] = None
    logo_url: Optional[str] = None
    channel_type: str = "channel-linear"


_CHANNELS_CACHE_SCHEMA_VERSION = 3


def channels_cache_path() -> Path:
    cache_dir = Path(user_cache_dir("satstash"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "channels.json"


def load_cached_channels(*, max_age_hours: int = 24) -> Optional[List[Channel]]:
    path = channels_cache_path()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        schema_version = int(raw.get("schema_version") or 0)
        if schema_version < _CHANNELS_CACHE_SCHEMA_VERSION:
            return None
        ts = raw.get("timestamp")
        if ts is None:
            return None
        fetched = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        if datetime.now(timezone.utc) - fetched > timedelta(hours=max_age_hours):
            return None
        items = raw.get("channels") or []
        out: List[Channel] = []
        for item in items:
            cid = item.get("id")
            name = item.get("name")
            if not cid or not name:
                continue
            out.append(
                Channel(
                    id=str(cid),
                    name=str(name),
                    number=item.get("number"),
                    description=item.get("description"),
                    genre=item.get("genre"),
                    logo_url=item.get("logo_url"),
                    channel_type=item.get("channel_type") or "channel-linear",
                )
            )

        # If an old cache was saved without channel numbers, force refresh.
        if out and all(ch.number is None for ch in out):
            return None
        return out
    except Exception:
        return None


def save_cached_channels(channels: List[Channel]) -> None:
    path = channels_cache_path()
    payload: Dict[str, Any] = {
        "schema_version": _CHANNELS_CACHE_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).timestamp(),
        "channels": [
            {
                "id": ch.id,
                "name": ch.name,
                "number": ch.number,
                "description": ch.description,
                "genre": ch.genre,
                "logo_url": ch.logo_url,
                "channel_type": ch.channel_type,
            }
            for ch in channels
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def clear_channels_cache() -> None:
    path = channels_cache_path()
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass
