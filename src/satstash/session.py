import json
import base64
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict

from platformdirs import user_cache_dir


class SatStashAuthError(RuntimeError):
    pass


class NotLoggedInError(SatStashAuthError):
    pass


class SessionExpiredError(SatStashAuthError):
    pass


@dataclass
class Session:
    bearer_token: str
    cookies: Dict[str, str]
    created_at: str
    expires_at: str

    def is_valid(self) -> bool:
        try:
            parts = (self.bearer_token or "").split(".")
            if len(parts) == 3:
                payload_b64 = parts[1]
                pad = "=" * (-len(payload_b64) % 4)
                payload_raw = base64.urlsafe_b64decode(payload_b64 + pad)
                payload = json.loads(payload_raw.decode("utf-8"))
                exp_ts = payload.get("exp")
                if exp_ts is not None:
                    exp_dt = datetime.fromtimestamp(float(exp_ts), tz=timezone.utc)
                    return datetime.now(timezone.utc) < exp_dt
            exp = datetime.fromisoformat(self.expires_at)
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) < exp
        except Exception:
            return False

    def expires_in_seconds(self) -> Optional[float]:
        try:
            parts = (self.bearer_token or "").split(".")
            if len(parts) == 3:
                payload_b64 = parts[1]
                pad = "=" * (-len(payload_b64) % 4)
                payload_raw = base64.urlsafe_b64decode(payload_b64 + pad)
                payload = json.loads(payload_raw.decode("utf-8"))
                exp_ts = payload.get("exp")
                if exp_ts is not None:
                    exp_dt = datetime.fromtimestamp(float(exp_ts), tz=timezone.utc)
                    return (exp_dt - datetime.now(timezone.utc)).total_seconds()
            exp = datetime.fromisoformat(self.expires_at)
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            return (exp - datetime.now(timezone.utc)).total_seconds()
        except Exception:
            return None

    def is_expiring_soon(self, *, threshold_seconds: int = 300) -> bool:
        remaining = self.expires_in_seconds()
        if remaining is None:
            return False
        return remaining <= float(threshold_seconds)


def session_path() -> Path:
    cache_dir = Path(user_cache_dir("satstash"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "session.json"


def load_session() -> Optional[Session]:
    path = session_path()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        sess = Session(
            bearer_token=str(raw.get("bearer_token") or ""),
            cookies=dict(raw.get("cookies") or {}),
            created_at=str(raw.get("created_at") or ""),
            expires_at=str(raw.get("expires_at") or ""),
        )
        if not sess.bearer_token:
            return None
        return sess
    except Exception:
        return None


def save_session(bearer_token: str, cookies: Dict[str, str], *, lifetime_hours: int = 12) -> Session:
    now = datetime.now(timezone.utc)
    sess = Session(
        bearer_token=bearer_token,
        cookies=cookies,
        created_at=now.isoformat(),
        expires_at=(now + timedelta(hours=lifetime_hours)).isoformat(),
    )
    path = session_path()
    path.write_text(json.dumps(asdict(sess), indent=2) + "\n", encoding="utf-8")
    return sess


def update_session(sess: Session) -> Session:
    path = session_path()
    path.write_text(json.dumps(asdict(sess), indent=2) + "\n", encoding="utf-8")
    return sess


def clear_session() -> None:
    path = session_path()
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass
