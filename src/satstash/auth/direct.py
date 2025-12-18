import json
from dataclasses import dataclass
from typing import Dict, Optional

import requests


@dataclass
class DirectAuthResult:
    bearer_token: str
    cookies: Dict[str, str]


class SiriusXMDirectAuth:
    """API-only auth modeled after m3u8XM.

    Flow:
      - device/v1/devices (x-sxm-tenant: sxm)
      - session/v1/sessions/anonymous
      - identity/v1/identities/authenticate/password
      - session/v1/sessions/authenticated

    Returns a bearer token that is usable for playback endpoints.
    """

    BASE_URL = "https://api.edge-gateway.siriusxm.com"
    USER_AGENT = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})

    def refresh_with_cookies(self, cookies: Dict[str, str]) -> Optional[DirectAuthResult]:
        sess = requests.Session()
        sess.headers.update({"User-Agent": self.USER_AGENT})
        if cookies:
            sess.cookies.update(cookies)

        sxmheaders = {"x-sxm-tenant": "sxm"}
        device_payload = {
            "devicePlatform": "web-desktop",
            "deviceAttributes": {
                "browser": {
                    "browserVersion": "7.74.0",
                    "userAgent": self.USER_AGENT,
                    "sdk": "web",
                    "app": "web",
                    "sdkVersion": "7.74.0",
                    "appVersion": "7.74.0",
                }
            },
            "grantVersion": "v2",
        }

        try:
            r = sess.post(
                f"{self.BASE_URL}/device/v1/devices",
                headers=sxmheaders,
                data=json.dumps(device_payload),
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            bearer = data.get("grant") or data.get("accessToken")
            if bearer:
                sess.headers.update({"Authorization": f"Bearer {bearer}"})

            r = sess.post(
                f"{self.BASE_URL}/session/v1/sessions/authenticated",
                headers=sxmheaders,
                data="{}",
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            bearer = data.get("accessToken") or data.get("grant")
            if bearer:
                sess.headers.update({"Authorization": f"Bearer {bearer}"})

            auth_header = sess.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return None
            bearer_token = auth_header.replace("Bearer ", "")
            out_cookies = {c.name: c.value for c in sess.cookies}
            return DirectAuthResult(bearer_token=bearer_token, cookies=out_cookies)
        except Exception:
            return None

    def authenticate(self, username: str, password: str) -> DirectAuthResult:
        sxmheaders = {"x-sxm-tenant": "sxm"}

        device_payload = {
            "devicePlatform": "web-desktop",
            "deviceAttributes": {
                "browser": {
                    "browserVersion": "7.74.0",
                    "userAgent": self.USER_AGENT,
                    "sdk": "web",
                    "app": "web",
                    "sdkVersion": "7.74.0",
                    "appVersion": "7.74.0",
                }
            },
            "grantVersion": "v2",
        }

        r = self.session.post(
            f"{self.BASE_URL}/device/v1/devices",
            headers=sxmheaders,
            data=json.dumps(device_payload),
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        bearer = data.get("grant") or data.get("accessToken")
        if bearer:
            self.session.headers.update({"Authorization": f"Bearer {bearer}"})

        r = self.session.post(
            f"{self.BASE_URL}/session/v1/sessions/anonymous",
            headers=sxmheaders,
            data="{}",
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        bearer = data.get("accessToken") or data.get("grant")
        if bearer:
            self.session.headers.update({"Authorization": f"Bearer {bearer}"})

        ident_payload = {"handle": username, "password": password}
        r = self.session.post(
            f"{self.BASE_URL}/identity/v1/identities/authenticate/password",
            data=json.dumps(ident_payload),
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        bearer = data.get("accessToken") or data.get("grant")
        if bearer:
            self.session.headers.update({"Authorization": f"Bearer {bearer}"})

        r = self.session.post(
            f"{self.BASE_URL}/session/v1/sessions/authenticated",
            data="{}",
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        bearer = data.get("accessToken") or data.get("grant")
        if bearer:
            self.session.headers.update({"Authorization": f"Bearer {bearer}"})

        auth_header = self.session.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise RuntimeError("Direct auth succeeded but did not yield a bearer token")

        bearer_token = auth_header.replace("Bearer ", "")
        cookies = {c.name: c.value for c in self.session.cookies}
        return DirectAuthResult(bearer_token=bearer_token, cookies=cookies)
