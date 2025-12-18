from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests


@dataclass
class TuneSourceResult:
    raw: Dict[str, Any]

    def master_url(self) -> Optional[str]:
        streams = self.raw.get("streams") or []
        if not streams:
            return None

        # tuneSource responses commonly contain multiple URLs (master + variants).
        # Prefer the true master playlist if present.
        candidates: list[str] = []
        preferred: list[str] = []

        for s in streams:
            urls = s.get("urls") or []
            for u in urls:
                try:
                    url = u.get("url")
                except Exception:
                    url = None
                if not url:
                    continue
                url_s = str(url).strip()
                if not url_s:
                    continue
                candidates.append(url_s)

                # Heuristics based on observed payloads.
                try:
                    typ = str(u.get("type") or u.get("urlType") or u.get("role") or "").strip().lower()
                except Exception:
                    typ = ""
                if "master" in typ:
                    preferred.append(url_s)
                    continue
                if "master" in url_s.lower():
                    preferred.append(url_s)

        if preferred:
            return preferred[0]
        if candidates:
            return candidates[0]
        return None


class SxmClient:
    BASE_URL = "https://api.edge-gateway.siriusxm.com"

    def __init__(
        self,
        *,
        bearer_token: str,
        cookies: Optional[Dict[str, str]] = None,
        on_unauthorized: Optional[Callable[[], Optional[Tuple[str, Dict[str, str]]]]] = None,
    ) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {bearer_token}",
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            }
        )
        if cookies:
            self.session.cookies.update(cookies)

        self._on_unauthorized = on_unauthorized

        self.last_status: Optional[int] = None

    def set_bearer(self, bearer_token: str) -> None:
        self.session.headers["Authorization"] = f"Bearer {bearer_token}"

    def _request(self, method: str, url: str, *, _retried: bool = False, **kwargs) -> requests.Response:
        r = self.session.request(method, url, timeout=20, **kwargs)
        self.last_status = r.status_code
        if r.status_code == 401 and (not _retried) and self._on_unauthorized is not None:
            updated = self._on_unauthorized()
            if updated:
                bearer, cookies = updated
                self.set_bearer(bearer)
                if cookies:
                    self.session.cookies.update(cookies)
                r = self.session.request(method, url, timeout=20, **kwargs)
                self.last_status = r.status_code
        return r

    def tune_source(
        self,
        *,
        entity_id: str,
        entity_type: str = "channel-linear",
        start_timestamp: Optional[str] = None,
        manifest_variant: str = "WEB",
        hls_version: str = "V3",
        mtc_version: str = "V2",
    ) -> TuneSourceResult:
        url = f"{self.BASE_URL}/playback/play/v1/tuneSource"
        if start_timestamp:
            ts = start_timestamp
        else:
            ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        payload = {
            "id": entity_id,
            "type": entity_type,
            "hlsVersion": hls_version,
            "manifestVariant": manifest_variant,
            "mtcVersion": mtc_version,
            "startTimestamp": ts,
        }

        r = self._request("POST", url, json=payload)
        r.raise_for_status()
        return TuneSourceResult(raw=r.json())

    def live_update(self, *, channel_id: str, start_timestamp: Optional[str] = None) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/playback/play/v1/liveUpdate"
        if start_timestamp:
            ts = start_timestamp
        else:
            ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        payload = {
            "channelId": channel_id,
            "hlsVersion": "V3",
            "manifestVariant": "WEB",
            "mtcVersion": "V2",
            "startTimestamp": ts,
        }
        r = self._request("POST", url, json=payload)
        r.raise_for_status()
        return r.json()

    def playback_state(self, *, entity_type: str, entity_id: str) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/playbackservices/v1/playback-state/{entity_type}/{entity_id}"
        r = self._request("GET", url)
        r.raise_for_status()
        return r.json()

    def channel_list(self) -> List[Dict[str, Any]]:
        return self.get_channels_browse()

    def get_channels_browse(self) -> List[Dict[str, Any]]:
        """Fetch channels via the browse API (known-good) rather than metadata/channellist.

        This matches the approach used by m3u8XM and avoids the now-broken
        /metadata/channellist/v3/web/get gateway route.
        """

        init_data = {
            "containerConfiguration": {
                "3JoBfOCIwo6FmTpzM1S2H7": {
                    "filter": {"one": {"filterId": "all"}},
                    "sets": {
                        "5mqCLZ21qAwnufKT8puUiM": {
                            "sort": {"sortId": "CHANNEL_NUMBER_ASC"}
                        }
                    },
                }
            },
            "pagination": {"offset": {"containerLimit": 3, "setItemsLimit": 50}},
            "deviceCapabilities": {"supportsDownloads": False},
        }

        page = self._browse_post(
            "browse/v1/pages/curated-grouping/403ab6a5-d3c9-4c2a-a722-a94a6a5fd056/view",
            init_data,
        )
        if not page:
            return []

        out: List[Dict[str, Any]] = []

        def parse_items(items: List[Dict[str, Any]]):
            for item in items:
                entity = item.get("entity") or {}
                cid = entity.get("id")
                texts = (entity.get("texts") or {})
                title = ((texts.get("title") or {}).get("default"))
                desc = ((texts.get("description") or {}).get("default"))
                decorations = item.get("decorations") or {}
                channel_number = decorations.get("channelNumber")
                try:
                    channel_number = int(channel_number) if channel_number is not None else None
                except Exception:
                    channel_number = None
                genre = decorations.get("genre") if isinstance(decorations.get("genre"), str) else None
                actions = item.get("actions") or {}
                play_actions = actions.get("play") or []
                channel_type = None
                play_id = None
                try:
                    channel_type = play_actions[0].get("entity", {}).get("type")
                    play_id = play_actions[0].get("entity", {}).get("id")
                except Exception:
                    channel_type = None
                    play_id = None

                images = entity.get("images") or {}
                logo_url = None
                try:
                    tile = images.get("tile", {})
                    a11 = tile.get("aspect_1x1", {})

                    # Common structure (matches liveUpdate parsing):
                    # tile.aspect_1x1.preferredImage.url / defaultImage.url
                    pref = (a11.get("preferredImage") or {}).get("url")
                    if isinstance(pref, str) and pref:
                        logo_url = pref
                    else:
                        default = (a11.get("defaultImage") or {}).get("url")
                        if isinstance(default, str) and default:
                            logo_url = default

                    # Older/alternate shapes observed in some browse payloads.
                    if not logo_url:
                        pref2 = (a11.get("preferred") or {}).get("url")
                        if isinstance(pref2, str) and pref2:
                            logo_url = pref2
                    if not logo_url:
                        default2 = (a11.get("default") or {}).get("url")
                        if isinstance(default2, str) and default2:
                            logo_url = default2
                    if not logo_url:
                        url = a11.get("url")
                        if isinstance(url, str) and url:
                            logo_url = url
                except Exception:
                    logo_url = None

                if not cid or not title:
                    continue

                # Prefer the play-action entity id for tuneSource.
                # Some browse entries wrap entities; the play action is authoritative.
                resolved_id = play_id or cid

                out.append(
                    {
                        "id": resolved_id,
                        "name": title,
                        "number": channel_number,
                        "genre": genre,
                        "description": desc,
                        "logo_url": logo_url,
                        "channel_type": channel_type or "channel-linear",
                    }
                )

        try:
            containers = (page.get("page") or {}).get("containers") or []
            first_sets = ((containers[0].get("sets") or [])[0].get("items") or [])
            parse_items(first_sets)
            total_size = (
                ((containers[0].get("sets") or [])[0].get("pagination") or {})
                .get("offset", {})
                .get("size")
            )
        except Exception:
            total_size = None

        if not total_size:
            return out

        for offset in range(50, int(total_size), 50):
            postdata = {
                "filter": {"one": {"filterId": "all"}},
                "sets": {
                    "5mqCLZ21qAwnufKT8puUiM": {
                        "sort": {"sortId": "CHANNEL_NUMBER_ASC"},
                        "pagination": {"offset": {"setItemsOffset": offset, "setItemsLimit": 50}},
                    }
                },
                "pagination": {"offset": {"setItemsLimit": 50}},
            }
            chunk = self._browse_post(
                "browse/v1/pages/curated-grouping/403ab6a5-d3c9-4c2a-a722-a94a6a5fd056/containers/3JoBfOCIwo6FmTpzM1S2H7/view",
                postdata,
                init_data,
            )
            if not chunk:
                continue
            try:
                items = ((chunk.get("container") or {}).get("sets") or [])[0].get("items") or []
                parse_items(items)
            except Exception:
                continue

        return out

    def _browse_post(self, path: str, payload: Dict[str, Any], init_payload: Dict[str, Any] | None = None) -> Optional[Dict[str, Any]]:
        url = f"{self.BASE_URL}/{path}"
        body = payload
        headers = {}
        if init_payload is not None:
            headers["X-SXM-BROWSE-INIT"] = "1"
        r = self._request("POST", url, json=body, headers=headers)
        if r.status_code not in (200, 201):
            return None
        try:
            return r.json()
        except Exception:
            return None

    def fetch_key_bytes(self, key_url: str) -> bytes:
        r = self._request("GET", key_url)
        r.raise_for_status()
        # SiriusXM commonly returns JSON with base64 key
        try:
            data = r.json()
            if isinstance(data, dict) and "key" in data:
                import base64

                return base64.b64decode(data["key"])
        except Exception:
            pass
        return r.content
