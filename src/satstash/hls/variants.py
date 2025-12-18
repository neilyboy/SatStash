from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests

from satstash.api.client import SxmClient


@dataclass
class VariantSelection:
    master_url: str
    variant_url: str


def select_variant(
    master_url: str,
    *,
    prefer: str = "256k",
    headers: Optional[dict] = None,
    client: Optional[SxmClient] = None,
) -> VariantSelection:
    if client is not None:
        r = client._request("GET", master_url, headers=headers or {"User-Agent": "Mozilla/5.0"})
    else:
        r = requests.get(master_url, headers=headers or {"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    text = r.text

    base = master_url.rsplit("/", 1)[0] + "/"
    lines = text.splitlines()

    def abs_url(path: str) -> str:
        return path if path.startswith("http") else base + path

    # Prefer 256k if present
    for i, line in enumerate(lines):
        if line.startswith("#EXT-X-STREAM-INF") and ("256k" in line or "BANDWIDTH=281600" in line):
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if nxt and not nxt.startswith("#"):
                    return VariantSelection(master_url=master_url, variant_url=abs_url(nxt))

    # Fallback: first m3u8
    for line in lines:
        s = line.strip()
        if s and (not s.startswith("#")) and s.endswith(".m3u8"):
            return VariantSelection(master_url=master_url, variant_url=abs_url(s))

    raise RuntimeError("Could not find a variant playlist in master")
