from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from urllib.parse import parse_qsl, urlsplit

from climadc.errors import ConfigurationError

_ALLOWED_RESPONSE_HEADERS = frozenset(
    {"cache-control", "content-type", "date", "etag", "last-modified"}
)
_SENSITIVE_QUERY_TOKENS = ("token", "key", "secret", "password", "auth", "cookie")


def _public_request_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ConfigurationError("raw-source request URL must be public HTTPS without credentials")
    for name, _ in parse_qsl(parsed.query, keep_blank_values=True):
        lowered = name.lower()
        if any(token in lowered for token in _SENSITIVE_QUERY_TOKENS):
            raise ConfigurationError(
                f"raw-source request URL contains sensitive query field {name}"
            )
    return url


@dataclass(frozen=True)
class RawHTTPResponse:
    """Exact HTTP response bytes plus a deliberately small, safe metadata envelope."""

    url: str
    body: bytes
    status: int
    headers: Mapping[str, str]
    capture_kind: str = "network"

    def __post_init__(self) -> None:
        safe_url = _public_request_url(self.url)
        if not isinstance(self.body, bytes) or not self.body:
            raise ConfigurationError("raw-source response body must be non-empty bytes")
        if not isinstance(self.status, int) or isinstance(self.status, bool) or self.status < 100:
            raise ConfigurationError("raw-source HTTP status must be an integer")
        headers: dict[str, str] = {}
        for name, value in self.headers.items():
            lowered = str(name).lower()
            if lowered not in _ALLOWED_RESPONSE_HEADERS:
                raise ConfigurationError(f"raw-source response header is not allowlisted: {name}")
            headers[lowered] = str(value)
        object.__setattr__(self, "url", safe_url)
        object.__setattr__(self, "headers", MappingProxyType(dict(sorted(headers.items()))))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.body).hexdigest()

    def json_object(self, provider: str) -> Mapping[str, object]:
        try:
            payload = json.loads(self.body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"{provider} response is not UTF-8 JSON: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ConfigurationError(f"{provider} response must be a JSON object")
        return payload


def fetch_json_response(
    url: str,
    *,
    provider: str,
    user_agent: str,
    timeout_seconds: float,
) -> RawHTTPResponse:
    safe_url = _public_request_url(url)
    request = urllib.request.Request(safe_url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read()
            raw_status = getattr(response, "status", None)
            status = int(response.getcode() if raw_status is None else raw_status)
            response_headers = getattr(response, "headers", {})
            headers = {
                str(name).lower(): str(value)
                for name, value in response_headers.items()
                if str(name).lower() in _ALLOWED_RESPONSE_HEADERS
            }
    except OSError as exc:
        raise ConfigurationError(f"{provider} request failed: {exc}") from exc
    captured = RawHTTPResponse(
        url=safe_url,
        body=body,
        status=status,
        headers=headers,
        capture_kind="network",
    )
    if not 200 <= captured.status < 300:
        raise ConfigurationError(f"{provider} request returned HTTP {captured.status}")
    captured.json_object(provider)
    return captured


def coerce_json_response(
    value: Mapping[str, object] | RawHTTPResponse,
    *,
    url: str,
    provider: str,
) -> tuple[Mapping[str, object], RawHTTPResponse]:
    """Preserve injected Mapping transports while marking their bytes as test-derived."""
    if isinstance(value, RawHTTPResponse):
        return value.json_object(provider), value
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{provider} transport must return a mapping or raw response")
    try:
        body = (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
                "utf-8"
            )
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{provider} injected payload is not finite JSON") from exc
    response = RawHTTPResponse(
        url=url,
        body=body,
        status=200,
        headers={"content-type": "application/json"},
        capture_kind="injected_mapping",
    )
    return value, response
