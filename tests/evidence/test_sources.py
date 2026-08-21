from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from climadc.errors import ConfigurationError
from climadc.evidence.sources import RawHTTPResponse, coerce_json_response, fetch_json_response


class _Response:
    status = 200
    headers: Mapping[str, str] = {
        "Content-Type": "application/json",
        "ETag": "fixture",
        "Set-Cookie": "must-not-be-captured",
    }

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return b'{"value":1}\n'

    def getcode(self) -> int:
        return self.status


def test_fetch_preserves_exact_bytes_status_hash_and_allowlisted_headers(monkeypatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: _Response())

    response = fetch_json_response(
        "https://data.example.test/api?latitude=1",
        provider="fixture",
        user_agent="climadc-test",
        timeout_seconds=1.0,
    )

    assert response.body == b'{"value":1}\n'
    assert response.status == 200
    assert response.json_object("fixture") == {"value": 1}
    assert set(response.headers) == {"content-type", "etag"}
    assert len(response.sha256) == 64


@pytest.mark.parametrize(
    "url",
    [
        "http://data.example.test/api",
        "https://user:secret@data.example.test/api",
        "https://data.example.test/api?api_key=secret",
        "https://data.example.test/api?authToken=secret",
    ],
)
def test_raw_response_rejects_unsafe_request_urls(url: str) -> None:
    with pytest.raises(ConfigurationError):
        RawHTTPResponse(url=url, body=b"{}", status=200, headers={})


def test_injected_mapping_is_canonicalized_and_marked_as_test_derived() -> None:
    payload = {"b": 2, "a": 1}

    parsed, response = coerce_json_response(
        payload,
        url="https://data.example.test/api",
        provider="fixture",
    )

    assert parsed is payload
    assert response.body == b'{"a":1,"b":2}\n'
    assert response.capture_kind == "injected_mapping"
    assert json.loads(response.body) == payload


def test_raw_response_rejects_nonfinite_or_nonobject_json() -> None:
    with pytest.raises(ConfigurationError, match="JSON object"):
        RawHTTPResponse(
            url="https://data.example.test/api", body=b"[]", status=200, headers={}
        ).json_object("fixture")
    with pytest.raises(ConfigurationError, match="finite JSON"):
        coerce_json_response(
            {"value": float("nan")},
            url="https://data.example.test/api",
            provider="fixture",
        )
