"""Tests for the retry/backoff behavior of the shared HTTP-JSON client."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from reglens.tools._http import DEFAULT_USER_AGENT, UrllibClient


class _Resp:
    """A minimal urlopen-style context manager json.load can read."""

    def __init__(self, payload):
        self._bytes = json.dumps(payload).encode()

    def read(self, *_):
        return self._bytes

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _http_error(code, retry_after=None):
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return urllib.error.HTTPError("http://x", code, "err", headers, None)


def _seq_urlopen(monkeypatch, items):
    """Patch urlopen to raise/return successive ``items`` and record call count."""
    state = {"i": 0}

    def fake(_req, timeout=None):
        item = items[state["i"]]
        state["i"] += 1
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return state


class TestRetry:
    def test_retries_then_succeeds(self, monkeypatch):
        state = _seq_urlopen(monkeypatch, [_http_error(429, "0"), _Resp({"ok": 1})])
        slept: list[float] = []
        client = UrllibClient(max_retries=2, sleep=slept.append)
        assert client.get_json("http://x") == {"ok": 1}
        assert state["i"] == 2 and len(slept) == 1  # one retry, one backoff

    def test_retry_after_header_wins(self, monkeypatch):
        _seq_urlopen(monkeypatch, [_http_error(429, "5"), _Resp({"ok": 1})])
        slept: list[float] = []
        UrllibClient(max_retries=2, backoff=0.5, sleep=slept.append).get_json("http://x")
        assert slept == [5.0]  # Retry-After (5) beats exponential backoff (0.5)

    def test_connection_error_retried(self, monkeypatch):
        _seq_urlopen(monkeypatch, [urllib.error.URLError("boom"), _Resp({"ok": 2})])
        assert UrllibClient(max_retries=2, sleep=lambda _: None).get_json("http://x") == {
            "ok": 2
        }

    def test_non_retryable_raises_immediately(self, monkeypatch):
        state = _seq_urlopen(monkeypatch, [_http_error(404), _Resp({"ok": 1})])
        with pytest.raises(urllib.error.HTTPError):
            UrllibClient(max_retries=3, sleep=lambda _: None).get_json("http://x")
        assert state["i"] == 1  # 404 is not retried

    def test_exhausts_retries_then_raises(self, monkeypatch):
        state = _seq_urlopen(monkeypatch, [_http_error(503) for _ in range(5)])
        with pytest.raises(urllib.error.HTTPError):
            UrllibClient(max_retries=2, sleep=lambda _: None).get_json("http://x")
        assert state["i"] == 3  # first attempt + 2 retries

    def test_user_agent_has_real_repo_url(self):
        assert "github.com/kpal002/RegLens" in DEFAULT_USER_AGENT
