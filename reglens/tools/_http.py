"""Minimal injectable HTTP-JSON client shared by the API-backed tools.

The deterministic tools that hit external APIs (Europe PMC, GTEx, GWAS Catalog,
Ensembl, ...) all go through the small :class:`HttpClient` protocol here. Real runs
use :class:`UrllibClient` (stdlib only — no new heavyweight dependency); tests inject
a fake client so the suite stays fully offline and deterministic.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol, runtime_checkable

# Identify ourselves politely to public APIs (some rate-limit anonymous traffic).
# A real contact URL matters for NCBI / Europe PMC etiquette and some rate limiters.
DEFAULT_USER_AGENT = (
    "RegLens/0.0.1 (https://github.com/kpal002/RegLens; regulatory-variant-interpreter)"
)

# Transient HTTP statuses worth retrying (rate limit + server-side hiccups).
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


@runtime_checkable
class HttpClient(Protocol):
    """A client that fetches a URL (with optional query params) and returns JSON."""

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """GET ``url`` with ``params`` and parse the response as JSON.

        Args:
            url: The base URL.
            params: Optional query parameters to URL-encode onto ``url``.

        Returns:
            The parsed JSON (dict or list).
        """
        ...


class UrllibClient:
    """An :class:`HttpClient` backed by the standard library ``urllib``.

    Retries transient failures (429 + 5xx, and connection errors) with exponential
    backoff, honoring a ``Retry-After`` header on 429/503. This keeps a large screen —
    dozens of variants each hitting Ensembl / GWAS Catalog / GTEx / Europe PMC — from
    silently dropping evidence limbs to a momentary blip or rate limit.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        user_agent: str = DEFAULT_USER_AGENT,
        max_retries: int = 3,
        backoff: float = 0.5,
        max_backoff: float = 20.0,
        sleep: Any = time.sleep,
    ) -> None:
        """Initialize the client.

        Args:
            timeout: Per-request timeout in seconds.
            user_agent: ``User-Agent`` header sent with each request.
            max_retries: Additional attempts after the first on a transient error.
            backoff: Base backoff (seconds); attempt ``n`` waits ``backoff * 2**n``.
            max_backoff: Cap on any single backoff wait.
            sleep: Sleep function (injectable so tests don't actually wait).
        """
        self.timeout = timeout
        self.user_agent = user_agent
        self.max_retries = max_retries
        self.backoff = backoff
        self.max_backoff = max_backoff
        self._sleep = sleep

    def _wait(self, attempt: int, retry_after: str | None) -> None:
        """Sleep before a retry — the server's ``Retry-After`` wins if present."""
        delay = self.backoff * (2**attempt)
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass  # HTTP-date form; fall back to exponential backoff
        self._sleep(min(delay, self.max_backoff))

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """See :meth:`HttpClient.get_json` — with retry/backoff on transient errors."""
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url, headers={"User-Agent": self.user_agent, "Accept": "application/json"}
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                if exc.code not in _RETRYABLE_STATUS or attempt == self.max_retries:
                    raise
                self._wait(attempt, exc.headers.get("Retry-After"))
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.max_retries:
                    raise
                self._wait(attempt, None)
        raise RuntimeError("unreachable")  # pragma: no cover


# Module-level default so tools work with zero setup but remain injectable.
DEFAULT_CLIENT: HttpClient = UrllibClient()


def resolve_client(client: HttpClient | None) -> HttpClient:
    """Return ``client`` if given, else the shared default client."""
    return client if client is not None else DEFAULT_CLIENT
