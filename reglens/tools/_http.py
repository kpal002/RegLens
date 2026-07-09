"""Minimal injectable HTTP-JSON client shared by the API-backed tools.

The deterministic tools that hit external APIs (Europe PMC, GTEx, GWAS Catalog,
Ensembl, ...) all go through the small :class:`HttpClient` protocol here. Real runs
use :class:`UrllibClient` (stdlib only — no new heavyweight dependency); tests inject
a fake client so the suite stays fully offline and deterministic.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Protocol, runtime_checkable

# Identify ourselves politely to public APIs (some rate-limit anonymous traffic).
DEFAULT_USER_AGENT = "RegLens/0.0.1 (https://github.com/; regulatory-variant-interpreter)"


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
    """An :class:`HttpClient` backed by the standard library ``urllib``."""

    def __init__(self, timeout: float = 30.0, user_agent: str = DEFAULT_USER_AGENT) -> None:
        """Initialize the client.

        Args:
            timeout: Per-request timeout in seconds.
            user_agent: ``User-Agent`` header sent with each request.
        """
        self.timeout = timeout
        self.user_agent = user_agent

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """See :meth:`HttpClient.get_json`."""
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url, headers={"User-Agent": self.user_agent, "Accept": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)


# Module-level default so tools work with zero setup but remain injectable.
DEFAULT_CLIENT: HttpClient = UrllibClient()


def resolve_client(client: HttpClient | None) -> HttpClient:
    """Return ``client`` if given, else the shared default client."""
    return client if client is not None else DEFAULT_CLIENT
