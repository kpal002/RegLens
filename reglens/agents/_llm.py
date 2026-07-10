"""Shared LLM plumbing for the reasoning layer: one hardened structured-JSON call.

Both the single-agent :mod:`reglens.agents.interpreter` and the multi-agent
:mod:`reglens.agents.multi_agent` layer issue the same kind of call — send a system
prompt + JSON context to Claude and get back a JSON object matching a schema. This
module centralizes that call (with the schema-constrained → prompted-JSON fallback)
so the fallback logic lives in exactly one place. The ``anthropic`` SDK is imported
lazily and a client may be injected for offline testing.
"""

from __future__ import annotations

import json
from typing import Any


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from model text, tolerating markdown fences/preamble.

    Args:
        text: Raw model output (ideally pure JSON, possibly fenced).

    Returns:
        The parsed object.

    Raises:
        ValueError: If no JSON object can be located.
    """
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        # Fall back to the outermost {...} span (handles ```json fences / preamble).
        start, end = stripped.find("{"), stripped.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"No JSON object found in model output: {text[:200]!r}") from None
        return json.loads(stripped[start : end + 1])


def is_structured_unsupported(exc: Exception) -> bool:
    """Whether ``exc`` indicates the schema-constrained output format was rejected.

    Triggers the prompted-JSON fallback for either a client-side ``TypeError`` (SDK
    doesn't accept ``output_config``) or an API error mentioning the output format.
    """
    if isinstance(exc, TypeError):
        return True
    blob = str(exc).lower()
    return any(k in blob for k in ("output_config", "json_schema", "output format", "format"))


def build_anthropic_client() -> Any:
    """Construct a zero-arg Anthropic client (resolves credentials from the env).

    Returns:
        An ``anthropic.Anthropic`` instance.

    Raises:
        ImportError: If the ``anthropic`` SDK is not installed.
    """
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "The anthropic SDK is required for the Claude reasoning layer. "
            "Install it with: pip install 'reglens[agents]'"
        ) from exc
    return anthropic.Anthropic()


class StructuredCaller:
    """Issues schema-constrained Claude calls, degrading to prompted JSON on failure.

    One instance is shared across all calls in a deliberation so a single downgrade
    (structured → prompted) is remembered rather than re-attempted every call.
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str = "claude-opus-4-8",
        max_tokens: int = 8000,
        use_structured: bool = True,
    ) -> None:
        """Initialize the caller.

        Args:
            client: An Anthropic-like client; constructed if ``None``.
            model: Anthropic model id.
            max_tokens: Output token ceiling (covers thinking + JSON).
            use_structured: Try ``output_config`` schema output first.
        """
        self.client = client if client is not None else build_anthropic_client()
        self.model = model
        self.max_tokens = max_tokens
        self.use_structured = use_structured

    def call(
        self, system: str, user_content: str, schema: dict[str, Any], prompted_suffix: str
    ) -> dict[str, Any]:
        """Run one call and return the parsed JSON object.

        Args:
            system: System prompt (schema path).
            user_content: The user message body (e.g. the evidence JSON).
            schema: JSON schema constraining the structured-output path.
            prompted_suffix: Appended to ``system`` on the prompted-JSON fallback,
                describing the expected JSON shape in prose.

        Returns:
            The parsed JSON object.
        """
        messages = [{"role": "user", "content": user_content}]
        if self.use_structured:
            try:
                return self._create(system, messages, schema=schema)
            except Exception as exc:  # noqa: BLE001 - narrow via is_structured_unsupported
                if not is_structured_unsupported(exc):
                    raise
                self.use_structured = False  # remember the downgrade
        return self._create(system + prompted_suffix, messages, schema=None)

    def _create(
        self, system: str, messages: list[dict[str, Any]], schema: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Issue one Messages API call and parse the JSON reply."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "thinking": {"type": "adaptive"},
            "system": system,
            "messages": messages,
        }
        if schema is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        response = self.client.messages.create(**kwargs)
        text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), "")
        return extract_json(text)
