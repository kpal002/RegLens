"""Guards for the real Anthropic request the agent layer builds.

Every other agent test injects a fake that returns a canned interpretation, so nothing
checks the *request* our code sends — a wrong model id, a bad ``thinking`` config, or a
broken ``output_config`` would pass CI silently. Two guards close that:

- ``TestRequestShape`` (offline, every commit) captures the kwargs ``StructuredCaller``
  passes to ``messages.create`` and pins the config. This fails the instant someone
  reintroduces ``budget_tokens``, mistypes the model, or drops the schema.
- ``test_live_structured_call`` (opt-in) makes ONE real API call — the honest check that
  the live API accepts our request. Marked ``live`` (deselected by default via
  ``-m 'not live'`` in pyproject) and skipped without ``ANTHROPIC_API_KEY``; run with
  ``pytest -m live``.
"""

from __future__ import annotations

import os
import types

import pytest

from reglens.agents._llm import DEFAULT_MODEL, StructuredCaller

_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "integer"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def _text_response(payload: str = '{"ok": 1}'):
    """A minimal Anthropic-style response: one text block carrying JSON."""
    return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=payload)])


class _CapturingClient:
    """Fake Anthropic client that records every ``messages.create`` kwargs dict."""

    def __init__(self, fail_first: bool = False, grammar_fail: int = 0):
        self.calls: list[dict] = []
        self._fail_first = fail_first
        self._grammar_fail = grammar_fail  # first N structured calls raise a grammar 400

    @property
    def messages(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._fail_first and len(self.calls) == 1:
            # Simulate an SDK/model that rejects the structured-output param.
            raise TypeError("unexpected keyword argument 'output_config'")
        if (self._grammar_fail and "output_config" in kwargs
                and len(self.calls) <= self._grammar_fail):
            # Transient server-side grammar compilation timeout (a 400).
            raise RuntimeError(
                "Error code: 400 - {'type': 'error', 'error': "
                "{'message': 'Grammar compilation timed out.'}}"
            )
        return _text_response()


class TestRequestShape:
    def test_structured_path_config(self):
        client = _CapturingClient()
        out = StructuredCaller(client=client).call("SYS", "USER", _SCHEMA, "\nSUFFIX")
        assert out == {"ok": 1}
        assert len(client.calls) == 1
        kw = client.calls[0]
        # The model resolves from the configured default (guards the wrong-id regression).
        assert kw["model"] == DEFAULT_MODEL and isinstance(DEFAULT_MODEL, str) and DEFAULT_MODEL
        # Adaptive thinking is the correct on-mode for Opus 4.8; budget_tokens 400s there.
        assert kw["thinking"] == {"type": "adaptive"}
        assert "budget_tokens" not in kw["thinking"]
        assert kw["max_tokens"] > 0
        # Structured output via output_config.format (the current API), not output_format.
        assert kw["output_config"] == {
            "format": {"type": "json_schema", "schema": _SCHEMA}
        }
        assert "output_format" not in kw
        assert kw["messages"] == [{"role": "user", "content": "USER"}]
        assert kw["system"] == "SYS"  # no prompted suffix on the structured path
        # Sampling params were removed on Opus 4.8/4.7 — sending them 400s.
        for banned in ("temperature", "top_p", "top_k"):
            assert banned not in kw

    def test_prompted_json_fallback(self):
        client = _CapturingClient(fail_first=True)
        caller = StructuredCaller(client=client)
        out = caller.call("SYS", "USER", _SCHEMA, "\nSUFFIX")
        assert out == {"ok": 1}
        assert len(client.calls) == 2
        first, second = client.calls
        assert "output_config" in first              # structured attempt
        assert "output_config" not in second         # fallback drops the schema
        assert second["system"] == "SYS\nSUFFIX"     # prose suffix appended
        assert second["thinking"] == {"type": "adaptive"}  # still adaptive on the fallback
        assert caller.use_structured is False         # downgrade remembered

    def test_model_override_flows_through(self, monkeypatch):
        client = _CapturingClient()
        StructuredCaller(client=client, model="claude-sonnet-5").call(
            "SYS", "USER", _SCHEMA, "\nSUFFIX"
        )
        assert client.calls[0]["model"] == "claude-sonnet-5"

    def test_grammar_timeout_retries_structured(self):
        # One transient grammar 400, then success — retried in place, stays structured.
        client = _CapturingClient(grammar_fail=1)
        caller = StructuredCaller(client=client, grammar_backoff=0, sleep=lambda s: None)
        out = caller.call("SYS", "USER", _SCHEMA, "\nSUFFIX")
        assert out == {"ok": 1}
        assert len(client.calls) == 2
        assert all("output_config" in c for c in client.calls)  # both structured
        assert caller.use_structured is True  # transient — NOT a permanent downgrade

    def test_grammar_timeout_falls_back_to_prompted(self):
        # Persistent grammar timeout: exhaust retries, then one-off prompted fallback.
        client = _CapturingClient(grammar_fail=99)
        caller = StructuredCaller(
            client=client, grammar_retries=2, grammar_backoff=0, sleep=lambda s: None
        )
        out = caller.call("SYS", "USER", _SCHEMA, "\nSUFFIX")
        assert out == {"ok": 1}
        # 3 structured attempts (initial + 2 retries) then 1 prompted fallback.
        assert len(client.calls) == 4
        assert all("output_config" in c for c in client.calls[:3])
        assert "output_config" not in client.calls[3]
        assert client.calls[3]["system"] == "SYS\nSUFFIX"
        assert caller.use_structured is True  # structured stays on for the next variant


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY"
)
def test_live_structured_call():
    """One real call — proves the live API accepts our request and returns parseable JSON."""
    pytest.importorskip("anthropic")
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    out = StructuredCaller(max_tokens=1024).call(
        "You are a test harness. Respond with the requested JSON only.",
        'Return exactly {"answer": "ok"}.',
        schema,
        '\nReturn ONLY a JSON object with a single key "answer".',
    )
    assert isinstance(out, dict) and "answer" in out
