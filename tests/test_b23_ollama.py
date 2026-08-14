"""Ollama backend for B23, exercised against a real local HTTP server.

The tests stand up a throwaway `http.server` speaking the two Ollama endpoints
the backend uses, so the request bodies, the digest-based provenance pin, the
context-window guard and the Qwen3 reasoning-block handling are all covered
without Ollama installed and without a model download.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from rsna_knee.b23_llm_labels import SYSTEM_PROMPT, parse_extraction_response
from rsna_knee.b23_local_llm import (
    BACKEND_OLLAMA,
    DECODING_GREEDY,
    DEFAULT_LOCAL_MODEL,
    OLLAMA_DEFAULT_NUM_CTX,
    SUGGESTED_LOCAL_MODELS,
    estimate_prompt_tokens,
    looks_openly_downloadable,
    make_ollama_backend,
    ollama_model_digest,
    strip_thinking,
)
from rsna_knee.constants import TARGETS

DIGEST = "f" * 64


def _findings_json():
    return json.dumps(
        {
            "findings": {
                target: {"state": "negated", "confidence": 0.9, "evidence": "normal"}
                for target in TARGETS
            }
        }
    )


class _FakeOllama(HTTPServer):
    """Minimal stand-in for `ollama serve`."""

    def __init__(self, *, installed=("qwen3:14b",), completion=None, reject_think=False):
        self.installed = list(installed)
        self.completion = completion if completion is not None else _findings_json()
        self.reject_think = reject_think
        self.chat_bodies: list[dict] = []
        super().__init__(("127.0.0.1", 0), _FakeHandler)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class _FakeHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep test output clean
        pass

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send(
                {
                    "models": [
                        {
                            "name": name,
                            "digest": f"sha256:{DIGEST}",
                            "size": 9_300_000_000,
                            "details": {"quantization_level": "Q4_K_M", "family": "qwen3"},
                        }
                        for name in self.server.installed
                    ]
                }
            )
        else:
            self._send({"error": "not found"}, status=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/chat":
            if self.server.reject_think and "think" in body:
                self._send({"error": "unknown field think"}, status=400)
                return
            self.server.chat_bodies.append(body)
            self._send({"message": {"role": "assistant", "content": self.server.completion}})
        else:
            self._send({"error": "not found"}, status=404)


@pytest.fixture
def fake_ollama():
    servers = []

    def _start(**kwargs):
        server = _FakeOllama(**kwargs)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        return server

    yield _start
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_default_model_is_the_ollama_qwen3_14b_build():
    assert DEFAULT_LOCAL_MODEL == "qwen3:14b"
    assert looks_openly_downloadable(DEFAULT_LOCAL_MODEL)
    assert "9.3 GB" in SUGGESTED_LOCAL_MODELS["qwen3:14b"]
    assert "5.2 GB" in SUGGESTED_LOCAL_MODELS["qwen3:8b"]


def test_digest_and_quantisation_come_from_the_running_daemon(fake_ollama):
    server = fake_ollama()
    digest, quantisation = ollama_model_digest("qwen3:14b", server.url)
    assert digest == DIGEST
    assert quantisation == "Q4_K_M"


def test_a_missing_model_is_an_error_rather_than_a_silent_pull(fake_ollama):
    server = fake_ollama(installed=("qwen3:8b",))
    with pytest.raises(RuntimeError, match="not installed"):
        ollama_model_digest("qwen3:14b", server.url)


def test_a_bare_name_resolves_to_the_latest_tag(fake_ollama):
    server = fake_ollama(installed=("mistral:latest",))
    digest, _ = ollama_model_digest("mistral", server.url)
    assert digest == DIGEST


def test_an_unreachable_daemon_says_so_clearly():
    with pytest.raises(RuntimeError, match="Is `ollama serve` running"):
        ollama_model_digest("qwen3:14b", "http://127.0.0.1:1")


def test_provenance_pins_the_installed_model_digest(fake_ollama):
    server = fake_ollama()
    _call, provenance = make_ollama_backend(SYSTEM_PROMPT, host=server.url)
    assert provenance.backend == BACKEND_OLLAMA
    assert provenance.model_id == "qwen3:14b"
    assert provenance.revision == DIGEST
    assert provenance.ollama_model_digest == DIGEST
    # weights_sha256 is reserved for a digest this code computes over weight
    # shards. Ollama's digest is not documented as a hash of the GGUF tensor
    # bytes, so claiming it in that field would overstate the guarantee.
    assert provenance.weights_sha256 is None
    assert provenance.quantisation == "Q4_K_M"
    assert provenance.decoding == DECODING_GREEDY
    assert provenance.reproducible


def test_generation_requests_greedy_json_with_an_explicit_context_window(fake_ollama):
    server = fake_ollama()
    call, _ = make_ollama_backend(SYSTEM_PROMPT, host=server.url, seed=2026)
    call("system text", "user text")

    body = server.chat_bodies[-1]
    assert body["stream"] is False
    assert body["format"] == "json"
    assert body["think"] is False  # reasoning off for an extraction task
    options = body["options"]
    assert options["temperature"] == 0.0
    assert options["seed"] == 2026
    assert options["num_ctx"] == OLLAMA_DEFAULT_NUM_CTX
    assert options["num_predict"] > 0


def test_older_daemons_that_reject_the_think_field_still_work(fake_ollama):
    server = fake_ollama(reject_think=True)
    call, _ = make_ollama_backend(SYSTEM_PROMPT, host=server.url)
    out = call("system text", "user text")
    assert set(parse_extraction_response(out)) == set(TARGETS)
    assert "think" not in server.chat_bodies[-1]


def test_a_qwen3_reasoning_block_never_reaches_the_parser(fake_ollama):
    completion = "<think>The report says the ACL is intact, so...</think>" + _findings_json()
    server = fake_ollama(completion=completion)
    call, _ = make_ollama_backend(SYSTEM_PROMPT, host=server.url)
    out = call("system text", "user text")
    assert "<think>" not in out
    assert set(parse_extraction_response(out)) == set(TARGETS)


def test_an_unterminated_reasoning_block_is_removed_entirely():
    assert strip_thinking("<think>ran out of tokens mid-thought") == ""
    assert strip_thinking("<think>a</think> tail") == "tail"
    assert strip_thinking("no block here") == "no block here"


def test_context_guard_refuses_rather_than_letting_ollama_truncate(fake_ollama):
    server = fake_ollama()
    # 512 tokens cannot hold the ~2k-token system prompt plus a report.
    call, _ = make_ollama_backend(SYSTEM_PROMPT, host=server.url, num_ctx=512)
    with pytest.raises(RuntimeError, match="would silently truncate"):
        call(SYSTEM_PROMPT, "a knee MRI report " * 200)
    assert server.chat_bodies == []  # nothing was sent


def test_the_default_context_window_fits_the_longest_observed_report():
    # Longest report in the sampled corpus is ~2,100 characters; the system
    # prompt is ~5,900. The default window must clear both with headroom.
    estimated = estimate_prompt_tokens(SYSTEM_PROMPT, "x" * 2100)
    assert estimated < OLLAMA_DEFAULT_NUM_CTX
    assert estimated * 2 < OLLAMA_DEFAULT_NUM_CTX


def test_a_full_json_schema_is_sent_rather_than_generic_json(fake_ollama):
    from rsna_knee.b23_llm_labels import make_backend

    server = fake_ollama()
    call, _ = make_backend(backend=BACKEND_OLLAMA, ollama_host=server.url)
    call("system text", "user text")

    schema = server.chat_bodies[-1]["format"]
    assert isinstance(schema, dict), "expected a schema, not the string 'json'"
    findings = schema["properties"]["findings"]
    # A missing target or an invented state becomes impossible at decode time.
    assert set(findings["required"]) == set(TARGETS)
    assert findings["properties"]["ACL"]["properties"]["state"]["enum"] == [
        "positive",
        "negated",
        "uncertain",
        "unmentioned",
    ]
