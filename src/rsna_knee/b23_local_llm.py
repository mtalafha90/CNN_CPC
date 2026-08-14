"""Locally hosted, openly downloadable LLM backends for B23 report labelling.

Competition reproducibility requires that the label-generating function be
identifiable and re-runnable. A hosted API cannot satisfy that: the served
weights behind a model name can change, and nothing in the export pins which
artefact produced a given label. B23 therefore labels with an openly
downloadable checkpoint executed locally, and records enough provenance that a
third party can reconstruct the exact labelling function:

```text
competition Report column
  -> local frozen open-weights LLM (repo id + commit revision + dtype + greedy)
  -> structured labels
  -> MRI model training
```

Four backends are provided. `ollama` is the default and the easiest to run on a
single laptop GPU. `local_transformers` is the reference path. `local_vllm` is
the same weights with faster batched decoding. `hosted_api` remains available
for development only and is explicitly marked non-reproducible, so an export
produced with it cannot be certified for competition use.

For Ollama the provenance pin is the model digest reported by `/api/tags`. That
digest identifies the installed model in Ollama's own content-addressed store
and changes when the model is re-pulled or re-quantised, which is what is needed
here. It is recorded as `ollama_model_digest`: Ollama does not document it as an
independently computed SHA-256 of the underlying GGUF tensor bytes, so B23 does
not claim that. `weights_sha256` stays reserved for a digest this code computes
itself over weight shards.

Determinism is enforced by greedy decoding rather than a temperature setting:
`do_sample=False` removes the sampler entirely, so the run does not depend on
RNG state or on how a particular library seeds it.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

BACKEND_OLLAMA = "ollama"
BACKEND_LOCAL_TRANSFORMERS = "local_transformers"
BACKEND_LOCAL_VLLM = "local_vllm"
BACKEND_HOSTED_API = "hosted_api"
REPRODUCIBLE_BACKENDS = (BACKEND_OLLAMA, BACKEND_LOCAL_TRANSFORMERS, BACKEND_LOCAL_VLLM)

OLLAMA_DEFAULT_HOST = "http://localhost:11434"
# Ollama defaults to a small context window and silently truncates anything
# longer. A ~6k-character system prompt plus a radiology report up to ~2.1k
# characters lands around 2.5-3k tokens, so the default would quietly cut the
# report in half and corrupt the labels invisibly. Set it explicitly.
OLLAMA_DEFAULT_NUM_CTX = 8192

DECODING_GREEDY = "greedy"

# Duplicated from b23_llm_labels to keep this module import-cycle free.
B23_STATE_NAMES = ("positive", "negated", "uncertain", "unmentioned")

# Openly downloadable instruction-tuned checkpoints suitable for multilingual
# structured extraction. The corpus contains at least English, Spanish, Dutch
# and Turkish, so multilingual competence matters more than raw size here.
#
# Licence notes matter for the competition's "publicly and equally accessible"
# standard, so they are recorded rather than assumed:
#   Qwen2.5 Instruct (0.5B-14B, 32B, 72B)  Apache-2.0, ungated
#   Mistral-7B-Instruct / Mixtral          Apache-2.0, ungated
#   Llama 3.1 / 3.3 Instruct               Llama Community Licence, click-through
#   Gemma 2 / 3 Instruct                   Gemma Terms of Use, click-through
#
# The Apache-2.0 families are the cleanest fit because they carry no acceptance
# gate at all. The default is the largest Qwen2.5 that fits comfortably on a
# single 24 GB card in 4-bit; override for a larger card.
# Default targets a 16 GB laptop card (e.g. RTX A4500 Laptop). Qwen3-14B at
# Q4_K_M is ~9.3 GB, leaving ~6.7 GB for the KV cache at num_ctx 8192.
DEFAULT_LOCAL_MODEL = "qwen3:14b"
SUGGESTED_LOCAL_MODELS = {
    "qwen3:8b": "Ollama Q4_K_M ~5.2 GB; fits 8 GB cards",
    "qwen3:14b": "Ollama Q4_K_M ~9.3 GB; default, fits 16 GB cards",
    "qwen3:32b": "Ollama Q4_K_M ~20 GB; needs 24 GB+",
    "Qwen/Qwen2.5-14B-Instruct": "transformers path, ~10 GB 4-bit",
    "Qwen/Qwen2.5-32B-Instruct": "transformers path, ~20 GB 4-bit",
}

# Hub-style identifiers (transformers/vLLM).
OPEN_LICENCE_PREFIXES = ("qwen/", "mistralai/", "meta-llama/", "google/gemma", "microsoft/phi")
# Ollama-style identifiers are `family:tag`, e.g. `qwen3:14b`. Only families
# whose weights are publicly downloadable are accepted.
OPEN_OLLAMA_FAMILIES = (
    "qwen3",
    "qwen2.5",
    "qwen2",
    "mistral",
    "mixtral",
    "llama3",
    "llama3.1",
    "llama3.2",
    "llama3.3",
    "gemma2",
    "gemma3",
    "phi3",
    "phi4",
)


@dataclass(frozen=True)
class ModelProvenance:
    """Everything needed to reconstruct the labelling function exactly."""

    backend: str
    model_id: str
    revision: str
    dtype: str
    quantisation: str
    decoding: str
    max_new_tokens: int
    seed: int
    prompt_sha256: str
    openly_downloadable: bool
    # Computed by this code over weight shards. Only set for the transformers
    # and vLLM paths, where the shards are on disk and can actually be hashed.
    weights_sha256: str | None = None
    # Ollama's own model identifier from /api/tags. Pins which installed model
    # ran, but is not a documented hash of the GGUF tensor bytes.
    ollama_model_digest: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def reproducible(self) -> bool:
        """True only for a pinned, openly downloadable, locally executed model."""
        return (
            self.backend in REPRODUCIBLE_BACKENDS
            and self.openly_downloadable
            and bool(self.revision)
            and self.revision != "unknown"
            and self.decoding == DECODING_GREEDY
        )

    def describe(self) -> str:
        lines = [
            f"  backend        {self.backend}",
            f"  model          {self.model_id}",
            f"  revision       {self.revision}",
            f"  dtype          {self.dtype} | quantisation {self.quantisation}",
            f"  decoding       {self.decoding} | max_new_tokens {self.max_new_tokens}",
            f"  prompt SHA-256 {self.prompt_sha256}",
            f"  reproducible   {self.reproducible}",
        ]
        if self.weights_sha256:
            lines.append(f"  weights SHA-256 {self.weights_sha256}")
        if self.ollama_model_digest:
            lines.append(f"  ollama digest  {self.ollama_model_digest}")
        return "\n".join(lines)


def prompt_sha256(system_prompt: str) -> str:
    """Pin the instruction half of the labelling function."""
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


def looks_openly_downloadable(model_id: str) -> bool:
    """Whether the identifier names a publicly downloadable checkpoint family.

    A local directory path is accepted only when it carries a resolved hub
    identifier, because a bare path tells a third party nothing about which
    weights were used.
    """
    lowered = str(model_id).strip().lower()
    if not lowered:
        return False
    if any(lowered.startswith(prefix) for prefix in OPEN_LICENCE_PREFIXES):
        return True
    # Ollama-style `family:tag`; the family alone must identify the weights.
    family = lowered.split(":", 1)[0]
    return family in OPEN_OLLAMA_FAMILIES


def _ollama_request(host: str, path: str, payload: dict | None = None, *, timeout: float = 600.0):
    """Call the local Ollama HTTP API without going through any proxy.

    Uses the standard library so the Ollama path adds no dependency, and builds
    an opener with proxies explicitly disabled: a machine with HTTPS_PROXY set
    would otherwise try to tunnel a localhost call through it.
    """
    import urllib.error
    import urllib.request

    url = f"{str(host).rstrip('/')}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"cannot reach Ollama at {host}: {exc}. Is `ollama serve` running?"
        ) from exc


def ollama_model_digest(model: str, host: str = OLLAMA_DEFAULT_HOST) -> tuple[str, str]:
    """Return the (digest, quantisation) of a locally installed Ollama model.

    The digest is Ollama's own identifier for the installed model, taken from
    `/api/tags`. It changes when the model is re-pulled or re-quantised, which
    is what pins provenance. It is NOT documented as a SHA-256 over the GGUF
    tensor bytes, so it is recorded as `ollama_model_digest` rather than as a
    weights hash.

    A model that is not installed is an error rather than a silent pull: an
    unnoticed pull could substitute different weights midway through a run.
    """
    payload = _ollama_request(host, "/api/tags", timeout=30.0)
    installed = {str(entry.get("name", "")): entry for entry in payload.get("models", [])}
    entry = installed.get(str(model))
    if entry is None and ":" not in str(model):
        entry = installed.get(f"{model}:latest")
    if entry is None:
        available = ", ".join(sorted(installed)) or "none"
        raise RuntimeError(
            f"Ollama model {model!r} is not installed (available: {available}). "
            f"Run: ollama pull {model}"
        )
    digest = str(entry.get("digest", "")).replace("sha256:", "")
    quantisation = str((entry.get("details") or {}).get("quantization_level", "unknown"))
    if not digest:
        raise RuntimeError(f"Ollama did not report a digest for {model!r}")
    return digest, quantisation


def strip_thinking(text: str) -> str:
    """Remove a reasoning model's <think> block from a completion.

    Qwen3 is a hybrid reasoning model and emits <think>...</think> before its
    answer unless thinking is disabled. The JSON parser must never see it. This
    is applied unconditionally because it is harmless for non-reasoning models
    and silent corruption otherwise.
    """
    import re

    cleaned = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL | re.IGNORECASE)
    # An unterminated block means the answer was cut off mid-reasoning.
    if "<think>" in cleaned.lower():
        cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def estimate_prompt_tokens(system: str, user: str) -> int:
    """Rough token estimate for a context-window guard.

    Deliberately conservative -- roughly 3 characters per token rather than the
    usual 4 -- because non-Latin scripts and clinical abbreviations tokenise
    worse than prose, and the failure being guarded against is silent
    truncation of the report.
    """
    return int((len(str(system)) + len(str(user))) / 3) + 64


def build_findings_schema(targets) -> dict:
    """Exact JSON Schema for the 12-target extraction.

    Ollama accepts a schema in `format`, not only the string "json". Since the
    schema is known exactly, constraining the decoder is strictly better than
    asking for generic JSON and rejecting malformed structures afterwards: a
    missing target or an invented state becomes impossible rather than a retry.
    """
    cell = {
        "type": "object",
        "properties": {
            "state": {"type": "string", "enum": list(B23_STATE_NAMES)},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "evidence": {"type": "string"},
        },
        "required": ["state", "confidence", "evidence"],
    }
    return {
        "type": "object",
        "properties": {
            "findings": {
                "type": "object",
                "properties": {str(target): cell for target in targets},
                "required": [str(target) for target in targets],
            }
        },
        "required": ["findings"],
    }


def make_ollama_backend(
    system_prompt: str,
    *,
    model: str = DEFAULT_LOCAL_MODEL,
    host: str = OLLAMA_DEFAULT_HOST,
    num_ctx: int = OLLAMA_DEFAULT_NUM_CTX,
    max_new_tokens: int = 2048,
    seed: int = 2026,
    think: bool = False,
    timeout: float = 600.0,
    schema: dict | None = None,
):
    """Default local backend: an installed Ollama model, greedy, JSON-formatted.

    `think=False` disables Qwen3's reasoning mode. Reasoning costs tokens and
    latency on what is an extraction task with explicit rules, and the answer
    still has to be plain JSON. The completion is stripped of any <think> block
    regardless, so a build that ignores the flag cannot corrupt the parse.
    """
    digest, quantisation = ollama_model_digest(model, host)
    provenance = ModelProvenance(
        backend=BACKEND_OLLAMA,
        model_id=str(model),
        revision=digest,
        dtype="gguf",
        quantisation=quantisation,
        decoding=DECODING_GREEDY,
        max_new_tokens=int(max_new_tokens),
        seed=int(seed),
        prompt_sha256=prompt_sha256(system_prompt),
        openly_downloadable=looks_openly_downloadable(model),
        ollama_model_digest=digest,
    )

    options = {
        "temperature": 0.0,  # Ollama's spelling of greedy decoding
        "top_p": 1.0,
        "top_k": 0,
        "seed": int(seed),
        "num_ctx": int(num_ctx),
        "num_predict": int(max_new_tokens),
    }

    def _call(system: str, user: str) -> str:
        estimated = estimate_prompt_tokens(system, user)
        if estimated > int(num_ctx):
            raise RuntimeError(
                f"prompt is about {estimated} tokens but num_ctx is {num_ctx}; Ollama "
                "would silently truncate the report and corrupt the labels. "
                "Raise --num-ctx."
            )
        body = {
            "model": str(model),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            # A full JSON Schema when we have one, so a missing target or an
            # invented state is impossible rather than a retry.
            "format": schema if schema is not None else "json",
            "options": options,
            "think": bool(think),
        }
        try:
            payload = _ollama_request(host, "/api/chat", body, timeout=timeout)
        except RuntimeError:
            # Older builds reject the unknown `think` field; retry without it.
            body.pop("think", None)
            payload = _ollama_request(host, "/api/chat", body, timeout=timeout)
        content = (payload.get("message") or {}).get("content", "")
        return strip_thinking(content)

    return _call, provenance


def resolve_revision(model_id: str, revision: str | None = None) -> str:
    """Resolve a hub repo to an exact commit SHA.

    Falls back to the caller-supplied revision when the hub is unreachable, so
    an air-gapped run can still pin provenance by passing the SHA explicitly.
    """
    if revision and revision != "main":
        return str(revision)
    try:
        from huggingface_hub import model_info
    except ImportError:
        return str(revision or "unknown")
    try:
        info = model_info(str(model_id), revision=revision or None)
    except Exception:
        return str(revision or "unknown")
    return str(getattr(info, "sha", None) or revision or "unknown")


def hash_local_weights(model_path: str | Path, *, chunk_bytes: int = 1 << 20) -> str:
    """Digest the weight shards of a downloaded checkpoint.

    Optional but decisive: it proves which bytes produced the labels even if a
    hub repository is later re-tagged. Files are hashed in sorted name order so
    the digest is stable across filesystems.
    """
    root = Path(model_path)
    if not root.is_dir():
        raise NotADirectoryError(f"expected a checkpoint directory, got {root}")
    shards = sorted(
        [p for p in root.rglob("*") if p.suffix in {".safetensors", ".bin"} and p.is_file()],
        key=lambda p: str(p.relative_to(root)),
    )
    if not shards:
        raise FileNotFoundError(f"no .safetensors or .bin weight shards under {root}")
    digest = hashlib.sha256()
    for shard in shards:
        digest.update(str(shard.relative_to(root)).encode("utf-8"))
        with shard.open("rb") as handle:
            while True:
                block = handle.read(chunk_bytes)
                if not block:
                    break
                digest.update(block)
    return digest.hexdigest()


def make_local_transformers_backend(
    system_prompt: str,
    *,
    model_id: str = DEFAULT_LOCAL_MODEL,
    revision: str | None = None,
    dtype: str = "bfloat16",
    quantisation: str = "none",
    max_new_tokens: int = 2048,
    seed: int = 2026,
    device_map: str = "auto",
    weights_sha256: str | None = None,
):
    """Reference local backend: transformers, greedy decoding, pinned revision."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "B23 local backend requires transformers: pip install -e '.[local-llm]'"
        ) from exc

    resolved = resolve_revision(model_id, revision)
    torch_dtype = getattr(torch, dtype)
    load_kwargs: dict = {"revision": resolved if resolved != "unknown" else None}
    if quantisation == "4bit":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
        )
    elif quantisation == "8bit":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantisation != "none":
        raise ValueError("quantisation must be one of: none, 8bit, 4bit")

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=load_kwargs["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch_dtype, device_map=device_map, **load_kwargs
    )
    model.eval()

    provenance = ModelProvenance(
        backend=BACKEND_LOCAL_TRANSFORMERS,
        model_id=str(model_id),
        revision=resolved,
        dtype=str(dtype),
        quantisation=str(quantisation),
        decoding=DECODING_GREEDY,
        max_new_tokens=int(max_new_tokens),
        seed=int(seed),
        prompt_sha256=prompt_sha256(system_prompt),
        openly_downloadable=looks_openly_downloadable(model_id),
        weights_sha256=weights_sha256,
    )

    def _call(system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,  # greedy: no RNG dependence at all
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion = generated[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(completion, skip_special_tokens=True)

    return _call, provenance


def make_local_vllm_backend(
    system_prompt: str,
    *,
    model_id: str = DEFAULT_LOCAL_MODEL,
    revision: str | None = None,
    dtype: str = "bfloat16",
    quantisation: str = "none",
    max_new_tokens: int = 2048,
    seed: int = 2026,
    gpu_memory_utilization: float = 0.90,
    weights_sha256: str | None = None,
):
    """Same weights as the reference path, batched for the full corpus.

    vLLM supports JSON-schema guided decoding, which removes most parse
    failures outright rather than relying on the retry loop.
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("B23 vLLM backend requires vllm: pip install vllm") from exc

    resolved = resolve_revision(model_id, revision)
    engine_kwargs: dict = {
        "model": str(model_id),
        "dtype": str(dtype),
        "seed": int(seed),
        "gpu_memory_utilization": float(gpu_memory_utilization),
    }
    if resolved and resolved != "unknown":
        engine_kwargs["revision"] = resolved
    if quantisation != "none":
        engine_kwargs["quantization"] = str(quantisation)
    engine = LLM(**engine_kwargs)
    tokenizer = engine.get_tokenizer()
    # temperature=0 is vLLM's spelling of greedy decoding.
    params = SamplingParams(temperature=0.0, max_tokens=int(max_new_tokens))

    provenance = ModelProvenance(
        backend=BACKEND_LOCAL_VLLM,
        model_id=str(model_id),
        revision=resolved,
        dtype=str(dtype),
        quantisation=str(quantisation),
        decoding=DECODING_GREEDY,
        max_new_tokens=int(max_new_tokens),
        seed=int(seed),
        prompt_sha256=prompt_sha256(system_prompt),
        openly_downloadable=looks_openly_downloadable(model_id),
        weights_sha256=weights_sha256,
    )

    def _call(system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = engine.generate([text], params)
        return outputs[0].outputs[0].text

    return _call, provenance


def make_hosted_api_backend(
    system_prompt: str,
    *,
    model_id: str = "claude-sonnet-5",
    max_new_tokens: int = 4096,
    api_key: str | None = None,
):
    """Development-only backend. NOT competition-reproducible.

    The served weights behind a hosted model name can change without notice, so
    provenance cannot be pinned. `ModelProvenance.reproducible` is False for
    this backend and `run_b23_export` refuses to certify such an export.
    """
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("hosted backend requires: pip install -e '.[hosted-llm]'") from exc

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("hosted backend requires ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=key)

    provenance = ModelProvenance(
        backend=BACKEND_HOSTED_API,
        model_id=str(model_id),
        revision="unknown",
        dtype="unknown",
        quantisation="unknown",
        decoding=DECODING_GREEDY,
        max_new_tokens=int(max_new_tokens),
        seed=0,
        prompt_sha256=prompt_sha256(system_prompt),
        openly_downloadable=False,
    )

    def _call(system: str, user: str) -> str:
        response = client.messages.create(
            model=str(model_id),
            max_tokens=int(max_new_tokens),
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )

    return _call, provenance


def load_provenance(path: str | Path) -> ModelProvenance:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "provenance" in payload:
        payload = payload["provenance"]
    known = set(ModelProvenance.__dataclass_fields__)
    return ModelProvenance(**{k: v for k, v in payload.items() if k in known})
