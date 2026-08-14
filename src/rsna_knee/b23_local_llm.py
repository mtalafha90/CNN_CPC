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

Three backends are provided. `local_transformers` is the reference path.
`local_vllm` is the same weights with faster batched decoding for the full
4,349-report corpus. `hosted_api` remains available for development only and is
explicitly marked non-reproducible, so an export produced with it cannot be
certified for competition use.

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

BACKEND_LOCAL_TRANSFORMERS = "local_transformers"
BACKEND_LOCAL_VLLM = "local_vllm"
BACKEND_HOSTED_API = "hosted_api"
REPRODUCIBLE_BACKENDS = (BACKEND_LOCAL_TRANSFORMERS, BACKEND_LOCAL_VLLM)

DECODING_GREEDY = "greedy"

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
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-14B-Instruct"
SUGGESTED_LOCAL_MODELS = {
    "Qwen/Qwen2.5-7B-Instruct": "~16 GB bf16, ~6 GB 4-bit; fastest, weakest",
    "Qwen/Qwen2.5-14B-Instruct": "~30 GB bf16, ~10 GB 4-bit; default",
    "Qwen/Qwen2.5-32B-Instruct": "~65 GB bf16, ~20 GB 4-bit; strong",
    "Qwen/Qwen2.5-72B-Instruct": "~145 GB bf16, ~42 GB 4-bit; strongest",
    "mistralai/Mistral-7B-Instruct-v0.3": "Apache-2.0 alternative family",
}

OPEN_LICENCE_PREFIXES = ("qwen/", "mistralai/", "meta-llama/", "google/gemma", "microsoft/phi")


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
    weights_sha256: str | None = None

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
    return any(lowered.startswith(prefix) for prefix in OPEN_LICENCE_PREFIXES)


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
