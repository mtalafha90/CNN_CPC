#!/usr/bin/env python
"""Audit-only local Ollama review of learned B27 routing tables.

This script is deliberately outside the training and submission path.  It makes
one local Qwen/Ollama call after B27 training and asks whether the *learned*
plane/contrast preferences are clinically interpretable.  Its output is never
fed back into the model, never changes labels or weights, and must not be used
to tune B27 after reading reused-gold performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsna_knee.b23_local_llm import make_ollama_backend

SYSTEM_PROMPT = """You are an audit-only musculoskeletal MRI methods reviewer.
You will receive learned B27 additive attention-logit biases for 12 knee MRI
abnormalities. Each target has three independent categorical bias tables:
plane (Sagittal/Coronal/Axial), fluid sensitivity (structural/fluid_sensitive),
and fat suppression (not_fat_suppressed/fat_suppressed).

Rules:
1. Review only clinical plausibility and interpretability of the learned routing.
2. Do NOT recommend changing weights, architecture, labels, thresholds, or epochs.
3. Do NOT infer predictive accuracy from these biases.
4. Compare categories mainly within the same target and metadata dimension.
5. Flag a preference as surprising only when there is a clear clinical reason.
6. Return concise JSON matching the requested schema.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "overall": {"type": "string"},
        "clinically_plausible_targets": {
            "type": "array",
            "items": {"type": "string"},
        },
        "surprising_targets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["target", "reason"],
            },
        },
        "cautions": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "overall",
        "clinically_plausible_targets",
        "surprising_targets",
        "cautions",
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser("Audit B27 learned routing with local Ollama")
    ap.add_argument("--routing", default="runs/b27_pathology_routing/routing_biases.json")
    ap.add_argument("--out", default="runs/b27_pathology_routing/ollama_route_review.json")
    ap.add_argument("--model", default="qwen3:14b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    routing_path = Path(args.routing)
    routing = json.loads(routing_path.read_text(encoding="utf-8"))
    if int(routing.get("parameter_count", -1)) != 84:
        raise ValueError("expected the frozen 84-parameter B27 routing table")

    call, provenance = make_ollama_backend(
        SYSTEM_PROMPT,
        model=args.model,
        host=args.host,
        num_ctx=8192,
        max_new_tokens=2048,
        seed=2026,
        think=False,
        timeout=float(args.timeout),
        schema=SCHEMA,
    )
    if not provenance.reproducible:
        raise RuntimeError("B27 route review requires a pinned reproducible local Ollama model")

    user = (
        "Review these learned B27 routing biases. They were learned from the MRI "
        "training objective; no pathology-plane preference was hard-coded.\n\n"
        + json.dumps(routing, ensure_ascii=False, separators=(",", ":"))
    )
    raw = call(SYSTEM_PROMPT, user)
    review = json.loads(raw)

    payload = {
        "role": "audit-only clinical plausibility review; not training or model selection",
        "routing_file": str(routing_path),
        "model_provenance": provenance.to_dict(),
        "review": review,
        "governance": (
            "Ollama output is descriptive only. It is not consumed by B27, is not part "
            "of competition inference, and cannot authorize post-hoc route tuning."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(provenance.describe())
    print(json.dumps(review, indent=2, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()
