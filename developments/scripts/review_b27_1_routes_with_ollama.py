#!/usr/bin/env python
"""Audit-only local Ollama review of learned B27.1 routing tables.

This is outside training and competition inference. It makes one local call and
cannot modify model parameters, labels, thresholds, epochs or model selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsna_knee.b23_local_llm import make_ollama_backend

SYSTEM_PROMPT = """You are an audit-only musculoskeletal MRI methods reviewer.
You will receive learned B27.1 additive attention-logit biases for 12 knee MRI
abnormalities. Each target has two categorical bias tables:
1) plane: Sagittal / Coronal / Axial
2) paired_sequence: structural_non_fat_suppressed / fluid_sensitive_fat_suppressed

The paired sequence representation was frozen because, on the complete training
surface, Fluid_Sensitive and Fat_Suppression were perfectly collinear and could
not be identified independently.

Rules:
1. Review only clinical plausibility and interpretability of the learned routing.
2. Do NOT recommend changing weights, architecture, labels, thresholds or epochs.
3. Do NOT infer predictive accuracy from routing biases.
4. Compare categories mainly within the same target and metadata dimension.
5. Flag a preference as surprising only when there is a clear clinical reason.
6. Treat the paired sequence category as one empirical acquisition axis, not as
   proof that fluid sensitivity and fat suppression are universally equivalent.
7. Return concise JSON matching the requested schema.
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
    "required": ["overall", "clinically_plausible_targets", "surprising_targets", "cautions"],
}


def main() -> None:
    ap = argparse.ArgumentParser("Audit B27.1 learned routing with local Ollama")
    ap.add_argument("--routing", default="runs/b27_1_pathology_routing/routing_biases.json")
    ap.add_argument("--out", default="runs/b27_1_pathology_routing/ollama_route_review.json")
    ap.add_argument("--model", default="qwen3:14b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    routing_path = Path(args.routing)
    routing = json.loads(routing_path.read_text(encoding="utf-8"))
    if int(routing.get("parameter_count", -1)) != 60:
        raise ValueError("expected the frozen 60-parameter B27.1 routing table")
    if set(routing).issuperset({"plane", "paired_sequence"}) is not True:
        raise ValueError("B27.1 routing file missing plane/paired_sequence tables")

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
        raise RuntimeError("B27.1 route review requires pinned reproducible local Ollama")

    user = (
        "Review these learned B27.1 routing biases. They were learned from the MRI "
        "training objective; no pathology-plane or sequence preference was hard-coded.\n\n"
        + json.dumps(routing, ensure_ascii=False, separators=(",", ":"))
    )
    review = json.loads(call(SYSTEM_PROMPT, user))
    payload = {
        "role": "audit-only clinical plausibility review; not training or model selection",
        "routing_file": str(routing_path),
        "model_provenance": provenance.to_dict(),
        "review": review,
        "governance": (
            "Ollama output is descriptive only. It is not consumed by B27.1, is not part "
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
