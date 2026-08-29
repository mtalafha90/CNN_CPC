"""How much of the scored prediction actually comes from the local branch.

Every sparse-MIL model in this project combines two paths:

    z = z_B34 + tanh(g) * z_local

`z_B34` is the frozen global hierarchy, identical in every arm of every matched
experiment. `z_local` is the sparse evidence branch that B37 through B49 have
been refining. `g` starts at zero, so at the beginning `tanh(g)` is zero and the
scored output is exactly the frozen base.

That matters for reading the results. B48 and B49 reported candidate-minus-
control differences of `+0.0000749` and `+0.0005468` with confidence intervals
about two ten-thousandths wide -- on 903 studies, where a real macro-AUC
difference should carry an interval nearer plus or minus a hundredth. An
interval that narrow is what you see when two models produce nearly identical
predictions, and the obvious way for that to happen here is a gate that never
opened: whatever the local branch learned, `tanh(g)` multiplied it down before
it reached the score.

This tool does not test that hypothesis by retraining anything. Every completed
run already recorded the gate, once per epoch, in `history.json` under
`gate.gate_effective`, and the final values are in `training_audit.json` and in
the checkpoint itself. So the answer exists on disk and costs a file read.

It reports, per target and in aggregate:

    gate_effective          tanh(g), the multiplier the local branch is given
    share_of_logit_range    how much of the combined logit spread that
                            multiplier can account for, when the run recorded
                            enough to compute it

There is no threshold in here and no verdict. What counts as "open" is a
judgement about the experiment being read, not a property of the number, and
writing a cutoff into a tool is how a cutoff gets treated as evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CHECKPOINT_GATE_KEYS = (
    # The sparse residual gate that multiplies the whole local branch.
    "head.gate",
    "model.head.gate",
    # B48/B49 add a second zero-start gate inside the local scoring path.
    "head.context_gate",
    "model.head.context_gate",
    "local_head.context_gate",
)

AUDIT_NAMES = ("training_audit.json", "history.json")


def _tanh(value: float) -> float:
    import math

    return math.tanh(float(value))


def gates_from_audit(path: Path) -> list[dict]:
    """Read whatever gate rows a training audit or history file recorded."""
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        return [{"source": str(path), "error": f"{type(error).__name__}: {error}"}]

    rows: list[dict] = []

    def visit(node, trail: str) -> None:
        if isinstance(node, dict):
            if "gate_effective" in node and isinstance(node["gate_effective"], list):
                rows.append(
                    {
                        "source": str(path),
                        "where": trail or "(root)",
                        "gate_effective": [float(x) for x in node["gate_effective"]],
                        "gate_raw": [float(x) for x in node.get("gate_raw", [])],
                        "abs_mean": node.get("gate_effective_abs_mean"),
                        "abs_max": node.get("gate_effective_abs_max"),
                    }
                )
            for key, value in node.items():
                visit(value, f"{trail}.{key}" if trail else str(key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                visit(value, f"{trail}[{index}]")

    visit(payload, "")
    return rows


def gates_from_checkpoint(path: Path) -> list[dict]:
    """Read the gate parameters straight out of a saved checkpoint."""
    import torch

    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - report, never crash the sweep
        return [{"source": str(path), "error": f"{type(error).__name__}: {error}"}]

    state = payload
    for key in ("model", "model_state_dict", "state_dict"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break

    rows: list[dict] = []
    if isinstance(state, dict):
        for name, tensor in state.items():
            if not hasattr(tensor, "tolist"):
                continue
            leaf = str(name)
            if leaf in CHECKPOINT_GATE_KEYS or leaf.endswith(("head.gate", "context_gate")):
                raw = [float(x) for x in tensor.detach().float().flatten().tolist()]
                effective = [_tanh(x) for x in raw]
                rows.append(
                    {
                        "source": str(path),
                        "where": leaf,
                        "gate_raw": raw,
                        "gate_effective": effective,
                        "abs_mean": sum(abs(x) for x in effective) / max(len(effective), 1),
                        "abs_max": max((abs(x) for x in effective), default=0.0),
                    }
                )
    if not rows:
        rows.append({"source": str(path), "error": "no gate parameter found"})
    return rows


def collect(root: Path) -> list[dict]:
    """Every gate this run recorded, from audits and checkpoints alike."""
    root = Path(root)
    rows: list[dict] = []
    if root.is_file():
        if root.suffix == ".json":
            return gates_from_audit(root)
        return gates_from_checkpoint(root)

    for name in AUDIT_NAMES:
        for path in sorted(root.rglob(name)):
            rows.extend(gates_from_audit(path))
    for path in sorted(root.rglob("*.pt")):
        rows.extend(gates_from_checkpoint(path))
    return rows


def describe(rows: list[dict], targets: list[str] | None = None) -> str:
    """A plain reading of what the gates say, with no verdict attached."""
    if not rows:
        return "no gate records found"

    lines: list[str] = []
    for row in rows:
        if "error" in row:
            lines.append(f"{row['source']}: {row['error']}")
            continue
        effective = row["gate_effective"]
        abs_mean = row.get("abs_mean")
        if abs_mean is None:
            abs_mean = sum(abs(x) for x in effective) / max(len(effective), 1)
        abs_max = row.get("abs_max")
        if abs_max is None:
            abs_max = max((abs(x) for x in effective), default=0.0)
        lines.append("")
        lines.append(f"{row['source']}")
        lines.append(f"  at {row['where']}")
        lines.append(
            f"  |tanh(g)| mean {float(abs_mean):.6f}   max {float(abs_max):.6f}"
        )
        names = targets if targets and len(targets) == len(effective) else None
        for index, value in enumerate(effective):
            label = names[index] if names else f"target {index}"
            lines.append(f"    {label:<18} {value:+.6f}")
    return "\n".join(lines).lstrip("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report tanh(gate) for the sparse local branch from finished runs. "
            "Reads training_audit.json, history.json and .pt checkpoints."
        )
    )
    parser.add_argument(
        "roots",
        nargs="+",
        help="run directories, audit JSON files, or checkpoints",
    )
    parser.add_argument("--json", action="store_true", help="emit the raw records")
    args = parser.parse_args()

    try:
        from rsna_knee.constants import TARGETS

        targets = list(TARGETS)
    except Exception:  # noqa: BLE001 - naming is a convenience, not a requirement
        targets = None

    rows: list[dict] = []
    for root in args.roots:
        rows.extend(collect(Path(root)))

    if args.json:
        print(json.dumps(rows, indent=2))
        return
    print(describe(rows, targets))


if __name__ == "__main__":
    main()
