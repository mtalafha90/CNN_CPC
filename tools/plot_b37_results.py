"""Create a visual report for the completed fixed B37 experiment.

The B37 training endpoint writes ``history.json`` at ``--run-root``.  The
diagnostic evaluation writes the expert-58 predictions and ``expert58.json``
under ``--run-root/expert58``.  This tool keeps those two stages separate: it
does not re-run training or evaluation, and it never uses the weak training
labels as a substitute for the expert labels in a confusion matrix.

Run the diagnostic evaluation once after the fixed E2 checkpoint is written::

    B37_ROOT=/media/talafha/Disk_1/CNN_CPC/runs/071_Experiment_B37_highres_448_sparse_mil/b37_highres_sparse_mil
    DATA_ROOT=/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection
    BASE=/media/talafha/Disk_1/CNN_CPC/runs/067_Experiment_LLM_FILL_ALL_b6_preserved6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all_ft1/train/llm-filled/model.pt
    PYTHONPATH=developments/src python -m rsna_knee.b37_highres_sparse_eval \\
      --config config/b37_highres_sparse_448.yaml \\
      --data-root "$DATA_ROOT" \\
      --checkpoint "$B37_ROOT/b37_model.pt" \\
      --base-checkpoint "$BASE" \\
      --out-root "$B37_ROOT/expert58"

Then generate the report::

    python tools/plot_b37_results.py \\
      --run-root "$B37_ROOT" \\
      --data-root "$DATA_ROOT"

The output directory defaults to ``<run-root>/report``.  The confusion-matrix
and case-review figures use only the reused, fully labelled Expert-58 surface;
they are development diagnostics, not independent test evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

# The script is normally run on a workstation or through systemd, where no GUI
# display server is available.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


STUDY_UID = "StudyInstanceUID"
TARGETS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)
PREDICTION_FILES = {
    "base_224": "base_224_predictions.csv",
    "b37_global_448": "b37_global_448_predictions.csv",
    "b37_combined": "b37_combined_predictions.csv",
}
MODEL_AUC_KEYS = {
    "base_224": "base_224_auc",
    "b37_global_448": "b37_global_448_auc",
    "b37_combined": "b37_combined_auc",
}
MODEL_LABELS = {
    "base_224": "Historical base (224)",
    "b37_global_448": "B37 global (448)",
    "b37_combined": "B37 combined sparse-MIL (448)",
}


class B37ReportError(ValueError):
    """The B37 output directory is incomplete or violates its file contract."""


def _read_json(path: Path) -> Any:
    """Read one UTF-8 JSON file with a useful path-specific error message."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise B37ReportError(f"required file is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise B37ReportError(f"invalid JSON in {path}: {error}") from error


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], path: Path) -> None:
    """Reject a CSV whose B37 contract columns are not all present."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise B37ReportError(f"{path} is missing required columns: {missing}")


def load_history(run_root: Path) -> pd.DataFrame:
    """Load B37's fixed-endpoint per-epoch history into a validated table."""
    path = run_root / "history.json"
    payload = _read_json(path)
    if not isinstance(payload, list) or not payload:
        raise B37ReportError(f"{path} must contain a non-empty epoch list")
    frame = pd.DataFrame(payload)
    required = ("epoch", "loss_total", "loss_combined", "loss_local_aux")
    _require_columns(frame, required, path)
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame["epoch"].duplicated().any():
        raise B37ReportError(f"{path} contains duplicate epoch numbers")
    return frame.sort_values("epoch").reset_index(drop=True)


def _gate_values(history: pd.DataFrame) -> np.ndarray:
    """Extract the twelve effective sparse-MIL gates from every epoch, if stored."""
    values: list[list[float]] = []
    for row in history.to_dict(orient="records"):
        gate_state = row.get("gate")
        if not isinstance(gate_state, dict):
            raise B37ReportError(
                "history.json does not contain B37 effective sparse-MIL gates"
            )
        # B37 writes ``model.head.state()`` directly into each epoch row, so
        # the current fixed-endpoint contract stores this field at the top
        # level.  The nested form is accepted only for earlier report drafts.
        gate = gate_state.get("gate_effective")
        if gate is None and isinstance(gate_state.get("head"), dict):
            gate = gate_state["head"].get("gate_effective")
        if not isinstance(gate, list) or len(gate) != len(TARGETS):
            raise B37ReportError(
                "each history gate must contain one value for every B37 target"
            )
        values.append([float(value) for value in gate])
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise B37ReportError("history.json contains a non-finite sparse-MIL gate")
    return array


def plot_training_losses(history: pd.DataFrame, output_path: Path) -> None:
    """Plot B37 total, combined, and local auxiliary losses versus epoch."""
    figure, axis = plt.subplots(figsize=(9, 5.5))
    styles = (
        ("loss_total", "Total loss", "#1f77b4"),
        ("loss_combined", "Combined loss", "#ff7f0e"),
        ("loss_local_aux", "Local auxiliary loss", "#2ca02c"),
    )
    for column, label, color in styles:
        axis.plot(
            history["epoch"],
            history[column],
            marker="o",
            linewidth=2.2,
            label=label,
            color=color,
        )
    axis.set_title("B37 training losses (fixed two-epoch endpoint)")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean loss")
    axis.set_xticks(history["epoch"].tolist())
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_gate_evolution(history: pd.DataFrame, output_path: Path) -> None:
    """Plot the learned effective sparse-MIL gate for every pathology target."""
    gate = _gate_values(history)
    figure, axis = plt.subplots(figsize=(11, 6.5))
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, len(TARGETS)))
    for index, target in enumerate(TARGETS):
        axis.plot(
            history["epoch"],
            gate[:, index],
            marker="o",
            linewidth=1.8,
            label=target,
            color=colors[index],
        )
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
    axis.set_title("B37 effective sparse-MIL gate by target")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("tanh(gate)")
    axis.set_xticks(history["epoch"].tolist())
    axis.grid(alpha=0.2)
    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def load_evaluation(evaluation_root: Path) -> dict[str, Any]:
    """Load the completed B37 Expert-58 diagnostic metadata."""
    path = evaluation_root / "expert58.json"
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise B37ReportError(f"{path} must contain an object")
    per_target = payload.get("per_target")
    if not isinstance(per_target, dict):
        raise B37ReportError(f"{path} does not contain per_target metrics")
    missing = [target for target in TARGETS if target not in per_target]
    if missing:
        raise B37ReportError(f"{path} does not cover all B37 targets: {missing}")
    return payload


def load_predictions(evaluation_root: Path, model_name: str) -> pd.DataFrame:
    """Load one B37 Expert-58 probability file and validate its probability surface."""
    path = evaluation_root / PREDICTION_FILES[model_name]
    try:
        frame = pd.read_csv(path)
    except FileNotFoundError as error:
        raise B37ReportError(f"required prediction file is missing: {path}") from error
    _require_columns(frame, (STUDY_UID, *TARGETS), path)
    frame = frame[[STUDY_UID, *TARGETS]].copy()
    frame[STUDY_UID] = frame[STUDY_UID].astype(str)
    if frame[STUDY_UID].duplicated().any():
        raise B37ReportError(f"{path} contains duplicate study UIDs")
    for target in TARGETS:
        frame[target] = pd.to_numeric(frame[target], errors="raise")
    probabilities = frame.loc[:, TARGETS].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise B37ReportError(f"{path} must contain finite probabilities in [0, 1]")
    return frame.sort_values(STUDY_UID).reset_index(drop=True)


def load_expert_truth(data_root: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    """Join prediction UIDs to fully observed expert truth in the release train CSV."""
    path = data_root / "train.csv"
    try:
        train = pd.read_csv(path)
    except FileNotFoundError as error:
        raise B37ReportError(f"required dataset table is missing: {path}") from error
    _require_columns(train, (STUDY_UID, *TARGETS), path)
    truth = train[[STUDY_UID, *TARGETS]].copy()
    truth[STUDY_UID] = truth[STUDY_UID].astype(str)
    if truth[STUDY_UID].duplicated().any():
        raise B37ReportError(f"{path} contains duplicate study UIDs")
    truth = truth.rename(columns={target: f"truth::{target}" for target in TARGETS})
    merged = predictions.merge(truth, on=STUDY_UID, how="left", validate="one_to_one")
    missing = merged[[f"truth::{target}" for target in TARGETS]].isna().any(axis=1)
    if missing.any():
        uids = merged.loc[missing, STUDY_UID].head(5).tolist()
        raise B37ReportError(
            "predictions do not map to complete expert labels in train.csv; "
            f"example UIDs: {uids}"
        )
    for target in TARGETS:
        column = f"truth::{target}"
        merged[column] = pd.to_numeric(merged[column], errors="raise")
        if not merged[column].isin((0, 1)).all():
            raise B37ReportError(f"expert truth for {target!r} must be exactly 0 or 1")
    return merged.sort_values(STUDY_UID).reset_index(drop=True)


def plot_auc_comparison(evaluation: dict[str, Any], output_path: Path) -> None:
    """Compare all predeclared B37 diagnostic AUCs without selecting a winner."""
    per_target = evaluation["per_target"]
    positions = np.arange(len(TARGETS))
    width = 0.25
    figure, axis = plt.subplots(figsize=(16, 6.5))
    colors = ("#7f7f7f", "#1f77b4", "#2ca02c")
    for offset, (model_name, color) in enumerate(zip(PREDICTION_FILES, colors)):
        key = MODEL_AUC_KEYS[model_name]
        values = [float(per_target[target][key]) for target in TARGETS]
        axis.bar(
            positions + (offset - 1) * width,
            values,
            width=width,
            label=MODEL_LABELS[model_name],
            color=color,
        )
    axis.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("ROC AUC")
    axis.set_title("Expert-58 diagnostic: per-target AUC comparison")
    axis.set_xticks(positions)
    axis.set_xticklabels(TARGETS, rotation=35, ha="right")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _case_table(joined: pd.DataFrame, threshold: float, n_examples: int) -> pd.DataFrame:
    """Build a long-form fixed-order case table for the first Expert-58 examples."""
    chosen = joined.sort_values(STUDY_UID).head(n_examples)
    rows: list[dict[str, Any]] = []
    for record in chosen.to_dict(orient="records"):
        for target in TARGETS:
            probability = float(record[target])
            rows.append(
                {
                    STUDY_UID: record[STUDY_UID],
                    "target": target,
                    "expert_truth": int(record[f"truth::{target}"]),
                    "probability": probability,
                    "predicted_positive": bool(probability >= threshold),
                }
            )
    return pd.DataFrame(rows)


def plot_case_examples(joined: pd.DataFrame, threshold: float, n_examples: int, output_path: Path) -> pd.DataFrame:
    """Plot a fixed, non-cherry-picked twelve-case expert classification review."""
    chosen = joined.sort_values(STUDY_UID).head(n_examples).reset_index(drop=True)
    truth = chosen[[f"truth::{target}" for target in TARGETS]].to_numpy(dtype=np.float64)
    probability = chosen.loc[:, TARGETS].to_numpy(dtype=np.float64)
    figure, axes = plt.subplots(1, 2, figsize=(19, max(7.5, 0.58 * len(chosen) + 3.0)))
    truth_image = axes[0].imshow(truth, vmin=0.0, vmax=1.0, cmap="Greys")
    probability_image = axes[1].imshow(probability, vmin=0.0, vmax=1.0, cmap="viridis")
    for row in range(len(chosen)):
        for column in range(len(TARGETS)):
            text_color = "white" if probability[row, column] < 0.42 else "black"
            marker = "+" if probability[row, column] >= threshold else ""
            axes[1].text(
                column,
                row,
                f"{probability[row, column]:.2f}{marker}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
            )
    for axis, title in zip(
        axes,
        ("Expert truth (white = positive)", "B37 combined probability (+ = classification positive)"),
    ):
        axis.set_title(title)
        axis.set_xticks(np.arange(len(TARGETS)))
        axis.set_xticklabels(TARGETS, rotation=45, ha="right", fontsize=8)
        axis.set_yticks(np.arange(len(chosen)))
        axis.set_yticklabels([f"Case {index + 1}" for index in range(len(chosen))], fontsize=8)
        axis.set_xlabel("Target")
    axes[0].set_ylabel("Expert-58 study (fixed StudyInstanceUID order)")
    figure.colorbar(truth_image, ax=axes[0], fraction=0.046, pad=0.04, ticks=(0, 1))
    figure.colorbar(probability_image, ax=axes[1], fraction=0.046, pad=0.04, label="Probability")
    figure.suptitle(
        f"B37 Expert-58 classification review: first {len(chosen)} studies, threshold={threshold:.2f}",
        y=1.02,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return _case_table(joined, threshold, len(chosen))


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    """Return a binary-performance ratio only when its denominator exists."""
    return None if denominator == 0 else float(numerator / denominator)


def confusion_rows(joined: pd.DataFrame, threshold: float, model_name: str) -> list[dict[str, Any]]:
    """Calculate one explicit 2x2 matrix and derived diagnostics for each target."""
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        truth = joined[f"truth::{target}"].to_numpy(dtype=np.int64)
        predicted = (joined[target].to_numpy(dtype=np.float64) >= threshold).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=(0, 1)).ravel()
        sensitivity = _safe_ratio(int(tp), int(tp + fn))
        specificity = _safe_ratio(int(tn), int(tn + fp))
        rows.append(
            {
                "model": model_name,
                "target": target,
                "threshold": float(threshold),
                "n_expert_studies": int(len(truth)),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "accuracy": _safe_ratio(int(tn + tp), int(len(truth))),
                "sensitivity": sensitivity,
                "specificity": specificity,
                "precision": _safe_ratio(int(tp), int(tp + fp)),
                "balanced_accuracy": None
                if sensitivity is None or specificity is None
                else float((sensitivity + specificity) / 2.0),
            }
        )
    return rows


def plot_confusion_matrices(joined: pd.DataFrame, threshold: float, model_name: str, output_path: Path) -> list[dict[str, Any]]:
    """Plot one clear binary confusion matrix per pathology target in a 3x4 grid."""
    rows = confusion_rows(joined, threshold, model_name)
    maximum = max(len(joined), 1)
    figure, axes = plt.subplots(3, 4, figsize=(16, 11.5), constrained_layout=True)
    image = None
    for axis, row in zip(axes.flat, rows):
        matrix = np.array(
            [
                [row["true_negative"], row["false_positive"]],
                [row["false_negative"], row["true_positive"]],
            ]
        )
        image = axis.imshow(matrix, vmin=0, vmax=maximum, cmap="Blues")
        for y_index in range(2):
            for x_index in range(2):
                value = int(matrix[y_index, x_index])
                color = "white" if value > maximum / 2.0 else "black"
                axis.text(x_index, y_index, str(value), ha="center", va="center", color=color, fontsize=11)
        axis.set_title(row["target"], fontsize=10)
        axis.set_xticks((0, 1), ("Pred. 0", "Pred. 1"), fontsize=8)
        axis.set_yticks((0, 1), ("True 0", "True 1"), fontsize=8)
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.76, label="Number of Expert-58 studies")
    figure.suptitle(
        f"{MODEL_LABELS[model_name]}: Expert-58 confusion matrices at threshold={threshold:.2f}",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return rows


def generate_b37_report(
    *,
    run_root: Path,
    data_root: Path,
    evaluation_root: Path | None = None,
    output_dir: Path | None = None,
    threshold: float = 0.50,
    n_examples: int = 12,
    confusion_model: str = "b37_combined",
) -> dict[str, Path]:
    """Write all B37 training and Expert-58 diagnostic report artifacts."""
    if not 0.0 < float(threshold) < 1.0:
        raise B37ReportError("threshold must be strictly between 0 and 1")
    if int(n_examples) < 1:
        raise B37ReportError("n_examples must be at least one")
    if confusion_model not in PREDICTION_FILES:
        raise B37ReportError(f"unknown confusion model: {confusion_model}")
    run_root = Path(run_root).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    evaluation_root = (
        run_root / "expert58" if evaluation_root is None else Path(evaluation_root).expanduser().resolve()
    )
    output_dir = run_root / "report" if output_dir is None else Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    history = load_history(run_root)
    paths: dict[str, Path] = {
        "training_losses": output_dir / "training_losses.png",
        "gate_evolution": output_dir / "sparse_gate_evolution.png",
        "auc_comparison": output_dir / "expert58_auc_comparison.png",
        "case_examples": output_dir / f"expert58_case_examples_{confusion_model}.png",
        "case_table": output_dir / f"expert58_case_examples_{confusion_model}.csv",
        "confusion_matrices": output_dir / f"expert58_confusion_matrices_{confusion_model}.png",
        "confusion_table": output_dir / f"expert58_confusion_matrices_{confusion_model}.csv",
        "summary": output_dir / "b37_report_summary.json",
    }
    plot_training_losses(history, paths["training_losses"])
    plot_gate_evolution(history, paths["gate_evolution"])

    evaluation = load_evaluation(evaluation_root)
    predictions = load_predictions(evaluation_root, confusion_model)
    joined = load_expert_truth(data_root, predictions)
    plot_auc_comparison(evaluation, paths["auc_comparison"])
    case_table = plot_case_examples(joined, float(threshold), int(n_examples), paths["case_examples"])
    case_table.to_csv(paths["case_table"], index=False)
    matrices = plot_confusion_matrices(joined, float(threshold), confusion_model, paths["confusion_matrices"])
    pd.DataFrame(matrices).to_csv(paths["confusion_table"], index=False)

    summary = {
        "run_root": str(run_root),
        "evaluation_root": str(evaluation_root),
        "data_root": str(data_root),
        "classification_model": confusion_model,
        "classification_threshold": float(threshold),
        "epochs": history["epoch"].astype(int).tolist(),
        "n_expert_studies": int(len(joined)),
        "expert58_role": evaluation.get("evaluation_role"),
        "base_224_macro_auc": evaluation.get("base_224_macro_auc"),
        "b37_global_448_macro_auc": evaluation.get("b37_global_448_macro_auc"),
        "b37_combined_macro_auc": evaluation.get("b37_combined_macro_auc"),
        "macro_delta_primary": evaluation.get("macro_delta_primary"),
        "artifacts": {name: str(path) for name, path in paths.items() if name != "summary"},
    }
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return paths


def parse_args() -> argparse.Namespace:
    """Parse the paths and fixed diagnostic display choices for this report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path, help="Completed B37 training directory containing history.json.")
    parser.add_argument("--data-root", required=True, type=Path, help="RSNA data release directory containing train.csv.")
    parser.add_argument("--evaluation-root", type=Path, default=None, help="Expert-58 evaluation directory; defaults to <run-root>/expert58.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Report output directory; defaults to <run-root>/report.")
    parser.add_argument("--threshold", type=float, default=0.50, help="Fixed probability threshold used for displayed classifications and confusion matrices.")
    parser.add_argument("--n-examples", type=int, default=12, help="Number of fixed-order Expert-58 studies shown in the case-review figure.")
    parser.add_argument("--confusion-model", choices=tuple(PREDICTION_FILES), default="b37_combined", help="Prediction surface used for the case review and confusion matrices.")
    return parser.parse_args()


def main() -> None:
    """Generate the B37 report and print every written artifact path."""
    args = parse_args()
    try:
        outputs = generate_b37_report(
            run_root=args.run_root,
            data_root=args.data_root,
            evaluation_root=args.evaluation_root,
            output_dir=args.out_dir,
            threshold=args.threshold,
            n_examples=args.n_examples,
            confusion_model=args.confusion_model,
        )
    except (B37ReportError, OSError) as error:
        raise SystemExit(f"B37 report failed: {error}") from error
    print("B37 visual report written:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
