"""Explanatory figures and plots of actual notebook-run results."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


def architecture():
    """Exact default architecture; arrows show tensors, not training history."""
    fig, ax = plt.subplots(figsize=(11, 12), layout="constrained")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(.04, .985, "Knee MRI → twelve finding probabilities", fontsize=19,
            weight="bold", va="top", color="#15334a")
    ax.text(.04, .95, "DINOv2 image features · slice context · attention across MRI series",
            fontsize=11, va="top", color="#526777")

    def box(x, y, w, h, title, detail, color="#eaf2f8"):
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                     boxstyle="round,pad=0.008,rounding_size=0.008",
                     facecolor=color, edgecolor="#8ba5b7", lw=1))
        ax.text(x, y+.012, title, ha="center", va="center", fontsize=11, weight="bold", color="#15334a")
        ax.text(x, y-.017, detail, ha="center", va="center", fontsize=9, color="#344e60")

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                   color="#526777", lw=1.4))

    centers = [.87, .75, .63, .51, .39, .27, .15]
    stages = [
        ("One study: K eligible MRI series", "Each series keeps its own rectangular geometry"),
        ("32 three-slice inputs per series", "[32, 3, H, W] · 90% central crop · area ≈ 448²"),
        ("DINOv2 ViT-S/14 encoder", "14 × 14 image patches → [32, 384] slice features"),
        ("One series transformer", "Summary token + 32 features → one 384-value vector"),
        ("Series vectors + scan metadata", "K vectors · [K, 384]"),
        ("Attention across series", "12 learned finding queries · 6 attention heads"),
        ("Twelve finding-specific linear heads", "12 logits → elementwise sigmoid → 12 probabilities"),
    ]
    for i, (y, (title, detail)) in enumerate(zip(centers, stages)):
        box(.345, y, .59, .075, title, detail, "#e7f4ee" if i == 6 else "#eaf2f8")
        if i:
            arrow((.345, centers[i-1]-.046), (.345, y+.046))
    box(.83, .39, .25, .09, "Scan metadata", "Plane / fluid / fat", "#fff2dd")
    arrow((.697, .39), (.65, .39))
    box(.83, .27, .25, .09, "Finding queries", "12 × 384 trainable values", "#fff2dd")
    arrow((.697, .27), (.65, .27))
    ax.text(.70, .62, "The encoder is fully\nfine-tuned from public\npretrained weights.",
            fontsize=10, color="#344e60", va="center", linespacing=1.6)
    ax.text(.70, .51, "No additional slice-\nposition or physical-\nspacing embedding.",
            fontsize=10, color="#344e60", va="center", linespacing=1.6)
    ax.text(.04, .055, "384 is the feature-vector width, not an image size. K varies by study.\n"
            "The three input channels are neighboring MRI slices, not RGB colors.",
            fontsize=11, color="#344e60", linespacing=1.7)
    return fig


def slice_example(item, *, series_index=0, center_index=None):
    volume = item["volumes"][series_index].detach().cpu().numpy()
    positions = item["slice_position"][series_index].detach().cpu().numpy()
    if center_index is None:
        center_index = len(volume) // 2
    if not 0 <= center_index < len(volume):
        raise IndexError("Choose a sampled center that exists in this series.")
    fig = plt.figure(figsize=(11, 6.5), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(5, 1))
    for channel, title in enumerate(("Previous slice", "Center slice", "Next slice")):
        ax = fig.add_subplot(grid[0, channel])
        ax.imshow(volume[center_index, channel], cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    ax = fig.add_subplot(grid[1, :])
    ax.scatter(positions, np.zeros_like(positions), color="#397ab0", label="Sampled centers")
    ax.scatter([positions[center_index]], [0], color="#dd7b30", s=90, label="Displayed triplet", zorder=3)
    ax.set(xlim=(-.03, 1.03), ylim=(-.4, .4), yticks=[], xlabel="Normalized through-series position")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.45))
    fig.suptitle(f"Actual training input · series {series_index + 1} · {tuple(volume.shape)}", weight="bold")
    return fig


def sigmoid():
    z = np.linspace(-7, 7, 400)
    p = 1 / (1 + np.exp(-z))
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot(z, p, color="#397ab0", lw=2)
    ax.scatter([-2, 0, 2], 1 / (1 + np.exp(-np.array([-2, 0, 2]))), color="#dd7b30", zorder=3)
    ax.axhline(.5, color="#b2bec8", ls="--", lw=1)
    ax.set(xlabel="Logit z (raw model score)", ylabel="Sigmoid probability", ylim=(-.02, 1.02),
           title="Sigmoid: a score becomes a number between 0 and 1")
    ax.grid(alpha=.15)
    return fig


def training_history(history):
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].plot(epochs, [r["train_loss"] for r in history], "o-", color="#397ab0")
    axes[0].set(title="Training loss", xlabel="Completed epoch", ylabel="Weighted binary cross-entropy")
    axes[1].plot(epochs, [r["validation"]["macro_auc"] for r in history], "o-", color="#32947b")
    axes[1].set(title="Scanner validation AUC", xlabel="Completed epoch", ylabel="Macro ROC AUC", ylim=(0, 1))
    for ax in axes:
        ax.set_xticks(epochs)
        ax.grid(alpha=.2)
    return fig


def target_auc(tables):
    findings = list(tables["validation"].index)
    x = np.arange(len(findings))
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    ax.barh(x+.18, tables["validation"].loc[findings, "AUC"], height=.34,
            label="Scanner validation", color="#397ab0")
    ax.barh(x-.18, tables["expert"].loc[findings, "AUC"], height=.34,
            label="Expert diagnostic", color="#77b8a4")
    ax.axvline(.5, ls="--", color="#a1abb3", lw=1)
    ax.set(yticks=x, yticklabels=findings, xlim=(0, 1), xlabel="ROC AUC", title="Fixed final model · AUC by finding")
    ax.invert_yaxis()
    ax.legend(loc="lower left")
    return fig


def save_figure(fig, run_root, name):
    from pathlib import Path
    out = Path(run_root) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=160, facecolor="white")
    fig.savefig(out / f"{name}.svg", facecolor="white")
