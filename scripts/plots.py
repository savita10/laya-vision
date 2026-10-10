"""Phase 4 figures: reliability diagrams and confusion matrices (matplotlib, Agg backend).

Bins come from vision.reliability_bins, which uses laya.common.ece_score's binning,
so a diagram and the ECE printed on it describe the same bins.
"""

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def reliability_diagram(path, panels, title):
    """panels: list of (label, bins from vision.reliability_bins, ece). One panel per entry,
    side by side: accuracy per confidence bin (bars), mean confidence per bin (dots), the
    diagonal, and bin counts below."""
    plt = _plt()
    fig, axes = plt.subplots(2, len(panels), figsize=(4.6 * len(panels), 5.6), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    for col, (label, bins, ece) in enumerate(panels):
        ax, axn = axes[0, col], axes[1, col]
        lo = np.array([b["lo"] for b in bins])
        width = bins[0]["hi"] - bins[0]["lo"]
        acc = np.array([b["accuracy"] if b["n"] else 0.0 for b in bins])
        conf = np.array([b["mean_confidence"] if b["n"] else np.nan for b in bins])
        n = np.array([b["n"] for b in bins])
        ax.bar(lo, acc, width=width, align="edge", color="#4c78a8", edgecolor="white", label="accuracy")
        filled = n > 0
        ax.plot((lo + width / 2)[filled], conf[filled], "o", color="#e45756", ms=4, label="mean confidence")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{label}  (ECE {ece:.3f})", fontsize=10)
        if col == 0:
            ax.set_ylabel("accuracy")
            ax.legend(loc="upper left", fontsize=8)
        axn.bar(lo, n, width=width, align="edge", color="#9d9d9d", edgecolor="white")
        axn.set_xlabel("confidence (max p)")
        if col == 0:
            axn.set_ylabel("count")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def confusion_png(path, matrix, classes, title):
    """Row-normalised 16x16 confusion matrix (rows = true class); cells show counts."""
    plt = _plt()
    m = np.asarray(matrix, dtype=float)
    norm = m / np.maximum(m.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if m[i, j]:
                ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="share of true class")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
