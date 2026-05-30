"""Generate the v1.2.0 accuracy figure (assets/accuracy_v1_2_0.png + .pdf).

A two-panel figure:
  Left: stacked-bar 3-way verdict distribution per split (dev/test) for VALID
        and HALLUCINATED entries, post-fix.
  Right: pre-fix vs post-fix FPR/leak comparison per split.

Numbers come from the apples-to-apples comparison against the
post-batch3-correction HALLMARK v1.0 gold; see bibtex_revalidation/
{dev_prefix.jsonl, dev_public_fixed.jsonl, test_prefix.jsonl,
test_public_fixed.jsonl}.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent.parent / "assets"
OUT_DIR.mkdir(exist_ok=True)

# Colorblind-safe palette (Wong 2011)
VERIFIED = "#0072B2"  # blue
CNV = "#E69F00"  # orange (could-not-verify)
PROBLEMATIC = "#D55E00"  # vermillion (FP on valid, catch on hallucinated)
LEAK = "#882255"  # dark purple (the bad-on-hallucinated case)

# Apples-to-apples numbers (post-correction gold incl. batch1-4, 32 mislabels)
# dev: 503 VALID, 616 HALLUCINATED
# test: 302 VALID, 529 HALLUCINATED
NUMBERS = {
    "dev": {
        "n_valid": 503,
        "n_hall": 616,
        "valid_pre": {"verified": 453, "cnv": 37, "problematic": 13},
        "valid_post": {"verified": 475, "cnv": 18, "problematic": 10},
        "hall_pre": {"verified": 3, "cnv": 231, "problematic": 382},
        "hall_post": {"verified": 4, "cnv": 149, "problematic": 463},
    },
    "test": {
        "n_valid": 302,
        "n_hall": 529,
        "valid_pre": {"verified": 243, "cnv": 32, "problematic": 27},
        "valid_post": {"verified": 280, "cnv": 15, "problematic": 7},
        "hall_pre": {"verified": 2, "cnv": 229, "problematic": 298},
        "hall_post": {"verified": 3, "cnv": 136, "problematic": 390},
    },
}


def stacked_bar(ax, split: str):
    """Stacked-bar 3-way verdict distribution per gold label (post-fix)."""
    d = NUMBERS[split]
    nv, nh = d["n_valid"], d["n_hall"]
    valid = d["valid_post"]
    hall = d["hall_post"]

    # Per-gold-label percentages
    v_verified = 100 * valid["verified"] / nv
    v_cnv = 100 * valid["cnv"] / nv
    v_prob = 100 * valid["problematic"] / nv
    h_caught = 100 * hall["problematic"] / nh
    h_cnv = 100 * hall["cnv"] / nh
    h_leak = 100 * hall["verified"] / nh

    bars = ["VALID\n(real refs)", "HALLUCINATED\n(bad refs)"]
    # Stack: bottom = correct, middle = abstain, top = wrong
    bottom = [v_verified, h_caught]
    middle = [v_cnv, h_cnv]
    top = [v_prob, h_leak]

    x = np.arange(len(bars))
    width = 0.55
    ax.bar(x, bottom, width, color=[VERIFIED, PROBLEMATIC], label="correct verdict", edgecolor="white", linewidth=0.5)
    ax.bar(x, middle, width, bottom=bottom, color=CNV, label="could-not-verify", edgecolor="white", linewidth=0.5)
    ax.bar(
        x,
        top,
        width,
        bottom=[bottom[i] + middle[i] for i in range(2)],
        color=[PROBLEMATIC, LEAK],
        label="wrong verdict (FP / leak)",
        edgecolor="white",
        linewidth=0.5,
    )

    # Annotate the headline cells
    ax.text(
        0,
        bottom[0] / 2,
        f"verified\n{bottom[0]:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontsize=9,
        fontweight="bold",
    )
    ax.text(0, bottom[0] + middle[0] / 2, f"CNV {middle[0]:.1f}%", ha="center", va="center", color="black", fontsize=8)
    ax.text(
        0,
        bottom[0] + middle[0] + top[0] / 2,
        f"FP {top[0]:.2f}%",
        ha="center",
        va="center",
        color="white",
        fontsize=8,
        fontweight="bold",
    )
    ax.text(
        1,
        bottom[1] / 2,
        f"caught\n{bottom[1]:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontsize=9,
        fontweight="bold",
    )
    ax.text(1, bottom[1] + middle[1] / 2, f"CNV {middle[1]:.1f}%", ha="center", va="center", color="black", fontsize=8)
    leak_pct = top[1]
    if leak_pct >= 0.2:
        ax.text(
            1,
            bottom[1] + middle[1] + top[1] / 2,
            f"LEAK {leak_pct:.2f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
        )
    else:
        ax.text(1, 102, f"leak {leak_pct:.2f}%", ha="center", va="center", color=LEAK, fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(bars)
    ax.set_ylim(0, 110)
    ax.set_ylabel("share of entries (%)")
    ax.set_title(f"{split}_public (n={nv + nh})", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fpr_leak_comparison(ax):
    """Pre-fix vs post-fix FPR + leak bars per split."""
    splits = ["dev_public", "test_public\n(held-out)"]
    fpr_pre = [
        100 * NUMBERS["dev"]["valid_pre"]["problematic"] / NUMBERS["dev"]["n_valid"],
        100 * NUMBERS["test"]["valid_pre"]["problematic"] / NUMBERS["test"]["n_valid"],
    ]
    fpr_post = [
        100 * NUMBERS["dev"]["valid_post"]["problematic"] / NUMBERS["dev"]["n_valid"],
        100 * NUMBERS["test"]["valid_post"]["problematic"] / NUMBERS["test"]["n_valid"],
    ]
    leak_pre = [
        100 * NUMBERS["dev"]["hall_pre"]["verified"] / NUMBERS["dev"]["n_hall"],
        100 * NUMBERS["test"]["hall_pre"]["verified"] / NUMBERS["test"]["n_hall"],
    ]
    leak_post = [
        100 * NUMBERS["dev"]["hall_post"]["verified"] / NUMBERS["dev"]["n_hall"],
        100 * NUMBERS["test"]["hall_post"]["verified"] / NUMBERS["test"]["n_hall"],
    ]

    x = np.arange(len(splits))
    width = 0.18
    # FPR bars
    ax.bar(
        x - 1.6 * width,
        fpr_pre,
        width,
        color=PROBLEMATIC,
        alpha=0.5,
        label="FPR pre-fix",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.bar(x - 0.6 * width, fpr_post, width, color=PROBLEMATIC, label="FPR post-fix", edgecolor="white", linewidth=0.5)
    # Leak bars
    ax.bar(
        x + 0.6 * width, leak_pre, width, color=LEAK, alpha=0.5, label="leak pre-fix", edgecolor="white", linewidth=0.5
    )
    ax.bar(x + 1.6 * width, leak_post, width, color=LEAK, label="leak post-fix", edgecolor="white", linewidth=0.5)

    for i, (pre, post) in enumerate(zip(fpr_pre, fpr_post)):
        ax.text(i - 1.6 * width, pre + 0.15, f"{pre:.2f}", ha="center", fontsize=8)
        ax.text(i - 0.6 * width, post + 0.15, f"{post:.2f}", ha="center", fontsize=8, fontweight="bold")
        # Δ arrow
        delta = (post - pre) / pre * 100
        ax.annotate(
            f"{delta:+.0f}%",
            xy=(i - 0.6 * width, post),
            xytext=(i - 1.1 * width, max(pre, post) + 1.2),
            fontsize=9,
            color=PROBLEMATIC,
            fontweight="bold",
            ha="center",
        )
    for i, (pre, post) in enumerate(zip(leak_pre, leak_post)):
        ax.text(i + 0.6 * width, pre + 0.15, f"{pre:.2f}", ha="center", fontsize=8)
        ax.text(i + 1.6 * width, post + 0.15, f"{post:.2f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("rate (%)")
    ax.set_title("v1.2.0 vs v1.0.0 baseline (corrected gold)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.32)
    ax_dev = fig.add_subplot(gs[0, 0])
    ax_test = fig.add_subplot(gs[0, 1])
    ax_cmp = fig.add_subplot(gs[0, 2])

    stacked_bar(ax_dev, "dev")
    stacked_bar(ax_test, "test")
    fpr_leak_comparison(ax_cmp)

    fig.suptitle(
        "bibtex-check v1.2.0 — HALLMARK v1.0 corrected gold (dev + held-out test)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    png = OUT_DIR / "accuracy_v1_2_0.png"
    pdf = OUT_DIR / "accuracy_v1_2_0.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {png} + {pdf}")


if __name__ == "__main__":
    main()
