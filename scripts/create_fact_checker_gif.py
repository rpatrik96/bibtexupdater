"""Create a lean animated GIF showing the reference fact-checker."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# --- Config ---
FIG_W, FIG_H = 12, 6.75
DPI = 150
BG = "#0d1117"
TEXT = "#c9d1d9"
MUTED = "#484f58"
GREEN = "#22c55e"
AMBER = "#f59e0b"
RED = "#ef4444"
BLUE = "#58a6ff"
CARD_BG = "#161b22"

REFS = [
    ("Smith et al. (2023)", "Deep Learning for Graph Neural...", "verified", GREEN, "Verified"),
    ("Jones et al. (2022)", "Transformer Architecture for...", "verified", GREEN, "Verified"),
    ("Chen et al. (2024)", "Neural Scaling Laws in Large...", "mismatch", AMBER, "Year Mismatch"),
    ("Doe et al. (2023)", "Quantum ML Framework for...", "hallucinated", RED, "Hallucinated"),
    ("Lee et al. (2021)", "Attention Mechanisms in Vision...", "verified", GREEN, "Verified"),
]


def draw_frame(fig, ax, n_scanned):
    """n_scanned: 0=title only, 1-5=entries scanned, 6=summary shown."""
    ax.clear()
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.5, 7)
    ax.set_facecolor(BG)
    fig.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(6, 6.6, "Reference Fact-Checker", fontsize=22, fontweight="bold", color=TEXT, ha="center")
    ax.text(6, 6.15, "bibtex-check", fontsize=12, color=BLUE, ha="center", fontfamily="monospace")

    # Draw entries
    box_h = 0.85
    gap = 0.15
    start_y = 5.2

    for i, (author, title, status, color, verdict) in enumerate(REFS):
        y = start_y - i * (box_h + gap)
        scanned = i < n_scanned

        # Entry box
        ec = color if scanned else MUTED
        lw = 2.5 if scanned and status == "hallucinated" else 1.8
        rect = mpatches.FancyBboxPatch(
            (0.3, y),
            7.5,
            box_h,
            boxstyle="round,pad=0.08",
            facecolor=CARD_BG,
            edgecolor=ec,
            linewidth=lw,
            zorder=2,
        )
        ax.add_patch(rect)

        # Red glow for hallucinated
        if scanned and status == "hallucinated":
            glow = mpatches.FancyBboxPatch(
                (0.22, y - 0.08),
                7.66,
                box_h + 0.16,
                boxstyle="round,pad=0.12",
                facecolor="none",
                edgecolor=RED,
                linewidth=1.5,
                alpha=0.3,
                zorder=1,
            )
            ax.add_patch(glow)

        # Author + title
        tc = TEXT if scanned else MUTED
        ax.text(0.6, y + 0.52, author, fontsize=9, fontweight="bold", color=tc, va="center", zorder=3)
        ax.text(3.5, y + 0.52, title, fontsize=8, color=MUTED if not scanned else "#8b949e", va="center", zorder=3)

        # Scanning indicator (only for currently scanning entry)
        if n_scanned > 0 and i == n_scanned - 1 and n_scanned <= len(REFS):
            # scan sweep line
            ax.plot([0.3, 7.8], [y + 0.1, y + 0.1], color=BLUE, lw=2, alpha=0.6, zorder=4)

        # Verdict badge
        if scanned:
            badge_x = 8.2
            badge_w = 3.2 if status != "verified" else 2.2
            badge = mpatches.FancyBboxPatch(
                (badge_x, y + 0.12),
                badge_w,
                0.6,
                boxstyle="round,pad=0.06",
                facecolor=color,
                edgecolor=color,
                alpha=0.2,
                zorder=3,
            )
            ax.add_patch(badge)
            symbol = {"verified": "+", "mismatch": "!", "hallucinated": "X"}[status]
            ax.text(
                badge_x + badge_w / 2,
                y + 0.42,
                f"{symbol} {verdict}",
                fontsize=8,
                fontweight="bold",
                color=color,
                ha="center",
                va="center",
                zorder=4,
            )

    # Sources bar
    ax.text(
        6, 0.55, "Crossref  |  DBLP  |  Semantic Scholar", fontsize=9, color=MUTED, ha="center", fontfamily="monospace"
    )

    # Summary (only in final state)
    if n_scanned > len(REFS):
        ax.text(
            6,
            0.1,
            "3 verified  \u2022  1 year mismatch  \u2022  1 hallucinated",
            fontsize=10,
            fontweight="bold",
            color=TEXT,
            ha="center",
        )


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # ~35 frames at 5 FPS = 7 seconds
    # 0=blank, 1-5=each entry scanned, 6=summary
    states = [0] * 3
    for i in range(1, 6):
        states += [i] * 3  # 0.6s per entry scan
    states += [6] * 12  # hold summary 2.4s

    def animate(i):
        draw_frame(fig, ax, states[i])

    anim = FuncAnimation(fig, animate, frames=len(states), interval=200)
    out = "/Users/patrik.reizinger/Documents/GitHub/bibtexupdater/assets/fact-checker.gif"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"Generating {len(states)} frames...")
    anim.save(out, writer=PillowWriter(fps=5), dpi=DPI)
    size_kb = os.path.getsize(out) / 1024
    print(f"Saved to {out} ({size_kb:.0f} KB)")
    plt.close()


if __name__ == "__main__":
    main()
