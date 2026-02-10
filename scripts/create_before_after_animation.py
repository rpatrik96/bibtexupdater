"""Create a lean animated GIF showing preprint → published transformation."""

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
ORANGE = "#f47b20"
GREEN = "#22c55e"
AMBER = "#f59e0b"
BLUE = "#58a6ff"
CARD_BG = "#161b22"

BEFORE_LINES = [
    ("@article", "{vaswani2017attention,", False),
    ("  title", "   = {Attention Is All You Need},", False),
    ("  author", "  = {Vaswani, Ashish and ...},", False),
    ("  journal", " = {arXiv:1706.03762},", True),  # changed
    ("  year", "    = {2017}", False),
    ("}", "", False),
]

AFTER_LINES = [
    ("@inproceedings", "{vaswani2017attention,", True),  # changed
    ("  title", "     = {Attention Is All You Need},", False),
    ("  author", "    = {Vaswani, Ashish and ...},", False),
    ("  booktitle", " = {NeurIPS},", True),  # changed
    ("  year", "      = {2017},", False),
    ("  doi", "       = {10.5555/3295222.3295349}", True),  # new
    ("}", "", False),
]


def draw_card(ax, lines, x, y, w, h, border_color, label, label_color, show_highlights=False):
    rect = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.12",
        facecolor=CARD_BG,
        edgecolor=border_color,
        linewidth=2.5,
        zorder=2,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h + 0.25,
        label,
        fontsize=12,
        fontweight="bold",
        color=label_color,
        ha="center",
        va="bottom",
        zorder=3,
    )

    line_h = h / (len(lines) + 1.5)
    for i, (key, val, changed) in enumerate(lines):
        yy = y + h - (i + 1.2) * line_h
        color = TEXT
        if show_highlights and changed:
            # highlight bar
            bar = mpatches.Rectangle(
                (x + 0.1, yy - line_h * 0.3),
                w - 0.2,
                line_h * 0.7,
                facecolor=AMBER if "doi" not in key else GREEN,
                alpha=0.2,
                zorder=2,
            )
            ax.add_patch(bar)
            color = AMBER if "doi" not in key else GREEN
        ax.text(x + 0.3, yy, key + val, fontsize=8, fontfamily="monospace", color=color, va="center", zorder=3)


def draw_frame(fig, ax, state):
    """state: 0=before only, 1=arrow, 2=both+highlights, 3=both+confidence"""
    ax.clear()
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.5, 6.5)
    ax.set_facecolor(BG)
    fig.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(5.75, 6.1, "Preprint \u2192 Published", fontsize=22, fontweight="bold", color=TEXT, ha="center")
    ax.text(5.75, 5.6, "bibtex-update", fontsize=12, color=BLUE, ha="center", fontfamily="monospace")

    # Before card (always shown)
    draw_card(ax, BEFORE_LINES, 0.2, 0.8, 4.8, 4.2, ORANGE, "BEFORE", ORANGE)

    if state >= 1:
        # Arrow
        ax.annotate(
            "",
            xy=(6.2, 2.9),
            xytext=(5.3, 2.9),
            arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 3, "mutation_scale": 20},
            zorder=5,
        )
        ax.text(
            5.75,
            3.4,
            "bibtex-update",
            fontsize=9,
            fontweight="bold",
            color=BLUE,
            ha="center",
            fontfamily="monospace",
            zorder=5,
        )

    if state >= 2:
        # After card
        draw_card(ax, AFTER_LINES, 6.5, 0.5, 5.2, 4.8, GREEN, "AFTER", GREEN, show_highlights=True)

    if state >= 3:
        # Confidence badge
        badge = mpatches.FancyBboxPatch(
            (8.0, 0.1),
            2.2,
            0.45,
            boxstyle="round,pad=0.06",
            facecolor=GREEN,
            edgecolor=GREEN,
            alpha=0.25,
            zorder=5,
        )
        ax.add_patch(badge)
        ax.text(
            9.1, 0.32, "Confidence: 97%", fontsize=8, fontweight="bold", color=GREEN, ha="center", va="center", zorder=6
        )


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # ~30 frames at 5 FPS = 6 seconds
    states = [0] * 5 + [1] * 5 + [2] * 5 + [3] * 15  # hold final state longer

    def animate(i):
        draw_frame(fig, ax, states[i])

    anim = FuncAnimation(fig, animate, frames=len(states), interval=200)
    out = "/Users/patrik.reizinger/Documents/GitHub/bibtexupdater/assets/before-after.gif"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"Generating {len(states)} frames...")
    anim.save(out, writer=PillowWriter(fps=5), dpi=DPI)
    size_kb = os.path.getsize(out) / 1024
    print(f"Saved to {out} ({size_kb:.0f} KB)")
    plt.close()


if __name__ == "__main__":
    main()
