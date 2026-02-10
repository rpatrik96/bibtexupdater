"""Create a lean animated GIF showing Zotero integration workflow."""

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
AMBER = "#f59e0b"
BLUE = "#58a6ff"
ZOTERO_RED = "#cc2936"
PURPLE = "#a855f7"
GREEN = "#3fb950"
CARD_BG = "#161b22"


def draw_frame(fig, ax, state):
    """state: 0=bib, 1=+arrow+cmd, 2=entries transform, 3=+zotero, 4=+collections, 5=final"""
    ax.clear()
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.5, 7)
    ax.set_facecolor(BG)
    fig.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(6, 6.6, "Zotero Integration", fontsize=22, fontweight="bold", color=TEXT, ha="center")
    ax.text(6, 6.15, "bibtex-update --zotero", fontsize=12, color=BLUE, ha="center", fontfamily="monospace")

    # --- Left: .bib file ---
    bib_box = mpatches.FancyBboxPatch(
        (0.3, 2.5),
        2.5,
        3.0,
        boxstyle="round,pad=0.12",
        facecolor=CARD_BG,
        edgecolor=AMBER,
        linewidth=2.5,
        zorder=2,
    )
    ax.add_patch(bib_box)
    ax.text(1.55, 5.1, ".bib", fontsize=16, fontweight="bold", color=AMBER, ha="center", zorder=3)

    # Entries
    if state < 2:
        entries = [("arXiv:2301.001", MUTED), ("bioRxiv:10.1101/...", MUTED), ("arXiv:2305.014", MUTED)]
    else:
        entries = [("NeurIPS 2023", GREEN), ("Nature Methods", GREEN), ("ICML 2023", GREEN)]

    for j, (label, color) in enumerate(entries):
        y = 4.3 - j * 0.55
        ax.text(
            1.55, y, label, fontsize=7, color=color, ha="center", fontfamily="monospace", fontweight="bold", zorder=3
        )

    # --- Center: command box ---
    if state >= 1:
        cmd_box = mpatches.FancyBboxPatch(
            (3.8, 3.0),
            3.6,
            2.0,
            boxstyle="round,pad=0.12",
            facecolor=CARD_BG,
            edgecolor=BLUE,
            linewidth=2.5,
            zorder=2,
        )
        ax.add_patch(cmd_box)
        ax.text(
            5.6,
            4.35,
            "bibtex-update",
            fontsize=11,
            fontweight="bold",
            color=BLUE,
            ha="center",
            fontfamily="monospace",
            zorder=3,
        )
        ax.text(5.6, 3.7, "--zotero", fontsize=10, color=BLUE, ha="center", fontfamily="monospace", alpha=0.8, zorder=3)

        # Arrow bib -> cmd
        ax.annotate(
            "",
            xy=(3.7, 4.0),
            xytext=(2.9, 4.0),
            arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 2.5, "mutation_scale": 18},
            zorder=4,
        )

    # --- Right: Zotero icon ---
    if state >= 3:
        z_box = mpatches.FancyBboxPatch(
            (8.5, 2.5),
            2.8,
            3.0,
            boxstyle="round,pad=0.12",
            facecolor=CARD_BG,
            edgecolor=ZOTERO_RED,
            linewidth=2.5,
            zorder=2,
        )
        ax.add_patch(z_box)
        ax.text(9.9, 4.5, "Z", fontsize=28, fontweight="bold", color=ZOTERO_RED, ha="center", va="center", zorder=3)
        ax.text(9.9, 3.5, "Zotero", fontsize=11, color=ZOTERO_RED, ha="center", zorder=3)

        # Arrow cmd -> zotero
        ax.annotate(
            "",
            xy=(8.4, 4.0),
            xytext=(7.5, 4.0),
            arrowprops={"arrowstyle": "-|>", "color": ZOTERO_RED, "lw": 2.5, "mutation_scale": 18},
            zorder=4,
        )

    # --- Bottom: AI collections ---
    if state >= 4:
        coll_box = mpatches.FancyBboxPatch(
            (3.5, 0.2),
            5.0,
            1.8,
            boxstyle="round,pad=0.12",
            facecolor=CARD_BG,
            edgecolor=PURPLE,
            linewidth=2,
            zorder=2,
        )
        ax.add_patch(coll_box)

        folders = [
            ("ML / Optimization", 1.65),
            ("Biology / Genomics", 1.25),
            ("NLP / Translation", 0.85),
        ]
        for label, y in folders:
            ax.text(4.0, y, label, fontsize=8, color=PURPLE, fontfamily="monospace", zorder=3)

        ax.text(
            7.8,
            1.2,
            "AI-powered\norganization",
            fontsize=8,
            color=MUTED,
            ha="center",
            va="center",
            style="italic",
            zorder=3,
        )
        ax.text(
            7.8,
            0.55,
            "Claude | OpenAI | Embeddings",
            fontsize=7,
            color=MUTED,
            ha="center",
            fontfamily="monospace",
            zorder=3,
        )


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # ~35 frames at 5 FPS = 7 seconds
    states = [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3 + [4] * 15  # hold final

    def animate(i):
        draw_frame(fig, ax, states[i])

    anim = FuncAnimation(fig, animate, frames=len(states), interval=200)
    out = "/Users/patrik.reizinger/Documents/GitHub/bibtexupdater/assets/zotero-sync.gif"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"Generating {len(states)} frames...")
    anim.save(out, writer=PillowWriter(fps=5), dpi=DPI)
    size_kb = os.path.getsize(out) / 1024
    print(f"Saved to {out} ({size_kb:.0f} KB)")
    plt.close()


if __name__ == "__main__":
    main()
