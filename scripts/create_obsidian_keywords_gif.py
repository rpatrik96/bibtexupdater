"""Create a lean animated GIF showing Obsidian AI auto-keywording."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# --- Config ---
FIG_W, FIG_H = 12, 6.75
DPI = 150
BG = "#0d1117"
TEXT = "#c9d1d9"
MUTED = "#484f58"
PURPLE = "#a855f7"
BLUE = "#58a6ff"
CARD_BG = "#161b22"

KEYWORDS = ["[[Transformer]]", "[[Self-Attention]]", "[[Seq-to-Seq]]", "[[NMT]]", "[[Enc-Dec]]"]
NODE_LABELS = ["Transformer", "Self-Attn", "Seq2Seq", "NMT", "Enc-Dec"]
NODE_COLORS = ["#58a6ff", "#a855f7", "#f97583", "#79c0ff", "#56d364"]
EDGES = [(0, 1), (0, 2), (0, 4), (1, 2), (2, 3), (3, 4)]


def draw_frame(fig, ax, n_keywords):
    """n_keywords: 0=note only, 1-5=keywords added, 6=final with backends."""
    ax.clear()
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.5, 7)
    ax.set_facecolor(BG)
    fig.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(6, 6.6, "AI Auto-Keywording", fontsize=22, fontweight="bold", color=TEXT, ha="center")
    ax.text(6, 6.15, "bibtex-obsidian-keywords", fontsize=12, color=PURPLE, ha="center", fontfamily="monospace")

    # --- Left: Obsidian note card ---
    note = mpatches.FancyBboxPatch(
        (0.3, 0.3),
        5.2,
        5.3,
        boxstyle="round,pad=0.12",
        facecolor=CARD_BG,
        edgecolor=PURPLE,
        linewidth=2.5,
        zorder=2,
    )
    ax.add_patch(note)

    # Paper title
    ax.text(2.9, 5.2, "Attention Is All You Need", fontsize=10, fontweight="bold", color=TEXT, ha="center", zorder=3)

    # Abstract preview (muted)
    abstract = [
        "The dominant sequence transduction",
        "models are based on complex recurrent",
        "or convolutional neural networks...",
    ]
    for j, line in enumerate(abstract):
        ax.text(0.6, 4.65 - j * 0.35, line, fontsize=7, color=MUTED, zorder=3)

    # Frontmatter section
    ax.text(0.6, 3.5, "---", fontsize=8, color=MUTED, fontfamily="monospace", zorder=3)
    ax.text(0.6, 3.15, "keywords:", fontsize=8, fontweight="bold", color=PURPLE, fontfamily="monospace", zorder=3)

    # Keywords appearing one by one
    kw_count = min(n_keywords, len(KEYWORDS))
    for k in range(kw_count):
        ax.text(
            0.8,
            2.75 - k * 0.38,
            KEYWORDS[k],
            fontsize=8,
            color=BLUE,
            fontfamily="monospace",
            fontweight="bold",
            zorder=3,
        )

    # AI backend indicator
    if n_keywords >= 1:
        ax.text(
            2.9,
            0.6,
            "Claude | OpenAI | Embeddings",
            fontsize=7,
            color=MUTED,
            ha="center",
            fontfamily="monospace",
            zorder=3,
        )

    # --- Right: Knowledge graph ---
    graph = mpatches.FancyBboxPatch(
        (6.2, 0.3),
        5.5,
        5.3,
        boxstyle="round,pad=0.12",
        facecolor=CARD_BG,
        edgecolor=BLUE,
        linewidth=2,
        alpha=0.5,
        zorder=2,
    )
    ax.add_patch(graph)
    ax.text(8.95, 5.2, "Knowledge Graph", fontsize=10, fontweight="bold", color=TEXT, ha="center", zorder=3)

    # Node positions (circular layout)
    angles = np.linspace(0, 2 * np.pi, len(NODE_LABELS), endpoint=False) - np.pi / 2
    cx, cy, r = 8.95, 2.8, 1.6
    positions = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]

    # Draw edges first (only between existing nodes)
    for e0, e1 in EDGES:
        if e0 < kw_count and e1 < kw_count:
            x0, y0 = positions[e0]
            x1, y1 = positions[e1]
            ax.plot([x0, x1], [y0, y1], color=MUTED, lw=1, alpha=0.4, zorder=3)

    # Draw nodes
    for k in range(kw_count):
        x, y = positions[k]
        circle = plt.Circle((x, y), 0.38, color=NODE_COLORS[k], alpha=0.85, zorder=4)
        ax.add_patch(circle)
        ax.text(
            x, y, NODE_LABELS[k], fontsize=6, fontweight="bold", color="#ffffff", ha="center", va="center", zorder=5
        )


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # ~35 frames at 5 FPS = 7 seconds
    states = [0] * 4  # show note
    for i in range(1, 6):
        states += [i] * 3  # each keyword: 0.6s
    states += [5] * 12  # hold final 2.4s

    def animate(i):
        draw_frame(fig, ax, states[i])

    anim = FuncAnimation(fig, animate, frames=len(states), interval=200)
    out = "/Users/patrik.reizinger/Documents/GitHub/bibtexupdater/assets/obsidian-keywords.gif"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"Generating {len(states)} frames...")
    anim.save(out, writer=PillowWriter(fps=5), dpi=DPI)
    size_kb = os.path.getsize(out) / 1024
    print(f"Saved to {out} ({size_kb:.0f} KB)")
    plt.close()


if __name__ == "__main__":
    main()
