"""Create an animated GIF showing the 9-stage resolution pipeline."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# --- Config ---
FIG_W, FIG_H = 12, 6.75  # 16:9
DPI = 200  # Higher DPI for sharper output
BG_COLOR = "#0d1117"  # GitHub dark
TEXT_COLOR = "#e6edf3"
MUTED_COLOR = "#484f58"
ACCENT_COLOR = "#58a6ff"
ARROW_COLOR = "#30363d"
ARROW_LIT_COLOR = "#58a6ff"

STAGES = [
    ("1", "arXiv API", "#f47b20"),
    ("1b", "OpenAlex", "#a855f7"),  # NEW
    ("1c", "Europe PMC", "#22c55e"),  # NEW
    ("2", "Crossref\nRelations", "#f59e0b"),
    ("3", "DBLP", "#3b82f6"),
    ("3b", "ACL\nAnthology", "#ec4899"),  # NEW
    ("4", "Semantic\nScholar", "#06b6d4"),
    ("5", "Crossref\nSearch", "#f59e0b"),
    ("6", "Google\nScholar", "#ef4444"),
]

NEW_STAGES = {"1b", "1c", "3b"}


def draw_frame(fig, ax, n_lit):
    """Draw the pipeline with n_lit stages illuminated."""
    ax.clear()
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-1.0, 6.5)
    ax.set_facecolor(BG_COLOR)
    fig.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title with subtle shadow effect
    ax.text(
        5.5,
        5.88,
        "bibtex-updater",
        fontsize=32,
        fontweight="bold",
        color="#000000",
        ha="center",
        va="center",
        fontfamily="monospace",
        alpha=0.3,
    )
    ax.text(
        5.5,
        5.9,
        "bibtex-updater",
        fontsize=32,
        fontweight="bold",
        color=TEXT_COLOR,
        ha="center",
        va="center",
        fontfamily="monospace",
    )
    ax.text(
        5.5,
        5.25,
        "9-stage preprint resolution pipeline",
        fontsize=15,
        color=MUTED_COLOR,
        ha="center",
        va="center",
        fontweight="normal",
    )

    # Layout: two rows — top row 5 stages, bottom row 4 stages (right-to-left)
    box_w, box_h = 1.8, 1.1
    top_y = 3.4
    bot_y = 1.2
    gap = 0.35

    # Top row positions (left to right): stages 0-4
    top_xs = [0.5 + i * (box_w + gap) for i in range(5)]
    # Bottom row positions (right to left): stages 5-8
    bot_xs = [0.5 + (3 - i) * (box_w + gap) for i in range(4)]
    # Offset bottom row to align right edge
    offset = top_xs[4] + box_w - (bot_xs[0] + box_w)
    bot_xs = [x + offset for x in bot_xs]

    positions = [(x, top_y) for x in top_xs] + [(x, bot_y) for x in bot_xs]

    for i, ((x, y), (stage_num, name, color)) in enumerate(zip(positions, STAGES)):
        lit = i < n_lit
        is_new = stage_num in NEW_STAGES

        # Box
        fc = color if lit else "#161b22"
        alpha = 0.92 if lit else 0.5
        ec = color if lit else "#30363d"
        lw = 2.5 if lit else 1.2

        rect = mpatches.FancyBboxPatch(
            (x, y),
            box_w,
            box_h,
            boxstyle="round,pad=0.12",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            alpha=alpha,
            zorder=3,
        )
        ax.add_patch(rect)

        # Glow effect when lit (double glow for more prominence)
        if lit:
            # Outer glow
            glow_outer = mpatches.FancyBboxPatch(
                (x - 0.12, y - 0.12),
                box_w + 0.24,
                box_h + 0.24,
                boxstyle="round,pad=0.18",
                facecolor="none",
                edgecolor=color,
                linewidth=1.5,
                alpha=0.2,
                zorder=2,
            )
            ax.add_patch(glow_outer)
            # Inner glow
            glow_inner = mpatches.FancyBboxPatch(
                (x - 0.06, y - 0.06),
                box_w + 0.12,
                box_h + 0.12,
                boxstyle="round,pad=0.15",
                facecolor="none",
                edgecolor=color,
                linewidth=2.0,
                alpha=0.4,
                zorder=2,
            )
            ax.add_patch(glow_inner)

        # Stage number
        num_color = "#ffffff" if lit else MUTED_COLOR
        ax.text(
            x + 0.18,
            y + box_h - 0.18,
            stage_num,
            fontsize=8,
            color=num_color,
            fontweight="bold",
            ha="left",
            va="top",
            alpha=0.7 if lit else 0.4,
            zorder=4,
        )

        # Stage name
        name_color = "#ffffff" if lit else MUTED_COLOR
        ax.text(
            x + box_w / 2,
            y + box_h / 2 - 0.02,
            name,
            fontsize=11 if "\n" not in name else 10,
            color=name_color,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=4,
        )

        # "NEW" badge for new stages
        if is_new and lit:
            badge_x = x + box_w - 0.15
            badge_y = y + box_h - 0.12
            badge = mpatches.FancyBboxPatch(
                (badge_x - 0.28, badge_y - 0.13),
                0.56,
                0.26,
                boxstyle="round,pad=0.05",
                facecolor="#22c55e",
                edgecolor="none",
                zorder=5,
                alpha=0.95,
            )
            ax.add_patch(badge)
            ax.text(
                badge_x,
                badge_y,
                "NEW",
                fontsize=7,
                color="#ffffff",
                fontweight="bold",
                ha="center",
                va="center",
                zorder=6,
            )

        # Arrow to next stage
        if i < len(STAGES) - 1:
            arrow_color = ARROW_LIT_COLOR if i < n_lit - 1 else ARROW_COLOR
            arrow_alpha = 0.8 if i < n_lit - 1 else 0.3

            if i < 4:
                # Top row: right arrow
                ax.annotate(
                    "",
                    xy=(positions[i + 1][0] - 0.05, y + box_h / 2),
                    xytext=(x + box_w + 0.05, y + box_h / 2),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": arrow_color,
                        "lw": 2.0,
                        "mutation_scale": 15,
                    },
                    alpha=arrow_alpha,
                    zorder=1,
                )
            elif i == 4:
                # Connector: top row last → bottom row first (down-right curve)
                ax.annotate(
                    "",
                    xy=(positions[5][0] + box_w + 0.05, bot_y + box_h / 2),
                    xytext=(x + box_w / 2, y - 0.05),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": arrow_color,
                        "lw": 2.0,
                        "mutation_scale": 15,
                        "connectionstyle": "arc3,rad=-0.3",
                    },
                    alpha=arrow_alpha,
                    zorder=1,
                )
            else:
                # Bottom row: LEFT arrow (right to left)
                nx, ny = positions[i + 1]
                ax.annotate(
                    "",
                    xy=(nx + box_w + 0.05, ny + box_h / 2),
                    xytext=(x - 0.05, y + box_h / 2),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": arrow_color,
                        "lw": 2.0,
                        "mutation_scale": 15,
                    },
                    alpha=arrow_alpha,
                    zorder=1,
                )

    # Bottom tagline
    if n_lit >= len(STAGES):
        ax.text(
            5.5,
            0.15,
            "pip install bibtex-updater    |    github.com/rpatrik96/bibtexupdater",
            fontsize=11,
            color=ACCENT_COLOR,
            ha="center",
            va="center",
            fontfamily="monospace",
            alpha=0.9,
        )

    # Preprint → Published animation at top right
    if n_lit >= len(STAGES):
        ax.text(
            10.8,
            5.25,
            "arXiv preprint → Published",
            fontsize=10,
            color="#22c55e",
            ha="right",
            va="center",
            fontweight="bold",
            alpha=0.85,
        )


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    n_stages = len(STAGES)
    # Frames: 2 blank, then each stage lights up (hold 3 frames each), then 8 frames of all lit
    frames = []
    # Initial blank
    frames.extend([0] * 3)
    # Light up one by one
    for i in range(1, n_stages + 1):
        frames.extend([i] * 3)
    # Hold final state
    frames.extend([n_stages] * 10)

    def animate(frame_idx):
        n_lit = frames[frame_idx]
        draw_frame(fig, ax, n_lit)

    anim = FuncAnimation(fig, animate, frames=len(frames), interval=180)

    out = "/Users/patrik.reizinger/Documents/GitHub/bibtexupdater/assets/pipeline.gif"
    import os

    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"Generating animation with {len(frames)} frames...")
    anim.save(out, writer=PillowWriter(fps=5), dpi=DPI)
    print(f"✓ Saved to {out}")
    print(f"  Resolution: {int(FIG_W * DPI)}x{int(FIG_H * DPI)} px")
    file_size = os.path.getsize(out) / 1024
    print(f"  File size: {file_size:.1f} KB")
    plt.close()


if __name__ == "__main__":
    main()
