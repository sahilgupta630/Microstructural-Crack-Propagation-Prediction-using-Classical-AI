"""
visualizer.py — Microstructure & Crack Path Visualization

Renders:
  1. The 2D microstructure grid (colored by phase)
  2. Applied stress field overlay
  3. A* crack propagation path
  4. Exploration frontier (nodes visited by A*)
  5. Annotated summary (cost, path length, outcome)
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from astar_search import CrackResult
from environment import (PHASE_COLORS, PHASE_NAMES, Microstructure)

# ─── Static Plot ──────────────────────────────────────────────────────────────


def plot_result(
    micro: Microstructure,
    result: CrackResult,
    title: str = "A*tomic Fracture — Crack Propagation Simulation",
    save_path: str | None = None,
    show: bool = True,
    show_exploration: bool = True,
    show_stress: bool = True,
):
    """
    Render the microstructure with the crack path overlaid.

    Parameters
    ----------
    micro : Microstructure
        The generated environment.
    result : CrackResult
        Output from A* search.
    title : str
        Plot title.
    save_path : str or None
        If provided, save the figure to this file.
    show : bool
        Whether to display the plot interactively.
    show_exploration : bool
        Whether to show explored nodes as a semi-transparent overlay.
    show_stress : bool
        Whether to overlay the stress field contours.
    """
    fig, axes = plt.subplots(
        1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    fig.suptitle(title, fontsize=16, fontweight="bold", color="#222")

    # ── Left panel: Microstructure + Crack Path ─────────────────────────
    ax1 = axes[0]

    # Build RGB image from phase grid
    rgb = np.zeros((micro.height, micro.width, 3))
    for phase_id, color in PHASE_COLORS.items():
        mask = micro.phase_grid == phase_id
        for c_idx in range(3):
            rgb[:, :, c_idx][mask] = color[c_idx]

    ax1.imshow(rgb, origin="upper", aspect="equal")

    # Stress field contours
    if show_stress:
        stress_overlay = ax1.contour(
            micro.stress_grid,
            levels=8,
            colors="white",
            linewidths=0.4,
            alpha=0.5,
        )
        ax1.clabel(stress_overlay, inline=True, fontsize=6, fmt="%.1f")

    # Explored nodes (semi-transparent yellow dots)
    if show_exploration and result.exploration_order:
        exp_rows = [p[0] for p in result.exploration_order]
        exp_cols = [p[1] for p in result.exploration_order]
        ax1.scatter(
            exp_cols,
            exp_rows,
            c="yellow",
            s=0.3,
            alpha=0.15,
            zorder=2,
            label=f"Explored ({result.nodes_explored} nodes)",
        )

    # Crack path (bold white → red gradient)
    if result.path:
        path_rows = [p[0] for p in result.path]
        path_cols = [p[1] for p in result.path]

        # Draw path shadow
        ax1.plot(
            path_cols, path_rows, color="black", linewidth=3.5, alpha=0.6, zorder=3
        )

        # Draw path itself with color gradient
        ax1.plot(
            path_cols, path_rows, color="#FF4444", linewidth=2.0, alpha=0.95, zorder=4
        )

        # Start and end markers
        ax1.plot(
            path_cols[0],
            path_rows[0],
            "o",
            color="#00FF88",
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=2,
            zorder=5,
            label="Crack initiation",
        )
        marker_color = "#FF0000" if result.outcome == "FRACTURE" else "#FFaa00"
        marker_label = (
            "Complete fracture" if result.outcome == "FRACTURE" else "Crack arrested"
        )
        ax1.plot(
            path_cols[-1],
            path_rows[-1],
            "X",
            color=marker_color,
            markersize=12,
            markeredgecolor="white",
            markeredgewidth=2,
            zorder=5,
            label=marker_label,
        )

    # Legend
    phase_patches = [
        mpatches.Patch(color=PHASE_COLORS[pid], label=name)
        for pid, name in PHASE_NAMES.items()
    ]
    ax1.legend(
        handles=phase_patches + ax1.get_legend_handles_labels()[0][-3:],
        loc="upper left",
        fontsize=7,
        framealpha=0.85,
    )

    ax1.set_xlim(-0.5, micro.width - 0.5)
    ax1.set_ylim(micro.height - 0.5, -0.5)
    ax1.set_xlabel("x (pixels)", fontsize=10)
    ax1.set_ylabel("y (pixels)", fontsize=10)
    ax1.set_title("Microstructure + Crack Path", fontsize=12)

    # ── Right panel: Info & Toughness map ───────────────────────────────
    ax2 = axes[1]

    # Toughness heatmap
    im = ax2.imshow(
        micro.toughness_grid,
        origin="upper",
        aspect="equal",
        cmap="inferno",
        vmin=0,
        vmax=4.0,
    )
    plt.colorbar(im, ax=ax2, label="Fracture Toughness K_IC", shrink=0.8)

    # Overlay path on toughness map too
    if result.path:
        path_rows = [p[0] for p in result.path]
        path_cols = [p[1] for p in result.path]
        ax2.plot(path_cols, path_rows, color="cyan", linewidth=1.5, alpha=0.9, zorder=4)

    ax2.set_title("Fracture Toughness Map", fontsize=12)
    ax2.set_xlabel("x (pixels)", fontsize=10)
    ax2.set_ylabel("y (pixels)", fontsize=10)

    # ── Summary text box ────────────────────────────────────────────────
    outcome_label = "[FRACTURE]" if result.outcome == "FRACTURE" else "[ARREST]"
    summary_text = (
        f"{outcome_label} Outcome: {result.outcome}\n"
        f"Path length: {len(result.path)} pixels\n"
        f"Total energy: {result.total_cost:.4f}\n"
        f"Nodes explored: {result.nodes_explored}\n"
        f"Grid: {micro.width}×{micro.height}"
    )
    fig.text(
        0.5,
        0.02,
        summary_text,
        ha="center",
        va="bottom",
        fontsize=10,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#ccc"),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved → {save_path}")

    if show:
        plt.show()

    plt.close(fig)


# ─── Animated Exploration ────────────────────────────────────────────────────


def animate_exploration(
    micro: Microstructure,
    result: CrackResult,
    save_path: str = "crack_animation.gif",
    interval: int = 10,
    step_size: int = 50,
):
    """
    Create an animated GIF showing A* exploring the microstructure
    step by step.

    Parameters
    ----------
    micro : Microstructure
    result : CrackResult
    save_path : str
        Output path for the GIF.
    interval : int
        Milliseconds between frames.
    step_size : int
        Number of explored nodes per frame (for speed).
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Base microstructure
    rgb = np.zeros((micro.height, micro.width, 3))
    for phase_id, color in PHASE_COLORS.items():
        mask = micro.phase_grid == phase_id
        for c_idx in range(3):
            rgb[:, :, c_idx][mask] = color[c_idx]

    ax.imshow(rgb, origin="upper", aspect="equal")
    ax.set_title(
        "A*tomic Fracture — Exploration Animation", fontsize=14, fontweight="bold"
    )

    # Scatter for explored nodes (updated each frame)
    scat = ax.scatter([], [], c="yellow", s=1.0, alpha=0.3, zorder=2)

    # Line for final path (drawn at end)
    (path_line,) = ax.plot([], [], color="#FF4444", linewidth=2.0, zorder=4)

    n_explore = len(result.exploration_order)
    n_frames = n_explore // step_size + 2  # +1 for final path frame

    def update(frame):
        idx = min(frame * step_size, n_explore)
        explored = result.exploration_order[:idx]
        if explored:
            cols = [p[1] for p in explored]
            rows = [p[0] for p in explored]
            scat.set_offsets(np.column_stack([cols, rows]))

        # On last frame, show the path
        if frame == n_frames - 1 and result.path:
            path_cols = [p[1] for p in result.path]
            path_rows = [p[0] for p in result.path]
            path_line.set_data(path_cols, path_rows)

        ax.set_xlabel(f"Nodes explored: {idx}/{n_explore}", fontsize=10)
        return scat, path_line

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval,
        blit=True,
    )
    anim.save(save_path, writer="pillow", fps=30)
    print(f"Animation saved → {save_path}")
    plt.close(fig)


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from astar_search import AStarCrackSearch

    micro = Microstructure(seed=42)
    searcher = AStarCrackSearch(micro)
    result = searcher.search()
    plot_result(micro, result, save_path="crack_result.png", show=False)
