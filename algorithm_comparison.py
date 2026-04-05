"""
algorithm_comparison.py — Multi-Algorithm Crack Propagation Comparison

Compare different search algorithms on the same microstructure:
  - A* Search (optimal + fast with admissible heuristic)
  - Dijkstra's Algorithm (optimal, no heuristic — A* with h=0)
  - Greedy Best-First Search (fast but suboptimal — uses only h(n))
  - BFS (finds shortest-hop path, ignores edge costs entirely)

This demonstrates WHY A* is the right choice: it finds the same optimal
path as Dijkstra but explores fewer nodes.
"""

import heapq
import math
import time
from collections import deque
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from astar_search import AStarCrackSearch, CrackResult
from environment import Microstructure

# ─── Dijkstra Search (A* with h=0) ───────────────────────────────────────────


class DijkstraSearch:
    """Dijkstra's algorithm — optimal but explores more nodes than A*."""

    def __init__(self, microstructure: Microstructure):
        self.micro = microstructure

    def search(self, start_row=None):
        micro = self.micro
        W, H = micro.width, micro.height
        if start_row is None:
            start_row = H // 2

        start = (start_row, 0)
        goal_col = W - 1

        counter = 0
        open_set = []
        heapq.heappush(open_set, (0.0, counter, start))
        counter += 1

        came_from = {}
        g_score = {start: 0.0}
        closed_set = set()
        exploration_order = []

        while open_set:
            g_current, _, current = heapq.heappop(open_set)
            if current in closed_set:
                continue
            closed_set.add(current)
            exploration_order.append(current)
            cr, cc = current

            if cc == goal_col:
                path = _reconstruct(came_from, current)
                return CrackResult(
                    path,
                    g_score[current],
                    "FRACTURE",
                    len(closed_set),
                    exploration_order,
                )

            for nr, nc in micro.get_neighbors(cr, cc):
                if (nr, nc) in closed_set or not micro.can_propagate(nr, nc):
                    continue
                cost = micro.edge_cost(cr, cc, nr, nc)
                if abs(nr - cr) + abs(nc - cc) == 2:
                    cost *= math.sqrt(2)
                tentative = g_score[current] + cost
                if tentative < g_score.get((nr, nc), float("inf")):
                    g_score[(nr, nc)] = tentative
                    came_from[(nr, nc)] = current
                    heapq.heappush(open_set, (tentative, counter, (nr, nc)))
                    counter += 1

        if exploration_order:
            farthest = max(exploration_order, key=lambda p: p[1])
            path = _reconstruct(came_from, farthest)
        else:
            path = [start]
        return CrackResult(
            path,
            g_score.get(path[-1], 0.0),
            "ARREST",
            len(closed_set),
            exploration_order,
        )


# ─── Greedy Best-First Search (only h, no g) ─────────────────────────────────


class GreedyBestFirstSearch:
    """Greedy best-first — fast but suboptimal. Uses only heuristic."""

    def __init__(self, microstructure: Microstructure):
        self.micro = microstructure

    def _heuristic(self, r, c):
        """Greedy heuristic: prefer nodes close to right boundary with low toughness."""
        dist = (self.micro.width - 1) - c
        local_toughness = self.micro.toughness_grid[r, c]
        return dist * local_toughness

    def search(self, start_row=None):
        micro = self.micro
        W, H = micro.width, micro.height
        if start_row is None:
            start_row = H // 2

        start = (start_row, 0)
        goal_col = W - 1

        counter = 0
        open_set = []
        heapq.heappush(open_set, (self._heuristic(*start), counter, start))
        counter += 1

        came_from = {}
        g_score = {start: 0.0}
        closed_set = set()
        exploration_order = []

        while open_set:
            _, _, current = heapq.heappop(open_set)
            if current in closed_set:
                continue
            closed_set.add(current)
            exploration_order.append(current)
            cr, cc = current

            if cc == goal_col:
                path = _reconstruct(came_from, current)
                return CrackResult(
                    path,
                    g_score[current],
                    "FRACTURE",
                    len(closed_set),
                    exploration_order,
                )

            for nr, nc in micro.get_neighbors(cr, cc):
                if (nr, nc) in closed_set or not micro.can_propagate(nr, nc):
                    continue
                cost = micro.edge_cost(cr, cc, nr, nc)
                if abs(nr - cr) + abs(nc - cc) == 2:
                    cost *= math.sqrt(2)
                tentative = g_score[current] + cost
                if (nr, nc) not in g_score:
                    g_score[(nr, nc)] = tentative
                    came_from[(nr, nc)] = current
                    heapq.heappush(
                        open_set, (self._heuristic(nr, nc), counter, (nr, nc))
                    )
                    counter += 1

        if exploration_order:
            farthest = max(exploration_order, key=lambda p: p[1])
            path = _reconstruct(came_from, farthest)
        else:
            path = [start]
        return CrackResult(
            path,
            g_score.get(path[-1], 0.0),
            "ARREST",
            len(closed_set),
            exploration_order,
        )


# ─── BFS (Unweighted — shortest hop count) ───────────────────────────────────


class BFSSearch:
    """BFS — finds shortest-hop path, completely ignores edge costs."""

    def __init__(self, microstructure: Microstructure):
        self.micro = microstructure

    def search(self, start_row=None):
        micro = self.micro
        W, H = micro.width, micro.height
        if start_row is None:
            start_row = H // 2

        start = (start_row, 0)
        goal_col = W - 1

        queue = deque([start])
        came_from = {}
        visited = {start}
        exploration_order = []

        while queue:
            current = queue.popleft()
            exploration_order.append(current)
            cr, cc = current

            if cc == goal_col:
                path = _reconstruct(came_from, current)

                # Calculate actual cost along the BFS path
                total_cost = sum(
                    micro.edge_cost(
                        path[i][0], path[i][1], path[i + 1][0], path[i + 1][1]
                    )
                    * (
                        math.sqrt(2)
                        if abs(path[i + 1][0] - path[i][0])
                        + abs(path[i + 1][1] - path[i][1])
                        == 2
                        else 1.0
                    )
                    for i in range(len(path) - 1)
                )
                return CrackResult(
                    path, total_cost, "FRACTURE", len(visited), exploration_order
                )

            for nr, nc in micro.get_neighbors(cr, cc):
                if (nr, nc) not in visited and micro.can_propagate(nr, nc):
                    visited.add((nr, nc))
                    came_from[(nr, nc)] = current
                    queue.append((nr, nc))

        if exploration_order:
            farthest = max(exploration_order, key=lambda p: p[1])
            path = _reconstruct(came_from, farthest)
        else:
            path = [start]
        total_cost = (
            sum(
                micro.edge_cost(path[i][0], path[i][1], path[i + 1][0], path[i + 1][1])
                for i in range(len(path) - 1)
            )
            if len(path) > 1
            else 0.0
        )
        return CrackResult(path, total_cost, "ARREST", len(visited), exploration_order)


# ─── Helper ──────────────────────────────────────────────────────────────────


def _reconstruct(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# ─── Run All Algorithms ──────────────────────────────────────────────────────


@dataclass
class ComparisonResult:
    """Results from comparing all algorithms."""

    algorithm: str
    result: CrackResult
    time_seconds: float


def compare_algorithms(
    microstructure: Microstructure,
    start_row: int | None = None,
) -> list[ComparisonResult]:
    """
    Run all 4 algorithms on the same microstructure and return results.

    Returns list of ComparisonResult, one per algorithm.
    """
    algorithms = [
        ("A* Search", AStarCrackSearch),
        ("Dijkstra", DijkstraSearch),
        ("Greedy Best-First", GreedyBestFirstSearch),
        ("BFS (Unweighted)", BFSSearch),
    ]

    results = []
    for name, cls in algorithms:
        searcher = cls(microstructure)
        t0 = time.time()
        result = searcher.search(start_row=start_row)
        elapsed = time.time() - t0
        results.append(ComparisonResult(name, result, elapsed))
        print(
            f"  {name:20s} | Cost: {result.total_cost:8.4f} | "
            f"Nodes: {result.nodes_explored:5d} | "
            f"Path: {len(result.path):4d}px | "
            f"Time: {elapsed:.4f}s | {result.outcome}"
        )

    return results


# ─── Comparison Visualization ─────────────────────────────────────────────────


def plot_comparison(
    microstructure: Microstructure,
    results: list[ComparisonResult],
    save_path: str | None = None,
    show: bool = True,
):
    """Generate 2×2 crack path comparison + summary bar charts."""
    from environment import PHASE_COLORS

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "Algorithm Comparison — Crack Propagation", fontsize=18, fontweight="bold"
    )

    # Build RGB base image
    micro = microstructure
    rgb = np.zeros((micro.height, micro.width, 3))
    for pid, color in PHASE_COLORS.items():
        mask = micro.phase_grid == pid
        for c_idx in range(3):
            rgb[:, :, c_idx][mask] = color[c_idx]

    # Color scheme for algorithms
    algo_colors = ["#FF4444", "#4488FF", "#FF8800", "#44CC44"]

    # ── Top row: 2×2 crack path panels ──
    for i, comp in enumerate(results):
        ax = fig.add_subplot(3, 2, i + 1)
        ax.imshow(rgb, origin="upper", aspect="equal")

        res = comp.result
        if res.path:
            pr = [p[0] for p in res.path]
            pc = [p[1] for p in res.path]
            ax.plot(pc, pr, color=algo_colors[i], linewidth=2.0, alpha=0.9)
            ax.plot(
                pc[0],
                pr[0],
                "o",
                color="#00FF88",
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=1.5,
            )
            end_color = "#FF0000" if res.outcome == "FRACTURE" else "#FFaa00"
            ax.plot(
                pc[-1],
                pr[-1],
                "X",
                color=end_color,
                markersize=10,
                markeredgecolor="white",
                markeredgewidth=1.5,
            )

        # Show explored area
        if res.exploration_order:
            er = [p[0] for p in res.exploration_order]
            ec = [p[1] for p in res.exploration_order]
            ax.scatter(ec, er, c=algo_colors[i], s=0.2, alpha=0.1)

        ax.set_title(
            f"{comp.algorithm}\nCost: {res.total_cost:.4f} | "
            f"Nodes: {res.nodes_explored} | Time: {comp.time_seconds:.4f}s",
            fontsize=10,
        )
        ax.set_xlim(-0.5, micro.width - 0.5)
        ax.set_ylim(micro.height - 0.5, -0.5)

    # ── Bottom row: Bar charts ──
    names = [c.algorithm for c in results]
    costs = [c.result.total_cost for c in results]
    nodes = [c.result.nodes_explored for c in results]
    [c.time_seconds for c in results]

    # Cost comparison
    ax_cost = fig.add_subplot(3, 2, 5)
    bars = ax_cost.bar(names, costs, color=algo_colors, alpha=0.85, edgecolor="white")
    ax_cost.set_ylabel("Total Path Cost (Energy)", fontsize=10)
    ax_cost.set_title("Path Cost Comparison (lower = better)", fontsize=11)

    # Highlight optimal
    min_cost = min(costs)
    for bar, cost in zip(bars, costs):
        if abs(cost - min_cost) < 1e-6:
            bar.set_edgecolor("#FFD700")
            bar.set_linewidth(3)
    ax_cost.tick_params(axis="x", labelsize=8)

    # Nodes explored comparison
    ax_nodes = fig.add_subplot(3, 2, 6)
    bars = ax_nodes.bar(names, nodes, color=algo_colors, alpha=0.85, edgecolor="white")
    ax_nodes.set_ylabel("Nodes Explored", fontsize=10)
    ax_nodes.set_title("Efficiency Comparison (lower = better)", fontsize=11)
    min_nodes = min(nodes)
    for bar, n in zip(bars, nodes):
        if n == min_nodes:
            bar.set_edgecolor("#FFD700")
            bar.set_linewidth(3)
    ax_nodes.tick_params(axis="x", labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Comparison plot saved → {save_path}")

    if show:
        plt.show()

    plt.close(fig)

    return fig


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating microstructure...")
    micro = Microstructure(seed=42)
    micro.summary()
    print()

    print("Running algorithm comparison:")
    results = compare_algorithms(micro)
    print()

    plot_comparison(micro, results, save_path="algorithm_comparison.png", show=False)
