"""
astar_search.py — Corrected A* Crack Propagation Search

Fixes applied (from analysis):
  1. Admissible heuristic: euclidean_distance × min_toughness_per_unit_length
  2. Consistent units: g(n) and h(n) both in energy-equivalent units
  3. Crack arrest pruning: skip nodes where K_applied < K_threshold
  4. Separate goal/arrest: goal = right boundary; no-path = arrest
  5. Edge cost = max(0, K_IC - K_applied) for physics-correct resistance
"""

import heapq
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from environment import Microstructure


@dataclass
class CrackResult:
    """Result of A* crack propagation search."""

    path: list[tuple[int, int]]
    total_cost: float
    outcome: str
    nodes_explored: int
    exploration_order: list[tuple[int, int]]

    def __str__(self):
        return (
            f"Crack Result: {self.outcome}\n"
            f"  Path length : {len(self.path)} pixels\n"
            f"  Total cost  : {self.total_cost:.4f}\n"
            f"  Nodes explored: {self.nodes_explored}"
        )


class AStarCrackSearch:
    """
    A* search for the minimum-energy crack propagation path through a
    2D polycrystalline microstructure.

    Goal: reach any node on the right boundary (col = width - 1).
    Arrest: no reachable path → crack stops.
    """

    def __init__(self, microstructure: Microstructure):
        self.micro = microstructure
        self.min_toughness = float(np.min(self.micro.toughness_grid))
        max_stress = float(np.max(self.micro.stress_grid))
        self.min_cost_per_unit = max(0.0, self.min_toughness - max_stress)

    def heuristic(self, r: int, c: int) -> float:
        """
        Admissible heuristic: Euclidean distance to the right boundary
        multiplied by the minimum possible cost-per-unit-distance.

        This NEVER overestimates because:
          - Euclidean distance is ≤ actual path length
          - min_cost_per_unit ≤ actual cost per unit on any edge
          - Product of two lower bounds is still a lower bound

        If min_cost_per_unit == 0 (some region is free to crack), h(n)=0
        everywhere → A* degrades to Dijkstra's, which is still optimal.
        """
        dist_to_boundary = (self.micro.width - 1) - c
        return dist_to_boundary * self.min_cost_per_unit

    def search(
        self,
        start_row: Optional[int] = None,
    ) -> CrackResult:
        """
        Run A* from a start node on the left boundary to the right boundary.

        Parameters
        ----------
        start_row : int or None
            Row index for the crack initiation point (left boundary, col=0).
            If None, uses the middle row.

        Returns
        -------
        CrackResult with path, cost, and outcome.
        """
        micro = self.micro
        W, H = micro.width, micro.height

        if start_row is None:
            start_row = H // 2

        start = (start_row, 0)
        goal_col = W - 1

        counter = 0
        open_set = []
        heapq.heappush(open_set, (self.heuristic(*start), counter, start))
        counter += 1

        came_from: dict[tuple, tuple] = {}
        g_score: dict[tuple, float] = {start: 0.0}
        closed_set: set[tuple] = set()
        exploration_order: list[tuple] = []

        while open_set:
            f_current, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            closed_set.add(current)
            exploration_order.append(current)
            cr, cc = current

            if cc == goal_col:
                path = self._reconstruct_path(came_from, current)
                return CrackResult(
                    path=path,
                    total_cost=g_score[current],
                    outcome="FRACTURE",
                    nodes_explored=len(closed_set),
                    exploration_order=exploration_order,
                )

            for nr, nc in micro.get_neighbors(cr, cc):
                if (nr, nc) in closed_set:
                    continue

                if not micro.can_propagate(nr, nc):
                    continue

                move_cost = micro.edge_cost(cr, cc, nr, nc)
                if abs(nr - cr) + abs(nc - cc) == 2:
                    move_cost *= math.sqrt(2)

                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get((nr, nc), float("inf")):
                    g_score[(nr, nc)] = tentative_g
                    f_score = tentative_g + self.heuristic(nr, nc)
                    came_from[(nr, nc)] = current
                    heapq.heappush(open_set, (f_score, counter, (nr, nc)))
                    counter += 1

        if exploration_order:
            farthest = max(exploration_order, key=lambda p: p[1])
            path = self._reconstruct_path(came_from, farthest)
        else:
            path = [start]

        return CrackResult(
            path=path,
            total_cost=g_score.get(path[-1], 0.0) if path else 0.0,
            outcome="ARREST",
            nodes_explored=len(closed_set),
            exploration_order=exploration_order,
        )

    def _reconstruct_path(
        self,
        came_from: dict[tuple, tuple],
        current: tuple,
    ) -> list[tuple]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


if __name__ == "__main__":
    micro = Microstructure(seed=42)
    micro.summary()
    print()

    searcher = AStarCrackSearch(micro)
    result = searcher.search()
    print(result)
    print(f"Path: ({result.path[0]}) → ({result.path[-1]})")
