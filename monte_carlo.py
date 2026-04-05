"""
monte_carlo.py — Monte Carlo Statistical Failure Analysis

Run N simulations with randomly generated microstructures to produce
statistical distributions of:
  - Failure probability (fracture vs arrest ratio)
  - Crack path energy distribution
  - Crack path length distribution
  - Algorithm performance metrics

This answers the question engineers actually care about:
"What is the PROBABILITY of this material failing?"
"""

import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from astar_search import AStarCrackSearch, CrackResult
from environment import Microstructure


@dataclass
class MonteCarloResult:
    """Aggregated results from Monte Carlo simulation."""

    n_runs: int
    n_fractures: int
    n_arrests: int
    energies: list[float]
    path_lengths: list[int]
    nodes_explored: list[int]
    times: list[float]
    outcomes: list[str]
    individual_results: list[CrackResult] = field(repr=False)

    @property
    def failure_probability(self) -> float:
        """P(fracture) = n_fractures / n_runs."""
        return self.n_fractures / self.n_runs if self.n_runs > 0 else 0.0

    @property
    def arrest_probability(self) -> float:
        return 1.0 - self.failure_probability

    @property
    def mean_energy(self) -> float:
        return float(np.mean(self.energies)) if self.energies else 0.0

    @property
    def std_energy(self) -> float:
        return float(np.std(self.energies)) if self.energies else 0.0

    @property
    def mean_path_length(self) -> float:
        return float(np.mean(self.path_lengths)) if self.path_lengths else 0.0

    def summary(self) -> str:
        fracture_energies = [
            e for e, o in zip(self.energies, self.outcomes) if o == "FRACTURE"
        ]
        return (
            f"Monte Carlo Analysis ({self.n_runs} runs)\n"
            f"{'─' * 45}\n"
            f"  Failure probability : {self.failure_probability:.1%}\n"
            f"  Fractures           : {self.n_fractures}\n"
            f"  Arrests             : {self.n_arrests}\n"
            f"  Mean energy (all)   : {self.mean_energy:.4f} ± {self.std_energy:.4f}\n"
            f"  Mean energy (frac.) : {np.mean(fracture_energies):.4f}"
            if fracture_energies
            else "" + "\n"
            f"  Mean path length    : {self.mean_path_length:.1f} px\n"
            f"  Mean nodes explored : {np.mean(self.nodes_explored):.0f}\n"
            f"  Total time          : {sum(self.times):.2f}s\n"
        )


class MonteCarloAnalysis:
    """
    Run multiple crack propagation simulations with randomly varying
    microstructures to build statistical failure distributions.
    """

    def __init__(
        self,
        n_runs: int = 100,
        grid_size: int = 100,
        n_grains: int = 80,
        phase_probs: tuple = (0.55, 0.30, 0.15),
        stress_max: float = 2.5,
        stress_min: float = 0.3,
        k_threshold: float = 0.1,
        base_seed: int = 0,
    ):
        self.n_runs = n_runs
        self.grid_size = grid_size
        self.n_grains = n_grains
        self.phase_probs = phase_probs
        self.stress_max = stress_max
        self.stress_min = stress_min
        self.k_threshold = k_threshold
        self.base_seed = base_seed

    def run(self, verbose: bool = True) -> MonteCarloResult:
        """Execute all simulation runs."""
        energies = []
        path_lengths = []
        nodes_explored = []
        times = []
        outcomes = []
        individual_results = []
        n_fractures = 0
        n_arrests = 0

        for i in range(self.n_runs):
            seed = self.base_seed + i

            t0 = time.time()

            micro = Microstructure(
                width=self.grid_size,
                height=self.grid_size,
                n_grains=self.n_grains,
                phase_probs=self.phase_probs,
                stress_max=self.stress_max,
                stress_min=self.stress_min,
                k_threshold=self.k_threshold,
                seed=seed,
            )

            searcher = AStarCrackSearch(micro)
            result = searcher.search()

            elapsed = time.time() - t0

            energies.append(result.total_cost)
            path_lengths.append(len(result.path))
            nodes_explored.append(result.nodes_explored)
            times.append(elapsed)
            outcomes.append(result.outcome)
            individual_results.append(result)

            if result.outcome == "FRACTURE":
                n_fractures += 1
            else:
                n_arrests += 1

            if verbose and (i + 1) % max(1, self.n_runs // 10) == 0:
                pct = (i + 1) / self.n_runs * 100
                print(
                    f"  [{pct:5.1f}%] Run {i+1}/{self.n_runs} — "
                    f"Fractures: {n_fractures}, Arrests: {n_arrests}"
                )

        return MonteCarloResult(
            n_runs=self.n_runs,
            n_fractures=n_fractures,
            n_arrests=n_arrests,
            energies=energies,
            path_lengths=path_lengths,
            nodes_explored=nodes_explored,
            times=times,
            outcomes=outcomes,
            individual_results=individual_results,
        )


def plot_monte_carlo(
    mc_result: MonteCarloResult,
    save_path: str | None = None,
    show: bool = True,
):
    """Generate statistical distribution plots from Monte Carlo results."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        f"Monte Carlo Failure Analysis ({mc_result.n_runs} Simulations)",
        fontsize=16,
        fontweight="bold",
    )

    energies = np.array(mc_result.energies)
    path_lens = np.array(mc_result.path_lengths)
    nodes = np.array(mc_result.nodes_explored)
    outcomes = np.array(mc_result.outcomes)

    frac_mask = outcomes == "FRACTURE"
    arrest_mask = outcomes == "ARREST"

    ax = axes[0, 0]
    if np.any(frac_mask):
        ax.hist(
            energies[frac_mask],
            bins=25,
            alpha=0.7,
            color="#FF4444",
            label="Fracture",
            edgecolor="white",
        )
    if np.any(arrest_mask):
        ax.hist(
            energies[arrest_mask],
            bins=25,
            alpha=0.7,
            color="#4488FF",
            label="Arrest",
            edgecolor="white",
        )
    ax.axvline(
        np.mean(energies),
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {np.mean(energies):.3f}",
    )
    ax.set_xlabel("Total Crack Energy")
    ax.set_ylabel("Frequency")
    ax.set_title("Energy Distribution")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    if np.any(frac_mask):
        ax.hist(
            path_lens[frac_mask],
            bins=25,
            alpha=0.7,
            color="#FF4444",
            label="Fracture",
            edgecolor="white",
        )
    if np.any(arrest_mask):
        ax.hist(
            path_lens[arrest_mask],
            bins=25,
            alpha=0.7,
            color="#4488FF",
            label="Arrest",
            edgecolor="white",
        )
    ax.set_xlabel("Path Length (pixels)")
    ax.set_ylabel("Frequency")
    ax.set_title("Path Length Distribution")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    sizes = [mc_result.n_fractures, mc_result.n_arrests]
    labels = [
        f"Fracture\n({mc_result.n_fractures})",
        f"Arrest\n({mc_result.n_arrests})",
    ]
    colors = ["#FF4444", "#4488FF"]
    explode = (0.05, 0.05)
    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
        textprops={"fontsize": 10},
    )
    ax.set_title("Failure Probability")

    ax = axes[1, 0]
    sorted_e = np.sort(energies)
    cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
    ax.plot(sorted_e, cdf, color="#FF4444", linewidth=2)
    ax.fill_between(sorted_e, cdf, alpha=0.15, color="#FF4444")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    median_e = np.median(energies)
    ax.axvline(
        median_e, color="#FF8800", linestyle="--", label=f"Median: {median_e:.3f}"
    )
    ax.set_xlabel("Total Crack Energy")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Energy CDF")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.hist(nodes, bins=25, alpha=0.75, color="#44CC44", edgecolor="white")
    ax.axvline(
        np.mean(nodes),
        color="black",
        linestyle="--",
        label=f"Mean: {np.mean(nodes):.0f}",
    )
    ax.set_xlabel("Nodes Explored")
    ax.set_ylabel("Frequency")
    ax.set_title("Search Efficiency Distribution")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    if np.any(frac_mask):
        ax.scatter(
            path_lens[frac_mask],
            energies[frac_mask],
            c="#FF4444",
            alpha=0.5,
            s=20,
            label="Fracture",
            edgecolors="none",
        )
    if np.any(arrest_mask):
        ax.scatter(
            path_lens[arrest_mask],
            energies[arrest_mask],
            c="#4488FF",
            alpha=0.5,
            s=20,
            label="Arrest",
            edgecolors="none",
        )
    ax.set_xlabel("Path Length (pixels)")
    ax.set_ylabel("Total Energy")
    ax.set_title("Energy vs Path Length")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Monte Carlo plot saved → {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def export_csv(mc_result: MonteCarloResult, path: str):
    """Export raw Monte Carlo data to CSV for external analysis."""
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run",
                "outcome",
                "energy",
                "path_length",
                "nodes_explored",
                "time_seconds",
            ]
        )
        for i in range(mc_result.n_runs):
            writer.writerow(
                [
                    i + 1,
                    mc_result.outcomes[i],
                    f"{mc_result.energies[i]:.6f}",
                    mc_result.path_lengths[i],
                    mc_result.nodes_explored[i],
                    f"{mc_result.times[i]:.6f}",
                ]
            )
    print(f"CSV exported → {path}")


if __name__ == "__main__":
    print("Running Monte Carlo Analysis (50 runs)...")
    print()

    mc = MonteCarloAnalysis(n_runs=50, grid_size=100, base_seed=0)
    result = mc.run()

    print()
    print(result.summary())

    plot_monte_carlo(result, save_path="monte_carlo_results.png", show=False)
    export_csv(result, "monte_carlo_data.csv")
