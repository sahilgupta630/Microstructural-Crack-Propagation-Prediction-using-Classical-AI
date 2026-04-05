"""
main.py — A*tomic Fracture: Crack Propagation Simulator

Entry point for all simulation modes:
  - Single run (default)
  - Algorithm comparison (--compare)
  - Monte Carlo analysis (--monte-carlo)
  - Real image input (--image)

Usage:
    python main.py
    python main.py --compare
    python main.py --monte-carlo --mc-runs 100
    python main.py --image microstructure.png
"""

import argparse
import time

from astar_search import AStarCrackSearch
from environment import Microstructure
from visualizer import plot_result


def run_single(args):
    """Standard single A* crack propagation run."""
    print("▸ Generating synthetic microstructure...")
    t0 = time.time()

    if args.image:
        from image_loader import microstructure_from_image

        micro = microstructure_from_image(
            args.image,
            grid_size=args.size,
            method=args.seg_method,
            stress_max=args.stress_max,
            stress_min=args.stress_min,
            k_threshold=args.k_threshold,
        )
        print(f"  Loaded from image: {args.image}")
    else:
        micro = Microstructure(
            width=args.size,
            height=args.size,
            n_grains=args.n_grains,
            phase_probs=(
                args.ferrite_ratio,
                args.martensite_ratio,
                args.inclusion_ratio,
            ),
            stress_max=args.stress_max,
            stress_min=args.stress_min,
            k_threshold=args.k_threshold,
            seed=args.seed,
        )

    t_env = time.time() - t0
    micro.summary()
    print(f"  Generated in {t_env:.3f}s\n")

    print("▸ Running A* crack propagation search...")
    t0 = time.time()
    searcher = AStarCrackSearch(micro)
    result = searcher.search(start_row=args.start_row)
    t_search = time.time() - t0
    print(result)
    print(f"  Search completed in {t_search:.3f}s\n")

    print("▸ Generating visualization...")
    plot_result(micro, result, save_path=args.output, show=not args.no_show)

    if args.animate:
        from visualizer import animate_exploration

        anim_path = args.output.rsplit(".", 1)[0] + "_animation.gif"
        animate_exploration(micro, result, save_path=anim_path)

    _print_summary(result, t_env, t_search)
    return micro, result


def run_comparison(args):
    """Run all 4 algorithms on the same microstructure."""
    from algorithm_comparison import compare_algorithms, plot_comparison

    print("▸ Generating microstructure for comparison...")
    if args.image:
        from image_loader import microstructure_from_image

        micro = microstructure_from_image(
            args.image,
            grid_size=args.size,
            method=args.seg_method,
            stress_max=args.stress_max,
            stress_min=args.stress_min,
            k_threshold=args.k_threshold,
        )
    else:
        micro = Microstructure(
            width=args.size,
            height=args.size,
            n_grains=args.n_grains,
            phase_probs=(
                args.ferrite_ratio,
                args.martensite_ratio,
                args.inclusion_ratio,
            ),
            stress_max=args.stress_max,
            stress_min=args.stress_min,
            k_threshold=args.k_threshold,
            seed=args.seed,
        )
    micro.summary()
    print()

    print("▸ Running algorithm comparison:")
    results = compare_algorithms(micro, start_row=args.start_row)
    print()

    save_path = args.output.rsplit(".", 1)[0] + "_comparison.png"
    plot_comparison(micro, results, save_path=save_path, show=not args.no_show)


def run_monte_carlo(args):
    """Run Monte Carlo statistical analysis."""
    from monte_carlo import MonteCarloAnalysis, export_csv, plot_monte_carlo

    print(f"▸ Running Monte Carlo Analysis ({args.mc_runs} runs)...")
    print()

    mc = MonteCarloAnalysis(
        n_runs=args.mc_runs,
        grid_size=args.size,
        n_grains=args.n_grains,
        phase_probs=(args.ferrite_ratio, args.martensite_ratio, args.inclusion_ratio),
        stress_max=args.stress_max,
        stress_min=args.stress_min,
        k_threshold=args.k_threshold,
        base_seed=args.seed,
    )
    result = mc.run()

    print()
    print(result.summary())

    save_path = args.output.rsplit(".", 1)[0] + "_monte_carlo.png"
    plot_monte_carlo(result, save_path=save_path, show=not args.no_show)

    csv_path = args.output.rsplit(".", 1)[0] + "_monte_carlo.csv"
    export_csv(result, csv_path)


def _print_summary(result, t_env, t_search):
    """Print final summary."""
    print()
    print("─" * 60)
    outcome_icon = "💥" if result.outcome == "FRACTURE" else "🛑"
    print(f"  {outcome_icon}  RESULT: {result.outcome}")
    print(f"  Path length   : {len(result.path)} pixels")
    print(f"  Total energy  : {result.total_cost:.4f}")
    print(f"  Nodes explored: {result.nodes_explored}")
    print(f"  Time (env)    : {t_env:.3f}s")
    print(f"  Time (search) : {t_search:.3f}s")
    print("─" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="A*tomic Fracture — Crack Propagation via AI Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  python main.py                                  Single A* run (default)
  python main.py --compare                        Compare A*/Dijkstra/Greedy/BFS
  python main.py --monte-carlo --mc-runs 100      Statistical failure analysis
  python main.py --image microstructure.png        Use real image input
  python main.py --compare --image micro.png       Compare algorithms on real image
        """,
    )

    # Mode selection
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run algorithm comparison (A* vs Dijkstra vs Greedy vs BFS)",
    )
    parser.add_argument(
        "--monte-carlo",
        action="store_true",
        help="Run Monte Carlo statistical failure analysis",
    )
    parser.add_argument(
        "--mc-runs",
        type=int,
        default=100,
        help="Number of Monte Carlo runs (default: 100)",
    )

    # Image input
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to SEM/EBSD image (replaces synthetic generation)",
    )
    parser.add_argument(
        "--seg-method",
        type=str,
        default="kmeans",
        choices=["kmeans", "otsu", "watershed"],
        help="Image segmentation method (default: kmeans)",
    )

    # Grid parameters
    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="Grid width/height in pixels (default: 100)",
    )
    parser.add_argument(
        "--n-grains",
        type=int,
        default=80,
        help="Number of Voronoi grains (default: 80)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    # Phase ratios
    parser.add_argument("--ferrite-ratio", type=float, default=0.55)
    parser.add_argument("--martensite-ratio", type=float, default=0.30)
    parser.add_argument("--inclusion-ratio", type=float, default=0.15)

    # Stress
    parser.add_argument("--stress-max", type=float, default=2.5)
    parser.add_argument("--stress-min", type=float, default=0.10)
    parser.add_argument("--k-threshold", type=float, default=0.1)

    # Search
    parser.add_argument("--start-row", type=int, default=None)

    # Output
    parser.add_argument("--output", type=str, default="crack_result.png")
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--no-show", action="store_true")

    args = parser.parse_args()

    print("=" * 60)
    print("  A*tomic Fracture — Crack Propagation Simulator")
    print("=" * 60)
    print()

    if args.compare:
        run_comparison(args)
    elif args.monte_carlo:
        run_monte_carlo(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
