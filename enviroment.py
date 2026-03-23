"""
environment.py — Synthetic 2D Polycrystalline Microstructure Generator

Generates a grid representing a metal microstructure with three phases:
  - Ferrite (soft, ductile matrix)       : K_IC ≈ 1.0 (moderate toughness)
  - Martensite (hard grain boundaries)   : K_IC ≈ 3.0 (high toughness)
  - Brittle inclusions (carbides/oxides) : K_IC ≈ 0.3 (low toughness, easy to crack)

Uses Voronoi tessellation to create realistic polygonal grain shapes.
Applied stress field decreases from left (crack initiation) to right (opposite boundary).
"""

import numpy as np
from scipy.spatial import Voronoi

PHASE_FERRITE = 0
PHASE_MARTENSITE = 1
PHASE_INCLUSION = 2

PHASE_NAMES = {
    PHASE_FERRITE: "Ferrite",
    PHASE_MARTENSITE: "Martensite",
    PHASE_INCLUSION: "Brittle Inclusion",
}

PHASE_TOUGHNESS = {
    PHASE_FERRITE: 1.0,
    PHASE_MARTENSITE: 3.0,
    PHASE_INCLUSION: 0.3,
}

PHASE_COLORS = {
    PHASE_FERRITE: (0.35, 0.55, 0.85),       
    PHASE_MARTENSITE: (0.25, 0.75, 0.45),     
    PHASE_INCLUSION: (0.90, 0.30, 0.25),      
}



class Microstructure:
    """A 2D synthetic polycrystalline microstructure."""

    def __init__(
        self,
        width: int = 100,
        height: int = 100,
        n_grains: int = 80,
        phase_probs: tuple = (0.55, 0.30, 0.15),
        inclusion_boundary_prob: float = 0.25,
        stress_max: float = 2.5,
        stress_min: float = 0.3,
        k_threshold: float = 0.1,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        width, height : int
            Grid dimensions (pixels).
        n_grains : int
            Number of Voronoi seed points → number of grains.
        phase_probs : tuple of 3 floats
            Probability weights for (ferrite, martensite, inclusion) per grain.
        inclusion_boundary_prob : float
            Extra probability of placing brittle inclusions at grain boundaries.
        stress_max : float
            Applied stress intensity at the left boundary (crack initiation side).
        stress_min : float
            Applied stress intensity at the right boundary (farthest from crack).
        k_threshold : float
            Minimum K_applied for crack propagation (below this → crack arrest).
        seed : int or None
            Random seed for reproducibility.
        """
        self.width = width
        self.height = height
        self.n_grains = n_grains
        self.phase_probs = np.array(phase_probs) / np.sum(phase_probs)
        self.inclusion_boundary_prob = inclusion_boundary_prob
        self.stress_max = stress_max
        self.stress_min = stress_min
        self.k_threshold = k_threshold
        self.rng = np.random.default_rng(seed)

        self.phase_grid = None
        self.toughness_grid = None
        self.stress_grid = None
        self.grain_id_grid = None
        self.voronoi_seeds = None

        self._generate()

    def _generate(self):
        """Build the full microstructure."""
        self._generate_grains()
        self._assign_phases()
        self._add_boundary_inclusions()
        self._build_toughness_grid()
        self._build_stress_field()


    def _generate_grains(self):
        """Create Voronoi-based grain regions."""
        seeds = self.rng.random((self.n_grains, 2))
        seeds[:, 0] *= self.width
        seeds[:, 1] *= self.height
        self.voronoi_seeds = seeds

        yy, xx = np.mgrid[0:self.height, 0:self.width]
        coords = np.stack([xx, yy], axis=-1).astype(float)

        dists = np.linalg.norm(
            coords[:, :, np.newaxis, :] - seeds[np.newaxis, np.newaxis, :, :],
            axis=-1,
        )
        self.grain_id_grid = np.argmin(dists, axis=-1)


    def _assign_phases(self):
        """Randomly assign each grain a phase (ferrite / martensite / inclusion)."""
        grain_phases = self.rng.choice(
            [PHASE_FERRITE, PHASE_MARTENSITE, PHASE_INCLUSION],
            size=self.n_grains,
            p=self.phase_probs,
        )
        self.phase_grid = grain_phases[self.grain_id_grid]


    def _add_boundary_inclusions(self):
        """Place extra brittle inclusions at grain boundary pixels."""
        padded = np.pad(self.grain_id_grid, 1, mode="edge")
        boundary = np.zeros((self.height, self.width), dtype=bool)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = padded[1 + dy : self.height + 1 + dy,
                             1 + dx : self.width + 1 + dx]
            boundary |= (shifted != self.grain_id_grid)

        inclusion_mask = (
            boundary
            & (self.rng.random((self.height, self.width)) < self.inclusion_boundary_prob)
        )
        self.phase_grid[inclusion_mask] = PHASE_INCLUSION

    def _build_toughness_grid(self):
        """Map phase IDs → K_IC values with slight random perturbation."""
        base = np.vectorize(PHASE_TOUGHNESS.get)(self.phase_grid).astype(float)
        noise = 1.0 + self.rng.uniform(-0.10, 0.10, size=base.shape)
        self.toughness_grid = base * noise

    def _build_stress_field(self):
        """
        Linear stress gradient: high at x=0 (crack source), low at x=width-1.
        K_applied(x) = stress_max - (stress_max - stress_min) * (x / (width - 1))
        """
        x = np.linspace(self.stress_max, self.stress_min, self.width)
        self.stress_grid = np.broadcast_to(x[np.newaxis, :], (self.height, self.width)).copy()

    def edge_cost(self, r1, c1, r2, c2):
        """
        Effective cost to propagate crack from (r1,c1) → (r2,c2).
        cost = max(0, K_IC(target) - K_applied(target))
        Zero means the crack propagates freely through that region.
        """
        k_ic = self.toughness_grid[r2, c2]
        k_app = self.stress_grid[r2, c2]
        return max(0.0, k_ic - k_app)

    def can_propagate(self, r, c):
        """Check if crack can physically propagate at (r, c)."""
        return self.stress_grid[r, c] >= self.k_threshold

    def get_neighbors(self, r, c):
        """8-connected neighbors within grid bounds."""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    neighbors.append((nr, nc))
        return neighbors

    def summary(self):
        """Print a text summary of the microstructure."""
        total = self.width * self.height
        counts = {name: np.sum(self.phase_grid == pid) for pid, name in PHASE_NAMES.items()}
        print(f"Microstructure: {self.width}×{self.height} ({total} pixels)")
        for name, count in counts.items():
            print(f"  {name:20s}: {count:5d} pixels ({100*count/total:.1f}%)")
        print(f"  Stress range: {self.stress_min:.2f} – {self.stress_max:.2f}")
        print(f"  K_threshold: {self.k_threshold:.2f}")



if __name__ == "__main__":
    micro = Microstructure(seed=42)
    micro.summary()
    print(f"\nSample toughness at (50,50): {micro.toughness_grid[50,50]:.3f}")
    print(f"Sample stress   at (50,50): {micro.stress_grid[50,50]:.3f}")
    print(f"Edge cost (50,50)→(50,51):  {micro.edge_cost(50,50,50,51):.3f}")
