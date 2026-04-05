"""
image_loader.py — Real Microstructure Image Input

Load SEM/EBSD images and segment them into metallurgical phases
for crack propagation simulation.

Supports:
  - Grayscale SEM images (auto-segment via K-means or Otsu thresholding)
  - EBSD phase maps (color-coded, map colors to phases)
  - Any image format supported by PIL (PNG, TIFF, JPEG, BMP)
"""

from pathlib import Path

import numpy as np
from PIL import Image
from skimage import filters, morphology, segmentation
from sklearn.cluster import KMeans

from environment import (PHASE_FERRITE, PHASE_INCLUSION, PHASE_MARTENSITE,
                         PHASE_TOUGHNESS, Microstructure)

DEFAULT_PROPERTY_MAP = {
    "dark": {
        "phase": PHASE_INCLUSION,
        "label": "Brittle Inclusion (dark regions)",
        "k_ic": PHASE_TOUGHNESS[PHASE_INCLUSION],
    },
    "medium": {
        "phase": PHASE_FERRITE,
        "label": "Ferrite (medium intensity)",
        "k_ic": PHASE_TOUGHNESS[PHASE_FERRITE],
    },
    "bright": {
        "phase": PHASE_MARTENSITE,
        "label": "Martensite (bright regions)",
        "k_ic": PHASE_TOUGHNESS[PHASE_MARTENSITE],
    },
}


def load_image(
    path: str | Path, target_size: tuple[int, int] | None = None
) -> np.ndarray:
    """
    Load an image and convert to grayscale numpy array.

    Parameters
    ----------
    path : str or Path
        Path to the image file.
    target_size : tuple (width, height) or None
        If provided, resize the image to this size.

    Returns
    -------
    np.ndarray : Grayscale image as float array in [0, 1].
    """
    img = Image.open(path)

    if img.mode != "L":
        img = img.convert("L")

    if target_size is not None:
        img = img.resize(target_size, Image.LANCZOS)

    return np.array(img, dtype=float) / 255.0


def segment_kmeans(image: np.ndarray, n_phases: int = 3) -> np.ndarray:
    """
    Segment grayscale image into phases using K-means clustering.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H, W) with values in [0, 1].
    n_phases : int
        Number of phases to segment into (default: 3).

    Returns
    -------
    np.ndarray : Phase label grid (H, W) with labels 0..n_phases-1,
                 ordered by intensity (0 = darkest, n-1 = brightest).
    """
    H, W = image.shape
    pixels = image.reshape(-1, 1)

    kmeans = KMeans(n_clusters=n_phases, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels).reshape(H, W)

    centers = kmeans.cluster_centers_.flatten()
    order = np.argsort(centers)
    remap = np.zeros(n_phases, dtype=int)
    for new_label, old_label in enumerate(order):
        remap[old_label] = new_label

    return remap[labels]


def segment_otsu(image: np.ndarray) -> np.ndarray:
    """
    Segment grayscale image into 3 phases using multi-Otsu thresholding.

    Returns
    -------
    np.ndarray : Phase label grid (H, W) with labels 0 (dark), 1 (medium), 2 (bright).
    """
    thresholds = filters.threshold_multiotsu(image, classes=3)
    labels = np.digitize(image, bins=thresholds)
    return labels.astype(int)


def segment_watershed(image: np.ndarray, n_phases: int = 3) -> np.ndarray:
    """
    Segment using watershed algorithm (better grain boundary detection).

    Returns
    -------
    np.ndarray : Phase label grid (H, W).
    """
    from skimage.filters import sobel

    gradient = sobel(image)

    markers = segment_kmeans(image, n_phases)

    labels = segmentation.watershed(gradient, markers=markers + 1)
    return labels - 1


def labels_to_phases(
    labels: np.ndarray,
    phase_mapping: dict | None = None,
) -> np.ndarray:
    """
    Convert segmentation labels to metallurgical phase IDs.

    Parameters
    ----------
    labels : np.ndarray
        Segmentation labels (0, 1, 2, ...).
    phase_mapping : dict or None
        Mapping from label → phase ID. If None, uses default:
        0 (dark) → INCLUSION, 1 (medium) → FERRITE, 2 (bright) → MARTENSITE.

    Returns
    -------
    np.ndarray : Phase grid with PHASE_* constants.
    """
    if phase_mapping is None:
        phase_mapping = {
            0: PHASE_INCLUSION,
            1: PHASE_FERRITE,
            2: PHASE_MARTENSITE,
        }

    phase_grid = np.zeros_like(labels, dtype=int)
    for label, phase_id in phase_mapping.items():
        phase_grid[labels == label] = phase_id

    return phase_grid


def clean_segmentation(phase_grid: np.ndarray, min_grain_size: int = 20) -> np.ndarray:
    """
    Remove small spurious regions from segmentation.

    Parameters
    ----------
    phase_grid : np.ndarray
        Phase grid to clean.
    min_grain_size : int
        Regions smaller than this (in pixels) are merged into neighbors.

    Returns
    -------
    np.ndarray : Cleaned phase grid.
    """
    cleaned = phase_grid.copy()
    for phase_id in np.unique(phase_grid):
        mask = phase_grid == phase_id
        cleaned_mask = morphology.remove_small_objects(mask, min_size=min_grain_size)
        removed = mask & ~cleaned_mask
        if np.any(removed):
            from scipy.ndimage import distance_transform_edt

            _, indices = distance_transform_edt(removed, return_indices=True)
            cleaned[removed] = cleaned[indices[0][removed], indices[1][removed]]

    return cleaned


def microstructure_from_image(
    image_path: str | Path,
    grid_size: int = 100,
    method: str = "kmeans",
    n_phases: int = 3,
    stress_max: float = 2.5,
    stress_min: float = 0.3,
    k_threshold: float = 0.1,
    clean: bool = True,
) -> Microstructure:
    """
    Create a Microstructure from a real SEM/EBSD image.

    Parameters
    ----------
    image_path : str or Path
        Path to the image file.
    grid_size : int
        Resize image to grid_size × grid_size.
    method : str
        Segmentation method: "kmeans", "otsu", or "watershed".
    n_phases : int
        Number of phases (for kmeans/watershed).
    stress_max, stress_min : float
        Applied stress field range.
    k_threshold : float
        Minimum stress for crack propagation.
    clean : bool
        Whether to clean up small spurious regions.

    Returns
    -------
    Microstructure : Ready for A* search.
    """
    image = load_image(image_path, target_size=(grid_size, grid_size))

    if method == "kmeans":
        labels = segment_kmeans(image, n_phases)
    elif method == "otsu":
        labels = segment_otsu(image)
    elif method == "watershed":
        labels = segment_watershed(image, n_phases)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")

    phase_grid = labels_to_phases(labels)

    if clean:
        phase_grid = clean_segmentation(phase_grid)
        micro = Microstructure(
            width=grid_size,
            height=grid_size,
            stress_max=stress_max,
            stress_min=stress_min,
            k_threshold=k_threshold,
        )

    micro.phase_grid = phase_grid
    micro._build_toughness_grid()

    return micro


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("No image provided. Generating synthetic test image...")

        test_micro = Microstructure(seed=42)
        test_img = (test_micro.phase_grid.astype(float) / 2.0 * 255).astype(np.uint8)
        test_path = "test_microstructure.png"
        Image.fromarray(test_img).save(test_path)
        print(f"Saved test image → {test_path}")

        micro = microstructure_from_image(test_path)
        micro.summary()
    else:
        micro = microstructure_from_image(sys.argv[1])
        micro.summary()
