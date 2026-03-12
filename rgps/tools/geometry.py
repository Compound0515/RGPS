import numpy as np
import math
from ase.geometry.cell import cellpar_to_cell


def cell_to_reciprocal(cell):
    """Return reciprocal cell matrix (2pi included) from real-space cell."""
    cell = np.asarray(cell, dtype=float)
    volume = np.abs(np.linalg.det(cell.T))
    if volume < 1e-10:
        raise ValueError("Cell volume is too small.")
    recip_cell = 2 * np.pi * np.linalg.inv(cell.T)
    return recip_cell


def cell_from_lengths_angles(a, b, c, alpha_deg, beta_deg, gamma_deg):
    """Return 3x3 cell matrix from lengths and angles (degrees)."""
    alpha, beta, gamma = map(math.radians, (alpha_deg, beta_deg, gamma_deg))
    ax, ay, az = a, 0.0, 0.0
    bx = b * math.cos(gamma)
    by = b * math.sin(gamma)
    bz = 0.0
    cx = c * math.cos(beta)
    sin_gamma = math.sin(gamma)
    cy = (
        0.0
        if abs(sin_gamma) < 1e-10
        else c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma
    )
    cz2 = c**2 - cx**2 - cy**2
    cz = math.sqrt(max(0.0, cz2))
    cell = np.array([[ax, ay, az], [bx, by, bz], [cx, cy, cz]], dtype=float)
    return cell


def random_cell_params(volume, min_ratio=2.0, max_ratio=4.0):
    """Generates random a, b, c, alpha, beta, gamma based on volume."""
    min_len, max_len = (min_ratio * volume) ** 0.33, (max_ratio * volume) ** 0.34
    a = np.random.uniform(min_len, max_len)
    b = np.random.uniform(min_len, max_len)
    c = np.random.uniform(min_len, max_len)

    centers = [60, 90, 120]
    weights = [0.2, 0.6, 0.2]
    angles = []

    for _ in range(3):
        center = np.random.choice(centers, p=weights)
        angle = np.random.normal(center, 5)
        angle = np.clip(angle, 30, 150)
        angles.append(angle)

    alpha, beta, gamma = angles
    cell_params = [a, b, c, alpha, beta, gamma]

    return cell_params


def safe_random_cell_params(volume, min_ratio=2.0, max_ratio=4.0, max_attempts=1000):
    """
    Generates random a, b, c, alpha, beta, gamma based on volume.
    Considering the relationship among alpha, beta, and gamma.
    """
    min_len = (min_ratio * volume) ** (1 / 3)
    max_len = (max_ratio * volume) ** (1 / 3)

    centers = [60, 90, 120]
    weights = [0.2, 0.6, 0.2]

    for _ in range(max_attempts):
        a = np.random.uniform(min_len, max_len)
        b = np.random.uniform(min_len, max_len)
        c = np.random.uniform(min_len, max_len)
        angles = []

        for _ in range(3):
            center = np.random.choice(centers, p=weights)
            angle = np.random.normal(center, 5)
            angle = np.clip(angle, 30, 150)
            angles.append(angle)

        alpha, beta, gamma = angles

        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)

        cos_alpha = np.cos(alpha_rad)
        cos_beta = np.cos(beta_rad)
        cos_gamma = np.cos(gamma_rad)

        cz_sq = (
            1
            - cos_alpha**2
            - cos_beta**2
            - cos_gamma**2
            + 2 * cos_alpha * cos_beta * cos_gamma
        )

        if cz_sq > 1e-10:
            try:
                cell_params = [a, b, c, alpha, beta, gamma]
                cell_matrix = cellpar_to_cell(cell_params)
                if not (np.isnan(cell_matrix).any() or np.isinf(cell_matrix).any()):
                    return cell_params

            except Exception as e:
                continue

    target_volume = volume * min_ratio
    length = target_volume ** (1 / 3)
    scales = np.random.uniform(0.9, 1.1, 3)
    lengths = length * scales / (np.prod(scales) ** (1 / 3))

    return [lengths[0], lengths[1], lengths[2], 90.0, 90.0, 90.0]


def get_min_max_distance(atoms):
    """Calculates min and max interatomic distances."""
    dists = atoms.get_all_distances(mic=True)
    sorted_dists = [sorted(row)[1] for row in dists if len(row) > 1]
    return min(sorted_dists), max(sorted_dists)


def get_volume_from_composition(elements_counts, radi_dict):
    volume = 0
    for element, count in elements_counts.items():
        volume += 4 / 3 * np.pi * radi_dict.get(element, 1.5) ** 3 * count
    return volume
