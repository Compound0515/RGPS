import numpy as np
from ase import Atoms

from rgps.data.atomic_data import ATOMIC_RADII
from rgps.tools.geometry import (
    safe_random_cell_params,
    get_volume_from_composition,
    get_min_max_distance,
)


def random_atoms_num(composition):
    """Selects specific atom counts from ranges."""
    elements_counts = {}
    symbols = ""
    n_atoms = 0

    for element, counts in composition.items():
        if isinstance(counts, list):
            count = np.random.randint(counts[0], counts[1] + 1)
        else:
            count = int(counts)
        if count > 0:
            elements_counts[element] = count
            symbols += f"{element}{count}"
            n_atoms += count

    return elements_counts, symbols, n_atoms


def gen_bulk_slab_core(composition, vacuum=0.0, tolerance_d=1.5, max_attempts=1000):
    """Core generator for bulk/slab structures."""
    for _ in range(max_attempts):
        elements_counts, symbols, n_atoms = random_atoms_num(composition)
        if n_atoms == 0:
            continue

        volume = get_volume_from_composition(
            elements_counts=elements_counts,
            radi_dict=ATOMIC_RADII,
        )

        cell_params = safe_random_cell_params(
            volume=volume,
        )

        atoms = Atoms(symbols, pbc=True)
        atoms.set_cell(cell_params)
        scaled_pos = np.random.uniform(0, 1, (len(atoms), 3))
        atoms.set_scaled_positions(scaled_pos)

        min_d, max_d = get_min_max_distance(atoms)
        if min_d > tolerance_d and max_d < 5.0:
            if vacuum > 0:
                atoms.center(vacuum, axis=2)
            return atoms

    return None


def gen_adsorption_core(ads_config, vacuum=0.0, tolerance_d=1.5, max_attempts=1000):
    """Generates adsorption structures (Slab + Adsorbate)."""
    adsorbent_elements_counts, adsorbent_symbols, adsorbent_n_atoms = random_atoms_num(
        ads_config["adsorbent"]
    )
    adsorbate_elements_counts, adsorbate_symbols, adsorbate_n_atom = random_atoms_num(
        ads_config["adsorbate"]
    )
    total_elements_counts = adsorbent_elements_counts.copy()

    for element, count in adsorbate_elements_counts.items():
        total_elements_counts[element] = total_elements_counts.get(element, 0) + count

    for _ in range(max_attempts):
        atoms = gen_bulk_slab_core(
            composition=total_elements_counts,
            vacuum=0.0,
            tolerance_d=tolerance_d,
        )
        if atoms is None:
            continue

        positions = atoms.get_positions()
        sorted_indices = np.argsort(positions[:, 2])
        adsorbent_indices = sorted_indices[:adsorbent_n_atoms]
        adsorbate_indices = sorted_indices[adsorbent_n_atoms:]
        adsorbent_pos = positions[adsorbent_indices]
        adsorbate_pos = positions[adsorbate_indices]
        np.random.shuffle(adsorbent_pos)
        np.random.shuffle(adsorbate_pos)
        new_pos = np.vstack([adsorbent_pos, adsorbate_pos])
        atoms.set_positions(new_pos)

        if vacuum > 0:
            atoms.center(vacuum, axis=2)

        return atoms

    return None


def gen_cluster_core(composition, vacuum=10.0, tolerance_d=1.5, max_attempts=1000):
    """Generates cluster structures (non-periodic/large vacuum)."""
    atoms = gen_bulk_slab_core(
        composition=composition,
        vacuum=0.0,
        tolerance_d=tolerance_d,
        max_attempts=max_attempts,
    )

    if atoms:
        atoms.center(vacuum=vacuum, axis=(0, 1, 2))

    return atoms
