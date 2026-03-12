import numpy as np

from rgps.tools.geometry import cell_from_lengths_angles


def perturb_random_core(atoms, perturb_ratio=0.05, perturb_cell_flag=False, rng=None):
    if rng is None:
        rng = np.random.RandomState()

    new_atoms = atoms.copy()
    scaled_pos = new_atoms.get_scaled_positions()
    delta_scaled_pos = rng.uniform(-perturb_ratio, perturb_ratio, size=scaled_pos.shape)
    new_atoms.set_scaled_positions((scaled_pos + delta_scaled_pos) % 1.0)
    meta = {"style": "random", "delta_scaled_pos": delta_scaled_pos.tolist()}

    if perturb_cell_flag:
        cell_params = new_atoms.cell.cellpar()
        lengths = cell_params[:3]
        angles = cell_params[3:]

        delta_scaled_lengths = rng.uniform(
            -perturb_ratio, perturb_ratio, size=lengths.shape
        )
        new_lengths = lengths * (1.0 + delta_scaled_lengths)

        delta_angles_lim = perturb_ratio * 30.0
        delta_angles = rng.uniform(
            -delta_angles_lim, delta_angles_lim, size=angles.shape
        )
        new_angles = angles + delta_angles

        new_cell = cell_from_lengths_angles(*new_lengths, *new_angles)
        new_atoms.set_cell(new_cell, scale_atoms=True)
        meta["delta_scaled_length"] = delta_scaled_lengths.tolist()
        meta["delta_angle"] = delta_angles.tolist()

    return new_atoms, meta


def perturb_eos_core(atoms, eos_factor):
    """Equation of State perturbation (isotropic scaling)."""
    new_atoms = atoms.copy()
    cell = new_atoms.get_cell()
    new_atoms.set_cell(cell * float(eos_factor), scale_atoms=True)
    return new_atoms, {"style": "eos", "eos_factor": float(eos_factor)}
