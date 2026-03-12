import re
import numpy as np

from rgps.tools.geometry import (
    cell_from_lengths_angles,
    cell_to_reciprocal,
)


def compute_kpoints(lengths, angles, kspacing):
    """Calculate k-mesh based on spacing for arbitrary cell."""
    cell = cell_from_lengths_angles(*lengths, *angles)
    recip_cell = cell_to_reciprocal(cell)
    recip_lengths = np.linalg.norm(recip_cell, axis=1)
    k_abc = [int(np.ceil(max(rl / kspacing, 1.0))) for rl in recip_lengths]
    return k_abc


def fill_template(template_content, atoms, struct_filename, kspacing):
    """
    Replaces placeholders in template string with structure data.
    Placeholders: cell_a, cell_b, cell_c, angle_a, ... coordfile, kpoints_a...
    """
    lengths = atoms.get_cell_lengths_and_angles()[:3]
    angles = atoms.get_cell_lengths_and_angles()[3:]
    k_a, k_b, k_c = compute_kpoints(
        lengths=lengths,
        angles=angles,
        kspacing=kspacing,
    )

    subs = {
        "cell_a": f"{lengths[0]:.10f}",
        "cell_b": f"{lengths[1]:.10f}",
        "cell_c": f"{lengths[2]:.10f}",
        "angle_a": f"{angles[0]:.8f}",
        "angle_b": f"{angles[1]:.8f}",
        "angle_c": f"{angles[2]:.8f}",
        "coordfile": struct_filename,
        "kpoints_a": str(k_a),
        "kpoints_b": str(k_b),
        "kpoints_c": str(k_c),
    }

    out = template_content
    for key, val in subs.items():
        out = re.sub(r"\b" + re.escape(key) + r"\b", val, out)

    return out
