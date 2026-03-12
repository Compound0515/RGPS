import json
import numpy as np
from pathlib import Path
from ase.io import read


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Error: Config file not found: {path}.")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_read_atoms(path, index=0):
    """Robust reader for various formats."""
    try:
        return read(str(path), index=index, format="extxyz")

    except Exception:
        return read(str(path), index=index)


def format_lattice_string(cell):
    return " ".join([f"{x:.10f}" for x in np.array(cell).flatten()])


def write_extxyz_frame(
    atoms, path, energy=None, max_force=None, step=None, append=True
):
    """
    Writes a single frame with robust Extended XYZ headers.
    """
    mode = "a" if append else "w"
    natoms = len(atoms)
    parts = []
    parts.append(f'Lattice="{format_lattice_string(atoms.get_cell())}"')

    if energy is not None:
        parts.append(f"energy={energy:.10f}")
    if energy is not None and natoms > 0:
        parts.append(f"energy_per_atom={energy/natoms:.10f}")
    if max_force is not None:
        parts.append(f"max_force={max_force:.8f}")
    if step is not None:
        parts.append(f"step={int(step)}")

    parts.append("Properties=species:S:1:pos:R:3")
    parts.append('pbc="T T T"')
    comment = " ".join(parts)

    with open(str(path), mode) as f:
        f.write(f"{natoms}\n{comment}\n")
        for s, p in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            f.write(f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")
