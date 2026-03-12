import numpy as np
from sklearn.linear_model import LinearRegression


def extract_info_key(atoms, keys, default=np.nan):
    """Helper to extract float values from atoms.info or attributes."""
    info = getattr(atoms, "info", {})

    for key in keys:
        if key in info:
            try:
                return np.float64(info[key])
            except (ValueError, TypeError):
                continue

    return default


def extract_total_energy(atoms):
    """Robustly extracts total energy from common keys or calculator."""
    energy = extract_info_key(
        atoms,
        ["energy", "free_energy", "total_energy"],
        default=np.nan,
    )
    if not np.isnan(energy):
        return energy

    try:
        return atoms.get_potential_energy()

    except Exception:
        return np.nan


def extract_max_force(atoms):
    """Extracts max_force from info."""
    return extract_info_key(
        atoms,
        ["max_force"],
        default=np.nan,
    )


class BindingEnergyCalculator:
    """
    Fits E0 values from a dataset using Linear Regression:
        E_total = Sum(N_i * E0_i) + E_b
    Then computes binding energy per atom for specific structures.
    """

    def __init__(self, elements):
        self.elements = sorted(elements)
        self.elements_map = {s: i for i, s in enumerate(self.elements)}
        self.e0s = None
        self.is_fitted = False

    def _get_elements_counts(self, atoms):
        """Returns atom count vector for a single structure."""
        elements_counts = np.zeros(len(self.elements))
        syms = atoms.get_chemical_symbols()
        for s in syms:
            if s in self.elements_map:
                elements_counts[self.elements_map[s]] += 1
        return elements_counts

    def fit(self, frames):
        """
        Fits E0 values using the provided list of frames.
        Frames must have total energy available.
        """
        X = []
        y = []
        valid_indices = []

        for i, atoms in enumerate(frames):
            e_total = extract_total_energy(atoms)
            if np.isnan(e_total):
                continue

            X.append(self._get_elements_counts(atoms))
            y.append(e_total)
            valid_indices.append(i)

        if not X:
            print("Error: No valid energies found for energy baseline fitting.")
            return

        X = np.array(X)
        y = np.array(y)
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, y)
        self.e0s = reg.coef_
        self.is_fitted = True

        print("\n--- Energy baseline fitting results ---")
        for element, e0 in zip(self.elements, self.e0s):
            print(f"\n{element}: {e0:.4f} eV")

    def predict_binding_energy_per_atom(self, atoms):
        """Calculates binding energy per atom for a single structure."""
        if not self.is_fitted:
            return np.nan

        e_total = extract_total_energy(atoms)

        if np.isnan(e_total):
            return np.nan

        elements_counts = self._get_elements_counts(atoms)
        n_atoms = len(atoms)

        if n_atoms == 0:
            return np.nan

        e_ref = np.dot(elements_counts, self.e0s)
        e_binding = e_total - e_ref

        return e_binding / n_atoms
