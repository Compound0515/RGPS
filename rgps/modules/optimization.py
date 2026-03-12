import numpy as np
from ase.optimize import BFGS, LBFGS
from ase.filters import UnitCellFilter

from rgps.tools.io import write_extxyz_frame


def opt_core(atoms, calculator, output_path, opt_config):
    """
    Runs optimization on an Atoms object using the provided calculator.
    Writes trajectory to output_path.
    """
    atoms.calc = calculator
    opt_atoms = atoms
    if opt_config.get("opt_style") in ["var_cell", "opt_cell"]:
        opt_atoms = UnitCellFilter(atoms)
    method = opt_config.get("method", "bfgs").lower()
    Optimizer = LBFGS if method == "lbfgs" else BFGS
    opt = Optimizer(opt_atoms, logfile=None)

    def write_frame(step_num):
        e = atoms.get_potential_energy()
        f_max = np.max(np.linalg.norm(atoms.get_forces(), axis=1))
        write_extxyz_frame(
            atoms,
            output_path,
            energy=e,
            max_force=f_max,
            step=step_num,
        )
        return e, f_max

    e_initial, f_max_initial = write_frame(0)
    e_previous = e_initial
    f_max_previous = f_max_initial
    converged = False
    max_steps = opt_config.get("steps", 200)
    e_tol = opt_config.get("energy_threshold", 1e-5)
    f_tol = opt_config.get("forces_threshold", 0.02)

    for i in range(1, max_steps + 1):
        opt.run(steps=1)
        e_current, f_max_current = write_frame(i)
        ediff = e_current - e_previous
        e_previous = e_current
        f_max_previous = f_max_current

        if abs(ediff) < e_tol and f_max_current < f_tol:
            converged = True
            break

    return {
        "converged": converged,
        "steps": i,
        "final_energy": e_previous,
        "final_ediff": ediff,
        "final_fmax": f_max_previous,
    }
