import json
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from ase.io import write

from rgps.tools.io import (
    load_json,
    safe_read_atoms,
)
from rgps.modules.perturbation import (
    perturb_random_core,
    perturb_eos_core,
)


def worker_perturb(args):
    idx, struct_path, perturb_config, output_path_base, seed, eos_val = args

    try:
        atoms = safe_read_atoms(
            struct_path,
        )
        style = perturb_config.get("perturb_style", "random")
        subdir = output_path_base / str(idx)
        subdir.mkdir(parents=True, exist_ok=True)
        output_filename = perturb_config.get("output_filename", "perturb.xyz")
        new_atoms = None
        meta = {}

        if style == "eos":
            eos_factor = eos_val if eos_val else 1.0
            new_atoms, meta = perturb_eos_core(
                atoms=atoms,
                eos_factor=eos_factor,
            )
        elif style == "random":
            rng = np.random.RandomState(seed)
            perturb_ratio = float(perturb_config.get("perturb_ratio", 0.05))
            perturb_cell_flag = bool(perturb_config.get("perturb_cell", False))
            new_atoms, meta = perturb_random_core(
                atoms=atoms,
                perturb_ratio=perturb_ratio,
                perturb_cell_flag=perturb_cell_flag,
                rng=rng,
            )
        else:
            perturb_types = ["random", "eos"]
            return f"Error: Unknown perturbation type. Only support {perturb_types} at present."

        write(subdir / output_filename, new_atoms, format="extxyz")
        meta["seed"] = seed
        with open(subdir / "perturb_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        return idx

    except Exception as e:
        return f"Task-{idx} failed: {str(e)}."


def execute(args):
    config = load_json(args.config)
    perturb_config = config.get("perturbation", {})
    print(f"Loaded configuration: {perturb_config}.")

    output_path_base = Path(config.get("work_dir")) / perturb_config.get(
        "job_name", "perturbation"
    )
    output_path_base.mkdir(parents=True, exist_ok=True)
    print(f"Output base-directory: {output_path_base.absolute()}.")

    input_path = Path(config.get("work_dir")) / "generation"
    if "input_path" in perturb_config:
        input_path = Path(perturb_config["input_path"])

    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist.")
        return

    input_filename = perturb_config.get("input_filename", "gen.xyz")
    subdirs = [x for x in input_path.iterdir() if x.is_dir()]

    if not subdirs:
        print(f"Error: No sub-directories found in {input_path}.")
        return

    n_perturb = int(perturb_config.get("numbers", 1))
    perturb_style = perturb_config.get("perturb_style", "random")
    eos_ratio = float(perturb_config.get("eos_ratio", 0.1))
    seed_offset = int(perturb_config.get("seed", 42))

    tasks = []
    global_idx = 0

    for d in subdirs:
        struct_path = d / input_filename

        if not struct_path.exists():
            continue

        eos_factors = []

        if perturb_style == "eos":
            factors = np.linspace(1.0 - eos_ratio, 1.0 + eos_ratio, n_perturb)
            eos_factors = factors

        for i in range(n_perturb):
            global_idx += 1
            idx = global_idx
            seed = seed_offset + idx
            eos_val = eos_factors[i] if perturb_style == "eos" else None
            tasks.append(
                (
                    idx,
                    struct_path,
                    perturb_config,
                    output_path_base,
                    seed,
                    eos_val,
                )
            )

    if not tasks:
        print(f"Error: No valid {input_filename} files found in sub-directories.")
        return

    print(f"Running perturbation on {len(tasks)} tasks...")

    results = []
    with Pool(perturb_config.get("nproc_total", 4)) as p:
        results = list(tqdm(p.imap(worker_perturb, tasks), total=len(tasks)))
    successes = [r for r in results if isinstance(r, int)]
    failures = [r for r in results if isinstance(r, str)]
    print("\n--- Perturbation Summary ---")
    print(f"Total tasks: {len(tasks)}")
    print(f"Successed: {len(successes)}")
    print(f"Failed: {len(failures)}")
    if failures:
        print("\nSample failures:")
        for failure_msg in failures[:5]:
            print(f"  {failure_msg}")
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more.")


def register(subparsers):
    p = subparsers.add_parser("perturb", help="Structure Perturbation")
    p.add_argument("config", help="Path to config.json")
