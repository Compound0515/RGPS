from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from ase.io import write

from rgps.tools.io import load_json
from rgps.modules.generation import (
    gen_bulk_slab_core,
    gen_adsorption_core,
    gen_cluster_core,
)


def worker_gen(args):
    idx, gen_config, output_path_base = args

    gen_types = ["bulk", "slab", "adsorption", "cluster"]
    found_types = [type for type in gen_types if type in gen_config]

    if len(found_types) != 1:
        raise ValueError(
            f"Only one type is allowed in the generation configuration: {gen_types}. "
            f"Found generation types {found_types}."
        )

    used_type = found_types[0]
    tolerance_d = gen_config.get("tolerance_d", 1.5)
    vacuum = gen_config.get("vacuum", 0.0)
    max_attempts = gen_config.get("max_attempts", 1000)

    if used_type == "bulk":
        composition = gen_config["bulk"]
        atoms = gen_bulk_slab_core(
            composition=composition,
            vacuum=0.0,
            tolerance_d=tolerance_d,
            max_attempts=max_attempts,
        )
    elif used_type == "slab":
        composition = gen_config["slab"]
        atoms = gen_bulk_slab_core(
            composition=composition,
            vacuum=vacuum,
            tolerance_d=tolerance_d,
            max_attempts=max_attempts,
        )
    elif used_type == "adsorption":
        ads_config = gen_config["adsorption"]
        atoms = gen_adsorption_core(
            ads_config=ads_config,
            vacuum=vacuum,
            tolerance_d=tolerance_d,
            max_attempts=max_attempts,
        )
    elif used_type == "cluster":
        composition = gen_config["cluster"]
        atoms = gen_cluster_core(
            composition=composition,
            vacuum=vacuum,
            tolerance_d=tolerance_d,
            max_attempts=max_attempts,
        )

    if atoms:
        subdir = output_path_base / str(idx)
        subdir.mkdir(exist_ok=True, parents=True)
        output_filename = gen_config.get("output_filename", "gen.xyz")
        write(subdir / output_filename, atoms, format="extxyz")
    else:
        return (
            f"Task-{idx} failed: no acceptable structure after {max_attempts} attempts."
        )

    return idx


def execute(args):
    config = load_json(args.config)
    gen_config = config.get("generation", {})
    print(f"Loaded configuration: {gen_config}.")

    output_path_base = Path(config.get("work_dir", ".")) / gen_config.get(
        "job_name", "generation"
    )
    output_path_base.mkdir(parents=True, exist_ok=True)
    print(f"Output base-directory: {output_path_base.absolute()}.")

    n_gen = int(gen_config.get("numbers", 10))
    tasks = [(idx, gen_config, output_path_base) for idx in range(1, n_gen + 1)]
    print(f"Running generation on {len(tasks)} tasks...")

    results = []
    with Pool(gen_config.get("nproc_total", 4)) as p:
        results = list(tqdm(p.imap(worker_gen, tasks), total=len(tasks)))
    successes = [r for r in results if isinstance(r, int)]
    failures = [r for r in results if isinstance(r, str)]
    print("\n--- Generation Summary ---")
    print(f"Total tasks: {n_gen}")
    print(f"Successed: {len(successes)}")
    print(f"Failed: {len(failures)}")
    if failures:
        print("\nSample failures:")
        for failure_msg in failures[:5]:
            print(f"  {failure_msg}")
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more.")
        print(
            "\nTip: If all generation tasks failed, check your density (volume), tolerance_d, or max_attempts."
        )


def register(subparsers):
    p = subparsers.add_parser("gen", help="Random Structure Generation")
    p.add_argument("config", help="Path to JSON config")
