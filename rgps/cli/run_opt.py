from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from ase.io import write

from rgps.tools.io import (
    load_json,
    safe_read_atoms,
)
from rgps.tools.calculators import get_calculator
from rgps.modules.optimization import opt_core


def worker_opt(args):
    idx, struct_path, opt_config, output_path_base = args

    try:
        atoms = safe_read_atoms(struct_path)
        calculator = get_calculator(opt_config["model_type"], opt_config["model_path"])

        subdir = output_path_base / str(idx)
        subdir.mkdir(parents=True, exist_ok=True)
        output_filename = opt_config.get("output_filename", "opt.xyz")
        output_path = subdir / output_filename
        final_path = subdir / "final.xyz"
        if output_path.exists():
            output_path.unlink()

        opt_results = opt_core(
            atoms=atoms,
            calculator=calculator,
            output_path=output_path,
            opt_config=opt_config,
        )

        write(final_path, atoms, format="extxyz")
        return opt_results

    except Exception as e:
        return f"Task-{idx} failed: {str(e)}."


def execute(args):
    config = load_json(args.config)
    opt_config = config.get("optimization", {})
    print(f"Loaded configuration: {opt_config}.")

    output_path_base = Path(config.get("work_dir", ".")) / opt_config.get(
        "job_name", "optimization"
    )
    output_path_base.mkdir(parents=True, exist_ok=True)
    print(f"Output base-directory: {output_path_base.absolute()}.")

    input_path = Path(config.get("work_dir")) / "perturbation"
    if "input_path" in opt_config:
        input_path = Path(opt_config["input_path"])

    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist.")
        return

    input_filename = opt_config.get("input_filename", "perturb.xyz")
    subdirs = [x for x in input_path.iterdir() if x.is_dir()]

    if not subdirs:
        print(f"Error: No sub-directories found in {input_path}.")
        return

    tasks = []
    global_idx = 0

    for d in subdirs:
        struct_path = d / input_filename

        if not struct_path.exists():
            continue

        global_idx += 1
        idx = global_idx
        tasks.append(
            (
                idx,
                struct_path,
                opt_config,
                output_path_base,
            )
        )

    if not tasks:
        print(f"Error: No valid {input_filename} files found in sub-directories.")
        return

    print(f"Running optimization on {len(tasks)} tasks...")

    results = []
    with Pool(opt_config.get("nproc_total", 4)) as p:
        results = list(tqdm(p.imap(worker_opt, tasks), total=len(tasks)))
    successes = [r for r in results if "failed" not in r]
    failures = [r for r in results if "failed" in r]
    print("\n--- Optimization Summary ---")
    print(f"Total tasks: {len(tasks)}")
    print(f"Successed: {len(successes)}")
    print(f"Failed: {len(failures)}")
    if failures:
        print("\nSample failures:")
        for failure_msg in failures[:5]:
            print(f"  {failure_msg}")
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more.")
        print("\nTips: If all optimization tasks failed, you can follow these steps:")
        print("\n1. Check 'model_path' in parameters.json.")
        print("\n2. Check if 'input_filename' matches previous step output.")
        print("\n3. Decrease 'nproc_total' if running out of GPU memory.")


def register(subparsers):
    p = subparsers.add_parser("opt", help="Structure Optimization")
    p.add_argument("config", help="Path to JSON config")
