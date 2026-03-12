import json
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from ase.io import read, write

from rgps.tools.analysis import (
    BindingEnergyCalculator,
    extract_max_force,
    extract_total_energy,
)
from rgps.tools.io import (
    load_json,
    safe_read_atoms,
)
from rgps.tools.vis import plot_selection_pca
from rgps.modules.selection import (
    compute_soap_descriptors_core,
    select_fps_core,
)


def worker_select(args):
    d, idx, struct_path, elements, soap_config, max_force_threshold = args

    try:
        frames = read(struct_path, index=":")

        if not frames:
            return {f"Error: No frames read from {struct_path}."}

        valid_frames = []
        valid_indices = []
        rejected_count = 0

        for i, atoms in enumerate(frames):
            atoms.pbc = True

            if max_force_threshold is not None:
                max_force = extract_max_force(atoms)
                if np.isnan(max_force) or max_force > max_force_threshold:
                    rejected_count += 1
                    continue

            valid_frames.append(atoms)
            valid_indices.append(i)

        if not valid_frames:
            return {
                "subdir": d.name,
                "status": "filtered",
                "rejected_count": rejected_count,
            }

        features = compute_soap_descriptors_core(
            frames=valid_frames,
            elements=elements,
            soap_config=soap_config,
        )

        return {
            "subdir": d.name,
            "status": "kept",
            "features": features,
            "frames": valid_frames,
            "original_indices": valid_indices,
            "rejected_count": rejected_count,
        }

    except Exception as e:
        original_path = Path(d)
        return f"Task-{idx} in {original_path.absolute()} failed: {str(e)}."


def execute(args):
    config = load_json(args.config)
    select_config = config.get("selection", {})
    print(f"Loaded configuration: {select_config}.")

    output_path_base = Path(config.get("work_dir")) / select_config.get(
        "job_name", "selection"
    )
    output_path_base.mkdir(parents=True, exist_ok=True)
    print(f"Output base-directory: {output_path_base.absolute()}")

    input_path = Path(config.get("work_dir")) / "optimization"
    if "input_path" in select_config:
        input_path = Path(select_config["input_path"])

    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist.")
        return

    input_filename = select_config.get("input_filename", "opt.xyz")
    subdirs = [x for x in input_path.iterdir() if x.is_dir()]

    if not subdirs:
        print(f"Error: No sub-directories found in {input_path}.")
        return

    elements_set = set()

    for d in subdirs:
        f = d / input_filename
        if f.exists():
            try:
                atoms = safe_read_atoms(f)
                symbols = atoms.get_chemical_symbols()
                elements_set.update(symbols)
            except Exception:
                continue

    elements = sorted(elements_set)

    if not elements:
        print("Error: Could not determine elements from input files.")
        return

    soap_config = select_config.get("soap", {})
    max_force_threshold = select_config.get("max_force_threshold", None)

    if max_force_threshold is not None:
        print(f"Applying max-force threshold: {max_force_threshold} eV/Ang.")

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
                d,
                idx,
                struct_path,
                elements,
                soap_config,
                max_force_threshold,
            )
        )

    if not tasks:
        print(f"Error: No valid {input_filename} files found in sub-directories.")
        return

    print(f"Running SOAP descriptor calculations on {len(tasks)} tasks...")

    results = []
    with Pool(select_config.get("nproc_total", 4)) as p:
        results = list(tqdm(p.imap(worker_select, tasks), total=len(tasks)))

    total_rejected_counts = 0
    total_features = []
    total_kept_frames = []
    mapping = []

    for r in results:
        if "failed" not in r and r.get("status") == "filtered":
            total_rejected_counts += r.get("rejected_count", 0)
            continue
        if "failed" not in r and r.get("status") == "kept":
            total_rejected_counts += r.get("rejected_count", 0)
            total_features.append(r["features"])
            total_kept_frames.extend(r["frames"])
            for i, original_idx in enumerate(r["original_indices"]):
                mapping.append((r["subdir"], original_idx))

    successes = [r for r in results if "failed" not in r]
    failures = [r for r in results if "failed" in r]
    print("\n--- Selection Summary ---")
    print(f"Total tasks: {len(tasks)}")
    print(f"Successed: {len(successes)}")
    print(f"Failed: {len(failures)}")
    print(f"Total frames kept: {len(total_kept_frames)}")
    print(f"Total frames filtered: {total_rejected_counts}")
    if failures:
        print("\nSample failures:")
        for failure_msg in failures[:5]:
            print(f"  {failure_msg}")
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more.")

    if not total_features:
        print("\nError: No candidates available for FPS selection.")
        return

    print("\nCalculating binding energies by fitting energy baseline...")
    calc = BindingEnergyCalculator(elements)
    calc.fit(total_kept_frames)
    binding_energies_per_atom = np.array(
        [calc.predict_binding_energy_per_atom(atoms) for atoms in total_kept_frames]
    )
    X = np.vstack(total_features)
    n_select = int(select_config.get("numbers", 10))
    print(
        f"\nRunning FPS selection for {n_select} structures among {len(total_kept_frames)} structures..."
    )
    selected_indices = select_fps_core(X, n_select)

    plot_selection_pca(
        features=X,
        property=binding_energies_per_atom,
        property_name=str("Binding energy (eV/atom)"),
        indices=selected_indices,
        output_path=output_path_base,
    )
    print(f"PCA plot saved to: {output_path_base}.")

    selected_frames = []
    meta_data = []
    output_filename = select_config.get("output_filename", "select.xyz")

    for rank, idx in enumerate(selected_indices):
        frame = total_kept_frames[idx]
        original_subdir, original_frame_idx = mapping[idx]
        subdir = output_path_base / str(rank + 1)
        subdir.mkdir(exist_ok=True)
        write(subdir / output_filename, frame, format="extxyz")
        selected_frames.append(frame)
        meta_data.append(
            {
                "rank": rank,
                "original_folder": original_subdir,
                "original_frame_index": original_frame_idx,
                "binding_energy_peratom": (
                    float(binding_energies_per_atom[idx])
                    if np.isfinite(binding_energies_per_atom[idx])
                    else None
                ),
            }
        )

    write(output_path_base / "select_global.xyz", selected_frames, format="extxyz")
    with open(output_path_base / "selection_map.json", "w") as f:
        json.dump(meta_data, f, indent=2)


def register(subparsers):
    p = subparsers.add_parser("select", help="Structure Selection (SOAP+FPS)")
    p.add_argument("config", help="Path to config.json")
