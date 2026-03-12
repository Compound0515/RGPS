import subprocess
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

from rgps.tools.io import load_json, safe_read_atoms
from rgps.modules.submission import fill_template


def worker_submit_prep(args):
    (
        d,
        idx,
        struct_filename,
        struct_path,
        template_content,
        modified_filename,
        kspacing,
    ) = args

    try:
        atoms = safe_read_atoms(struct_path)
        content = fill_template(
            template_content=template_content,
            atoms=atoms,
            struct_filename=struct_filename,
            kspacing=kspacing,
        )

        output_modified_path = d / modified_filename
        with open(output_modified_path, "w") as f:
            f.write(content)
        return str(output_modified_path)

    except Exception as e:
        original_path = Path(d)
        return f"Task-{idx} in {original_path.absolute()} failed: {str(e)}."


def worker_submit_exec(args):
    d, cmd = args
    try:
        with open(d / "run.log", "w") as log:
            subprocess.run(
                cmd,
                cwd=str(d),
                shell=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        return True

    except Exception:
        return False


def execute(args):
    config = load_json(args.config)
    submit_config = config.get("submission", {})
    print(f"Loaded configuration: {submit_config}.")

    input_path = Path(config.get("work_dir")) / "selection"
    if "input_path" in submit_config:
        input_path = Path(submit_config["input_path"])

    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist.")
        return

    template_path = Path(submit_config.get("template_path", "./template.inp"))
    with open(template_path, "r") as f:
        template_content = f.read()
    modified_filename = submit_config.get("modified_filename", "input.inp")
    struct_filename = submit_config.get("struct_filename", "select.xyz")
    kspacing = float(submit_config.get("kspacing", 0.2))
    subdirs = [x for x in input_path.iterdir() if x.is_dir()]

    if not subdirs:
        print(f"Error: No sub-directories found in {input_path}.")
        return

    prep_tasks = []
    global_idx = 0

    for d in subdirs:
        struct_path = d / struct_filename

        if not struct_path.exists():
            continue

        global_idx += 1
        idx = global_idx
        prep_tasks.append(
            (
                d,
                idx,
                struct_filename,
                struct_path,
                template_content,
                modified_filename,
                kspacing,
            )
        )

    if not prep_tasks:
        print(f"Error: No valid {struct_filename} files found in sub-directories.")
        return

    print(f"Preparing input files for {len(prep_tasks)} tasks...")

    results = []
    with Pool() as p:
        results = list(
            tqdm(p.imap(worker_submit_prep, prep_tasks), total=len(prep_tasks))
        )
    successes = [r for r in results if "failed" not in r]
    failures = [r for r in results if "failed" in r]
    print("\n--- Input Preparation Summary ---")
    print(f"Total tasks: {len(prep_tasks)}")
    print(f"Successed: {len(successes)}")
    print(f"Failed: {len(failures)}")
    if failures:
        print("\nSample failures:")
        for failure_msg in failures[:5]:
            print(f"  {failure_msg}")
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more.")

    command = submit_config.get("command")

    if command:
        print("Submitting jobs...")
        cmd = command.replace("{modified_name}", modified_filename)
        cmd = cmd.replace("{nproc_single}", str(submit_config.get("nproc_single", 4)))
        exec_tasks = [(d, cmd) for d in subdirs]
        print(f"Running calculation on {len(exec_tasks)} tasks...")
        with Pool(submit_config.get("nproc_total", 16)) as p:
            p.starmap(worker_submit_exec, exec_tasks)


def register(subparsers):
    p = subparsers.add_parser(
        "submit", help="Calculation Submittal (CP2K-only at present)"
    )
    p.add_argument("config", help="Path to config.json")
