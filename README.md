# RGPS - Random Generation and Perturbation of Structures

The RGPS Workflow Tool is a command-line interface designed to automate the pipeline for generating, perturbing, optimizing, selecting, and submitting atomic structures for calculation.

📦 Installation & Setup
Before running the tool, you need to set up a Python virtual environment and install the required dependencies.

1. Open your terminal and navigate to the project directory, then run:

   ```Bash
   conda create -n [customized name] python=3.12
   ```

2. Activate the virtual environment:

   ```Bash
   conda activate [customized name]
   ```

3. Install required packages:

   ```Bash
   pip install -r requirement.txt
   pip install -e .
   ```

4. If you need other modules such as les for *MACELES* type model, you can install the module by youself.

🚀 Workflow Overview
The tool is divided into five sequential subcommands, representing each step in the pipeline. Each step requires a JSON configuration file to run.

The default workflow moves data through the following stages:

1. Generation (gen): Generates initial atomic structures (bulk, slab, adsorption, or cluster).
2. Perturbation (perturb): Applies random or equation-of-state (EOS) perturbations to the generated structures.
3. Optimization (opt): Relaxes the perturbed structures using a specified machine learning calculator/model.
4. Selection (select): Filters structures based on a maximum force threshold (optional), computes SOAP descriptors, and performs Farthest Point Sampling (FPS) to select a diverse subset.
5. Submission (submit): Prepares and submits the selected structures for calculation (currently supports CP2K).

🛠️ Basic Usage
Run the main script and pass the desired subcommand along with the path to your configuration JSON file:

```Bash
python path/to/main.py <command> path/to/parameter.json
```

Available commands: gen, perturb, opt, select, submit.

⚙️ Configuration (parameter.json)
The entire workflow is driven by a single JSON configuration file. Keys preceded by an underscore (like _input_path) are ignored by the parser, acting effectively as comments or disabled settings.

Here is a full example configuration:

JSON
{
    "work_dir": "./",

    "generation": {
        "job_name": "generation",
        "output_filename": "gen.xyz",
        "numbers": 32,
        "bulk": {
            "O": 16,
            "Ti": 8
        },
        "_slab": {
            "Zn": [12, 24],
            "Pd": 12
        },
        "_cluster": {
            "Pt": 13
        },
        "_adsorption": {
            "adsorbent": {
                "Au": 32
            },
            "adsorbate": {
                "C": 1, 
                "O": 1
            }
        },
        "tolerance_d": 1.0,
        "vacuum": 0.0,
        "max_attempts": 1000,
        "nproc_total": 16
    },

    "perturbation": {
        "job_name": "perturbation",
        "_input_path": "./generation",
        "_input_filename": "structure.xyz",
        "output_filename": "perturb.xyz",
        "numbers": 16,
        "perturb_style": "random",
        "_perturb_style": "eos",
        "perturb_cell": true,
        "perturb_ratio": 0.05,
        "_eos_ratio": 0.1,
        "seed": 42,
        "nproc_total": 16
    },

    "optimization": {
        "job_name": "optimization",
        "_input_path": "./perturbation",
        "_input_filename": "perturb.xyz",
        "output_filename": "opt.xyz",
        "model_type": "mace",
        "_model_type": "dp",
        "_model_type": "nep",
        "model_path": "./mace-omat-0-small.model",
        "steps": 200,
        "opt_style": "var_cell",
        "_opt_style": "fix_cell",
        "method": "bfgs",
        "_method": "cg",
        "_method": "newton",
        "energy_threshold": 1E-5,
        "forces_threshold": 0.05,
        "nproc_total": 16
    },

    "selection": {
        "job_name": "selection",
        "_input_path": "./optimization",
        "_input_filename": "opt.xyz",
        "output_filename": "select.xyz",
        "numbers": 32,
        "soap": {
            "r_cut": 6.0,
            "n_max": 8,
            "l_max": 6
        },
        "max_force_threshold": 50.0,
        "nproc_total": 16
    },

    "submission": {
        "job_name": "cp2k_calculation",
        "_input_path": "./selection",
        "template_path": "./template.inp",
        "modified_filename": "input.inp",
        "struct_filename": "select.xyz",
        "kspacing": 0.2,
        "nproc_single": 4,
        "nproc_total": 16,
        "command": "mpirun -np {nproc_single} cp2k.popt -i {modified_name} -o cp2k.out"
    }
}

🧩 Customizing Your Workflow (Running Standalone Steps)
You do not have to run the entire pipeline from start to finish. You can easily execute individual steps.

1. Executing a Single Step
   To run just one step (for example, optimization), use its specific command:
   python main.py opt parameter.json

2. Overriding Default Input Paths
   By default, each module looks for its input in the default output folder of the previous step in the pipeline (e.g., optimization automatically looks for the perturbation directory).
   If you are skipping steps or using external data, you must explicitly define where to find the input files. To do this, remove the underscore from _input_path and _input_filename in your JSON file and point them to your custom directory and file names.

3. Notice for the Selection Step
   If you are running the select command standalone, pay close attention to the max_force_threshold parameter. This value sets the maximum allowable atomic force (e.g., 50.0 eV/Å).
   If an atom in a frame has a force exceeding this threshold, the entire frame is completely rejected before SOAP descriptors or Farthest Point Sampling (FPS) are calculated.
   If you set this value too low on unoptimized, highly perturbed structures, the script may reject every single frame, resulting in a failure where no candidates are available for selection.
   The max_force_parameter acts on info_key "max_force". So if the input trajectories have no "max_force" info_key, please do not set max_force_parameter parameter.

4. Notice for the Submission Step
   At present, the submission step only supports CP2K calculation with input template in a specific format. See template.inp for details.
