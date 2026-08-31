<!--
Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
# Physics-Informed Neural ODE (PINODE): Embedding Physics into Models using Collocation Points

This repository includes source code for training and using the Physics-Informed Neural ODE (PINODE) Operator for modeling complex dynamics systems published in our Nature Scientific Reports paper,
**Physics-Informed Neural ODE (PINODE): Embedding Physics into Models using Collocation Points**
by Aleksei Sholokhov, Yuying Liu, Hassan Mansour, and Saleh Nabi.

[Please click here to read the paper.](https://www.nature.com/articles/s41598-023-36799-6.pdf)

## Installation

Clone the repository, use Conda to create only the Python environment, and let
pip install the project dependencies. This avoids a potentially slow Conda
solve involving the scientific Python and PyTorch packages:

    git clone https://github.com/merlresearch/PhysicsInformedNeuralODE.git
    cd PhysicsInformedNeuralODE
    conda create -n pinode python=3.9 pip -y
    conda activate pinode
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -e .

Alternatively, skip Conda and use a Python 3.9 virtual environment:

    python3.9 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -e .

The full [environment.yml](environment.yml) remains available for users who
prefer a Conda-managed installation.

## Repository structure

    .
    ├── src/
    │   └── pinode/                    # Importable Python package
    │       ├── core/                  # PINODE, DMD, SINDy, and collocation components
    │       ├── burgers/               # Burgers data, training program, and configurations
    │       └── duffing/               # Duffing data, training program, and configurations
    ├── examples/
    │   ├── burgers_pinode.py          # Burgers saved-model evaluation
    │   ├── duffing_pinode.py          # Duffing saved-model evaluation
    │   └── saved_outputs/             # Bundled experiment artifacts
    ├── requirements.txt               # Python dependencies
    ├── environment.yml                # Optional full Conda environment
    ├── pyproject.toml                 # Package and editable-install metadata
    ├── CONTRIBUTING.md                # Contribution policy
    └── LICENSE.md                     # AGPL-3.0-or-later license

The project uses a `src` layout. Installing it with
`python -m pip install -e .` exposes `src/pinode` as the top-level `pinode`
package. Imports and module commands therefore start with `pinode`; neither
`src` nor the repository directory name is part of the Python import path.
Generated training runs and evaluation outputs are written to `runs/` and
`examples/outputs/`, respectively, and are not source directories.

## Training

The training programs use [Sacred](https://sacred.readthedocs.io/) to record
configuration, metrics, and model artifacts. Runs are stored locally under
`runs/burgers/` and `runs/duffing/` by default, so no database or SSH tunnel is
required.

During training, each program first generates simulated trajectories and
physics collocation points for its dynamical system. It also fits a Dynamic
Mode Decomposition (DMD) baseline. The PINODE then learns an encoder, latent
dynamics, and decoder using a configurable combination of:

- a data-driven trajectory-prediction and snapshot-reconstruction loss; and
- a physics-informed latent-closure and collocation-reconstruction loss.

At each configured logging interval, the program evaluates predictions on the
held-out trajectories over both the training time horizon (the metrics ending
in `_short`) and the longer test time horizon. It also records errors relative
to the DMD baseline. Burgers additionally reports results by initial-condition
family (harmonic, Gaussian, and bump), while Duffing reports results by phase
space region (left, right, and outer lobes).

Train the Burgers model with its default configuration:

    python -m pinode.burgers.pinode_burgers

Train the Duffing model with its default configuration:

    python -m pinode.duffing.hi_dim_duffing

Both programs select CUDA automatically when it is available and otherwise use
the CPU. Sacred configuration values can be overridden after `with`; for
example, this runs a five-epoch Burgers training job explicitly on the CPU:

    python -m pinode.burgers.pinode_burgers with device=cpu n_epochs=5

For an NVIDIA GPU, install a CUDA-enabled PyTorch build and use `device=cuda`.

### Training outputs

Each command creates a numbered Sacred run directory:

    runs/
    ├── burgers/<run-id>/
    └── duffing/<run-id>/

A run contains the resolved configuration, run status and result, captured
console output, logged metric histories, and a `model` artifact containing the
trained PyTorch `state_dict`. The Burgers run also contains a
`collocation_provider` artifact and retains the model checkpoint with the best
held-out error over the training time horizon. The Duffing run stores the model
state at the end of training. The exact Sacred filenames normally include
`config.json`, `run.json`, `metrics.json`, and `cout.txt`, in addition to the
artifacts.

The returned Sacred result is the final full-horizon test loss. The model
artifact is a state dictionary rather than a standalone serialized model, so
reloading it requires constructing a `PINODE` with the saved configuration and
then calling `load_state_dict`.

Set `PINODE_SACRED_RUNS_DIR` to change the local run directory. To additionally
record runs in MongoDB, set `PINODE_MONGO_URL` and, optionally,
`PINODE_MONGO_DB`:

    PINODE_MONGO_URL=mongodb://localhost:27017 PINODE_MONGO_DB=sacred \
        python -m pinode.burgers.pinode_burgers

## Testing saved models

The testing stage is an offline evaluation and figure-reproduction workflow.
The example scripts load the bundled `experiment_*.pickle` files from
`examples/saved_outputs/`; each pickle contains a trained model together with
its configuration, Sacred metrics, and run record. The scripts regenerate the
required Burgers or Duffing reference data, run predictions from the saved
models, compare their errors and latent dynamics with relevant baselines and
experiment variants, and produce the analyses and figures used by the project.
No model parameters are optimized during this stage.

Run the saved-model evaluations with:

    python examples/burgers_pinode.py
    python examples/duffing_pinode.py

### Testing outputs

By default, each invocation creates a timestamped directory such as
`examples/outputs/burgers_pinode_<timestamp>/` or
`examples/outputs/duffing_pinode_<timestamp>/` with this structure:

    <output-dir>/
    ├── figures/       # Generated evaluation and comparison figures (primarily PDF)
    ├── logs/
    │   ├── run.log    # Console output, progress, and saved-file messages
    │   └── run_details.json  # Command, environment, and resolved paths
    ├── data/          # Generated analysis data (for example, Burgers RMSE JSON)
    └── animations/    # Optional Burgers MP4 output

Some directories can be empty when an analysis does not produce that output
type. The Burgers animation is written only when `--save-animations` is passed
and `ffmpeg` is available. These outputs are evaluation products; they do not
include a newly trained model checkpoint.

For a reproducible output location, pass `--output-dir`; for example:

    python examples/burgers_pinode.py --output-dir examples/outputs/burgers_test
    python examples/duffing_pinode.py --output-dir examples/outputs/duffing_test

The saved output pickle files under `examples/saved_outputs/` are provided as experiment artifacts for reproducing figures and comparisons from the paper.

Use `--output-dir` to choose a specific output location and `--saved-outputs-dir` to read experiment artifacts from a non-default directory. The Burgers script can also save its animation with `--save-animations` when `ffmpeg` is available.

If you use any part of this code for your work, we ask that you include the following citation:

    @article{Sholokhov_2023SciRep,
      author =	 {Aleksei Sholokhov and Yuying Liu and Hassan Mansour and Saleh Nabi},
      title =	 {Physics-Informed Neural ODE (PINODE): Embedding Physics into Models using Collocation Points},
      journal =	 {Scientific Reports},
      volume = 13,
      number = 10166,
      year =	 2023,
      url = https://doi.org/10.1038/s41598-023-36799-6,
      doi = 10.1038/s41598-023-36799-6,
      issn = 2045-2322,
      month =	 June
    }


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## License

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as noted below and in the licenses for packages listed in environment.yml:

```
Copyright (C) 2022-2023 Mitsubishi Electric Research Laboratories (MERL).

SPDX-License-Identifier: AGPL-3.0-or-later
```

`src/pinode/core/dfdri.py` was adapted from https://github.com/rkotynski/D_FDRI/

(`GPL-3.0-or-later` license as found in [LICENSES/GPL-3.0-or-later.txt](LICENSES/GPL-3.0-or-later.txt))

```
Copyright (C) 2021 Anna Pastuszczak, Rafal Stojek, Piotr Wrobel, Rafal Kotynski

SPDX-License-Identifier: GPL-3.0-or-later
```
