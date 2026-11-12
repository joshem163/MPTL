# Multi-parameter Persistent Topological Laplacian (MPTL)

Topological and spectral representations for molecular graph classification and protein–ligand binding-affinity prediction.

This repository contains two experimental pipelines:

- **CASF:** Protein–ligand binding-affinity prediction using element-specific interaction representations and tree-based regression.
- **OGBG:** Molecular graph classification using topological and spectral features with an MLP or CNN classifier.

## Supported Methods

| Method | Filtration | Laplacian spectral features |
|--------|------------|-----------------------------|
| PH | Single parameter | No |
| PTL | Single parameter | Yes |
| MPH | Multiple parameters | No |
| MPTL | Multiple parameters | Yes |

For CASF, the multiparameter representation combines protein–ligand distance and fixed weighted degree. For OGBG, it combines node and edge filtration functions.

Spectral descriptors include the smallest positive eigenvalue, largest eigenvalue, and mean positive eigenvalue.

## Installation

Clone the repository:

```bash
git clone https://github.com/joshem163/MPTL.git
cd MPTL
```

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the CASF dependencies:

```bash
python -m pip install numpy scipy pandas scikit-learn matplotlib joblib ripser gudhi rdkit
```

Install the additional OGBG dependencies:

```bash
python -m pip install torch torch-geometric networkx ogb
```

For GPU execution, install a PyTorch build compatible with your CUDA environment. The OGBG trainer uses CUDA when available and otherwise falls back to CPU.

Optional CASF regressors require:

```bash
python -m pip install lightgbm xgboost
```

## CASF Experiments

### Data Preparation

Obtain the PDBbind refined sets, corresponding CASF core sets, and affinity index files separately. These datasets are not included in this repository.

The loader expects the following structure under the data root:

| Path | Contents |
|------|----------|
| `CASF-2007/refined-set/` | Refined-set complex folders |
| `CASF-2007/core-set/` | Core-set complex folders |
| `CASF-2007/index/` | Refined affinity index file |

Use the same layout for `CASF-2013` and `CASF-2016`.

Each complex folder must contain:

- `ID_protein.pdb`
- `ID_ligand.mol2` or `ID_ligand.sdf`

Here, `ID` denotes the four-character complex identifier. The loader searches the folders recursively.

The training set contains refined-set complexes excluding core-set identifiers. The core set is used for testing.

### Run CASF Experiments

From the repository root:

```bash
cd CASF
```

Before running, update these paths in `submit_job.sh`:

- `CODE_DIR`: Absolute path to the repository’s `CASF` directory.
- `DATA_ROOT`: Directory containing the CASF datasets.
- `OUTPUT_ROOT`: Directory for experiment outputs.

The command format is:

```bash
bash submit_job.sh <YEAR> <single|multi> [options]
```

Run the four methods on CASF-2007:

```bash
# PH
bash submit_job.sh 2007 single

# PTL
bash submit_job.sh 2007 single --spectral

# MPH
bash submit_job.sh 2007 multi

# MPTL
bash submit_job.sh 2007 multi --spectral
```

Replace `2007` with `2013` or `2016` to run another benchmark.

By default, CASF experiments use scikit-learn GBM and 10 random seeds (`0` through `9`).

Additional examples:

```bash
# MPTL with GBM and a 12 × 12 grid
bash submit_job.sh 2013 multi --spectral --model gbm --grid-size 12 --n-seeds 10

# MPTL with LightGBM
bash submit_job.sh 2016 multi --spectral --model lgbm --n-seeds 10

# Explicit random seeds
bash submit_job.sh 2007 multi --spectral --seeds 0 1 2 3 4
```

Supported regressors:

| Argument | Regressor |
|----------|-----------|
| `--model gbm` | scikit-learn Gradient Boosting |
| `--model lgbm` | LightGBM |
| `--model xgboost` | XGBoost |
| `--model rf` | Random Forest |

Running with `bash` executes the experiment directly. On a SLURM cluster, use:

```bash
sbatch submit_job.sh 2007 multi --spectral
```

Update the SLURM resource directives and log paths before submission, and ensure the log directory exists.

### Direct Python Execution

Alternatively, run the Python entry point directly:

```bash
python run_experiment.py \
  --dataset 2007 \
  --feature-type multi \
  --spectral \
  --model gbm \
  --data-root /absolute/path/to/data \
  --output-root /absolute/path/to/outputs \
  --n-seeds 10
```

Additional options include:

- `--refined-dir`: Override the refined-set directory.
- `--core-dir`: Override the core-set directory.
- `--index-file`: Specify the affinity index file.
- `--experiment NAME`: Separate experiment configurations.
- `--grid-size`: Set the multiparameter grid resolution.
- `--workers`: Set the number of baseline extraction workers.

The pipeline supports separate stages:

```bash
--stage prepare
--stage extract
--stage train
```

The default is `--stage all`. Complete preparation and extraction before running the training-only stage.

View all options:

```bash
python run_experiment.py --help
```

### CASF Outputs

Outputs are organized by dataset, experiment, and representation under the selected output root.

Saved outputs include:

- Dataset manifests and feature caches.
- Feature-extraction settings.
- `run_config.json`: Model parameters, seeds, feature settings, and software versions.
- `metrics_by_seed.csv`: Metrics for each random seed.
- `summary.json`: Aggregate metrics.
- Per-seed predictions and evaluation plots.
- Fitted models when model saving is enabled.

Evaluation metrics include Pearson correlation, Spearman correlation, RMSE, and MAE in the supplied logarithmic-affinity units, plus energy-scaled RMSE.

## OGBG Experiments

### Run OGBG Experiments

From the repository root:

```bash
cd OGBG
```

If currently in the `CASF` directory:

```bash
cd ../OGBG
```

Run MPTL on ToxCast:

```bash
python train_ogb.py --dataset ogbg-moltoxcast --epochs 500 --filtration atom --direction both
```

**Use `--direction`, singular.**

This command uses:

- MPTL features.
- An MLP classifier.
- The atom-feature-sum node filtration.
- The electronegativity-difference edge filtration.
- Concatenated sublevel and superlevel node filtrations.
- Five runs with seeds `42` through `46`.

To specify these settings explicitly and save the results:

```bash
mkdir -p ogb_results

python train_ogb.py \
  --dataset ogbg-moltoxcast \
  --mode MPTL \
  --model mlp \
  --epochs 500 \
  --filtration atom \
  --edge_filtration en_diff \
  --direction both \
  --runs 5 \
  --seed 42 \
  --filename ogb_results/moltoxcast_MPTL_atom_both.pt
```

OGB downloads the dataset under the directory specified by `--root`, which defaults to `dataset`.

Experiments use the training, validation, and test partitions supplied by OGB and the official OGB evaluator. ROC-AUC is reported as a percentage.

### Supported Datasets

- `ogbg-molsider`
- `ogbg-molclintox`
- `ogbg-molbace`
- `ogbg-molbbbp`
- `ogbg-moltoxcast`
- `ogbg-moltox21`
- `ogbg-molhiv`

### Compare Representation Variants

```bash
# PH
python train_ogb.py --dataset ogbg-moltoxcast --mode PH --epochs 500 --filtration atom --direction both

# PTL
python train_ogb.py --dataset ogbg-moltoxcast --mode PTL --epochs 500 --filtration atom --direction both

# MPH
python train_ogb.py --dataset ogbg-moltoxcast --mode MPH --epochs 500 --filtration atom --direction both

# MPTL
python train_ogb.py --dataset ogbg-moltoxcast --mode MPTL --epochs 500 --filtration atom --direction both
```

### Filtration Options

| Argument | Value | Description |
|----------|-------|-------------|
| `--filtration` | `atomN` | First OGB atom-feature column |
| `--filtration` | `atom` | Sum of the supplied atom-feature entries |
| `--filtration` | `atom_norm` | Sum of per-graph normalized atom-feature entries |
| `--filtration` | `hks` | Heat kernel signature |
| `--edge_filtration` | `en_diff` | Electronegativity difference |
| `--edge_filtration` | `bondtype` | OGB bond-type category |
| `--direction` | `sub` | Sublevel node filtration |
| `--direction` | `super` | Superlevel node filtration |
| `--direction` | `both` | Concatenate both node-filtration directions |

Edge filtrations apply to MPH and MPTL and always use sublevel thresholds.

The requested numbers of thresholds can be changed using:

```bash
--n_node_thresh 10 --n_edge_thresh 10
```

Duplicate quantile thresholds are removed, so the realized grid can contain fewer thresholds. The `bondtype` option uses the unique observed bond categories.

In the current implementation, OGB thresholds are fitted across all input graphs without labels. CASF weighted-degree thresholds are fitted using training complexes only.

### Training Options

The direct Python defaults are:

| Setting | Default |
|---------|---------|
| Representation | MPTL |
| Classifier | MLP |
| Epochs | 500 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | `1e-4` |
| Hidden dimension | 256 |
| Number of linear layers | 3 |
| Dropout | 0.5 |
| Runs | 5 |
| Initial seed | 42 |
| Validation-selection warmup | 200 epochs |

Use `--model cnn` for convolutional processing of MPH or MPTL grids.

Model selection uses validation scores after the specified warmup. The reported test score corresponds to the selected epoch.

To allow validation-based selection across all epochs:

```bash
--warmup_epochs 0
```

If the warmup is at least the total number of epochs, only the final epoch is eligible.

### Save and Collect Results

Pass `--filename PATH.pt` to save aggregate results, filtration thresholds, feature shape, and timing information.

The saved file is a results summary, not a trained model checkpoint.

Collect saved summaries:

```bash
python collect_results.py --results_dir ogb_results --save results.csv
```

Results are summarized using the mean and population standard deviation across individual runs.

For SLURM batch experiments, configure `submit_ogb.sh` before use. Its training settings differ from the direct Python defaults.

View all training options:

```bash
python train_ogb.py --help
```

## Reproducibility

Record the repository commit, command-line arguments, software environment, dataset release, and random seeds for each experiment.

Repeated runs retain the same data partitions. Configuration defaults and optional models do not by themselves identify the exact settings used for a reported result.

## License

This repository is distributed under the [MIT License](LICENSE).

Dataset access and use are governed by the respective data providers.
