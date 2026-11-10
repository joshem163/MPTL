"""Unified configuration for all experiment modes.

Modes:
  PH   - Single persistent homology (distance filtration only)
  PTL  - Persistent topological Laplacian (distance filtration + spectral)
  MPH  - Multi-parameter persistent homology (distance x weighted degree)
  MPTL - Multi-parameter persistent topological Laplacian (distance x degree + spectral)

Run via run_experiment.py:
  python run_experiment.py --dataset 2013 --feature-type single
  python run_experiment.py --dataset 2013 --feature-type single --spectral
  python run_experiment.py --dataset 2013 --feature-type multi
  python run_experiment.py --dataset 2013 --feature-type multi --spectral
"""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT   = PROJECT_DIR / 'data'
OUTPUT_DIR  = PROJECT_DIR / 'outputs'

# ── Baseline / single-parameter feature settings ──────────────────────────────
FEATURES = {
    'schema_version': 1,
    'protein_elements': ['C', 'N', 'O', 'S'],
    'ligand_elements':  ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I'],
    'pair_cutoff':   12.0,
    'kernel_cutoff':  6.0,
    'alpha_cutoff':   9.0,
    'distance_bins':      [0, 2.5, 3, 3.5, 4.5, 6, 12],
    'alpha_birth_bins':   [0, 2, 3, 4, 5, 6, 9],
    'kernels': [
        {'kind': 'lorentz', 'tau': 0.5, 'power': 3.0},
        {'kind': 'lorentz', 'tau': 1.0, 'power': 3.0},
        {'kind': 'lorentz', 'tau': 2.0, 'power': 3.0},
    ],
    'include_global_kernel':  True,
    'include_alpha':          True,
    'include_carbon_alpha':   True,
    'alpha_units':           'radius',
    'include_protein_hetatm': False,
    'ligand_preference':     ['mol2', 'sdf'],
    'max_global_kernel_atoms': 1200,
    'vdw_radii': {
        'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80,
        'P': 1.80, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98,
        'Se': 1.90, 'B': 1.92, 'Si': 2.10, 'Zn': 1.39,
        'Mg': 1.73, 'Ca': 2.31, 'Fe': 1.56, 'Na': 2.27,
        'K':  2.75, 'Mn': 1.61, 'Cu': 1.40, 'Co': 1.52,
        'Ni': 1.63, 'Cd': 1.58, 'Hg': 1.55,
    },
}

# ── Multi-parameter bifiltration settings ─────────────────────────────────────
# Distance axis: GRID_SIZE evenly-spaced thresholds from 0 to REFERENCE_CUTOFF Å
GRID_SIZE           = 12          # number of steps on both distance and degree axes
REFERENCE_CUTOFF    = 12.0        # Å; upper bound of distance axis
REFERENCE_KERNEL    = {'kind': 'lorentz', 'tau': 1.0, 'power': 3.0}
KEEP_OTHER_FEATURES = True        # retain non-pair baseline features alongside MP features

# ── Spectral settings (PTL and MPTL) ──────────────────────────────────────────
# Channels extracted from each graph's Laplacian at every grid cell:
#   lambda_min_positive  – smallest non-zero eigenvalue (Fiedler-like connectivity)
#   lambda_max           – largest eigenvalue (spectral radius, geometric spread)
#   lambda_mean_positive – mean of positive eigenvalues (average energy)
SPECTRAL_ATOL               = 1e-10   # absolute zero-eigenvalue tolerance
SPECTRAL_RTOL               = 1e-8    # relative tolerance multiplier
SPECTRAL_MAX_COMPONENT_SIZE = 4096    # max vertices per component for dense solver

# Ordered list of spectral channel names appended after H0, H1 in MPTL/PTL grids.
# Changing this list changes the feature schema version; adjust SCHEMA_VERSION below.
SPECTRAL_CHANNELS = ['lambda_min_positive', 'lambda_max', 'lambda_mean_positive']

# ── Training ──────────────────────────────────────────────────────────────────
WORKERS     = 1                 # baseline extraction processes; increase on HPC
SAVE_BARCODES = True
SAVE_MODELS   = True
SEEDS = list(range(10))         # use list(range(50)) for full paper replication

# Model parameters match Table 2 of the UGA_Research document.
# learning_rate=0.005 is the value used to produce Tables 4 and 5.
# The CLI --learning-rate flag overrides this at runtime.
MODEL_PARAMS = dict(
    n_estimators=20000,
    max_depth=8,
    min_samples_split=6,
    learning_rate=0.005,
    loss='squared_error',
    subsample=0.7,
    max_features='sqrt',
)

STRICT_2007_COUNTS = False      # set True to enforce 1105/195 split counts
