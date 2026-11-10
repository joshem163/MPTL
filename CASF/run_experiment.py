"""Single entry point for all four experiment modes × four model choices.

Modes (--feature-type × --spectral):
  single            → PH   (baseline persistent homology)
  single --spectral → PTL  (persistent topological Laplacian)
  multi             → MPH  (multi-parameter persistent homology)
  multi  --spectral → MPTL (multi-parameter topological Laplacian)

Models (--model):
  gbm      sklearn GradientBoostingRegressor  [default; matches published results]
  lgbm     LightGBM                           [recommended for MPH / MPTL]
  xgboost  XGBoost                            [strong L1/L2 regularization]
  rf       Random Forest                      [fast sanity-check baseline]

Hyperparameter flags apply to the chosen model. Unrecognised flags for a given
model are silently ignored (e.g. --num-leaves has no effect for gbm).

Examples
--------
  # Reproduce published GBM results
  python run_experiment.py --dataset 2013 --feature-type multi --model gbm

  # LightGBM on MPTL (recommended for 26K-dim features)
  python run_experiment.py --dataset 2013 --feature-type multi --spectral --model lgbm

  # XGBoost quick test
  python run_experiment.py --dataset 2007 --feature-type multi --model xgboost --n-seeds 3

  # Random Forest sanity check
  python run_experiment.py --dataset 2007 --feature-type single --model rf --n-seeds 5

SLURM:
  sbatch submit_job.sh 2013 multi --spectral --model lgbm
"""
import argparse
import copy
import math
import os
import re
from contextlib import contextmanager
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dataset_year(value):
    aliases = {'07':'2007','13':'2013','16':'2016',
               '2007':'2007','2013':'2013','2016':'2016'}
    value = value.lower().replace('casf-', '')
    if value not in aliases:
        raise argparse.ArgumentTypeError('Choose 07/13/16 or 2007/2013/2016')
    return aliases[value]


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('Must be a positive integer')
    return value


def _build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # ── Required ───────────────────────────────────────────────────────────────
    p.add_argument('--dataset', type=_dataset_year, required=True,
                   help='CASF year: 07/2007, 13/2013, or 16/2016')

    # ── Mode ──────────────────────────────────────────────────────────────────
    p.add_argument('--feature-type', '--features',
                   choices=['single', 'multi'], default='multi',
                   help='single → PH/PTL;  multi → MPH/MPTL')
    p.add_argument('--spectral', action='store_true',
                   help='Add Laplacian spectral channels (PTL or MPTL)')

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument('--model',
                   choices=['gbm', 'lgbm', 'xgboost', 'rf'],
                   default='gbm',
                   help='Regression model (default: gbm = sklearn GBM)')

    # ── Paths ─────────────────────────────────────────────────────────────────
    p.add_argument('--data-root', type=Path,
                   default=Path(os.environ.get(
                       'CASF_DATA_ROOT',
                       Path(__file__).resolve().parent / 'data')),
                   help='Parent of CASF-2007/2013/2016; or set $CASF_DATA_ROOT')
    p.add_argument('--output-root', type=Path,
                   default=Path(os.environ.get(
                       'CASF_OUTPUT_ROOT',
                       Path(__file__).resolve().parent / 'outputs')))
    p.add_argument('--experiment', default='default',
                   help='Experiment namespace (subfolder under output-root/CASF-YEAR/)')
    p.add_argument('--index-file',  type=Path)
    p.add_argument('--refined-dir', type=Path)
    p.add_argument('--core-dir',    type=Path)
    p.add_argument('--exclude-id',  nargs='*', default=[])

    # ── Pipeline control ───────────────────────────────────────────────────────
    p.add_argument('--stage', choices=['all', 'prepare', 'extract', 'train'],
                   default='all')
    p.add_argument('--workers', type=_positive_int, default=1)

    # ── Seeds ─────────────────────────────────────────────────────────────────
    seeds = p.add_mutually_exclusive_group()
    seeds.add_argument('--n-seeds',  type=_positive_int, default=10)
    seeds.add_argument('--seeds',    nargs='+', type=int)

    # ── MP grid ────────────────────────────────────────────────────────────────
    p.add_argument('--grid-size', type=_positive_int, default=12)

    # ── Shared hyperparameters (all tree models) ───────────────────────────────
    p.add_argument('--n-estimators',  type=_positive_int, default=None,
                   help='Number of trees/estimators (default from model preset)')
    p.add_argument('--learning-rate', type=float, default=None,
                   help='Learning rate (boosting models only)')
    p.add_argument('--subsample',     type=float, default=None,
                   help='Row subsampling fraction (gbm/xgboost)')
    p.add_argument('--max-features',
                   choices=['sqrt', 'log2', 'all'], default=None,
                   help='Column sampling per split (gbm/rf)')

    # ── GBM-specific ──────────────────────────────────────────────────────────
    p.add_argument('--max-depth',         type=_positive_int, default=None)
    p.add_argument('--min-samples-split', type=_positive_int, default=None)

    # ── LightGBM-specific ─────────────────────────────────────────────────────
    p.add_argument('--num-leaves',       type=_positive_int, default=None,
                   help='Max leaves per tree (lgbm); default 127')
    p.add_argument('--min-child-samples', type=_positive_int, default=None,
                   help='Min samples in a leaf (lgbm); default 10')
    p.add_argument('--feature-fraction', type=float, default=None,
                   help='Column fraction per tree (lgbm); default 0.3')

    # ── XGBoost-specific ──────────────────────────────────────────────────────
    p.add_argument('--colsample-bytree', type=float, default=None,
                   help='Column fraction per tree (xgboost); default 0.3')
    p.add_argument('--min-child-weight', type=_positive_int, default=None,
                   help='Min child weight (xgboost); default 5')

    # ── Shared regularization (lgbm / xgboost) ────────────────────────────────
    p.add_argument('--reg-alpha',  type=float, default=None,
                   help='L1 regularization (lgbm/xgboost); default 0.1')
    p.add_argument('--reg-lambda', type=float, default=None,
                   help='L2 regularization (lgbm/xgboost); default 1.0')

    # ── Memory guard ──────────────────────────────────────────────────────────
    p.add_argument('--max-global-kernel-atoms', type=_positive_int, default=1200)

    # ── Toggles ───────────────────────────────────────────────────────────────
    p.add_argument('--no-save-models',   action='store_true')
    p.add_argument('--no-save-barcodes', action='store_true')

    return p


def _build_params(args, model_name: str, default_params_fn) -> dict:
    """Start from the model preset and apply any CLI overrides."""
    params = default_params_fn(model_name)

    # Shared overrides
    overrides = {
        'n_estimators':  args.n_estimators,
        'learning_rate': args.learning_rate,
        'subsample':     args.subsample,
    }
    # max_features: gbm/rf use this key; lgbm uses feature_fraction
    if args.max_features is not None:
        if model_name in ('gbm', 'rf'):
            overrides['max_features'] = (None if args.max_features == 'all'
                                         else args.max_features)
    # GBM
    if args.max_depth         is not None: overrides['max_depth']         = args.max_depth
    if args.min_samples_split is not None: overrides['min_samples_split'] = args.min_samples_split
    # LightGBM
    if args.num_leaves        is not None: overrides['num_leaves']        = args.num_leaves
    if args.min_child_samples is not None: overrides['min_child_samples'] = args.min_child_samples
    if args.feature_fraction  is not None: overrides['feature_fraction']  = args.feature_fraction
    # XGBoost
    if args.colsample_bytree  is not None: overrides['colsample_bytree']  = args.colsample_bytree
    if args.min_child_weight  is not None: overrides['min_child_weight']  = args.min_child_weight
    # Shared regularization
    if args.reg_alpha         is not None: overrides['reg_alpha']         = args.reg_alpha
    if args.reg_lambda        is not None: overrides['reg_lambda']        = args.reg_lambda

    for k, v in overrides.items():
        if v is not None and k in params:
            params[k] = v

    return params


# ─────────────────────────────────────────────────────────────────────────────
# Advisory experiment lock
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def _experiment_lock(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                'Another job is using this experiment. '
                'Use a different --experiment or wait.')
        try:
            f.seek(0); f.truncate(); f.write(str(os.getpid())); f.flush()
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    p    = _build_parser()
    args = p.parse_args(argv)

    # ── Validation ────────────────────────────────────────────────────────────
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.experiment):
        p.error('--experiment must be a simple name without path separators')
    if args.grid_size < 2:
        p.error('--grid-size must be at least 2')
    if args.learning_rate is not None:
        if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            p.error('--learning-rate must be a finite positive number')
    if args.subsample is not None and not (0 < args.subsample <= 1):
        p.error('--subsample must be in (0, 1]')

    seeds = args.seeds if args.seeds is not None else list(range(args.n_seeds))
    if len(seeds) != len(set(seeds)) or any(s < 0 or s >= 2**32 for s in seeds):
        p.error('Seeds must be distinct integers in [0, 2**32)')

    # ── Lazy imports ──────────────────────────────────────────────────────────
    import config
    from data_loader import prepare_dataset
    from features import extract_baseline, prepare_mp_features, checked_mp_settings
    from train import run_training, default_params
    from utils import write_json

    # ── Configure modules ─────────────────────────────────────────────────────
    config.FEATURES = copy.deepcopy(config.FEATURES)
    config.FEATURES['max_global_kernel_atoms'] = args.max_global_kernel_atoms
    config.SAVE_BARCODES  = not args.no_save_barcodes
    config.SAVE_MODELS    = not args.no_save_models
    config.SEEDS          = seeds
    config.GRID_SIZE      = args.grid_size
    config.REFERENCE_CUTOFF = 12.0
    config.STRICT_2007_COUNTS = False

    model_name = args.model
    params     = _build_params(args, model_name, default_params)

    # ── Directory layout ───────────────────────────────────────────────────────
    args.data_root = args.data_root.expanduser().resolve()
    experiment     = (args.output_root.expanduser().resolve()
                      / ('CASF-' + args.dataset) / args.experiment)

    spectral_suffix = '_spectral' if args.spectral else ''
    model_suffix    = f'_{model_name}' if model_name != 'gbm' else ''

    if args.feature_type == 'single':
        baseline_dir = experiment / ('single' + spectral_suffix)
        target_dir   = baseline_dir
    else:
        baseline_dir = experiment / 'single'   # plain PH baseline always here
        target_dir   = experiment / (
            f'multi_{args.grid_size}x{args.grid_size}{spectral_suffix}')

    mode = {('single', False): 'PH',
            ('single', True):  'PTL',
            ('multi',  False): 'MPH',
            ('multi',  True):  'MPTL'}[(args.feature_type, args.spectral)]

    print(f'\n{"="*62}')
    print(f'  Mode     : {mode}')
    print(f'  Model    : {model_name.upper()}')
    print(f'  Dataset  : CASF-{args.dataset}')
    print(f'  Seeds    : {seeds}')
    print(f'  Output   : {target_dir}')
    if args.feature_type == 'multi':
        print(f'  Grid     : {args.grid_size}×{args.grid_size}')
    if args.spectral:
        print(f'  Spectral : {config.SPECTRAL_CHANNELS}')
    print(f'  Params   : {params}')
    print(f'{"="*62}\n', flush=True)

    with _experiment_lock(experiment / '.pipeline.lock'):

        # ── [1/3] Dataset ──────────────────────────────────────────────────────
        print('[1/3] Preparing dataset manifest…', flush=True)
        prepare_dataset(args, baseline_dir)
        if args.stage == 'prepare':
            return baseline_dir

        # ── [2/3] Features ─────────────────────────────────────────────────────
        if args.stage in ('all', 'extract'):
            ph_spectral = (args.feature_type == 'single' and args.spectral)
            label = 'PTL (PH + spectral)' if ph_spectral else 'PH (baseline)'
            print(f'\n[2/3] Extracting {label} features (workers={args.workers})…',
                  flush=True)
            extract_baseline(
                baseline_dir / 'manifest.csv', baseline_dir,
                config.FEATURES, workers=args.workers,
                save_barcodes=config.SAVE_BARCODES,
                use_spectral=ph_spectral)

            if args.feature_type == 'multi':
                label = ('MPTL (Betti + spectral)' if args.spectral
                         else 'MPH (Betti only)')
                print(f'\n[2/3] Building {label} features…', flush=True)
                prepare_mp_features(baseline_dir, target_dir,
                                    use_spectral=args.spectral)

        if args.stage == 'extract':
            return target_dir

        # ── [3/3] Training ─────────────────────────────────────────────────────
        print(f'\n[3/3] Training [{model_name.upper()}]…', flush=True)

        if args.feature_type == 'multi':
            settings      = checked_mp_settings(target_dir, baseline_dir, args.spectral)
            save_barcodes = False
            manifest      = target_dir / 'manifest.csv'
            feat_dir      = target_dir
        else:
            # For PTL (single + spectral), the features were extracted with
            # _use_spectral_ptl=True embedded in the settings so the signature
            # matches. Pass the same effective settings here.
            settings      = copy.deepcopy(config.FEATURES)
            settings['_use_spectral_ptl'] = bool(args.spectral)
            save_barcodes = config.SAVE_BARCODES
            manifest      = baseline_dir / 'manifest.csv'
            feat_dir      = baseline_dir

        run_dir = run_training(
            manifest, feat_dir, settings, params, seeds,
            model_name=model_name,
            save_models=config.SAVE_MODELS,
            save_barcodes=save_barcodes)

        invocation = {k: str(v) if isinstance(v, Path) else v
                      for k, v in vars(args).items()}
        write_json(run_dir / 'invocation.json', invocation)
        write_json(target_dir / 'latest_run.json',
                   {'run_dir': str(run_dir), 'mode': mode, 'model': model_name})

        print(f'\nResults : {run_dir}')
        print(f'Summary : {run_dir / "summary.json"}', flush=True)
        return run_dir


if __name__ == '__main__':
    main()
