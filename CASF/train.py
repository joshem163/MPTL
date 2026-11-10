"""Model training for all four experiment modes.

Supported models (--model flag):
─────────────────────────────────────────────────────────────────────────────
  gbm      sklearn GradientBoostingRegressor  (default; matches Tables 4/5)
  lgbm     LightGBM LGBMRegressor             (recommended for MPH / MPTL)
  xgboost  XGBoost  XGBRegressor              (good L1/L2 regularization)
  rf       sklearn RandomForestRegressor       (fast sanity-check baseline)

Hyperparameter rationale for high-dimensional Betti / spectral grids
─────────────────────────────────────────────────────────────────────────────
All tree models use max_features='sqrt' (or equivalent), so the effective
number of candidate features per split is √D regardless of total D. This
is why MPH (10 K dims) already outperforms PH (450 dims): the model
explores the grid cheaply.

LightGBM extras that help the MP/MPTL case:
  - num_leaves=127 (deeper than max_depth=8 trees; captures grid interactions)
  - min_child_samples=10 (replaces min_samples_split; prevents overfitting)
  - feature_fraction (= max_features) per tree, not per split → faster
  - bagging_fraction + bagging_freq (= subsample)
  - reg_alpha (L1): drives uninformative grid cells to zero importance
  - reg_lambda (L2): smooths the remaining weights
  - EFB (exclusive feature bundling) is ON by default: auto-bundles
    mutually-exclusive grid cells, reducing effective dimensionality.

XGBoost extras:
  - reg_alpha (L1) + reg_lambda (L2): explicit sparsity on grid features
  - tree_method='hist': fast histogram-based splits (needed for 26 K dims)
  - colsample_bytree (= max_features per tree)

CNN note (for future graph-classification work on MUTAG/BZR):
  The Betti + spectral grids saved in grids/ID.npz can be reshaped to
  (n_channels, R, Q) images — 2 Betti + 3 spectral = 5 channels at 12×12.
  That tensor is already written by prepare_mp_features and is ready to
  feed a CNN without any change to the extraction pipeline.
"""
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config
from features import load_features
from utils import digest, file_hash, versions, write_json, write_npz, save_evaluation


# ─────────────────────────────────────────────────────────────────────────────
# Per-model default hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

def default_params(model_name: str) -> dict:
    """Return sensible defaults for each model in the high-dimensional MP setting.

    These are starting points, not exhaustive tuned values. All can be
    overridden via CLI flags in run_experiment.py.
    """
    if model_name == 'gbm':
        # Original paper settings (Tables 4 / 5).
        return dict(
            n_estimators=20000,
            max_depth=8,
            min_samples_split=6,
            learning_rate=0.005,
            loss='squared_error',
            subsample=0.7,
            max_features='sqrt',
        )

    if model_name == 'lgbm':
        # Tuned for high-D grids (MPH / MPTL).
        # EFB is ON by default → auto-bundles correlated grid cells.
        # num_leaves=127 allows richer trees than depth=8 alone.
        # reg_alpha drives sparse uninformative grid features to zero.
        return dict(
            n_estimators=5000,
            learning_rate=0.005,
            num_leaves=127,          # 2^7 - 1; richer than max_depth=8
            min_child_samples=10,    # min samples in a leaf
            feature_fraction=0.3,   # ≈ sqrt(26K)/26K; per-tree column sampling
            bagging_fraction=0.7,   # row subsampling (= subsample)
            bagging_freq=1,         # apply bagging every tree
            reg_alpha=0.1,          # L1: pushes uninformative grid cells to 0
            reg_lambda=1.0,         # L2: smooth weights
            n_jobs=-1,
            verbose=-1,
        )

    if model_name == 'xgboost':
        # Histogram splits for speed at 26 K dims.
        # L1 + L2 regularization controls grid feature sparsity.
        return dict(
            n_estimators=5000,
            learning_rate=0.005,
            max_depth=8,
            subsample=0.7,
            colsample_bytree=0.3,   # ≈ sqrt fraction; per-tree column sampling
            reg_alpha=0.1,          # L1
            reg_lambda=1.0,         # L2
            min_child_weight=5,     # min sum of instance weight in a child
            tree_method='hist',     # fast histogram algorithm (needed for 26 K)
            n_jobs=-1,
            verbosity=0,
        )

    if model_name == 'rf':
        # Fast sanity-check baseline; no boosting sequential bias.
        # Fewer trees than GBM because RF converges faster.
        return dict(
            n_estimators=1000,
            max_depth=None,         # fully grown trees
            min_samples_split=6,
            max_features='sqrt',
            n_jobs=-1,
        )

    raise ValueError(f'Unknown model: {model_name!r}. '
                     'Choose gbm | lgbm | xgboost | rf')


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_name: str, params: dict, seed: int):
    """Instantiate a regressor from its name and params dict."""
    if model_name == 'gbm':
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(**params, random_state=seed)

    if model_name == 'lgbm':
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            raise ImportError('LightGBM not installed. Run: pip install lightgbm')
        return LGBMRegressor(**params, random_state=seed)

    if model_name == 'xgboost':
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError('XGBoost not installed. Run: pip install xgboost')
        return XGBRegressor(**params, random_state=seed)

    if model_name == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**params, random_state=seed)

    raise ValueError(f'Unknown model: {model_name!r}')


def _get_importance(model, model_name: str, n_features: int) -> np.ndarray:
    """Extract feature importances uniformly across model types."""
    if model_name == 'lgbm':
        imp = model.booster_.feature_importance(importance_type='gain')
    elif model_name == 'xgboost':
        imp = model.feature_importances_        # gain-based by default
    else:
        imp = model.feature_importances_        # sklearn standard attribute
    imp = np.asarray(imp, dtype=float)
    if imp.shape[0] != n_features:
        # LightGBM with EFB may bundle features; pad to full length
        padded = np.zeros(n_features)
        padded[:imp.shape[0]] = imp
        return padded
    total = imp.sum()
    return imp / total if total > 0 else imp


# ─────────────────────────────────────────────────────────────────────────────
# Training entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_training(manifest_path, output_dir, settings, params, seeds,
                 model_name='gbm', save_models=True, save_barcodes=True):
    """Train one model per seed, evaluate on core-set, and write a summary.

    Parameters
    ----------
    manifest_path : Path   CSV with pdb_id, split, pKd columns.
    output_dir    : Path   Directory containing the features/ sub-folder.
    settings      : dict   Feature settings (stored for provenance).
    params        : dict   Model hyperparameters (model-specific).
    seeds         : list   Distinct integer random seeds.
    model_name    : str    'gbm' | 'lgbm' | 'xgboost' | 'rf'
    save_models   : bool   Persist .joblib files.
    save_barcodes : bool   Passed to load_features for signature matching.
    """
    table, X, y, names, signature = load_features(
        manifest_path, output_dir, settings, save_barcodes)
    train_mask = table.split.to_numpy() == 'train'
    test_mask  = table.split.to_numpy() == 'test'
    if train_mask.sum() < 2 or test_mask.sum() < 2:
        raise ValueError('Need at least two training and two test complexes')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Provide distinct random seeds')

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out    = Path(output_dir) / 'runs' / run_id
    out.mkdir(parents=True, exist_ok=False)

    table.to_csv(out / 'manifest.csv', index=False)
    write_npz(out / 'dataset.npz',
              X=X, y=y,
              ids=table.pdb_id.to_numpy(str),
              split=table.split.to_numpy(str),
              names=np.asarray(names),
              signature=np.asarray(signature))
    write_json(out / 'run_config.json', {
        'model': model_name,
        'feature_settings': settings,
        'feature_signature': signature,
        'model_params': params,
        'seeds': list(seeds),
        'versions': versions(),
        'n_train': int(train_mask.sum()),
        'n_test':  int(test_mask.sum()),
        'n_features': len(names),
        'target': 'PDBbind logarithmic affinity',
        'selection_policy': 'All seeds reported; no best-core-seed selection.',
    })

    snap = out / 'source_snapshot'
    snap.mkdir()
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copy2(source, snap / source.name)

    pd.DataFrame({'feature_index': range(len(names)),
                  'feature_name': names}).to_csv(out / 'feature_names.csv', index=False)

    reports    = []
    importance = np.zeros(len(names))

    print(f'\n[{model_name.upper()}] {len(names)} features, '
          f'{train_mask.sum()} train, {test_mask.sum()} test, '
          f'{len(seeds)} seed(s)', flush=True)

    for seed in seeds:
        print(f'  seed={seed} …', end=' ', flush=True)
        model = build_model(model_name, params, seed)
        model.fit(X[train_mask], y[train_mask])
        prediction = model.predict(X[test_mask])
        report = save_evaluation(
            table.pdb_id.to_numpy()[test_mask],
            y[test_mask], prediction, out, f'seed{seed}')
        reports.append({'seed': int(seed), **report})
        importance += _get_importance(model, model_name, len(names)) / len(seeds)

        if save_models:
            joblib.dump({
                'model': model,
                'model_name': model_name,
                'feature_names': names,
                'feature_signature': signature,
                'settings': settings,
                'seed': int(seed),
            }, out / f'model_seed{seed}.joblib', compress=3)
        pd.DataFrame(reports).to_csv(out / 'metrics_by_seed.csv', index=False)
        r = report.get('pearson_r')
        rmse = report.get('rmse_pKd')
        print(f'r={r:.4f}  RMSE={rmse:.4f}', flush=True)

    # ── Aggregate summary ──────────────────────────────────────────────────────
    summary = {}
    for key in ('pearson_r', 'spearman_rho', 'rmse_pKd', 'mae_pKd',
                'rmse_kcal_mol_at_298K'):
        v = np.asarray([r[key] for r in reports if r[key] is not None])
        summary[key] = {
            'mean':             float(v.mean()),
            'median':           float(np.median(v)),
            'std_population':   float(v.std()),
            'min':              float(v.min()),
            'max':              float(v.max()),
        } if len(v) else None
    write_json(out / 'summary.json', summary)

    print(f'\nSUMMARY [{model_name.upper()}] across {len(reports)} seed(s)')
    for metric, stats in summary.items():
        if stats is not None:
            print(f'  {metric}: {stats["mean"]:.4f} ± {stats["std_population"]:.4f}')

    pd.DataFrame({'feature_name': names, 'mean_gain_importance': importance})\
      .sort_values('mean_gain_importance', ascending=False)\
      .to_csv(out / 'feature_importances.csv', index=False)

    print(f'Complete: {out}', flush=True)
    return out
