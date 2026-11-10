"""Shared utilities: provenance hashing, atomic file output, metrics and evaluation plots."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


# ── Hashing / provenance ───────────────────────────────────────────────────────

def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def versions():
    import platform
    result = {'python': platform.python_version()}
    for name in ['numpy', 'scipy', 'pandas', 'scikit-learn',
                 'ripser', 'gudhi', 'rdkit', 'joblib']:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = 'not installed'
    return result


# ── Atomic file I/O ────────────────────────────────────────────────────────────

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(temp, path)


# ── Metrics and evaluation ─────────────────────────────────────────────────────

def metrics(y_true, y_pred):
    from scipy.stats import pearsonr, spearmanr
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all() or not len(y):
        raise ValueError('Targets and predictions must be nonempty, finite, matching shapes')
    variable = len(y) >= 2 and np.ptp(y) > 0 and np.ptp(p) > 0
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    return {
        'n': len(y),
        'pearson_r':          float(pearsonr(y, p).statistic)  if variable else None,
        'spearman_rho':       float(spearmanr(y, p).statistic) if variable else None,
        'rmse_pKd':           rmse,
        'mae_pKd':            float(np.mean(np.abs(p - y))),
        'rmse_kcal_mol_at_298K': 1.3633 * rmse,
    }


def save_evaluation(ids, y, pred, directory, prefix):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = metrics(y, pred)
    pd.DataFrame({
        'pdb_id': ids,
        'experimental_pKd': y,
        'predicted_pKd': pred,
        'residual_pred_minus_true': np.asarray(pred) - np.asarray(y),
    }).to_csv(directory / f'{prefix}_predictions.csv', index=False)
    write_json(directory / f'{prefix}_metrics.json', report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(y, pred, s=20, alpha=0.7, color='#176b87')
    lo = min(np.min(y), np.min(pred)) - 0.3
    hi = max(np.max(y), np.max(pred)) + 0.3
    ax.plot([lo, hi], [lo, hi], '--', color='gray', linewidth=1)
    ax.set(xlabel='Experimental pKd', ylabel='Predicted pKd', xlim=(lo, hi), ylim=(lo, hi))
    r = report['pearson_r']
    title = (f'r = {r:.3f}, RMSE = {report["rmse_pKd"]:.3f}' if r is not None
             else f'RMSE = {report["rmse_pKd"]:.3f}')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(directory / f'{prefix}_scatter.png', dpi=180)
    plt.close(fig)
    return report
