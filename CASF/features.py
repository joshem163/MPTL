"""Feature extraction for all four experiment modes.

Feature composition per mode
─────────────────────────────────────────────────────────────────────────────
All four modes share the same 126-dimensional "other" baseline features:
  - global kernel PH:  54 features  (H0–H2 birth/lifetime sums, 3 kernels × 3 roles)
  - alpha complex PH:  72 features  (H1/H2 lifetime-birth histograms, 2 groups × 3 roles)

On top of that each mode adds its own pair-level features:

  PH  (single, no spectral)
    + 324 pair_ features: binned H0 deaths + kernel lifetime sums, 36 element pairs
    Total: 324 + 126 = 450

  PTL (single, --spectral)
    + 324 pair_ features (same as PH)
    + 36 × n_spectral_ch single-parameter Laplacian scalars (one per element pair)
      default: 3 channels → 108 spectral features
    Total: 324 + 108 + 126 = 558

  MPH (multi, no spectral)
    + 36 × 2 × R × Q  MP Betti features  (H0 + H1 grids)
      default 12×12 grid → 36 × 2 × 144 = 10 368
    Total: 10 368 + 126 = 10 494

  MPTL (multi, --spectral)
    + 36 × 2 × R × Q  MP Betti features  (H0 + H1 grids)
    + 36 × n_spectral_ch × R × Q  MP spectral features
      default 12×12, 3 ch → 36 × 3 × 144 = 15 552
    Total: 10 368 + 15 552 + 126 = 26 046

The 126-dimensional "other" block is always appended at the end of the
feature vector in all four modes. The pair_ features of the PH baseline
are dropped in MPH/MPTL and replaced by the corresponding MP grids.

Entry points
─────────────────────────────────────────────────────────────────────────────
  extract_baseline(manifest, output_dir, settings, workers, save_barcodes)
      → Extracts PH (and PTL spectral scalars when use_spectral=True).
        Always writes the full 450- (or 558-) dim vector including the 126.

  prepare_mp_features(baseline_dir, mp_dir, use_spectral)
      → Reads the baseline, drops pair_ features, prepends MP grids,
        appends the retained 126-dim "other" block.

  load_features(manifest_path, output_dir, settings, save_barcodes)
      → (table, X, y, names, signature)  used by train.py
"""
import copy
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import config
from data_loader import read_protein, read_ligand, combine
from topology import (
    nearby, interaction_deaths, kernel_distance, kernel_diagrams,
    alpha_diagrams, global_summary, alpha_summary,
    pair_graphs, fit_degree_axes,
    vectorize_betti, vectorize_spectral,
    spectral_grid, SPECTRAL_CHANNEL_NAMES,
)
from utils import digest, file_hash, versions, write_json, write_npz


# ─────────────────────────────────────────────────────────────────────────────
# Signature helpers
# ─────────────────────────────────────────────────────────────────────────────

def extraction_signature(settings, save_barcodes):
    root = Path(__file__).parent
    return digest({
        'settings': settings,
        'save_barcodes': save_barcodes,
        'versions': versions(),
        'source': {name: file_hash(root / name)
                   for name in ['features.py', 'topology.py',
                                'data_loader.py', 'utils.py']},
    })


def _inputs_for(row):
    return {name: {'path': row[name], 'sha256': file_hash(row[name])}
            for name in ('protein_path', 'mol2_path', 'sdf_path') if row.get(name)}


def _kernel_tag(spec):
    return f'{spec["kind"]}_tau{spec["tau"]:g}_p{spec["power"]:g}'


# ─────────────────────────────────────────────────────────────────────────────
# PH baseline feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_atoms(protein, ligand, settings, save_barcodes=True, use_spectral=False):
    """Extract features for one complex.

    Returns (x, names, bars) where:
      - x      : 1D float64 feature vector
      - names  : list of feature name strings matching x
      - bars   : dict of barcode arrays (only populated when save_barcodes=True)

    Feature layout
    ──────────────
    PH  (use_spectral=False):
      [pair_ block: 324 features] + [other block: 126 features]

    PTL (use_spectral=True):
      [pair_ block: 324 features]
      + [spectral block: 36 pairs × n_ch scalars]
      + [other block: 126 features]

    The spectral block computes one Laplacian spectrum per element pair using
    the FULL distance graph at pair_cutoff (single-parameter, no degree filter).
    Channels: lambda_min_positive, lambda_max, lambda_mean_positive.
    """
    values, names, bars = [], [], {}
    radii = settings['vdw_radii']
    bins  = settings['distance_bins']

    def add(prefix, vector, labels):
        if len(vector) != len(labels):
            raise ValueError('Internal schema length mismatch')
        names.extend(f'{prefix}/{label}' for label in labels)
        values.extend(vector)

    def record(prefix, diagrams):
        if save_barcodes:
            for dim, diagram in enumerate(diagrams):
                bars[f'{prefix}__H{dim}'] = np.asarray(diagram).reshape(-1, 2)

    # ── pair_ block (PH interaction features, 324 dims) ───────────────────────
    for pe in settings['protein_elements']:
        for le in settings['ligand_elements']:
            l = ligand.select(ligand.elements == le)
            p = nearby(protein.select(protein.elements == pe), l, settings['pair_cutoff'])
            deaths = interaction_deaths(p.xyz, l.xyz)
            prefix = f'pair_{pe}_{le}'
            add(prefix + '/euclidean',
                np.histogram(deaths, bins=bins)[0],
                [f'death_bin_{i}_{a:g}_{b:g}'
                 for i, (a, b) in enumerate(zip(bins[:-1], bins[1:]))])
            dgm = np.column_stack([np.zeros(len(deaths)), deaths])
            essential_count = 1 if len(p) and len(l) else len(p) + len(l)
            if essential_count:
                dgm = np.vstack([dgm, np.tile([0, np.inf], (essential_count, 1))])
            record(prefix + '__euclidean', [dgm])
            for spec in settings['kernels']:
                kd  = kernel_distance(deaths, radii[pe] + radii[le], spec)
                tag = _kernel_tag(spec)
                add(prefix + '/' + tag, [kd.sum()], ['finite_H0_lifetime_sum'])
                diagram = np.column_stack([np.zeros(len(kd)), kd])
                if essential_count:
                    diagram = np.vstack([diagram, np.tile([0, np.inf], (essential_count, 1))])
                record(prefix + '__' + tag, [diagram])

    # ── spectral block (PTL only, 36 × n_ch dims) ─────────────────────────────
    if use_spectral:
        for pe in settings['protein_elements']:
            for le in settings['ligand_elements']:
                l = ligand.select(ligand.elements == le)
                p = nearby(protein.select(protein.elements == pe),
                           l, settings['pair_cutoff'])
                if len(p) == 0 or len(l) == 0:
                    # No atoms for this pair: all spectral channels = 0
                    for ch in SPECTRAL_CHANNEL_NAMES:
                        add(f'ptl_{pe}_{le}/{ch}', [0.0], ['scalar'])
                    continue
                from scipy.spatial.distance import cdist as _cdist
                dist_mat = _cdist(p.xyz, l.xyz)
                # Single-parameter: admit all vertices, vary only distance.
                # Use a single threshold = pair_cutoff (the full graph).
                edges = dist_mat <= settings['pair_cutoff']
                from topology import _spectral_stats
                lmin, lmax, lmean, valid = _spectral_stats(edges)
                for ch, val in zip(SPECTRAL_CHANNEL_NAMES, [lmin, lmax, lmean]):
                    add(f'ptl_{pe}_{le}/{ch}', [val], ['scalar'])

    # ── other block (global kernel + alpha, 126 dims) ─────────────────────────
    if settings['include_global_kernel']:
        p = nearby(protein, ligand, settings['kernel_cutoff'])
        c = combine(p, ligand)
        labels = [f'H{k}_{stat}' for k in range(3)
                  for stat in ('birth_sum', 'lifetime_sum')]
        for spec in settings['kernels']:
            tag = _kernel_tag(spec)
            dp  = kernel_diagrams(p, radii, spec, settings['max_global_kernel_atoms'])
            dc  = kernel_diagrams(c, radii, spec, settings['max_global_kernel_atoms'])
            sp, sc = global_summary(dp), global_summary(dc)
            for role, summary in [('protein', sp), ('complex', sc),
                                   ('complex_minus_protein', sc - sp)]:
                add(f'global_kernel/{tag}/{role}', summary, labels)
            record('global_kernel__' + tag + '__protein', dp)
            record('global_kernel__' + tag + '__complex', dc)

    abins  = settings['alpha_birth_bins']
    alabel = [f'H{k}_lifetime_birth_bin_{i}_{a:g}_{b:g}'
              for k in (1, 2)
              for i, (a, b) in enumerate(zip(abins[:-1], abins[1:]))]
    selections = []
    if settings['include_alpha']:
        selections.append(('all', protein, ligand))
    if settings['include_carbon_alpha']:
        selections.append(('carbon',
                           protein.select(protein.elements == 'C'),
                           ligand.select(ligand.elements == 'C')))
    for group, p0, l in selections:
        p = nearby(p0, l, settings['alpha_cutoff'])
        c = combine(p, l)
        dp = alpha_diagrams(p, settings['alpha_units'])
        dc = alpha_diagrams(c, settings['alpha_units'])
        sp, sc = alpha_summary(dp, abins), alpha_summary(dc, abins)
        for role, summary in [('protein', sp), ('complex', sc),
                               ('complex_minus_protein', sc - sp)]:
            add(f'alpha/{group}/{role}', summary, alabel)
        record(f'alpha__{group}__protein', dp)
        record(f'alpha__{group}__complex', dc)

    x = np.asarray(values, dtype=np.float64)
    if len(names) != len(set(names)) or not np.isfinite(x).all():
        raise ValueError('Feature names must be unique and values finite')
    return x, names, bars


def extract_one(row, output_dir, settings, signature, save_barcodes=True,
                use_spectral=False):
    start  = time.monotonic()
    output = Path(output_dir) / 'features'
    key    = row['pdb_id']
    path   = output / f'{key}.npz'
    inputs = _inputs_for(row)
    cache_key = digest({'signature': signature, 'inputs': inputs})
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if (str(old['cache_key']) == cache_key
                    and (output / f'{key}.json').is_file()
                    and (not save_barcodes
                         or (output / f'{key}_barcodes.npz').is_file())):
                return {'pdb_id': key, 'status': 'cached',
                        'seconds': time.monotonic() - start}
    protein = read_protein(row['protein_path'], settings['include_protein_hetatm'])
    ligand, ligand_path, parse_warnings = read_ligand(row, settings['ligand_preference'])
    x, names, bars = extract_atoms(protein, ligand, settings,
                                   save_barcodes, use_spectral)
    if save_barcodes:
        write_npz(output / f'{key}_barcodes.npz', **bars)
    write_json(output / f'{key}.json', {
        'pdb_id': key, 'inputs': inputs, 'ligand_used': ligand_path,
        'parse_warnings': parse_warnings, 'settings': settings,
        'use_spectral': use_spectral,
        'signature': signature, 'cache_key': cache_key,
        'n_features': len(x), 'seconds': time.monotonic() - start,
    })
    write_npz(path, x=x, names=np.asarray(names),
              cache_key=np.asarray(cache_key), signature=np.asarray(signature))
    return {'pdb_id': key, 'status': 'computed',
            'seconds': time.monotonic() - start}


def extract_baseline(manifest_path, output_dir, settings,
                     workers=1, save_barcodes=True, limit=None,
                     use_spectral=False):
    """Extract PH (or PTL when use_spectral=True) features. Safe to re-run (cached).

    use_spectral=True adds 36 × n_ch single-parameter Laplacian scalars (PTL mode).
    The 126-dim 'other' block is always included regardless of use_spectral.
    """
    table = pd.read_csv(manifest_path, dtype={'pdb_id': str}, keep_default_na=False)
    if table.pdb_id.duplicated().any():
        raise ValueError('Duplicate complex IDs in manifest')
    rows = table.to_dict('records')
    if limit is not None:
        rows = rows[:limit]

    # Encode use_spectral into settings so the signature changes between PH and PTL.
    effective_settings = copy.deepcopy(settings)
    effective_settings['_use_spectral_ptl'] = bool(use_spectral)
    signature  = extraction_signature(effective_settings, save_barcodes)
    results, errors = [], []

    def handle(row, result=None, error=None):
        if error is not None:
            errors.append({'pdb_id': row['pdb_id'], 'error': str(error),
                           'traceback': ''.join(traceback.format_exception(error))})
            print(f'FAILED {row["pdb_id"]}: {error}', flush=True)
        else:
            results.append(result)
            print(f'[{len(results)+len(errors)}/{len(rows)}] '
                  f'{result["pdb_id"]}: {result["status"]} '
                  f'({result["seconds"]:.1f}s)', flush=True)
        write_json(Path(output_dir) / 'extraction_report.json',
                   {'signature': signature, 'results': results, 'errors': errors})

    if workers == 1:
        for row in rows:
            print(f'Processing {row["pdb_id"]}...', flush=True)
            try:
                handle(row, extract_one(row, output_dir, effective_settings,
                                        signature, save_barcodes, use_spectral))
            except Exception as exc:
                handle(row, error=exc)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(extract_one, r, output_dir, effective_settings,
                                   signature, save_barcodes, use_spectral): r
                       for r in rows}
            for future in as_completed(futures):
                row = futures[future]
                try:
                    handle(row, future.result())
                except Exception as exc:
                    handle(row, error=exc)
    if errors:
        raise RuntimeError(
            f'{len(errors)} complexes failed. See extraction_report.json.')
    write_json(Path(output_dir) / 'feature_settings.json',
               {'settings': effective_settings, 'signature': signature,
                'use_spectral': use_spectral, 'versions': versions()})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Feature loading (used by train.py for all modes)
# ─────────────────────────────────────────────────────────────────────────────

def load_features(manifest_path, output_dir, settings, save_barcodes=True):
    """Load pre-extracted feature matrices. Returns (table, X, y, names, signature)."""
    table = pd.read_csv(manifest_path, dtype={'pdb_id': str}, keep_default_na=False)
    if table.pdb_id.duplicated().any() or not set(table.split).issubset({'train', 'test'}):
        raise ValueError('Manifest has duplicate IDs or invalid split labels')
    signature = extraction_signature(settings, save_barcodes)
    vectors, names = [], None
    for row in table.to_dict('records'):
        path = Path(output_dir) / 'features' / f'{row["pdb_id"]}.npz'
        if not path.exists():
            raise FileNotFoundError(f'Missing {path}; extract ALL complexes before training.')
        inputs   = _inputs_for(row)
        expected = digest({'signature': signature, 'inputs': inputs})
        with np.load(path, allow_pickle=False) as d:
            if str(d['cache_key']) != expected or str(d['signature']) != signature:
                raise ValueError(f'Stale features for {row["pdb_id"]}; rerun extraction.')
            current = d['names'].tolist()
            if names is None:
                names = current
            if current != names:
                raise ValueError(f'Feature schema mismatch for {row["pdb_id"]}')
            vectors.append(d['x'].copy())
    X = np.asarray(vectors, dtype=float)
    y = pd.to_numeric(table.pKd, errors='raise').to_numpy(float)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite features or labels')
    return table, X, y, names, signature


# ─────────────────────────────────────────────────────────────────────────────
# MP feature preparation (MPH and MPTL)
# ─────────────────────────────────────────────────────────────────────────────

def _mp_request_id(baseline_dir, use_spectral):
    """Build the request dict and its digest for cache invalidation."""
    base = Path(baseline_dir).resolve()
    r    = np.asarray([config.REFERENCE_CUTOFF * i / (config.GRID_SIZE - 1)
                       for i in range(config.GRID_SIZE)], float)
    # Load from baseline PH (not PTL): pair_ features are always present there.
    ph_settings = copy.deepcopy(config.FEATURES)
    ph_settings['_use_spectral_ptl'] = False
    table, X, y, names, signature = load_features(
        base / 'manifest.csv', base, ph_settings, config.SAVE_BARCODES)
    root = Path(__file__).parent
    channel_info = {
        'channels': list(SPECTRAL_CHANNEL_NAMES),
        'atol': config.SPECTRAL_ATOL,
        'rtol': config.SPECTRAL_RTOL,
        'max_component_vertices': config.SPECTRAL_MAX_COMPONENT_SIZE,
    } if use_spectral else {}
    request = {
        'method':                 ('MPTL_distance_x_degree_v1' if use_spectral
                                   else 'MPH_distance_x_degree_v1'),
        'spectral':               channel_info,
        'baseline_signature':     signature,
        'baseline_manifest_hash': file_hash(base / 'manifest.csv'),
        'distance_axis_angstrom': r.tolist(),
        'degree_grid_size':       config.GRID_SIZE,
        'reference_cutoff':       float(config.REFERENCE_CUTOFF),
        'reference_kernel':       config.REFERENCE_KERNEL,
        'keep_other_features':    True,   # always True; 126-dim block always retained
        'code_hashes': {name: file_hash(root / name)
                        for name in ['features.py', 'topology.py']},
    }
    return table, X, names, request, digest(request)


def prepare_mp_features(baseline_dir, mp_dir, use_spectral=False):
    """Fit degree axes on training data and extract MPH or MPTL features.

    The 126-dim 'other' (global kernel + alpha) block from the PH baseline is
    always appended to the MP feature vector in both MPH and MPTL modes.
    The 324-dim pair_ block from the baseline is replaced by MP Betti grids
    (and spectral grids in MPTL).

    Parameters
    ----------
    baseline_dir : Path   PH baseline output directory (NOT PTL).
    mp_dir       : Path   Output directory for MP features.
    use_spectral : bool   False → MPH, True → MPTL.
    """
    base = Path(baseline_dir).resolve()
    out  = Path(mp_dir).resolve()
    if base == out or out in base.parents or base in out.parents:
        raise ValueError('mp_dir must be separate and non-nested from baseline_dir')

    distance_axis = np.asarray(
        [config.REFERENCE_CUTOFF * i / (config.GRID_SIZE - 1)
         for i in range(config.GRID_SIZE)], float)

    table, baseline_X, baseline_names, request, request_id = \
        _mp_request_id(base, use_spectral)
    if set(table.split) != {'train', 'test'}:
        raise ValueError('Baseline manifest must contain train and test splits')

    out.mkdir(parents=True, exist_ok=True)
    state_path = out / 'feature_settings.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else None

    if state is not None and state.get('request_id') == request_id:
        axes = {pair: np.asarray(axis)
                for pair, axis in
                state['settings']['mp_addon']['degree_thresholds'].items()}
        print('Reusing training-fitted weighted-degree axes.', flush=True)
    else:
        pairs   = [f'{a}_{b}' for a in config.FEATURES['protein_elements']
                   for b in config.FEATURES['ligand_elements']]
        samples = {pair: [] for pair in pairs}
        train_rows = table.loc[table.split == 'train'].to_dict('records')
        print(f'Fitting degree axes on {len(train_rows)} training complexes…',
              flush=True)
        for i, row in enumerate(train_rows, 1):
            p = read_protein(row['protein_path'],
                             config.FEATURES['include_protein_hetatm'])
            l, _, _ = read_ligand(row, config.FEATURES['ligand_preference'])
            for pair, _, strength in pair_graphs(
                    p, l, config.FEATURES,
                    config.REFERENCE_CUTOFF, config.REFERENCE_KERNEL):
                samples[pair].append(strength)
            if i == 1 or i % 50 == 0 or i == len(train_rows):
                print(f'  Axis fitting {i}/{len(train_rows)}', flush=True)
        axes = fit_degree_axes(samples, config.GRID_SIZE)

    settings = copy.deepcopy(config.FEATURES)
    settings['mp_addon'] = {
        **request,
        'degree_thresholds': {k: a.tolist() for k, a in axes.items()},
    }
    signature = extraction_signature(settings, False)

    # Retain only the non-pair_ baseline features (126-dim "other" block).
    # The pair_ features are replaced by the MP Betti (and spectral) grids.
    keep           = np.asarray([not n.startswith('pair_') for n in baseline_names])
    retained_names = np.asarray(baseline_names)[keep].tolist()
    n_retained     = int(keep.sum())

    if not any(n.startswith('pair_') for n in baseline_names):
        raise ValueError('Baseline is missing expected pair_ feature names')
    if n_retained == 0:
        raise ValueError('No non-pair_ features found in baseline; '
                         'expected ~126 global_kernel/alpha features.')

    mode_label = 'MPTL' if use_spectral else 'MPH'
    n_pairs    = len(axes)
    n_betti_ch = 2
    n_spec_ch  = len(SPECTRAL_CHANNEL_NAMES) if use_spectral else 0
    n_mp       = n_pairs * (n_betti_ch + n_spec_ch) * config.GRID_SIZE * config.GRID_SIZE

    print(f'[{mode_label}] Feature layout per complex:', flush=True)
    print(f'  MP grids   : {n_pairs} pairs × '
          f'({n_betti_ch} Betti + {n_spec_ch} spectral) × '
          f'{config.GRID_SIZE}×{config.GRID_SIZE} = {n_mp}', flush=True)
    print(f'  Other (126): {n_retained} retained from baseline', flush=True)
    print(f'  Total      : {n_mp + n_retained}', flush=True)

    table.to_csv(out / 'manifest.csv', index=False)
    write_json(state_path, {
        'request_id':             request_id,
        'settings':               settings,
        'signature':              signature,
        'retained_feature_names': retained_names,
        'n_retained':             n_retained,
        'constant_degree_axes':   [p for p, a in axes.items() if np.ptp(a) == 0],
        'use_spectral':           use_spectral,
        'spectral_channels':      list(SPECTRAL_CHANNEL_NAMES) if use_spectral else [],
        'feature_layout': {
            'mp_features':       n_mp,
            'retained_other':    n_retained,
            'total':             n_mp + n_retained,
        },
    })

    results, errors = [], []
    for index, row in enumerate(table.to_dict('records')):
        key         = row['pdb_id']
        path        = out / 'features' / f'{key}.npz'
        tensor_path = out / 'grids'    / f'{key}.npz'
        start       = time.monotonic()
        try:
            inputs    = _inputs_for(row)
            cache_key = digest({'signature': signature, 'inputs': inputs})
            cached    = False
            if (path.exists() and tensor_path.exists()
                    and path.with_suffix('.json').exists()):
                with np.load(path, allow_pickle=False) as old:
                    cached = str(old['cache_key']) == cache_key

            if not cached:
                p = read_protein(row['protein_path'],
                                 config.FEATURES['include_protein_hetatm'])
                l, _, _ = read_ligand(row, config.FEATURES['ligand_preference'])
                graphs  = list(pair_graphs(p, l, config.FEATURES,
                                           config.REFERENCE_CUTOFF,
                                           config.REFERENCE_KERNEL))

                if use_spectral:
                    xmp, mpnames, b_tensor, s_tensor, v_tensor = \
                        vectorize_spectral(graphs, axes, distance_axis)
                    write_npz(tensor_path,
                              betti=b_tensor,
                              lambda_min=s_tensor[:, 0],
                              lambda_max=s_tensor[:, 1],
                              lambda_mean=s_tensor[:, 2],
                              spectral_valid=v_tensor,
                              channel_names=np.asarray(SPECTRAL_CHANNEL_NAMES),
                              pair_names=np.asarray(list(axes)),
                              distance_axis=distance_axis,
                              degree_thresholds=np.asarray(
                                  [axes[p] for p in axes]),
                              cache_key=np.asarray(cache_key))
                else:
                    xmp, mpnames, b_tensor = \
                        vectorize_betti(graphs, axes, distance_axis)
                    write_npz(tensor_path,
                              betti=b_tensor,
                              pair_names=np.asarray(list(axes)),
                              distance_axis=distance_axis,
                              degree_thresholds=np.asarray(
                                  [axes[p] for p in axes]),
                              cache_key=np.asarray(cache_key))

                # Append the retained 126-dim "other" block
                x     = np.concatenate([xmp, baseline_X[index, keep]])
                names = mpnames + retained_names
                if not np.isfinite(x).all() or len(set(names)) != len(names):
                    raise ValueError('Feature values or names invalid')
                write_json(path.with_suffix('.json'), {
                    'pdb_id':            key,
                    'cache_key':         cache_key,
                    'signature':         signature,
                    'mode':              mode_label,
                    'n_mp_features':     len(xmp),
                    'n_retained_other':  n_retained,
                    'n_features':        len(x),
                    'grid_shape':        list(b_tensor.shape),
                    'spectral_channels': (list(SPECTRAL_CHANNEL_NAMES)
                                          if use_spectral else []),
                    'seconds':           time.monotonic() - start,
                })
                write_npz(path, x=x, names=np.asarray(names),
                          cache_key=np.asarray(cache_key),
                          signature=np.asarray(signature))

            status = 'cached' if cached else 'computed'
            results.append({'pdb_id': key, 'status': status,
                             'seconds': time.monotonic() - start})
            print(f'[{index+1}/{len(table)}] {key}: {status} '
                  f'({time.monotonic()-start:.2f}s)', flush=True)
        except Exception as exc:
            errors.append({'pdb_id': key, 'error': str(exc)})
            print(f'FAILED {key}: {exc}', flush=True)
        write_json(out / 'extraction_report.json',
                   {'results': results, 'errors': errors,
                    'signature': signature, 'mode': mode_label})

    if errors:
        raise RuntimeError(
            f'{len(errors)} complexes failed; '
            f'see {out / "extraction_report.json"}')

    print(f'[{mode_label}] Done: {n_mp} MP + {n_retained} other = '
          f'{n_mp + n_retained} features per complex.', flush=True)
    return out


def checked_mp_settings(mp_dir, baseline_dir, use_spectral):
    """Verify MP settings have not changed before training. Returns settings dict."""
    _, _, _, request, request_id = _mp_request_id(
        Path(baseline_dir).resolve(), use_spectral)
    path = Path(mp_dir) / 'feature_settings.json'
    if not path.exists():
        raise FileNotFoundError('Run prepare_mp_features before training.')
    state = json.loads(path.read_text())
    if state['request_id'] != request_id:
        raise ValueError(
            'MP settings or baseline data changed. Rerun prepare_mp_features.')
    if state.get('use_spectral', False) != use_spectral:
        raise ValueError(
            f'Stored features used spectral={state["use_spectral"]} '
            f'but requested spectral={use_spectral}. '
            'Rerun with the correct --spectral flag.')
    return state['settings']
