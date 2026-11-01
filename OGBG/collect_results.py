"""Collect OGB experiment JSON results and print a formatted summary table.

Usage:
  python collect_results.py                        # reads ./ogb_results/
  python collect_results.py --results_dir /path/   # custom directory
  python collect_results.py --save table.csv       # also save CSV
  python collect_results.py --metric TestMean      # default

Output example:
  Dataset        Mode   Filtration  Direction  Test (mean±std)   Val (mean±std)
  ogbg-molbace   PH     hks         sub        75.23 ± 1.45      74.11 ± 1.32
  ogbg-molbace   PH     hks         both       76.10 ± 1.21      75.03 ± 1.18
  ...
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


MODES_ORDER = ['PH', 'PTL', 'MPH', 'MPTL']
DIRS_ORDER  = ['sub', 'both']
DATASETS_ORDER = [
    'ogbg-molbace', 'ogbg-molbbbp', 'ogbg-molclintox', 'ogbg-molsider',
    'ogbg-moltoxcast', 'ogbg-moltox21', 'ogbg-molhiv',
]


def load_results(results_dir: Path) -> list:
    records = []
    missing = []

    all_files = sorted(list(results_dir.glob('*.json')) +
                       list(results_dir.glob('*.pt')))
    if not all_files:
        print(f'No result files (.json or .pt) found in {results_dir}')
        return records, missing

    for path in all_files:
        try:
            if path.suffix == '.pt':
                import torch
                r = torch.load(path, weights_only=False)
            else:
                with open(path) as f:
                    r = json.load(f)
            records.append({
                'Dataset':    r.get('dataset', '?'),
                'Mode':       r.get('mode', '?'),
                'Model':      r.get('model', '?'),
                'Filtration': r.get('filtration', '?'),
                'EdgeFilt':   r.get('edge_filtration', '-'),
                'Direction':  r.get('direction', 'sub'),
                'Metric':     r.get('metric', '?'),
                'TestMean':   r.get('TestMean', float('nan')),
                'TestStd':    r.get('TestStd',  float('nan')),
                'ValMean':    r.get('ValMean',  float('nan')),
                'ValStd':     r.get('ValStd',   float('nan')),
                'TrainMean':  r.get('TrainMean',float('nan')),
                'Runs':       r.get('runs', '?'),
                'FeatShape':  str(r.get('feat_shape', '?')),
                'FeatTimeSec':r.get('feat_time_s', float('nan')),
                'File':       path.name,
            })
        except Exception as e:
            missing.append((path.name, str(e)))

    return records, missing


def build_table(records: list) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Sort by dataset, mode, direction
    mode_order = {m: i for i, m in enumerate(MODES_ORDER)}
    dir_order  = {d: i for i, d in enumerate(DIRS_ORDER)}
    ds_order   = {d: i for i, d in enumerate(DATASETS_ORDER)}

    df['_ds_rank']   = df['Dataset'].map(lambda x: ds_order.get(x, 99))
    df['_mode_rank'] = df['Mode'].map(lambda x: mode_order.get(x, 99))
    df['_dir_rank']  = df['Direction'].map(lambda x: dir_order.get(x, 99))
    df = df.sort_values(['_ds_rank', '_mode_rank', '_dir_rank'])
    df = df.drop(columns=['_ds_rank', '_mode_rank', '_dir_rank'])
    df = df.reset_index(drop=True)

    return df


def format_score(mean, std):
    if pd.isna(mean):
        return 'N/A'
    return f'{mean:.2f} ± {std:.2f}'


def print_table(df: pd.DataFrame):
    if df.empty:
        print('No results to display.')
        return

    # One table per dataset
    for dataset in df['Dataset'].unique():
        sub = df[df['Dataset'] == dataset].copy()
        metric = sub['Metric'].iloc[0]

        print(f'\n{"="*72}')
        print(f'  {dataset}   (metric: {metric})')
        print(f'{"="*72}')
        print(f'{"Mode":<8} {"Filtration":<12} {"Direction":<10} '
              f'{"Test (mean±std)":>18} {"Val (mean±std)":>18}')
        print('-' * 72)

        for _, row in sub.iterrows():
            print(f'{row["Mode"]:<8} {row["Filtration"]:<12} '
                  f'{row["Direction"]:<10} '
                  f'{format_score(row["TestMean"], row["TestStd"]):>18} '
                  f'{format_score(row["ValMean"], row["ValStd"]):>18}')

    # Overall best per dataset
    print(f'\n{"="*72}')
    print('  BEST TEST SCORE PER DATASET')
    print(f'{"="*72}')
    print(f'{"Dataset":<22} {"Mode":<8} {"Direction":<10} {"Best Test":>12}')
    print('-' * 72)
    for dataset in df['Dataset'].unique():
        sub = df[df['Dataset'] == dataset]
        best_row = sub.loc[sub['TestMean'].idxmax()]
        print(f'{dataset:<22} {best_row["Mode"]:<8} '
              f'{best_row["Direction"]:<10} '
              f'{format_score(best_row["TestMean"], best_row["TestStd"]):>12}')


def save_csv(df: pd.DataFrame, path: str):
    """Save a clean CSV with just the key columns."""
    out = df[[
        'Dataset', 'Mode', 'Model', 'Filtration', 'EdgeFilt',
        'Direction', 'Metric', 'Runs',
        'TestMean', 'TestStd', 'ValMean', 'ValStd', 'TrainMean',
        'FeatShape', 'FeatTimeSec',
    ]].copy()
    out['Test'] = out.apply(
        lambda r: format_score(r['TestMean'], r['TestStd']), axis=1)
    out['Val']  = out.apply(
        lambda r: format_score(r['ValMean'],  r['ValStd']),  axis=1)
    out.to_csv(path, index=False)
    print(f'\nTable saved to {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results_dir', type=Path,
                        default=Path('./ogb_results'),
                        help='Directory containing result JSON files')
    parser.add_argument('--save', type=str, default='',
                        help='Save table to this CSV file path')
    parser.add_argument('--show_missing', action='store_true',
                        help='Print files that could not be loaded')
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f'Results directory not found: {args.results_dir}')
        print('Run submit_ogb.sh first, then collect after jobs finish.')
        return

    records, missing = load_results(args.results_dir)

    if missing:
        print(f'\nWARNING: {len(missing)} file(s) could not be loaded:')
        if args.show_missing:
            for fname, err in missing:
                print(f'  {fname}: {err}')
        else:
            print('  Run with --show_missing for details.')

    # Check for missing expected jobs
    expected = [
        f'{ds.split("-")[1]}_{mode}_{filt}_{direc}.json'
        for ds in DATASETS_ORDER
        for mode in MODES_ORDER
        for filt in ['atom_norm', 'hks']   # approximate; varies by dataset
        for direc in DIRS_ORDER
    ]
    found_files = {r['File'] for r in records}
    not_done = [e for e in expected if e not in found_files]
    if not_done:
        print(f'\n{len(not_done)} expected result(s) not yet found '
              f'(jobs may still be running).')

    if not records:
        return

    df = build_table(records)
    print_table(df)

    if args.save:
        save_csv(df, args.save)


if __name__ == '__main__':
    main()
