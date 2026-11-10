"""Training script for OGB molecular graph datasets.

Supports all four topological feature modes × all model types.
Model selection is based on validation set performance (OGB evaluator).

Usage examples
──────────────
  # PH with MLP
  python train_ogb.py --dataset ogbg-molbace --mode PH --model mlp

  # PTL with MLP
  python train_ogb.py --dataset ogbg-molbace --mode PTL --model mlp

  # MPH with CNN
  python train_ogb.py --dataset ogbg-molbace --mode MPH --model cnn

  # MPTL with CNN (recommended)
  python train_ogb.py --dataset ogbg-molbace --mode MPTL --model cnn

  # MPTL with MLP (flat)
  python train_ogb.py --dataset ogbg-molbace --mode MPTL --model mlp

  # Change filtration
  python train_ogb.py --dataset ogbg-molbace --mode PH --model mlp --filtration atomN
  python train_ogb.py --dataset ogbg-molbace --mode PTL --model mlp --filtration hks
"""

import argparse
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# ── PyTorch 2.6 compatibility fix ─────────────────────────────────────────────
# torch.load changed default weights_only=True in 2.6, breaking OGB dataset
# loading which uses torch_geometric classes. Patch it back to False.
if tuple(int(x) for x in torch.__version__.split('.')[:2]) >= (2, 6):
    _original_torch_load = torch.load
    def _patched_torch_load(f, *args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _original_torch_load(f, *args, **kwargs)
    torch.load = _patched_torch_load
# ──────────────────────────────────────────────────────────────────────────────

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch.utils.data import DataLoader, Dataset

from topo_features_ogb import extract_all_features
from models_ogb import MLP, CNN2D, HybridCNN, build_model


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer_problem_type(dataset) -> str:
    task_type = dataset.task_type.lower()
    if 'binary' in task_type or dataset.num_tasks > 1:
        return 'multi-label'
    return 'single-label'


# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TopoDataset(Dataset):
    """Wraps precomputed topological feature tensors and labels."""
    def __init__(self, features, labels):
        self.features = features
        self.labels   = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def collate_fn(batch):
    """Stack feature tensors and labels from a list of (feat, label) pairs."""
    feats, labels = zip(*batch)
    return torch.stack(feats), torch.stack(labels)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(logits, labels, problem_type: str) -> torch.Tensor:
    if problem_type == 'single-label':
        return F.nll_loss(logits, labels.view(-1).long())

    labels = labels.float()
    if labels.ndim == 1:
        labels = labels.unsqueeze(-1)
    mask = ~torch.isnan(labels)
    if mask.sum() == 0:
        return logits.sum() * 0.0
    labels_clean = torch.where(mask, labels, torch.zeros_like(labels))
    loss_mat = F.binary_cross_entropy_with_logits(
        logits, labels_clean, reduction='none')
    return (loss_mat * mask.float()).sum() / mask.float().sum()


# ─────────────────────────────────────────────────────────────────────────────
# Train / evaluate
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, problem_type):
    model.train()
    total_loss, total_n = 0.0, 0

    for feats, labels in loader:
        feats  = feats.to(device)
        labels = labels.to(device)
        if labels.ndim == 3 and labels.size(1) == 1:
            labels = labels.squeeze(1)

        optimizer.zero_grad()

        # CNN expects (B, C, R, Q); MLP accepts both flat and grid (flattens internally)
        if isinstance(model, (CNN2D, HybridCNN)):
            logits = model(feats)
        else:
            logits = model(feats)

        loss = compute_loss(logits, labels, problem_type)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * feats.size(0)
        total_n    += feats.size(0)

    return total_loss / total_n


@torch.no_grad()
def evaluate_epoch(model, loader, device, evaluator,
                   problem_type, metric_name):
    model.eval()
    y_true, y_pred = [], []

    for feats, labels in loader:
        feats  = feats.to(device)
        labels = labels.to(device)
        if labels.ndim == 3 and labels.size(1) == 1:
            labels = labels.squeeze(1)

        logits = model(feats)

        if problem_type == 'single-label':
            pred = logits.argmax(dim=-1, keepdim=True)
            y_true.append(labels.view(-1, 1).cpu())
            y_pred.append(pred.cpu())
        else:
            if labels.ndim == 1:
                labels = labels.unsqueeze(-1)
            y_true.append(labels.cpu())
            y_pred.append(logits.cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    return evaluator.eval({'y_true': y_true, 'y_pred': y_pred})[metric_name]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Topological feature MLP/CNN for OGB datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    # Dataset
    parser.add_argument('--dataset',   type=str, default='ogbg-molbbbp')
    parser.add_argument('--root',      type=str, default='dataset',
                        help='Root directory where OGB datasets are stored')
    parser.add_argument('--device',    type=int, default=0)

    # Mode and model
    parser.add_argument('--mode',  choices=['PH', 'PTL', 'MPH', 'MPTL'],
                        default='MPTL',
                        help='Topological feature mode')
    parser.add_argument('--model', choices=['mlp', 'cnn'],
                        default='mlp',
                        help='Model type (cnn recommended for MPH/MPTL)')
    parser.add_argument('--filtration',
                        choices=['atomN', 'atom', 'atom_norm', 'hks'],
                        default='atom',
                        help='Node filtration. atom_norm recommended: normalizes '
                             'each OGB feature column to [0,1] before summing '
                             'so atomic number does not dominate. sider, and bace: hks, others: atom')
    parser.add_argument('--edge_filtration',
                        choices=['bondtype', 'en_diff'],
                        default='en_diff',
                        help='Edge filtration for MPH/MPTL. bondtype recommended: '
                             'uses OGB bond type (single/double/triple/aromatic), '
                             'spreads across 4 clean levels. en_diff collapses '
                             'to 4-5 values for typical drug molecules.')
    parser.add_argument('--direction',
                        choices=['sub', 'super', 'both'],
                        default='both',
                        help='Node filtration direction. '
                             'sub=sublevel (light atoms first), '
                             'super=superlevel (heavy/complex atoms first), '
                             'both=concatenate sub+super (doubles feature size). '
                             'Edge filtration is always sublevel.')

    # Filtration grid
    parser.add_argument('--n_node_thresh', type=int, default=10,
                        help='Number of node filtration thresholds')
    parser.add_argument('--n_edge_thresh', type=int, default=10,
                        help='Number of edge (EN-diff) thresholds (MPH/MPTL)')

    # Model architecture
    parser.add_argument('--hidden_dim',   type=int,   default=256)
    parser.add_argument('--num_layers',   type=int,   default=3,
                        help='MLP layers (ignored for CNN)')
    parser.add_argument('--drop_ratio',   type=float, default=0.5)
    parser.add_argument('--conv_channels',type=str,   default='32,64',
                        help='CNN conv channel sizes, comma-separated')

    # Training
    parser.add_argument('--epochs',        type=int,   default=500)
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--runs',          type=int,   default=5)
    parser.add_argument('--seed',          type=int,   default=42)
    parser.add_argument('--num_workers',   type=int,   default=0)
    parser.add_argument('--warmup_epochs', type=int,   default=200,
                        help='Ignore val scores before this epoch for model '
                             'selection. Useful for noisy datasets like '
                             'ClinTox where early val scores are unstable. '
                             'E.g. --warmup_epochs 30')
    parser.add_argument('--val_window',    type=int,   default=1,
                        help='Smooth val score over this many epochs before '
                             'model selection. E.g. --val_window 5 uses a '
                             '5-epoch moving average. Helps on noisy datasets.')

    # Output
    parser.add_argument('--filename',   type=str,   default='')

    args = parser.parse_args()

    device = torch.device(
        f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    conv_channels = tuple(int(c) for c in args.conv_channels.split(','))

    # ── Load OGB dataset ──────────────────────────────────────────────────────
    dataset   = PygGraphPropPredDataset(name=args.dataset, root=args.root)
    evaluator = Evaluator(args.dataset)
    split_idx = dataset.get_idx_split()

    problem_type = infer_problem_type(dataset)
    metric_name  = dataset.eval_metric
    num_tasks    = dataset.num_tasks

    print(f'\nDataset         : {args.dataset}')
    print(f'Mode            : {args.mode}')
    print(f'Model           : {args.model}')
    print(f'Node filtration : {args.filtration}  [{args.direction}]')
    print(f'Edge filtration : {args.edge_filtration} (MPH/MPTL only, always sublevel)')
    print(f'Task type       : {dataset.task_type}  →  {problem_type}')
    print(f'Eval metric     : {metric_name}')
    print(f'Num tasks       : {num_tasks}')

    if problem_type == 'single-label':
        all_labels = torch.cat([dataset[i].y.view(-1)
                                for i in range(len(dataset))])
        num_classes = int(all_labels.max().item()) + 1
    else:
        num_classes = num_tasks

    # ── Extract topological features ──────────────────────────────────────────
    print(f'\nExtracting {args.mode} features…', flush=True)
    t0 = time.time()
    features, node_thresh, edge_thresh, labels = extract_all_features(
        dataset,
        mode=args.mode,
        n_node_thresh=args.n_node_thresh,
        n_edge_thresh=args.n_edge_thresh,
        filtration=args.filtration,
        edge_filtration=args.edge_filtration,
        direction=args.direction,
    )
    feat_time = time.time() - t0
    print(f'Feature extraction: {feat_time:.1f}s', flush=True)
    print(features[2])

    # Labels tensor
    y = torch.stack([lbl.view(-1).float() for lbl in labels])

    # ── Shape info ────────────────────────────────────────────────────────────
    feat_shape = tuple(features[0].shape)
    print(f'\nFeature shape per graph: {feat_shape}')
    print(f'  → {int(torch.tensor(feat_shape).prod())} total values per graph')

    if args.mode in ('MPH', 'MPTL'):
        C, R, Q = feat_shape
        print(f'  Grid: {C} channels × {R} node thresholds × {Q} edge thresholds')
        if args.mode == 'MPTL':
            print('  Channels: β0, β1, L0(min,max,mean), L1(min,max,mean)')

    if args.model == 'cnn' and args.mode in ('PH', 'PTL'):
        print('WARNING: CNN requires grid features (MPH/MPTL). '
              'Switching to MLP for PH/PTL.', flush=True)
        args.model = 'mlp'

    # ── Splits ────────────────────────────────────────────────────────────────
    tr_idx  = split_idx['train'].tolist()
    val_idx = split_idx['valid'].tolist()
    te_idx  = split_idx['test'].tolist()

    X_tr  = [features[i] for i in tr_idx]
    X_val = [features[i] for i in val_idx]
    X_te  = [features[i] for i in te_idx]
    y_tr  = y[tr_idx]
    y_val = y[val_idx]
    y_te  = y[te_idx]

    val_ds  = TopoDataset(X_val,  y_val)
    test_ds = TopoDataset(X_te,   y_te)

    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             collate_fn=collate_fn)

    # ── Training runs ─────────────────────────────────────────────────────────
    all_train_scores, all_val_scores, all_test_scores, all_times = [], [], [], []

    for run in range(args.runs):
        run_seed = args.seed + run
        set_seed(run_seed)
        print(f"\n{'='*60}")
        print(f"Run {run+1}/{args.runs}  seed={run_seed}")
        print(f"{'='*60}")

        g = torch.Generator()
        g.manual_seed(run_seed)
        train_ds     = TopoDataset(X_tr, y_tr)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  generator=g, collate_fn=collate_fn)

        model = build_model(
            model_type=args.model,
            mode=args.mode,
            feat_shape=feat_shape,
            out_channels=num_classes,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.drop_ratio,
            conv_channels=conv_channels,
            problem_type=problem_type,
        ).to(device)

        model.reset_parameters()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        train_curve, val_curve, test_curve = [], [], []
        t_start = time.time()

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(
                model, train_loader, optimizer, device, problem_type)

            train_score = evaluate_epoch(
                model, train_loader, device, evaluator,
                problem_type, metric_name)
            val_score = evaluate_epoch(
                model, val_loader, device, evaluator,
                problem_type, metric_name)
            test_score = evaluate_epoch(
                model, test_loader, device, evaluator,
                problem_type, metric_name)

            train_curve.append(train_score)
            val_curve.append(val_score)
            test_curve.append(test_score)

            if epoch % 20 == 0 or epoch == 1:
                print(f'  Ep {epoch:03d} | loss={train_loss:.4f} '
                      f'train={train_score:.4f} '
                      f'val={val_score:.4f} '
                      f'test={test_score:.4f}', flush=True)

        t_end = time.time()
        all_times.append(t_end - t_start)

        # Model selection: best epoch by validation score
        # Apply smoothing window first, then ignore warmup epochs.
        val_arr = np.array(val_curve)
        if args.val_window > 1:
            # Moving average over val_window epochs
            kernel = np.ones(args.val_window) / args.val_window
            val_smooth = np.convolve(val_arr, kernel, mode='same')
            # Fix edges: use raw values where window doesn't fully overlap
            val_smooth[:args.val_window - 1] = val_arr[:args.val_window - 1]
        else:
            val_smooth = val_arr
        # Ignore epochs before warmup
        start_idx = min(args.warmup_epochs, len(val_smooth) - 1)
        best_epoch = int(np.argmax(val_smooth[start_idx:])) + start_idx
        best_train = train_curve[best_epoch]
        best_val   = val_curve[best_epoch]
        best_test  = test_curve[best_epoch]   # test at best-val epoch

        all_train_scores.append(best_train * 100)
        all_val_scores.append(best_val   * 100)
        all_test_scores.append(best_test  * 100)

        print(f'\nRun {run+1} | best epoch={best_epoch+1} '
              f'| val={best_val:.4f} | test={best_test:.4f}')

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS  [{args.mode} / {args.model.upper()} / "
          f"{args.filtration} / {args.dataset}]")
    print(f"{'='*60}")
    print(f"Train : {np.mean(all_train_scores):.2f} ± {np.std(all_train_scores):.2f}")
    print(f"Valid : {np.mean(all_val_scores):.2f}  ± {np.std(all_val_scores):.2f}")
    print(f"Test  : {np.mean(all_test_scores):.2f}  ± {np.std(all_test_scores):.2f}")
    print(f"Feature extraction : {feat_time:.1f}s")
    print(f"Train time/run     : {np.mean(all_times):.1f} ± {np.std(all_times):.1f}s")

    if args.filename:
        torch.save({
            'mode':           args.mode,
            'model':          args.model,
            'filtration':     args.filtration,
            'edge_filtration':args.edge_filtration,
            'direction':      args.direction,
            'dataset':        args.dataset,
            'problem_type':   problem_type,
            'metric':         metric_name,
            'TrainMean':      float(np.mean(all_train_scores)),
            'TrainStd':       float(np.std(all_train_scores)),
            'ValMean':        float(np.mean(all_val_scores)),
            'ValStd':         float(np.std(all_val_scores)),
            'TestMean':       float(np.mean(all_test_scores)),
            'TestStd':        float(np.std(all_test_scores)),
            'feat_time_s':    float(feat_time),
            'train_time_mean_s': float(np.mean(all_times)),
            'feat_shape':     list(feat_shape),
            'node_thresholds':node_thresh.tolist(),
            'edge_thresholds':edge_thresh.tolist() if edge_thresh is not None else [],
            'runs':           args.runs,
            'seed':           args.seed,
        }, args.filename)
        print(f'Results saved to {args.filename}')


if __name__ == '__main__':
    main()
