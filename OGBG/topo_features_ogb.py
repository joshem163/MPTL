"""Topological feature extraction for OGB molecular graph datasets.

Four modes
──────────────────────────────────────────────────────────────────────────────
  PH   – single-parameter persistent homology
         filtration: atomic number (node function)
         features:   Betti-0, Betti-1 vectors over thresholds

  PTL  – persistent topological Laplacian (single-parameter)
         filtration: atomic number
         features:   PH features + Laplacian spectral features per threshold
                     (dim-0 and dim-1 Laplacians)

  MPH  – multi-parameter persistent homology
         filtration 1: atomic number (node)
         filtration 2: electronegativity difference (edge)
         features:   Betti-0, Betti-1 grids over (node_thresh × edge_thresh)

  MPTL – multi-parameter topological Laplacian
         filtration: same as MPH
         features:   MPH features + Laplacian spectral grids

Laplacian spectral features (PTL / MPTL)
──────────────────────────────────────────────────────────────────────────────
For each (node_thresh, edge_thresh) cell and each homology dimension k ∈ {0,1}:
  - lambda_min_k   : smallest NON-ZERO eigenvalue of L_k
  - lambda_max_k   : largest eigenvalue of L_k
  - lambda_mean_k  : mean of NON-ZERO eigenvalues of L_k

zero_count (number of zero eigenvalues) is NOT stored as a separate feature
because it equals β_k exactly — already captured by the Betti channels.

The graph is treated as a general simplicial complex (NOT bipartite).
Triangles (2-simplices) are filled in using clique detection, so H1 can die
via triangle boundaries — unlike the CASF bipartite case where H1 never dies.

Node filtration options
──────────────────────────────────────────────────────────────────────────────
  atomN     : raw atomic number (data.x[:, 0])
  atom      : raw sum of all 9 OGB node feature columns — dominated by
              atomic number since it ranges 1–118 vs 0–5 for others
  atom_norm : normalized sum — each of the 9 feature columns scaled to
              [0, 1] before summing, so no single column dominates.
              Gives a genuine composite chemical fingerprint.
              Recommended over raw 'atom' for spread thresholds.
  hks       : Heat Kernel Signature at t=0.1 — topology-sensitive

Edge filtration options (MPH / MPTL)
──────────────────────────────────────────────────────────────────────────────
  en_diff   : |EN(u) - EN(v)| Pauling electronegativity difference.
              Collapses to 4–5 unique values for typical drug molecules
              (most bonds are C–C, C–N, C–O).
  bondtype  : OGB bond type from edge_attr[:, 0].
              0=single, 1=double, 2=triple, 3=aromatic.
              Clean discrete values, spreads well, chemically meaningful.
              Recommended over en_diff.

Electronegativity table (Pauling scale, common elements in OGB molecules):
  H=2.20, C=2.55, N=3.04, O=3.44, F=3.98, P=2.19, S=2.58, Cl=3.16,
  Br=2.96, I=2.66, B=2.04, Si=1.90, Se=2.55, Te=2.10
"""

import numpy as np
import torch
import networkx as nx
from collections import Counter
from scipy.linalg import eigh
from torch_geometric.utils import to_networkx


# ─────────────────────────────────────────────────────────────────────────────
# Electronegativity table (Pauling scale)
# ─────────────────────────────────────────────────────────────────────────────

# OGB atomic number → element (1-indexed, first column of node features)
ATOMIC_NUM_TO_EN = {
    1:  2.20,   # H
    5:  2.04,   # B
    6:  2.55,   # C
    7:  3.04,   # N
    8:  3.44,   # O
    9:  3.98,   # F
    14: 1.90,   # Si
    15: 2.19,   # P
    16: 2.58,   # S
    17: 3.16,   # Cl
    34: 2.55,   # Se
    35: 2.96,   # Br
    52: 2.10,   # Te
    53: 2.66,   # I
}
DEFAULT_EN = 2.20   # fallback for unknown elements


def get_en(atomic_num: int) -> float:
    return ATOMIC_NUM_TO_EN.get(int(atomic_num), DEFAULT_EN)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold fitting
# ─────────────────────────────────────────────────────────────────────────────

def process_thresholds(values: torch.Tensor, N: int) -> torch.Tensor:
    """Return N quantile-spaced thresholds from a flat value tensor."""
    if N < 2:
        raise ValueError('N must be at least 2')
    vals = values.float().view(-1).numpy()
    quantiles = np.linspace(0.0, 1.0, N)
    thresholds = np.quantile(vals, quantiles)
    return torch.tensor(np.unique(thresholds), dtype=torch.float32)


def fit_node_thresholds(dataset, N: int, filtration: str = 'atomN') -> torch.Tensor:
    """Fit N node filtration thresholds over the whole dataset."""
    all_vals = []
    for data in dataset:
        vals = _node_filtration_values(data, filtration)
        all_vals.append(vals)
    return process_thresholds(torch.cat(all_vals), N)


def fit_edge_thresholds(dataset, N: int,
                        edge_filtration: str = 'bondtype') -> torch.Tensor:
    """Fit N edge filtration thresholds over the whole dataset.

    For bondtype: returns the sorted unique bond type values (0–3) directly
    rather than quantile-fitting, since the values are already discrete and
    well-spread (single/double/triple/aromatic).
    """
    all_vals = []
    for data in dataset:
        vals = _edge_filtration_values(data, edge_filtration)
        if vals.numel() > 0:
            all_vals.append(vals)
    if not all_vals:
        return torch.zeros(N)
    if edge_filtration == 'bondtype':
        # Use all unique bond type values; ignore N (always 4 or fewer)
        unique = torch.unique(torch.cat(all_vals)).float()
        return unique
    return process_thresholds(torch.cat(all_vals), N)


# ─────────────────────────────────────────────────────────────────────────────
# Per-graph filtration value extraction
# ─────────────────────────────────────────────────────────────────────────────

def _node_filtration_values(data, filtration: str) -> torch.Tensor:
    """Return per-node scalar filtration values.

    atomN     : raw atomic number — clean but ignores other chemistry
    atom      : raw feature sum — dominated by atomic number (range 1–118)
    atom_norm : normalized feature sum — each of the 9 OGB columns scaled
                to [0,1] by its dataset-wide max before summing.
                Gives equal weight to all chemical attributes.
    hks       : Heat Kernel Signature at t=0.1 — topology-sensitive
    """
    if filtration == 'atomN':
        return data.x[:, 0].float()

    if filtration == 'atom':
        return data.x.sum(dim=1).float()

    if filtration == 'atom_norm':
        feat = data.x.float()
        # Per-column max across this graph's atoms (safe: no dataset-wide stats needed
        # because we normalize per-graph; thresholds are fitted across all graphs)
        col_max = feat.max(dim=0).values.clamp(min=1.0)
        return (feat / col_max).sum(dim=1)

    if filtration == 'hks':
        graph = to_networkx(data, to_undirected=True)
        return _compute_hks(graph, t=0.1)

    raise ValueError(f'Unknown filtration: {filtration!r}. '
                     'Choose atomN | atom | atom_norm | hks')


def _edge_filtration_values(data, edge_filtration: str = 'bondtype') -> torch.Tensor:
    """Return per-edge scalar filtration values (undirected: one value per bond).

    bondtype : OGB edge_attr[:, 0] — 0=single, 1=double, 2=triple, 3=aromatic.
               Clean discrete values; spreads naturally across 4 levels.
    en_diff  : |EN(u) - EN(v)| Pauling electronegativity difference.
               Collapses to few values for typical drug molecules.
    """
    edge_index = data.edge_index
    seen = set()
    vals = []

    if edge_filtration == 'bondtype':
        if data.edge_attr is None:
            raise ValueError('edge_attr is None — dataset may not have bond features.')
        for i in range(edge_index.size(1)):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                vals.append(float(data.edge_attr[i, 0]))
        return torch.tensor(vals, dtype=torch.float32) if vals else torch.empty(0)

    if edge_filtration == 'en_diff':
        atomic_nums = data.x[:, 0].long()
        for i in range(edge_index.size(1)):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                diff = abs(get_en(atomic_nums[u].item()) -
                           get_en(atomic_nums[v].item()))
                vals.append(diff)
        return torch.tensor(vals, dtype=torch.float32) if vals else torch.empty(0)

    raise ValueError(f'Unknown edge_filtration: {edge_filtration!r}. '
                     'Choose bondtype | en_diff')


def _compute_hks(graph: nx.Graph, t: float) -> torch.Tensor:
    """Heat Kernel Signature diagonal at time t."""
    L = nx.laplacian_matrix(graph).toarray().astype(np.float64)
    eigvals, eigvecs = np.linalg.eigh(L)
    diag = np.sum(eigvecs ** 2 * np.exp(-t * eigvals)[None, :], axis=1)
    return torch.tensor(diag, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Laplacian spectral utilities (Rips / clique complex, NOT bipartite)
# ─────────────────────────────────────────────────────────────────────────────

_ATOL = 1e-8   # eigenvalue zero tolerance


def _hodge_laplacian_dim0(G: nx.Graph):
    """Standard graph Laplacian L0 = D - A."""
    nodes = sorted(G.nodes())
    if not nodes:
        return np.zeros((0, 0))
    A = nx.to_numpy_array(G, nodelist=nodes)
    D = np.diag(A.sum(axis=1))
    return D - A


def _hodge_laplacian_dim1(G: nx.Graph):
    """L1 = B1^T B1 + B2 B2^T  where B1 = edge boundary, B2 = triangle boundary.

    Triangles are ALL triangles (cliques of size 3) present in G.
    This is the correct Hodge Laplacian for the clique complex.
    """
    nodes  = sorted(G.nodes())
    edges  = sorted(G.edges())   # undirected, no duplicates
    tris   = [tuple(sorted(c))
              for c in nx.enumerate_all_cliques(G) if len(c) == 3]

    nE = len(edges)
    if nE == 0:
        return np.zeros((0, 0))

    edge_idx = {e: i for i, e in enumerate(edges)}

    # B1: node_boundary × edge  (nV × nE)
    nV = len(nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}
    B1 = np.zeros((nV, nE))
    for j, (u, v) in enumerate(edges):
        B1[node_idx[u], j] = -1.0
        B1[node_idx[v], j] =  1.0

    # B2: edge_boundary × triangle  (nE × nT)
    nT = len(tris)
    B2 = np.zeros((nE, nT)) if nT > 0 else np.zeros((nE, 0))
    for k, (a, b, c) in enumerate(tris):
        # oriented: edges (a,b), (a,c), (b,c)
        for u, v, sign in [(a, b, 1), (a, c, -1), (b, c, 1)]:
            e = (u, v) if (u, v) in edge_idx else (v, u)
            if e in edge_idx:
                B2[edge_idx[e], k] = sign

    L1 = B1.T @ B1 + B2 @ B2.T
    return L1


def _spectral_features(L: np.ndarray) -> dict:
    """Extract spectral features from a Hodge Laplacian matrix.

    Note: zero_count (number of zero eigenvalues) equals the Betti number
    by definition, so it is NOT returned here — it is already stored as
    the β0 / β1 channel to avoid redundancy.

    Returns
    -------
    lambda_min  : float  smallest NON-ZERO eigenvalue  (0 if none)
    lambda_max  : float  largest eigenvalue             (0 if empty)
    lambda_mean : float  mean of NON-ZERO eigenvalues  (0 if none)
    """
    if L.shape[0] == 0:
        return dict(lambda_min=0.0, lambda_max=0.0, lambda_mean=0.0)
    eigvals = eigh(L, eigvals_only=True, check_finite=False, driver='ev')
    eigvals = np.maximum(eigvals, 0.0)    # numerical clean-up
    nonzero     = eigvals[eigvals >= _ATOL]
    lambda_min  = float(nonzero.min())  if len(nonzero) else 0.0
    lambda_max  = float(eigvals.max())  if len(eigvals) else 0.0
    lambda_mean = float(nonzero.mean()) if len(nonzero) else 0.0
    return dict(lambda_min=lambda_min,
                lambda_max=lambda_max,
                lambda_mean=lambda_mean)


def _spectral_feature_vector(G: nx.Graph) -> np.ndarray:
    """6-dim spectral feature vector.

    Layout: [L0: min, max, mean] + [L1: min, max, mean]

    zero_count is intentionally excluded: it equals β0 (L0) and β1 (L1),
    which are already stored as dedicated Betti channels.
    """
    L0 = _hodge_laplacian_dim0(G)
    L1 = _hodge_laplacian_dim1(G)
    s0 = _spectral_features(L0)
    s1 = _spectral_features(L1)
    return np.array([
        s0['lambda_min'], s0['lambda_max'], s0['lambda_mean'],
        s1['lambda_min'], s1['lambda_max'], s1['lambda_mean'],
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Subgraph construction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_subgraph(data, active_nodes: set) -> nx.Graph:
    """Build NetworkX graph on active_nodes with edges from data.edge_index."""
    G = nx.Graph()
    G.add_nodes_from(active_nodes)
    ei = data.edge_index
    for i in range(ei.size(1)):
        u, v = int(ei[0, i]), int(ei[1, i])
        if u in active_nodes and v in active_nodes and u != v:
            G.add_edge(u, v)
    return G


def _betti(G: nx.Graph):
    """β0 and β1 for a general graph (triangles counted via clique complex)."""
    if G.number_of_nodes() == 0:
        return 0, 0
    b0 = nx.number_connected_components(G)
    # For the clique complex: β1 = E - V + C - triangles (rank of B2)
    nV = G.number_of_nodes()
    nE = G.number_of_edges()
    tris = sum(1 for c in nx.enumerate_all_cliques(G) if len(c) == 3)
    b1 = max(0, nE - nV + b0 - tris)
    return b0, b1


# ─────────────────────────────────────────────────────────────────────────────
# Active-node selection (sublevel / superlevel)
# ─────────────────────────────────────────────────────────────────────────────

def _active_nodes(node_vals_list: list, threshold: float,
                  direction: str) -> set:
    """Return node indices active at this threshold.

    sublevel  : f(v) <= t  — simple/light atoms enter first
    superlevel: f(v) >= t  — complex/heavy atoms enter first
    """
    if direction == 'sub':
        return {i for i, v in enumerate(node_vals_list) if v <= threshold}
    if direction == 'super':
        return {i for i, v in enumerate(node_vals_list) if v >= threshold}
    raise ValueError(f'direction must be sub or super, got {direction!r}')


def _build_edge_map(data, edge_vals: torch.Tensor) -> dict:
    """Build {(u,v): edge_val} map for undirected edges."""
    ei = data.edge_index
    edge_map = {}
    ev_idx = 0
    seen = set()
    edge_list = edge_vals.tolist()
    for i in range(ei.size(1)):
        u, v = int(ei[0, i]), int(ei[1, i])
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            if ev_idx < len(edge_list):
                edge_map[key] = edge_list[ev_idx]
                ev_idx += 1
    return edge_map


# ─────────────────────────────────────────────────────────────────────────────
# PH feature extraction (single-parameter)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ph_features(data, node_vals: torch.Tensor,
                        node_thresholds: torch.Tensor,
                        direction: str = 'sub') -> torch.Tensor:
    """PH: Betti-0 and Betti-1 over node thresholds.

    direction : 'sub' | 'super' | 'both'
      sub   → (2R,)   sublevel only
      super → (2R,)   superlevel only
      both  → (4R,)   [sublevel | superlevel] concatenated

    Layout (both): [β0_sub…, β1_sub…, β0_super…, β1_super…]
    """
    def _one_direction(direc):
        T = node_thresholds.tolist()
        vals = node_vals.tolist()
        b0s, b1s = [], []
        for t in T:
            active = _active_nodes(vals, t, direc)
            G = _build_subgraph(data, active)
            b0, b1 = _betti(G)
            b0s.append(b0)
            b1s.append(b1)
        return torch.tensor(b0s + b1s, dtype=torch.float32)

    if direction == 'both':
        return torch.cat([_one_direction('sub'), _one_direction('super')])
    return _one_direction(direction)


# ─────────────────────────────────────────────────────────────────────────────
# PTL feature extraction (single-parameter + Laplacian)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ptl_features(data, node_vals: torch.Tensor,
                         node_thresholds: torch.Tensor,
                         direction: str = 'sub') -> torch.Tensor:
    """PTL: Betti + 6 Laplacian spectral features per threshold.

    Per threshold layout: [β0, β1, L0:min,max,mean, L1:min,max,mean] = 8 values

    direction : 'sub' | 'super' | 'both'
      sub   → (8R,)
      super → (8R,)
      both  → (16R,)  [sublevel | superlevel] concatenated
    """
    def _one_direction(direc):
        T = node_thresholds.tolist()
        vals = node_vals.tolist()
        rows = []
        for t in T:
            active = _active_nodes(vals, t, direc)
            G = _build_subgraph(data, active)
            b0, b1 = _betti(G)
            spec = _spectral_feature_vector(G)
            rows.append([float(b0), float(b1)] + spec.tolist())
        return torch.tensor(rows, dtype=torch.float32).view(-1)

    if direction == 'both':
        return torch.cat([_one_direction('sub'), _one_direction('super')])
    return _one_direction(direction)


# ─────────────────────────────────────────────────────────────────────────────
# MPH feature extraction (multi-parameter)
# ─────────────────────────────────────────────────────────────────────────────

def compute_mph_features(data, node_vals: torch.Tensor,
                         edge_vals: torch.Tensor,
                         node_thresholds: torch.Tensor,
                         edge_thresholds: torch.Tensor,
                         direction: str = 'sub') -> torch.Tensor:
    """MPH: Betti-0 and Betti-1 on (node_thresh × edge_thresh) grid.

    Edge filtration is always sublevel (bondtype ascending).
    Only node filtration direction varies.

    direction : 'sub' | 'super' | 'both'
      sub   → (2, R, Q)
      super → (2, R, Q)
      both  → (4, R, Q)  channels: [β0_sub, β1_sub, β0_super, β1_super]
    """
    def _one_direction(direc):
        R = len(node_thresholds)
        Q = len(edge_thresholds)
        vals = node_vals.tolist()
        edge_map = _build_edge_map(data, edge_vals)
        et = edge_thresholds.tolist()
        nt = node_thresholds.tolist()

        grid_b0 = np.zeros((R, Q), dtype=np.float32)
        grid_b1 = np.zeros((R, Q), dtype=np.float32)

        for i, nt_val in enumerate(nt):
            active_nodes = _active_nodes(vals, nt_val, direc)
            for j, et_val in enumerate(et):
                G = nx.Graph()
                G.add_nodes_from(active_nodes)
                for (u, v), ev in edge_map.items():
                    if u in active_nodes and v in active_nodes and ev <= et_val:
                        G.add_edge(u, v)
                b0, b1 = _betti(G)
                grid_b0[i, j] = b0
                grid_b1[i, j] = b1

        return np.stack([grid_b0, grid_b1], axis=0)   # (2, R, Q)

    if direction == 'both':
        sub   = _one_direction('sub')    # (2, R, Q)
        super_ = _one_direction('super') # (2, R, Q)
        return torch.from_numpy(np.concatenate([sub, super_], axis=0))  # (4, R, Q)
    return torch.from_numpy(_one_direction(direction))


# ─────────────────────────────────────────────────────────────────────────────
# MPTL feature extraction (multi-parameter + Laplacian)
# ─────────────────────────────────────────────────────────────────────────────

def compute_mptl_features(data, node_vals: torch.Tensor,
                          edge_vals: torch.Tensor,
                          node_thresholds: torch.Tensor,
                          edge_thresholds: torch.Tensor,
                          direction: str = 'sub') -> torch.Tensor:
    """MPTL: Betti + spectral grids over (node_thresh × edge_thresh).

    Per cell: [β0, β1, L0:min,max,mean, L1:min,max,mean] = 8 channels

    Edge filtration is always sublevel (bondtype ascending).
    Only node filtration direction varies.

    direction : 'sub' | 'super' | 'both'
      sub   → (8,  R, Q)
      super → (8,  R, Q)
      both  → (16, R, Q)  [sublevel channels | superlevel channels]

    Channel layout (both, 16 channels):
      0   : β0_sub        8  : β0_super
      1   : β1_sub        9  : β1_super
      2-4 : L0 sub        10-12: L0 super
      5-7 : L1 sub        13-15: L1 super
    """
    def _one_direction(direc):
        R = len(node_thresholds)
        Q = len(edge_thresholds)
        vals = node_vals.tolist()
        edge_map = _build_edge_map(data, edge_vals)
        et = edge_thresholds.tolist()
        nt = node_thresholds.tolist()

        n_ch = 8    # 2 Betti + 3 L0 + 3 L1
        grid = np.zeros((n_ch, R, Q), dtype=np.float32)

        for i, nt_val in enumerate(nt):
            active_nodes = _active_nodes(vals, nt_val, direc)
            for j, et_val in enumerate(et):
                G = nx.Graph()
                G.add_nodes_from(active_nodes)
                for (u, v), ev in edge_map.items():
                    if u in active_nodes and v in active_nodes and ev <= et_val:
                        G.add_edge(u, v)
                b0, b1 = _betti(G)
                spec = _spectral_feature_vector(G)   # 6-dim
                grid[0, i, j] = b0
                grid[1, i, j] = b1
                grid[2:, i, j] = spec

        return grid   # (8, R, Q)

    if direction == 'both':
        sub    = _one_direction('sub')    # (8, R, Q)
        super_ = _one_direction('super')  # (8, R, Q)
        return torch.from_numpy(np.concatenate([sub, super_], axis=0))  # (16, R, Q)
    return torch.from_numpy(_one_direction(direction))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-level extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_features(dataset, mode: str,
                         n_node_thresh: int = 10,
                         n_edge_thresh: int = 10,
                         filtration: str = 'atom_norm',
                         edge_filtration: str = 'bondtype',
                         direction: str = 'sub'):
    """Extract topological features for every graph in dataset.

    Parameters
    ----------
    dataset        : OGB PygGraphPropPredDataset
    mode           : 'PH' | 'PTL' | 'MPH' | 'MPTL'
    n_node_thresh  : number of node filtration thresholds
    n_edge_thresh  : number of edge filtration thresholds (MPH/MPTL only);
                     ignored for bondtype (always uses all unique bond types)
    filtration     : 'atom_norm' (recommended) | 'atomN' | 'atom' | 'hks'
    edge_filtration: 'bondtype'  (recommended) | 'en_diff'
    direction      : 'sub' | 'super' | 'both'
                     Controls node filtration direction only.
                     Edge filtration is always sublevel.
                     'both' concatenates sub and super features:
                       PH   : (4R,)    MPH  : (4,  R, Q)
                       PTL  : (16R,)   MPTL : (16, R, Q)

    Returns
    -------
    features        : list of tensors
    node_thresholds : Tensor (R,)
    edge_thresholds : Tensor (Q,) or None for PH/PTL
    labels          : list of Tensors
    """
    mode = mode.upper()
    if mode not in ('PH', 'PTL', 'MPH', 'MPTL'):
        raise ValueError(f'Unknown mode: {mode!r}')
    if direction not in ('sub', 'super', 'both'):
        raise ValueError(f'direction must be sub | super | both, got {direction!r}')

    print(f'Fitting node thresholds ({filtration}, direction={direction})…',
          flush=True)
    node_thresholds = fit_node_thresholds(dataset, n_node_thresh, filtration)
    print(f'  node thresholds: {node_thresholds.tolist()}')

    edge_thresholds = None
    if mode in ('MPH', 'MPTL'):
        print(f'Fitting edge thresholds ({edge_filtration}, always sublevel)…',
              flush=True)
        edge_thresholds = fit_edge_thresholds(dataset, n_edge_thresh,
                                              edge_filtration)
        print(f'  edge thresholds: {edge_thresholds.tolist()}')

    features, labels = [], []
    N = len(dataset)

    for idx, data in enumerate(dataset):
        if idx % 500 == 0:
            print(f'  [{idx}/{N}] extracting…', flush=True)

        node_vals = _node_filtration_values(data, filtration)
        label = data.y

        if mode == 'PH':
            feat = compute_ph_features(data, node_vals, node_thresholds,
                                       direction)
        elif mode == 'PTL':
            feat = compute_ptl_features(data, node_vals, node_thresholds,
                                        direction)
        elif mode == 'MPH':
            edge_vals = _edge_filtration_values(data, edge_filtration)
            feat = compute_mph_features(data, node_vals, edge_vals,
                                        node_thresholds, edge_thresholds,
                                        direction)
        elif mode == 'MPTL':
            edge_vals = _edge_filtration_values(data, edge_filtration)
            feat = compute_mptl_features(data, node_vals, edge_vals,
                                         node_thresholds, edge_thresholds,
                                         direction)

        features.append(feat)
        labels.append(label)

    return features, node_thresholds, edge_thresholds, labels
