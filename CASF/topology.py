"""Topological and spectral computation for all four experiment modes.

Four modes are supported:
  PH   – barcodes only (kernel/alpha diagrams, H0 interaction deaths)
  PTL  – PH + single-parameter Laplacian spectral channels per element pair
  MPH  – multi-parameter Betti grid (distance × weighted degree)
  MPTL – MPH + Laplacian spectral channels at every grid cell

Spectral channels (PTL / MPTL)
───────────────────────────────
For every (distance threshold, degree threshold) cell the bipartite graph
Laplacian L = D – A is diagonalized per connected component (float64, dense
eigvalsh).  Three scalars are extracted and stored as separate channel grids:

  lambda_min_positive  – smallest eigenvalue above zero (Fiedler-like)
  lambda_max           – largest eigenvalue (spectral radius, encodes density)
  lambda_mean_positive – mean of all positive eigenvalues (average energy)

Missing values (no edges, or numerically degenerate) are stored as 0.0 with
a boolean validity mask that is saved for inspection but not fed to the model.
"""
import numpy as np
from scipy.linalg import eigvalsh
from scipy.sparse import csr_matrix, bmat
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

import config

# Channel names produced by the spectral layer; order is fixed by schema.
SPECTRAL_CHANNEL_NAMES = tuple(config.SPECTRAL_CHANNELS)

# Reuse tolerances from config so everything is in one place.
ATOL               = config.SPECTRAL_ATOL
RTOL               = config.SPECTRAL_RTOL
MAX_COMPONENT_SIZE = config.SPECTRAL_MAX_COMPONENT_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers shared by PH and MPH
# ─────────────────────────────────────────────────────────────────────────────

def nearby(protein, ligand, cutoff):
    """Return the subset of protein atoms within cutoff of any ligand atom."""
    from data_loader import Atoms
    if not len(ligand) or not len(protein):
        return protein.select(np.zeros(len(protein), dtype=bool))
    keep = np.zeros(len(protein), dtype=bool)
    for start in range(0, len(protein), 2048):
        keep[start:start+2048] = cdist(protein.xyz[start:start+2048], ligand.xyz).min(axis=1) <= cutoff
    return protein.select(keep)


def kernel_distance(distance, radius_sum, spec):
    x = np.asarray(distance) / (spec['tau'] * np.asarray(radius_sum))
    if spec['tau'] <= 0 or spec['power'] <= 0:
        raise ValueError('Kernel tau and power must be positive')
    with np.errstate(over='ignore'):
        z = np.power(x, spec['power'])
    if spec['kind'] == 'lorentz':
        return 1.0 - 1.0 / (1.0 + z)
    if spec['kind'] == 'exponential':
        return -np.expm1(-z)
    raise ValueError(f'Unknown kernel: {spec["kind"]}')


# ─────────────────────────────────────────────────────────────────────────────
# PH: standard single-parameter persistence (H0 interaction + kernel/alpha)
# ─────────────────────────────────────────────────────────────────────────────

def interaction_deaths(protein_xyz, ligand_xyz):
    """Finite H0 deaths of the complete weighted bipartite graph."""
    m, n = len(protein_xyz), len(ligand_xyz)
    if not m or not n:
        return np.empty(0)
    weights = cdist(protein_xyz, ligand_xyz).ravel()
    order   = np.argsort(weights, kind='stable')
    parent  = np.arange(m + n)
    size    = np.ones(m + n, dtype=int)

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    deaths = []
    for k in order:
        a, b = root(int(k // n)), root(m + int(k % n))
        if a != b:
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a
            size[a] += size[b]
            deaths.append(weights[k])
            if len(deaths) == m + n - 1:
                break
    return np.asarray(deaths)


def kernel_diagrams(atoms, radii, spec, max_atoms):
    if len(atoms) == 0:
        return [np.empty((0, 2)) for _ in range(3)]
    if len(atoms) > max_atoms:
        raise MemoryError(
            f'Global Rips H2 requested for {len(atoms)} atoms (guard={max_atoms}). '
            'Increase max_global_kernel_atoms deliberately.')
    missing = set(atoms.elements) - set(radii)
    if missing:
        raise ValueError(f'Add explicit vdW radii for: {sorted(missing)}')
    from ripser import ripser
    r = np.asarray([radii[e] for e in atoms.elements])
    matrix = kernel_distance(cdist(atoms.xyz, atoms.xyz), r[:, None] + r[None, :], spec)
    np.fill_diagonal(matrix, 0)
    return ripser(matrix, distance_matrix=True, maxdim=2, coeff=2, thresh=np.inf)['dgms']


def alpha_diagrams(atoms, units='radius'):
    if units not in {'radius', 'diameter'}:
        raise ValueError('alpha_units must be radius or diameter')
    if not len(atoms):
        return [np.empty((0, 2)) for _ in range(3)]
    import gudhi
    points = np.unique(atoms.xyz, axis=0)
    tree = gudhi.AlphaComplex(points=points, precision='safe').create_simplex_tree()
    tree.persistence(homology_coeff_field=2, min_persistence=0, persistence_dim_max=True)
    factor = 1.0 if units == 'radius' else 2.0
    return [factor * np.sqrt(np.maximum(tree.persistence_intervals_in_dimension(k), 0))
            for k in range(3)]


def finite_diagram(diagram):
    d = np.asarray(diagram, dtype=float).reshape(-1, 2)
    return d[np.isfinite(d).all(axis=1)]


def global_summary(diagrams):
    result = []
    for d in diagrams:
        d = finite_diagram(d)
        result.extend([float(d[:, 0].sum()), float((d[:, 1] - d[:, 0]).sum())])
    return np.asarray(result)


def alpha_summary(diagrams, bins):
    result = []
    for dim in (1, 2):
        d = finite_diagram(diagrams[dim])
        result.extend(np.histogram(d[:, 0], bins=bins, weights=d[:, 1] - d[:, 0])[0])
    return np.asarray(result, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# MP: multi-parameter bifiltration graphs
# ─────────────────────────────────────────────────────────────────────────────

def pair_graphs(protein, ligand, settings, cutoff, kernel):
    """Yield (pair_name, cross_distances [m×n], fixed_node_strengths [m+n])."""
    radii = settings['vdw_radii']
    for pe in settings['protein_elements']:
        for le in settings['ligand_elements']:
            l = ligand.select(ligand.elements == le)
            p = nearby(protein.select(protein.elements == pe), l, settings['pair_cutoff'])
            distance = cdist(p.xyz, l.xyz)
            weight   = 1. - kernel_distance(distance, radii[pe] + radii[le], kernel)
            weight[distance > cutoff] = 0.
            strengths = np.concatenate([weight.sum(axis=1), weight.sum(axis=0)])
            yield f'{pe}_{le}', distance, strengths


def fit_degree_axes(training_strengths, grid_size):
    """Per-pair descending degree thresholds fitted ONLY on training vertices."""
    if grid_size < 2:
        raise ValueError('GRID_SIZE must be at least 2')
    axes = {}
    for pair, arrays in training_strengths.items():
        nonempty = [np.asarray(a, float) for a in arrays if len(a)]
        values   = np.concatenate(nonempty) if nonempty else np.empty(0)
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f'Invalid weighted degrees in training pair {pair}')
        axis = (np.quantile(values, np.linspace(1., 0., grid_size)) if len(values)
                else np.zeros(grid_size))
        axis[-1] = 0.
        axes[pair] = axis
    return axes


def betti_grid(distance, strengths, distance_axis, degree_axis):
    """Integer Betti grid (2, R, Q) – H0 then H1."""
    distance     = np.asarray(distance, float)
    q            = np.asarray(strengths, float)
    r            = np.asarray(distance_axis, float)
    a            = np.asarray(degree_axis, float)
    if distance.ndim != 2:
        raise ValueError('distance must be a protein-by-ligand matrix')
    m, n = distance.shape
    if q.shape != (m + n,) or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError('strengths must be finite nonneg, one per vertex')
    if not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError('Distances must be finite and nonnegative')
    if r.ndim != 1 or a.ndim != 1 or not len(r) or not len(a):
        raise ValueError('Axes must be nonempty 1D arrays')
    if (not np.isfinite(r).all() or not np.isfinite(a).all()
            or np.any(np.diff(r) < 0) or np.any(np.diff(a) > 0)):
        raise ValueError('Distance axis must increase; degree thresholds must decrease')

    grid  = np.zeros((2, len(r), len(a)), dtype=np.int32)
    flat  = distance.ravel()
    order = np.argsort(flat, kind='stable')
    order = order[flat[order] <= r[-1]]

    for j, threshold in enumerate(a):
        active     = q >= threshold
        vertices   = int(active.sum())
        components = vertices
        parent     = np.arange(m + n)
        size       = np.ones(m + n, dtype=int)
        eligible   = order[active[order // n] & active[m + order % n]] if len(order) else order
        cursor, edges = 0, 0

        def root(v):
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v

        for i, radius in enumerate(r):
            while cursor < len(eligible) and flat[eligible[cursor]] <= radius:
                k    = eligible[cursor]
                u, v = root(int(k // n)), root(m + int(k % n))
                edges += 1
                if u != v:
                    if size[u] < size[v]:
                        u, v = v, u
                    parent[v] = u
                    size[u]  += size[v]
                    components -= 1
                cursor += 1
            grid[0, i, j] = components
            grid[1, i, j] = edges - vertices + components

    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Spectral computation (PTL / MPTL)
# ─────────────────────────────────────────────────────────────────────────────

def _spectral_stats(cross_edges,
                    atol=ATOL, rtol=RTOL,
                    max_component_vertices=MAX_COMPONENT_SIZE):
    """Return (lambda_min, lambda_max, lambda_mean, valid) for one bipartite graph.

    All three statistics are over the *positive* spectrum (eigenvalues > tolerance).
    Missing/no-edges case returns (0, 0, 0, False).
    """
    cross = np.asarray(cross_edges)
    if cross.ndim != 2 or not np.isin(cross, [0, 1]).all():
        raise ValueError('cross_edges must be a 2D binary matrix')
    if not cross.any():
        return 0.0, 0.0, 0.0, False

    # Drop isolated vertices before building the full bipartite adjacency.
    cross    = cross[np.any(cross, axis=1)][:, np.any(cross, axis=0)]
    block    = csr_matrix(cross, dtype=np.float64)
    adjacency = bmat([[None, block], [block.T, None]], format='csr')
    degrees  = np.asarray(adjacency.sum(axis=1)).ravel()
    tolerance = atol + rtol * (2.0 * float(degrees.max()))

    count, labels = connected_components(adjacency, directed=False)
    all_positive = []
    for component in range(count):
        ids = np.flatnonzero(labels == component)
        if len(ids) > max_component_vertices:
            raise MemoryError(
                f'L0 component has {len(ids)} vertices; limit is {max_component_vertices}.')
        lap = -adjacency[ids][:, ids].toarray()
        lap[np.diag_indices(len(ids))] = degrees[ids]
        eigenvalues = eigvalsh(lap, check_finite=False, driver='evr')
        if not np.isfinite(eigenvalues).all() or eigenvalues[0] < -tolerance:
            raise ArithmeticError('Invalid spectrum from unweighted graph Laplacian')
        all_positive.extend(eigenvalues[eigenvalues > tolerance].tolist())

    if not all_positive:
        # Edges exist but graph is bipartite-disconnected in a way that yields
        # no positive eigenvalues above tolerance – treat as missing.
        return 0.0, 0.0, 0.0, False

    pos = np.asarray(all_positive)
    return float(pos.min()), float(pos.max()), float(pos.mean()), True


def spectral_grid(distance, strengths, distance_axis, degree_axis):
    """Return channels (n_spectral_ch, R, Q) and validity (R, Q).

    Channels are in the order defined by SPECTRAL_CHANNEL_NAMES:
      [0] lambda_min_positive
      [1] lambda_max
      [2] lambda_mean_positive
    """
    distance  = np.asarray(distance, float)
    strengths = np.asarray(strengths, float)
    m, n      = distance.shape
    R, Q      = len(distance_axis), len(degree_axis)

    lmin  = np.zeros((R, Q), dtype=np.float64)
    lmax  = np.zeros((R, Q), dtype=np.float64)
    lmean = np.zeros((R, Q), dtype=np.float64)
    valid = np.zeros((R, Q), dtype=bool)
    cache = {}

    for j, threshold in enumerate(degree_axis):
        active    = strengths >= threshold
        endpoints = active[:m, None] & active[None, m:]
        for i, radius in enumerate(distance_axis):
            edges = (distance <= radius) & endpoints
            key   = np.packbits(edges.ravel()).tobytes()
            if key not in cache:
                cache[key] = _spectral_stats(edges)
            mn, mx, me, ok = cache[key]
            lmin[i, j]  = mn
            lmax[i, j]  = mx
            lmean[i, j] = me
            valid[i, j] = ok

    channels = np.stack([lmin, lmax, lmean], axis=0)  # (3, R, Q)
    return channels, valid


# ─────────────────────────────────────────────────────────────────────────────
# Vectorization helpers
# ─────────────────────────────────────────────────────────────────────────────

def vectorize_betti(graphs, axes, distance_axis):
    """MPH vectorizer: returns (flat_vector, names, tensor [pair, 2, R, Q])."""
    grids, names = [], []
    for pair, distance, strengths in graphs:
        grid = betti_grid(distance, strengths, distance_axis, axes[pair])
        grids.append(grid)
        names.extend(
            f'mp_pair_{pair}/H{k}/distance_{i}/degree_{j}'
            for k in range(2)
            for i in range(len(distance_axis))
            for j in range(len(axes[pair])))
    tensor = np.stack(grids)
    return tensor.ravel().astype(float), names, tensor


def vectorize_spectral(graphs, axes, distance_axis):
    """MPTL vectorizer: returns (flat_vector, names, betti_tensor, spec_tensor, validity_tensor).

    Flat vector contains:
      For each element pair:
        H0 grid (R×Q) + H1 grid (R×Q) + spectral channels (n_ch × R × Q)
    """
    betti_grids, spec_grids, valid_grids, names = [], [], [], []
    for pair, distance, strengths in graphs:
        deg_axis = axes[pair]
        R, Q     = len(distance_axis), len(deg_axis)

        b_grid = betti_grid(distance, strengths, distance_axis, deg_axis)
        s_grid, s_valid = spectral_grid(distance, strengths, distance_axis, deg_axis)

        betti_grids.append(b_grid)
        spec_grids.append(s_grid)
        valid_grids.append(s_valid)

        for k in range(2):
            names.extend(f'mp_pair_{pair}/H{k}/distance_{i}/degree_{j}'
                         for i in range(R) for j in range(Q))
        for ch_name in SPECTRAL_CHANNEL_NAMES:
            names.extend(f'mp_pair_{pair}/{ch_name}/distance_{i}/degree_{j}'
                         for i in range(R) for j in range(Q))

    b_tensor = np.stack(betti_grids)   # (pairs, 2, R, Q)
    s_tensor = np.stack(spec_grids)    # (pairs, n_ch, R, Q)
    v_tensor = np.stack(valid_grids)   # (pairs, R, Q)

    # Interleave per pair: H0+H1 grids then spectral channels, matching names order above.
    flat = np.concatenate([
        np.concatenate([b_tensor[p].ravel().astype(float),
                        s_tensor[p].ravel()])
        for p in range(len(betti_grids))
    ])
    return flat, names, b_tensor, s_tensor, v_tensor
