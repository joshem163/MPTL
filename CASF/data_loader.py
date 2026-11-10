"""Structure reading, label loading and dataset preparation.

Covers:
  - Atoms dataclass and PDB/MOL2/SDF parsers (structure I/O)
  - PDBbind index parsing and label loading
  - Refined-set / core-set split and manifest construction
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from utils import file_hash, write_json

# ── Atom container ─────────────────────────────────────────────────────────────

@dataclass
class Atoms:
    xyz: np.ndarray
    elements: np.ndarray

    def select(self, mask):
        return Atoms(self.xyz[mask], self.elements[mask])

    def __len__(self):
        return len(self.elements)


def make_atoms(xyz, elements):
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    if not np.isfinite(xyz).all():
        raise ValueError('Nonfinite coordinates')
    return Atoms(xyz, np.asarray(elements, dtype=str))


def combine(a, b):
    return make_atoms(np.vstack([a.xyz, b.xyz]), np.concatenate([a.elements, b.elements]))


# ── Structure parsers ──────────────────────────────────────────────────────────

def read_protein(path, include_hetatm=False):
    """Parse a PDB file, returning heavy atoms from the first MODEL (or all ATOM records)."""
    chosen = {}
    started_model = False
    for line in Path(path).read_text(errors='replace').splitlines():
        record = line[:6].strip()
        if record == 'MODEL':
            if started_model:
                break
            started_model = True
        if record == 'ENDMDL':
            break
        if record != 'ATOM' and not (include_hetatm and record == 'HETATM'):
            continue
        if line[17:20].strip() in {'HOH', 'WAT', 'DOD'}:
            continue
        element = line[76:78].strip().capitalize()
        if not element:
            atom_name = line[12:16]
            letters = ''.join(c for c in atom_name if c.isalpha())
            if not letters:
                raise ValueError(f'Cannot infer element in {path}: {line}')
            ion_symbols = {'ZN': 'Zn', 'CA': 'Ca', 'MG': 'Mg', 'FE': 'Fe',
                           'NA': 'Na', 'K': 'K',  'MN': 'Mn', 'CU': 'Cu',
                           'CO': 'Co', 'NI': 'Ni', 'CD': 'Cd', 'HG': 'Hg'}
            residue = line[17:20].strip().upper()
            if residue in ion_symbols and atom_name.strip().upper() == residue:
                element = ion_symbols[residue]
            else:
                element = (letters[0].upper() if record == 'ATOM' or atom_name.startswith(' ')
                           else letters[:2].capitalize())
        if element in {'H', 'D', 'T'}:
            continue
        xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        occupancy = float(line[54:60].strip() or '0')
        altloc = line[16:17]
        key = (record, line[21:22], line[22:27], line[17:20], line[12:16])
        rank = (occupancy, altloc in {' ', ''}, altloc == 'A')
        if key not in chosen or rank > chosen[key][0]:
            chosen[key] = (rank, xyz, element)
    if not chosen:
        raise ValueError(f'No protein heavy atoms in {path}')
    return make_atoms([v[1] for v in chosen.values()], [v[2] for v in chosen.values()])


def read_mol2(path):
    xyz, elements = [], []
    reading = False
    for line in Path(path).read_text(errors='replace').splitlines():
        if line.strip().upper() == '@<TRIPOS>ATOM':
            reading = True
            continue
        if reading and line.startswith('@<TRIPOS>'):
            break
        if reading and line.strip():
            fields = line.split()
            if len(fields) < 6:
                raise ValueError(f'Malformed MOL2 atom line: {line}')
            element = fields[5].split('.')[0].capitalize()
            if element in {'H', 'D', 'T'}:
                continue
            if element in {'Du', 'Lp', 'Any', 'Hal', 'Het', 'Hev'}:
                raise ValueError(f'Unsupported non-element MOL2 atom type: {fields[5]}')
            xyz.append([float(v) for v in fields[2:5]])
            elements.append(element)
    if not elements:
        raise ValueError(f'No heavy atoms in MOL2 {path}')
    return make_atoms(xyz, elements)


def read_sdf(path):
    from rdkit import Chem
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False, strictParsing=True)
    mol = supplier[0] if len(supplier) else None
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f'Cannot parse first SDF molecule: {path}')
    conf = mol.GetConformer()
    atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not atoms:
        raise ValueError(f'No heavy atoms in {path}')
    return make_atoms([list(conf.GetAtomPosition(a.GetIdx())) for a in atoms],
                      [a.GetSymbol() for a in atoms])


def read_ligand(row, preference):
    errors = []
    for ext in preference:
        path = row.get(f'{ext}_path', '')
        if not path or not Path(path).is_file():
            continue
        try:
            return (read_mol2(path) if ext == 'mol2' else read_sdf(path)), str(path), errors
        except Exception as exc:
            errors.append(f'{ext}: {exc}')
    raise ValueError('No readable ligand: ' + '; '.join(errors))


# ── Label loading ──────────────────────────────────────────────────────────────

ID_PATTERN = re.compile(r'^[A-Za-z0-9]{4}$')


def parse_index(path):
    """Parse PDBbind INDEX file columns: ID resolution release-year -logKd/Ki …"""
    values = {}
    for number, line in enumerate(Path(path).read_text(errors='replace').splitlines(), 1):
        fields = line.split()
        if not fields or line.lstrip().startswith('#') or not ID_PATTERN.fullmatch(fields[0]):
            continue
        if len(fields) < 4:
            continue
        try:
            year, value = int(fields[2]), float(fields[3])
        except ValueError:
            continue
        if not (1900 <= year <= 2100) or not np.isfinite(value):
            continue
        key = fields[0].lower()
        if key in values and not np.isclose(values[key], value, atol=1e-6, rtol=0):
            raise ValueError(f'Conflicting affinity for {key} in {path}:{number}')
        values[key] = value
    return values


def load_labels(data_root, index_files=None, labels_csv=None):
    labels, sources = {}, []
    if labels_csv is not None:
        table = pd.read_csv(labels_csv, dtype={'pdb_id': str})
        if not {'pdb_id', 'pKd'}.issubset(table.columns):
            raise ValueError('Label CSV must have columns pdb_id, pKd.')
        entries = [(str(r.pdb_id).lower().strip(), float(r.pKd)) for r in table.itertuples()]
        sources = [Path(labels_csv)]
    else:
        sources = ([Path(p) for p in index_files] if index_files is not None else
                   sorted(p for p in Path(data_root).rglob('*')
                          if p.is_file() and p.name.upper().startswith('INDEX')
                          and not p.name.startswith('.')))
        entries = [(key, value) for path in sources for key, value in parse_index(path).items()]
    for key, value in entries:
        if not ID_PATTERN.fullmatch(key) or not np.isfinite(value):
            raise ValueError(f'Invalid label: {key}={value}')
        if key in labels and not np.isclose(labels[key], value, atol=1e-6, rtol=0):
            raise ValueError(f'Conflicting labels for {key}.')
        labels[key] = value
    if not labels:
        raise ValueError('No affinity labels found. Supply a PDBbind INDEX file or LABELS_CSV.')
    return labels, [{'path': str(p.resolve()), 'sha256': file_hash(p)} for p in sources]


# ── Complex discovery ──────────────────────────────────────────────────────────

def discover_complexes(root, allowed_ids=None):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Directory not found: {root}')
    found = {}
    for protein in sorted(root.rglob('*_protein.pdb')):
        if any(part.startswith('.') for part in protein.relative_to(root).parts):
            continue
        stem = protein.name[:-len('_protein.pdb')]
        pdb_id = stem.lower()
        if not ID_PATTERN.fullmatch(pdb_id):
            continue
        if allowed_ids is not None and pdb_id not in allowed_ids:
            continue
        ligands = [protein.parent / f'{stem}_ligand.{ext}' for ext in ('mol2', 'sdf')]
        if not any(p.is_file() for p in ligands):
            raise FileNotFoundError(f'{pdb_id}: no ligand mol2/sdf beside {protein}')
        if pdb_id in found:
            raise ValueError(f'Duplicate {pdb_id} within {root}')
        found[pdb_id] = {'pdb_id': pdb_id, 'protein_path': str(protein),
                         'mol2_path': str(ligands[0]) if ligands[0].is_file() else '',
                         'sdf_path':  str(ligands[1]) if ligands[1].is_file() else ''}
    if not found:
        raise ValueError(f'No ID_protein.pdb files under {root}')
    return found


# ── Dataset preparation ────────────────────────────────────────────────────────

def resolve_paths(args):
    root = args.data_root.expanduser().resolve() / ('CASF-' + args.dataset)
    refined = (args.refined_dir or root / 'refined-set').expanduser().resolve()
    core    = (args.core_dir    or root / 'core-set'   ).expanduser().resolve()
    if not refined.is_dir() or not core.is_dir():
        raise FileNotFoundError(
            f'Expected refined and core folders:\n{refined}\n{core}\n'
            'Use --refined-dir / --core-dir to override.')
    if args.index_file:
        index = args.index_file.expanduser().resolve()
    else:
        candidates = sorted({p.resolve() for folder in (root, root/'index', root/'INDEX')
                             if folder.is_dir() for p in folder.iterdir()
                             if p.is_file() and 'index' in p.name.lower()
                             and 'refined' in p.name.lower() and not p.name.startswith('.')})
        if len(candidates) != 1:
            raise ValueError('Expected ONE refined INDEX file. Use --index-file.\n'
                             + '\n'.join(map(str, candidates)))
        index = candidates[0]
    if not index.is_file():
        raise FileNotFoundError(index)
    return root, refined, core, index


def prepare_dataset(args, baseline_dir):
    """Build the train/test manifest. Safe to call multiple times (idempotent)."""
    root, refined_dir, core_dir, index = resolve_paths(args)
    labels, sources = load_labels(root, index_files=[index])
    excluded = {s.lower() for s in args.exclude_id}
    unknown = excluded - labels.keys()
    if unknown:
        print(f'Explicit exclusions already absent: {sorted(unknown)}', flush=True)
    core    = discover_complexes(core_dir)
    if excluded & core.keys():
        raise ValueError(f'Cannot exclude test/core IDs: {sorted(excluded & core.keys())}')
    if core.keys() - labels.keys():
        raise ValueError(f'Core IDs missing labels: {sorted(core.keys() - labels.keys())}')
    requested = set(labels) - excluded
    refined   = discover_complexes(refined_dir, allowed_ids=requested)
    train_ids = sorted(requested - core.keys())
    missing   = set(train_ids) - refined.keys()
    if missing:
        raise ValueError(f'{len(missing)} training structures missing: {sorted(missing)}')
    if len(train_ids) < 2 or len(core) < 2:
        raise ValueError('Need at least two training and two core complexes')
    rows  = [{**refined[k], 'split': 'train', 'pKd': labels[k]} for k in train_ids]
    rows += [{**core[k],    'split': 'test',  'pKd': labels[k]} for k in sorted(core)]
    table = pd.DataFrame(rows)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    identity = {'dataset': args.dataset, 'root': str(root),
                'refined_dir': str(refined_dir), 'core_dir': str(core_dir),
                'index_file': str(index), 'index_sha256': file_hash(index),
                'exclude_ids': sorted(excluded)}
    identity_path = baseline_dir / 'dataset_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Dataset/index/exclusions changed. Choose a new --experiment name.')
    manifest   = baseline_dir / 'manifest.csv'
    content    = table.to_csv(index=False)
    if manifest.exists() and manifest.read_text() != content:
        raise ValueError('Manifest content changed. Choose a new --experiment name.')
    if not manifest.exists():
        manifest.write_text(content)
    write_json(identity_path, identity)
    write_json(baseline_dir / 'dataset_report.json', {
        **identity,
        'train_count': len(train_ids), 'test_count': len(core),
        'refined_structure_count_used': len(refined),
        'index_count': len(labels),
        'overlap_excluded_from_training': sorted(refined.keys() & core.keys()),
        'core_ids_not_in_refined_folder': sorted(core.keys() - refined.keys()),
        'label_sources': sources,
    })
    print(f'Dataset {args.dataset}: train={len(train_ids)}, test={len(core)}', flush=True)
    return table
