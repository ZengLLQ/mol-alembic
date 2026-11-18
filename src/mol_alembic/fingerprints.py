"""Morgan circular (ECFP) fingerprints, similarity matrices, Butina clustering, and MaxMin diversity."""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple
import numpy as np
from numpy.typing import NDArray

from .chemistry import perceive_sssr
from .types import BondOrder, Molecule


def _splitmix64(x: int) -> int:
    """Deterministic 64-bit SplitMix64 integer hash mixer."""
    z = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF


def _hash_tuple(values: Sequence[int]) -> int:
    """Hash a sequence of integers into a 64-bit unsigned identifier."""
    acc = 0x243F6A8885A308D3
    for val in values:
        acc = _splitmix64(acc ^ (val & 0xFFFFFFFFFFFFFFFF))
    return acc


def _initial_atom_invariant(mol: Molecule, atom_idx: int) -> int:
    """Compute Morgan/ECFP Daylight atom invariant tuple hash."""
    atom = mol.atoms[atom_idx]
    deg = mol.degree(atom_idx)
    total_conn = deg + atom.total_h
    return _hash_tuple(
        [
            atom.atomic_number,
            deg,
            total_conn,
            atom.formal_charge + 10,
            atom.total_h,
            1 if atom.is_in_ring else 0,
            1 if atom.is_aromatic else 0,
        ]
    )


def morgan_fingerprint_counts(
    mol: Molecule,
    radius: int = 2,
    use_bond_orders: bool = True,
) -> Tuple[Dict[int, int], Dict[int, Set[Tuple[int, int]]]]:
    """Compute sparse Morgan circular fingerprint feature counts and bit provenance."""
    if not any(a.is_in_ring for a in mol.atoms) and mol.num_bonds >= mol.num_atoms:
        perceive_sssr(mol)

    n = mol.num_atoms
    if n == 0:
        return {}, {}

    current_ids = [_initial_atom_invariant(mol, i) for i in range(n)]
    atom_bond_sets: List[Set[int]] = [set() for _ in range(n)]
    seen_env_keys: Set[Tuple[int, frozenset[int]]] = set()

    counts: Dict[int, int] = {}
    provenance: Dict[int, Set[Tuple[int, int]]] = {}

    for i in range(n):
        fid = current_ids[i]
        counts[fid] = counts.get(fid, 0) + 1
        provenance.setdefault(fid, set()).add((i, 0))

    for r in range(1, radius + 1):
        next_ids = list(current_ids)
        next_bond_sets = [set(s) for s in atom_bond_sets]

        for i in range(n):
            nbr_pairs: List[Tuple[int, int]] = []
            merged_bonds = set(atom_bond_sets[i])
            for j, bond in mol.neighbors(i):
                b_code = (
                    4
                    if bond.is_aromatic or bond.order == BondOrder.AROMATIC
                    else int(round(bond.order.value))
                )
                if not use_bond_orders:
                    b_code = 1
                nbr_pairs.append((b_code, current_ids[j]))
                merged_bonds.add(bond.index)
                merged_bonds.update(atom_bond_sets[j])

            nbr_pairs.sort()
            flat_seq: List[int] = [r, current_ids[i]]
            for b_code, n_id in nbr_pairs:
                flat_seq.append(b_code)
                flat_seq.append(n_id)
            new_id = _hash_tuple(flat_seq)
            next_ids[i] = new_id
            next_bond_sets[i] = merged_bonds

            env_key = (new_id, frozenset(merged_bonds))
            if env_key not in seen_env_keys:
                seen_env_keys.add(env_key)
                counts[new_id] = counts.get(new_id, 0) + 1
                provenance.setdefault(new_id, set()).add((i, r))

        current_ids = next_ids
        atom_bond_sets = next_bond_sets

    return counts, provenance


def morgan_bit_vector(
    mol: Molecule,
    radius: int = 2,
    nbits: int = 1024,
    counted: bool = False,
) -> NDArray[np.float64]:
    """Fold Morgan circular invariants into a fixed-length binary or count NumPy vector."""
    counts, _ = morgan_fingerprint_counts(mol, radius=radius)
    vec = np.zeros(nbits, dtype=np.float64)
    for fid, cnt in counts.items():
        bit = fid % nbits
        if counted:
            vec[bit] += float(cnt)
        else:
            vec[bit] = 1.0
    return vec


def topological_torsion_vector(mol: Molecule, nbits: int = 512) -> NDArray[np.float64]:
    """Compute 4-atom linear path topological torsion folded count fingerprint."""
    n = mol.num_atoms
    vec = np.zeros(nbits, dtype=np.float64)
    if n < 4:
        return vec

    seen_paths: Set[Tuple[int, ...]] = set()
    for a0 in range(n):
        for a1, _ in mol.neighbors(a0):
            for a2, _ in mol.neighbors(a1):
                if a2 == a0:
                    continue
                for a3, _ in mol.neighbors(a2):
                    if a3 == a1 or a3 == a0:
                        continue
                    canon = min((a0, a1, a2, a3), (a3, a2, a1, a0))
                    if canon in seen_paths:
                        continue
                    seen_paths.add(canon)
                    sigs = [
                        (
                            mol.atoms[idx].atomic_number * 100
                            + mol.degree(idx) * 10
                            + (1 if mol.atoms[idx].is_aromatic else 0)
                        )
                        for idx in canon
                    ]
                    fid = _hash_tuple(sigs) % nbits
                    vec[fid] += 1.0
    return vec


def tanimoto_similarity(fp_a: NDArray[np.float64], fp_b: NDArray[np.float64]) -> float:
    """Compute continuous/binary Tanimoto (Jaccard) similarity in [0.0, 1.0]."""
    dot = float(np.dot(fp_a, fp_b))
    norm_a = float(np.dot(fp_a, fp_a))
    norm_b = float(np.dot(fp_b, fp_b))
    denom = norm_a + norm_b - dot
    if denom <= 1e-12:
        return 1.0 if (norm_a <= 1e-12 and norm_b <= 1e-12) else 0.0
    return max(0.0, min(1.0, dot / denom))


def dice_similarity(fp_a: NDArray[np.float64], fp_b: NDArray[np.float64]) -> float:
    """Compute Dice-Sorensen similarity between two fingerprint vectors."""
    dot = float(np.dot(fp_a, fp_b))
    denom = float(np.dot(fp_a, fp_a) + np.dot(fp_b, fp_b))
    if denom <= 1e-12:
        return 1.0
    return max(0.0, min(1.0, (2.0 * dot) / denom))


def tversky_similarity(
    fp_a: NDArray[np.float64],
    fp_b: NDArray[np.float64],
    alpha: float = 0.7,
    beta: float = 0.3,
) -> float:
    """Compute asymmetric Tversky substructure-biased similarity."""
    dot = float(np.dot(fp_a, fp_b))
    a_only = max(0.0, float(np.dot(fp_a, fp_a)) - dot)
    b_only = max(0.0, float(np.dot(fp_b, fp_b)) - dot)
    denom = dot + alpha * a_only + beta * b_only
    if denom <= 1e-12:
        return 1.0
    return max(0.0, min(1.0, dot / denom))


def pairwise_tanimoto_matrix(fps: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute vectorized NxN Tanimoto similarity matrix for stacked fingerprints (N, B)."""
    if fps.ndim != 2:
        raise ValueError("Expected 2D fingerprint matrix")
    gram = fps @ fps.T
    norms = np.diag(gram)[:, None]
    denom = norms + norms.T - gram
    sim = np.where(denom > 1e-12, gram / np.maximum(denom, 1e-12), 1.0)
    return np.clip(sim, 0.0, 1.0)


def butina_clustering(
    fps: NDArray[np.float64],
    similarity_threshold: float = 0.35,
) -> List[List[int]]:
    """Taylor-Butina sphere-exclusion clustering ordered by neighbor density."""
    n = fps.shape[0]
    if n == 0:
        return []
    sim_mat = pairwise_tanimoto_matrix(fps)
    neighbor_lists: List[List[int]] = []
    for i in range(n):
        nbrs = [j for j in range(n) if j != i and sim_mat[i, j] >= similarity_threshold]
        neighbor_lists.append(nbrs)

    order = sorted(range(n), key=lambda i: (-len(neighbor_lists[i]), i))
    assigned = [False] * n
    clusters: List[List[int]] = []

    for seed in order:
        if assigned[seed]:
            continue
        cluster = [seed]
        assigned[seed] = True
        for nbr in neighbor_lists[seed]:
            if not assigned[nbr]:
                assigned[nbr] = True
                cluster.append(nbr)
        clusters.append(cluster)

    clusters.sort(key=lambda c: (-len(c), c[0]))
    return clusters


def maxmin_diversity_selection(
    fps: NDArray[np.float64],
    k: int,
    seed_idx: int = 0,
) -> List[int]:
    """Select k maximally diverse compound indices using greedy MaxMin Tanimoto distance."""
    n = fps.shape[0]
    if n == 0 or k <= 0:
        return []
    k_eff = min(k, n)
    sim_mat = pairwise_tanimoto_matrix(fps)
    dist_mat = 1.0 - sim_mat

    selected: List[int] = [seed_idx]
    min_dist_to_selected = dist_mat[seed_idx].copy()
    selected_set = {seed_idx}

    while len(selected) < k_eff:
        best_idx = -1
        best_dist = -1.0
        for i in range(n):
            if i not in selected_set and min_dist_to_selected[i] > best_dist:
                best_dist = float(min_dist_to_selected[i])
                best_idx = i
        if best_idx < 0:
            break
        selected.append(best_idx)
        selected_set.add(best_idx)
        min_dist_to_selected = np.minimum(min_dist_to_selected, dist_mat[best_idx])

    return selected
