"""Molecular graph tensor featurization, normalized Laplacian, shortest-path matrices, and batch collation."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import numpy as np
from numpy.typing import NDArray

from .chemistry import compute_conjugation, compute_gasteiger_charges, perceive_sssr
from .types import ELEMENT_TABLE, BondOrder, GraphTensors, Hybridization, Molecule


_ELEMENT_SLOTS: Tuple[str, ...] = ("C", "N", "O", "S", "F", "Cl", "Br", "P", "B")
ATOM_FEATURE_DIM = 24
EDGE_FEATURE_DIM = 8


def _encode_atom_features(mol: Molecule, atom_idx: int) -> NDArray[np.float64]:
    """Build 24-dimensional atom feature vector including electronic & topological invariants."""
    atom = mol.atoms[atom_idx]
    vec = np.zeros(ATOM_FEATURE_DIM, dtype=np.float64)

    if atom.symbol in _ELEMENT_SLOTS:
        vec[_ELEMENT_SLOTS.index(atom.symbol)] = 1.0
    else:
        vec[9] = 1.0

    vec[10] = min(1.0, mol.degree(atom_idx) / 4.0)
    vec[11] = min(1.0, atom.total_h / 4.0)
    vec[12] = max(-1.0, min(1.0, float(atom.formal_charge) / 2.0))
    vec[13] = max(-1.0, min(1.0, atom.partial_charge))

    if atom.hybridization == Hybridization.SP:
        vec[14] = 1.0
    elif atom.hybridization == Hybridization.SP2:
        vec[15] = 1.0
    elif atom.hybridization == Hybridization.SP3:
        vec[16] = 1.0

    vec[17] = 1.0 if atom.is_aromatic else 0.0
    vec[18] = 1.0 if atom.is_in_ring else 0.0
    vec[19] = 1.0 if any(s in (3, 4, 5) for s in atom.ring_sizes) else 0.0
    vec[20] = 1.0 if 6 in atom.ring_sizes else 0.0

    vec[21] = 1.0 if (atom.symbol in ("N", "O", "S") and atom.total_h > 0) else 0.0
    vec[22] = 1.0 if (atom.symbol in ("N", "O") and atom.formal_charge <= 0) else 0.0
    info = ELEMENT_TABLE[atom.symbol]
    vec[23] = (info.electronegativity - 2.0) / 2.0
    return vec


def _encode_bond_features(mol: Molecule, bond_idx: int) -> NDArray[np.float64]:
    """Build 8-dimensional directed bond feature vector."""
    bond = mol.bonds[bond_idx]
    vec = np.zeros(EDGE_FEATURE_DIM, dtype=np.float64)

    if bond.order == BondOrder.SINGLE:
        vec[0] = 1.0
    elif bond.order == BondOrder.DOUBLE:
        vec[1] = 1.0
    elif bond.order == BondOrder.TRIPLE:
        vec[2] = 1.0
    else:
        vec[3] = 1.0

    vec[4] = 1.0 if bond.is_aromatic else 0.0
    vec[5] = 1.0 if bond.is_in_ring else 0.0
    vec[6] = 1.0 if bond.is_conjugated else 0.0
    vec[7] = float(bond.order.value) / 3.0
    return vec


def _floyd_warshall(n: int, edges: Sequence[Tuple[int, int]]) -> NDArray[np.int64]:
    """Compute all-pairs shortest-path topological distances."""
    inf_val = n + 10
    dist = np.full((n, n), inf_val, dtype=np.int64)
    np.fill_diagonal(dist, 0)
    for u, v in edges:
        dist[u, v] = 1
        dist[v, u] = 1
    for k in range(n):
        cand = dist[:, k : k + 1] + dist[k : k + 1, :]
        dist = np.minimum(dist, cand)
    return dist


def featurize_molecule(mol: Molecule) -> GraphTensors:
    """Convert a Molecule into normalized GraphTensors for GNN message passing."""
    work = mol.copy()
    perceive_sssr(work)
    compute_conjugation(work)
    compute_gasteiger_charges(work)

    n = work.num_atoms
    x = np.zeros((n, ATOM_FEATURE_DIM), dtype=np.float64)
    for i in range(n):
        x[i] = _encode_atom_features(work, i)

    src_list: List[int] = []
    dst_list: List[int] = []
    edge_feats: List[NDArray[np.float64]] = []
    undirected_pairs: List[Tuple[int, int]] = []

    adj = np.zeros((n, n), dtype=np.float64)
    for bond in work.bonds:
        u, v = bond.begin_idx, bond.end_idx
        bf = _encode_bond_features(work, bond.index)
        src_list.extend([u, v])
        dst_list.extend([v, u])
        edge_feats.extend([bf, bf.copy()])
        adj[u, v] = 1.0
        adj[v, u] = 1.0
        undirected_pairs.append((u, v))

    if src_list:
        edge_index = np.array([src_list, dst_list], dtype=np.int64)
        e_mat = np.vstack(edge_feats)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        e_mat = np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float64)

    a_hat = adj + np.eye(n, dtype=np.float64)
    deg = np.sum(a_hat, axis=1)
    inv_sqrt = np.where(deg > 0.0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    norm_adj = (inv_sqrt[:, None] * a_hat) * inv_sqrt[None, :]
    spd = _floyd_warshall(n, undirected_pairs)

    return GraphTensors(
        node_features=x,
        edge_index=edge_index,
        edge_features=e_mat,
        adjacency=adj,
        norm_adjacency=norm_adj,
        shortest_paths=spd,
        smiles=work.smiles,
    )


def collate_graph_batch(graphs: Sequence[GraphTensors]) -> Dict[str, NDArray[np.float64] | NDArray[np.int64]]:
    """Block-diagonalize a mini-batch of molecular GraphTensors for vectorized inference."""
    if not graphs:
        raise ValueError("Cannot collate empty graph list")

    total_nodes = sum(g.node_features.shape[0] for g in graphs)
    x_cat = np.vstack([g.node_features for g in graphs])
    norm_adj_blk = np.zeros((total_nodes, total_nodes), dtype=np.float64)
    batch_idx = np.zeros(total_nodes, dtype=np.int64)

    src_parts: List[NDArray[np.int64]] = []
    dst_parts: List[NDArray[np.int64]] = []
    e_parts: List[NDArray[np.float64]] = []

    offset = 0
    for g_id, g in enumerate(graphs):
        n = g.node_features.shape[0]
        norm_adj_blk[offset : offset + n, offset : offset + n] = g.norm_adjacency
        batch_idx[offset : offset + n] = g_id
        if g.edge_index.shape[1] > 0:
            src_parts.append(g.edge_index[0] + offset)
            dst_parts.append(g.edge_index[1] + offset)
            e_parts.append(g.edge_features)
        offset += n

    if src_parts:
        edge_index = np.vstack([np.concatenate(src_parts), np.concatenate(dst_parts)])
        edge_features = np.vstack(e_parts)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_features = np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float64)

    return {
        "node_features": x_cat,
        "norm_adjacency": norm_adj_blk,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "batch_index": batch_idx,
    }
