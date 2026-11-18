"""3D distance-geometry conformer embedding and Vina/GBSA empirical receptor-ligand docking."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from .chemistry import compute_gasteiger_charges, count_rotatable_bonds, perceive_sssr
from .types import ELEMENT_TABLE, BondOrder, DockingPoseResult, Molecule


@dataclass(frozen=True)
class PocketSite:
    """Receptor binding-pocket interaction hotspot."""

    coords: Tuple[float, float, float]
    kind: str
    vdw_radius: float = 1.7
    partial_charge: float = 0.0


def default_kinase_pocket() -> List[PocketSite]:
    """Construct an ATP-hinge / DFG-like synthetic receptor binding cavity."""
    return [
        PocketSite((3.2, 1.4, 0.5), "acceptor", vdw_radius=1.52, partial_charge=-0.45),
        PocketSite((3.4, -1.3, 0.4), "donor", vdw_radius=1.55, partial_charge=0.32),
        PocketSite((-2.6, 1.8, -0.8), "hydrophobic", vdw_radius=1.80, partial_charge=0.0),
        PocketSite((-3.4, -0.6, 1.1), "hydrophobic", vdw_radius=1.80, partial_charge=0.0),
        PocketSite((-1.8, -2.1, -1.2), "hydrophobic", vdw_radius=1.75, partial_charge=0.0),
        PocketSite((0.4, 2.8, 1.2), "donor", vdw_radius=1.55, partial_charge=0.38),
        PocketSite((0.8, -2.9, -0.9), "acceptor", vdw_radius=1.52, partial_charge=-0.42),
        PocketSite((0.0, 0.0, 4.8), "steric_wall", vdw_radius=1.85, partial_charge=0.0),
        PocketSite((0.0, 0.0, -4.8), "steric_wall", vdw_radius=1.85, partial_charge=0.0),
        PocketSite((6.2, 0.0, 0.0), "steric_wall", vdw_radius=1.90, partial_charge=0.0),
        PocketSite((-6.4, 0.0, 0.0), "steric_wall", vdw_radius=1.90, partial_charge=0.0),
    ]


def _bond_target_length(mol: Molecule, u: int, v: int, order: BondOrder) -> float:
    """Estimate covalent equilibrium bond length in Angstroms from covalent radii."""
    r1 = ELEMENT_TABLE[mol.atoms[u].symbol].covalent_radius
    r2 = ELEMENT_TABLE[mol.atoms[v].symbol].covalent_radius
    base = r1 + r2
    if order == BondOrder.DOUBLE:
        return base - 0.18
    if order == BondOrder.TRIPLE:
        return base - 0.30
    if order == BondOrder.AROMATIC:
        return base - 0.12
    return base


def embed_3d_conformer(mol: Molecule, seed: int = 42) -> Tuple[NDArray[np.float64], float]:
    """Generate a 3D molecular conformer via Metric Matrix Distance Geometry and BFGS refinement."""
    n = mol.num_atoms
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64), 0.0
    if n == 1:
        return np.zeros((1, 3), dtype=np.float64), 0.0

    perceive_sssr(mol)
    lower = np.full((n, n), 2.2, dtype=np.float64)
    upper = np.full((n, n), 12.0, dtype=np.float64)
    np.fill_diagonal(lower, 0.0)
    np.fill_diagonal(upper, 0.0)

    bond_targets: List[Tuple[int, int, float]] = []
    for bond in mol.bonds:
        u, v = bond.begin_idx, bond.end_idx
        d0 = _bond_target_length(mol, u, v, bond.order)
        lower[u, v] = lower[v, u] = d0 - 0.03
        upper[u, v] = upper[v, u] = d0 + 0.03
        bond_targets.append((u, v, d0))

    angle_targets: List[Tuple[int, int, float]] = []
    for center in range(n):
        nbrs = mol.neighbors(center)
        theta = 2.0944 if (mol.atoms[center].is_aromatic or len(nbrs) <= 3) else 1.9106
        for idx_a in range(len(nbrs)):
            u, b1 = nbrs[idx_a]
            d1 = _bond_target_length(mol, center, u, b1.order)
            for idx_b in range(idx_a + 1, len(nbrs)):
                v, b2 = nbrs[idx_b]
                d2 = _bond_target_length(mol, center, v, b2.order)
                d13 = math.sqrt(max(0.5, d1 * d1 + d2 * d2 - 2.0 * d1 * d2 * math.cos(theta)))
                if d13 < upper[u, v]:
                    lower[u, v] = lower[v, u] = d13 - 0.08
                    upper[u, v] = upper[v, u] = d13 + 0.08
                    angle_targets.append((u, v, d13))

    for k in range(n):
        upper = np.minimum(upper, upper[:, k : k + 1] + upper[k : k + 1, :])

    rng = np.random.default_rng(seed)
    frac = rng.uniform(0.35, 0.65, size=(n, n))
    frac = 0.5 * (frac + frac.T)
    d_target = lower + frac * np.maximum(0.0, upper - lower)
    np.fill_diagonal(d_target, 0.0)

    d_sq = d_target * d_target
    j_mat = np.eye(n) - np.ones((n, n)) / float(n)
    gram = -0.5 * (j_mat @ d_sq @ j_mat)
    eigvals, eigvecs = np.linalg.eigh(gram)
    top_idx = np.argsort(eigvals)[::-1][:3]
    vals = np.maximum(0.0, eigvals[top_idx])
    vecs = eigvecs[:, top_idx]
    coords_init = np.zeros((n, 3), dtype=np.float64)
    k_cols = min(3, vecs.shape[1])
    coords_init[:, :k_cols] = vecs[:, :k_cols] * np.sqrt(vals[:k_cols])[None, :]
    coords_init += rng.normal(0.0, 0.05, size=(n, 3))

    bonded_set = {(min(u, v), max(u, v)) for u, v, _ in bond_targets}

    def stress_fn(flat: NDArray[np.float64]) -> float:
        pos = flat.reshape((n, 3))
        e = 0.0
        for u, v, d0 in bond_targets:
            dist = float(np.linalg.norm(pos[u] - pos[v]))
            e += 120.0 * (dist - d0) ** 2
        for u, v, d13 in angle_targets:
            dist = float(np.linalg.norm(pos[u] - pos[v]))
            e += 30.0 * (dist - d13) ** 2
        diff = pos[:, None, :] - pos[None, :, :]
        dmat = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-8)
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in bonded_set and dmat[i, j] < 2.1:
                    e += 25.0 * (2.1 - float(dmat[i, j])) ** 2
        return e

    res = minimize(stress_fn, coords_init.ravel(), method="BFGS", options={"maxiter": 80})
    pos_opt = res.x.reshape((n, 3))
    pos_opt -= np.mean(pos_opt, axis=0, keepdims=True)
    for idx, atom in enumerate(mol.atoms):
        atom.coords = (float(pos_opt[idx, 0]), float(pos_opt[idx, 1]), float(pos_opt[idx, 2]))
    return pos_opt, float(res.fun)


def _rotation_matrix_from_euler(rx: float, ry: float, rz: float) -> NDArray[np.float64]:
    """Construct SO(3) rotation matrix from Z-Y-X Euler angles."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return np.asarray(rz_m @ ry_m @ rx_m, dtype=np.float64)


def score_ligand_pose(
    mol: Molecule,
    coords: NDArray[np.float64],
    pocket: Optional[List[PocketSite]] = None,
    internal_strain: float = 0.0,
) -> DockingPoseResult:
    """Evaluate Vina-inspired empirical binding score + Still GBSA desolvation in kcal/mol."""
    sites = pocket or default_kinase_pocket()
    compute_gasteiger_charges(mol)
    rot_bonds = count_rotatable_bonds(mol)

    gauss1 = 0.0
    gauss2 = 0.0
    repulsion = 0.0
    hydrophobic = 0.0
    hbond = 0.0
    gbsa = 0.0
    contacts = 0

    for idx, atom in enumerate(mol.atoms):
        r_a = ELEMENT_TABLE[atom.symbol].vdw_radius
        pos_a = coords[idx]
        is_hbd = atom.symbol in ("N", "O", "S") and atom.total_h > 0
        is_hba = atom.symbol in ("N", "O") and atom.formal_charge <= 0
        is_hydro = atom.symbol in ("C", "F", "Cl", "Br", "S") and not is_hbd and not is_hba

        born_r = r_a + 0.45
        gbsa += 0.25 * (atom.partial_charge ** 2) * (1.0 / born_r)

        for site in sites:
            d = float(np.linalg.norm(pos_a - np.array(site.coords)))
            d_surf = d - (r_a + site.vdw_radius)
            if d < 4.5 and site.kind != "steric_wall":
                contacts += 1

            gauss1 += -0.035579 * math.exp(-((d_surf / 0.5) ** 2))
            gauss2 += -0.005156 * math.exp(-(((d_surf - 3.0) / 2.0) ** 2))
            if d_surf < 0.0:
                repulsion += 0.840245 * (d_surf ** 2)

            if site.kind == "hydrophobic" and is_hydro:
                if d_surf <= 0.5:
                    hydrophobic += -0.045
                elif d_surf < 1.5:
                    hydrophobic += -0.045 * (1.5 - d_surf)

            if ((site.kind == "acceptor" and is_hbd) or (site.kind == "donor" and is_hba)) and d_surf <= 0.6:
                hb_w = max(0.0, min(1.0, (0.6 - d_surf) / 1.3))
                hbond += -0.5874 * hb_w

            if abs(site.partial_charge) > 1e-6:
                eps_eff = 4.0 * max(1.5, d)
                gbsa += 1.65 * (atom.partial_charge * site.partial_charge) / eps_eff

    inter_score = 4.2 * gauss1 + 5.5 * gauss2 + 2.8 * repulsion + 6.5 * hydrophobic + 2.2 * hbond + gbsa
    torsion_pen = 0.29 * rot_bonds
    total = inter_score + torsion_pen + 0.02 * internal_strain
    le = -total / max(1, mol.num_atoms)

    return DockingPoseResult(
        smiles=mol.smiles,
        coords=coords.copy(),
        total_score_kcal_mol=round(total, 4),
        gauss1_term=round(4.2 * gauss1, 4),
        gauss2_term=round(5.5 * gauss2, 4),
        repulsion_term=round(2.8 * repulsion, 4),
        hydrophobic_term=round(6.5 * hydrophobic, 4),
        hbond_term=round(2.2 * hbond, 4),
        torsion_penalty=round(torsion_pen, 4),
        gbsa_desolvation=round(gbsa, 4),
        internal_strain=round(internal_strain, 4),
        pocket_contacts=contacts,
        ligand_efficiency=round(le, 4),
    )


def dock_ligand(
    mol: Molecule,
    pocket: Optional[List[PocketSite]] = None,
    seed: int = 42,
    n_restarts: int = 6,
) -> DockingPoseResult:
    """Embed a 3D conformer and optimize rigid-body 6-DOF pose inside a receptor pocket."""
    work = mol.copy()
    base_coords, strain = embed_3d_conformer(work, seed=seed)
    sites = pocket or default_kinase_pocket()
    rng = np.random.default_rng(seed)

    def pose_energy(pose_vec: NDArray[np.float64]) -> float:
        tx, ty, tz, rx, ry, rz = pose_vec
        rot = _rotation_matrix_from_euler(float(rx), float(ry), float(rz))
        transformed = base_coords @ rot.T + np.array([tx, ty, tz])
        return score_ligand_pose(work, transformed, pocket=sites, internal_strain=strain).total_score_kcal_mol

    best_vec = np.zeros(6, dtype=np.float64)
    best_score = pose_energy(best_vec)

    for _restart in range(n_restarts):
        init = np.array(
            [
                rng.uniform(-1.5, 1.5),
                rng.uniform(-1.5, 1.5),
                rng.uniform(-1.0, 1.0),
                rng.uniform(-math.pi, math.pi),
                rng.uniform(-math.pi / 2.0, math.pi / 2.0),
                rng.uniform(-math.pi, math.pi),
            ],
            dtype=np.float64,
        )
        res = minimize(pose_energy, init, method="Powell", options={"maxiter": 60, "xtol": 1e-2, "ftol": 1e-2})
        if float(res.fun) < best_score:
            best_score = float(res.fun)
            best_vec = res.x.copy()

    tx, ty, tz, rx, ry, rz = best_vec
    rot_opt = _rotation_matrix_from_euler(float(rx), float(ry), float(rz))
    final_coords = base_coords @ rot_opt.T + np.array([tx, ty, tz])
    return score_ligand_pose(work, final_coords, pocket=sites, internal_strain=strain)
