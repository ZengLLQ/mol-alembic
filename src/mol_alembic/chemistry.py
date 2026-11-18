"""Ring perception (SSSR), Huckel aromaticity, Kekulization, Gasteiger charges, and QED/TPSA/logP."""

from __future__ import annotations

from collections import deque
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .types import (
    ELEMENT_TABLE,
    BondOrder,
    Hybridization,
    KekulizationError,
    MolecularDescriptors,
    Molecule,
    RingSystem,
)


def _shortest_path_excluding_edge(
    mol: Molecule,
    start: int,
    target: int,
    excluded_bond_idx: int,
) -> Optional[Tuple[List[int], List[int]]]:
    """Find shortest path (atom_indices, bond_indices) between start and target without excluded_bond_idx."""
    queue: deque[int] = deque([start])
    parent_atom: Dict[int, Optional[int]] = {start: None}
    parent_bond: Dict[int, Optional[int]] = {start: None}

    while queue:
        u = queue.popleft()
        if u == target:
            atoms_path: List[int] = []
            bonds_path: List[int] = []
            cur: Optional[int] = target
            while cur is not None:
                atoms_path.append(cur)
                b_idx = parent_bond[cur]
                if b_idx is not None:
                    bonds_path.append(b_idx)
                cur = parent_atom[cur]
            atoms_path.reverse()
            bonds_path.reverse()
            return atoms_path, bonds_path

        for v, bond in mol.neighbors(u):
            if bond.index == excluded_bond_idx:
                continue
            if v not in parent_atom:
                parent_atom[v] = u
                parent_bond[v] = bond.index
                queue.append(v)
    return None


def _gf2_independent(basis_masks: List[int], candidate_mask: int) -> bool:
    """Return True if candidate_mask is linearly independent over GF(2) from basis_masks."""
    x = candidate_mask
    for pivot in basis_masks:
        msb = pivot.bit_length() - 1
        if (x >> msb) & 1:
            x ^= pivot
    return x != 0


def _gf2_insert(basis_masks: List[int], candidate_mask: int) -> None:
    """Insert candidate_mask into an echelon GF(2) basis in place."""
    x = candidate_mask
    for pivot in basis_masks:
        msb = pivot.bit_length() - 1
        if (x >> msb) & 1:
            x ^= pivot
    if x != 0:
        basis_masks.append(x)
        basis_masks.sort(reverse=True)


def perceive_sssr(mol: Molecule) -> List[RingSystem]:
    """Compute a Smallest Set of Smallest Rings (SSSR) via Horton shortest cycles and GF(2) pruning."""
    n_atoms = mol.num_atoms
    n_bonds = mol.num_bonds
    for atom in mol.atoms:
        atom.is_in_ring = False
        atom.ring_sizes = ()
    for bond in mol.bonds:
        bond.is_in_ring = False

    if n_atoms == 0 or n_bonds == 0:
        return []

    visited: Set[int] = set()
    components = 0
    for idx in range(n_atoms):
        if idx not in visited:
            components += 1
            dq: deque[int] = deque([idx])
            visited.add(idx)
            while dq:
                u = dq.popleft()
                for v, _ in mol.neighbors(u):
                    if v not in visited:
                        visited.add(v)
                        dq.append(v)

    expected_dim = n_bonds - n_atoms + components
    if expected_dim <= 0:
        return []

    candidate_cycles: Dict[int, Tuple[int, Tuple[int, ...], Tuple[int, ...]]] = {}

    for root in range(n_atoms):
        dist: Dict[int, int] = {root: 0}
        parent: Dict[int, Optional[int]] = {root: None}
        parent_bond: Dict[int, Optional[int]] = {root: None}
        dq = deque([root])
        while dq:
            u = dq.popleft()
            for v, bond in mol.neighbors(u):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    parent_bond[v] = bond.index
                    dq.append(v)

        for bond in mol.bonds:
            u, v = bond.begin_idx, bond.end_idx
            if u not in dist or v not in dist:
                continue
            if parent.get(v) == u or parent.get(u) == v:
                continue
            path_u_atoms: List[int] = []
            path_u_bonds: List[int] = []
            cur: Optional[int] = u
            while cur is not None:
                path_u_atoms.append(cur)
                if parent_bond[cur] is not None:
                    path_u_bonds.append(parent_bond[cur] or 0)
                cur = parent[cur]

            path_v_atoms: List[int] = []
            path_v_bonds: List[int] = []
            cur = v
            while cur is not None:
                path_v_atoms.append(cur)
                if parent_bond[cur] is not None:
                    path_v_bonds.append(parent_bond[cur] or 0)
                cur = parent[cur]

            set_u = set(path_u_atoms)
            lca = None
            for a_idx in path_v_atoms:
                if a_idx in set_u:
                    lca = a_idx
                    break
            if lca is None:
                continue

            cut_u = path_u_atoms.index(lca)
            cut_v = path_v_atoms.index(lca)
            cyc_atoms_list = path_u_atoms[: cut_u + 1] + list(reversed(path_v_atoms[:cut_v]))
            if len(set(cyc_atoms_list)) != len(cyc_atoms_list):
                continue
            cyc_bonds_list = path_u_bonds[:cut_u] + [bond.index] + path_v_bonds[:cut_v]
            mask = 0
            for b_idx in cyc_bonds_list:
                mask |= 1 << b_idx
            if mask not in candidate_cycles:
                candidate_cycles[mask] = (len(cyc_atoms_list), tuple(cyc_atoms_list), tuple(cyc_bonds_list))

    for bond in mol.bonds:
        res = _shortest_path_excluding_edge(mol, bond.begin_idx, bond.end_idx, bond.index)
        if res is not None:
            a_path, b_path = res
            cyc_bonds_list = b_path + [bond.index]
            mask = 0
            for b_idx in cyc_bonds_list:
                mask |= 1 << b_idx
            if mask not in candidate_cycles:
                candidate_cycles[mask] = (len(a_path), tuple(a_path), tuple(cyc_bonds_list))

    sorted_candidates = sorted(candidate_cycles.items(), key=lambda kv: (kv[1][0], kv[0]))
    basis: List[int] = []
    sssr_raw: List[Tuple[int, Tuple[int, ...], Tuple[int, ...]]] = []

    for mask, info in sorted_candidates:
        if _gf2_independent(basis, mask):
            _gf2_insert(basis, mask)
            sssr_raw.append(info)
            if len(sssr_raw) == expected_dim:
                break

    ring_atom_sizes: Dict[int, List[int]] = {i: [] for i in range(n_atoms)}
    for size, cyc_atoms, cyc_bonds in sssr_raw:
        for a_idx in cyc_atoms:
            mol.atoms[a_idx].is_in_ring = True
            ring_atom_sizes[a_idx].append(size)
        for b_idx in cyc_bonds:
            mol.bonds[b_idx].is_in_ring = True

    for a_idx, sizes in ring_atom_sizes.items():
        mol.atoms[a_idx].ring_sizes = tuple(sorted(sizes))

    rings: List[RingSystem] = []
    for size, cyc_atoms, cyc_bonds in sssr_raw:
        pi_count, is_arom = _evaluate_ring_aromaticity(mol, cyc_atoms, cyc_bonds)
        rings.append(
            RingSystem(
                atom_indices=cyc_atoms,
                bond_indices=cyc_bonds,
                size=size,
                is_aromatic=is_arom,
                pi_electrons=pi_count,
            )
        )

    for ring in rings:
        if ring.is_aromatic:
            for a_idx in ring.atom_indices:
                mol.atoms[a_idx].is_aromatic = True
                mol.atoms[a_idx].hybridization = Hybridization.SP2
            for b_idx in ring.bond_indices:
                mol.bonds[b_idx].is_aromatic = True
                mol.bonds[b_idx].order = BondOrder.AROMATIC

    return rings


def _atom_pi_contribution(mol: Molecule, atom_idx: int) -> Optional[int]:
    """Return pi-electron count contributed by atom_idx to an aromatic ring, or None if non-planar."""
    atom = mol.atoms[atom_idx]
    deg = mol.degree(atom_idx)
    nbrs = mol.neighbors(atom_idx)

    has_arom_bond = any(b.is_aromatic or b.order == BondOrder.AROMATIC for _, b in nbrs)
    has_endo_double = any(
        b.order == BondOrder.DOUBLE and mol.atoms[nbr].is_in_ring for nbr, b in nbrs
    )
    has_exo_carbonyl = any(
        b.order == BondOrder.DOUBLE and not mol.atoms[nbr].is_in_ring and mol.atoms[nbr].symbol in ("O", "S", "N")
        for nbr, b in nbrs
    )

    if atom.symbol == "C":
        if atom.formal_charge == 0:
            if has_exo_carbonyl:
                return 0
            if atom.is_aromatic or has_arom_bond or has_endo_double:
                return 1
            return None
        if atom.formal_charge == 1:
            return 0
        if atom.formal_charge == -1:
            return 2
        return None

    if atom.symbol == "N":
        if atom.formal_charge == 0:
            if deg == 2 and atom.total_h == 0 and (atom.is_aromatic or has_arom_bond or has_endo_double):
                return 1
            if (deg == 3 or atom.total_h >= 1) and (atom.is_aromatic or has_arom_bond or deg + atom.total_h == 3):
                if atom.is_aromatic or has_arom_bond:
                    return 2
            return None
        if atom.formal_charge == 1 and (atom.is_aromatic or has_endo_double):
            return 1
        return None

    if atom.symbol in ("O", "S"):
        if atom.formal_charge == 0 and deg == 2 and atom.total_h == 0:
            if atom.is_aromatic or has_arom_bond:
                return 2
            if all(
                mol.atoms[n_idx].is_aromatic
                or any(b2.order == BondOrder.DOUBLE for _, b2 in mol.neighbors(n_idx))
                for n_idx, _ in nbrs
            ):
                return 2
        return None

    if atom.symbol == "B" and atom.formal_charge == 0:
        return 0 if (atom.is_aromatic or has_arom_bond) else None

    return None


def _evaluate_ring_aromaticity(
    mol: Molecule,
    cyc_atoms: Tuple[int, ...],
    cyc_bonds: Tuple[int, ...],
) -> Tuple[int, bool]:
    """Evaluate Huckel 4n+2 rule on a candidate cycle."""
    if len(cyc_atoms) < 5 or len(cyc_atoms) > 7:
        if all(mol.bonds[b].is_aromatic for b in cyc_bonds):
            return len(cyc_atoms), True
        return 0, False

    if all(mol.bonds[b].is_aromatic or mol.atoms[a].is_aromatic for a in cyc_atoms for b in cyc_bonds):
        pi_vals = [_atom_pi_contribution(mol, a) for a in cyc_atoms]
        if all(v is not None for v in pi_vals):
            total_pi = sum(v for v in pi_vals if v is not None)
            if total_pi >= 2 and (total_pi - 2) % 4 == 0:
                return total_pi, True
        return len(cyc_atoms), True

    pi_vals = [_atom_pi_contribution(mol, a) for a in cyc_atoms]
    if any(v is None for v in pi_vals):
        return 0, False
    total_pi = sum(v for v in pi_vals if v is not None)
    is_huckel = total_pi >= 2 and (total_pi - 2) % 4 == 0
    return total_pi, is_huckel


def kekulize_molecule(mol: Molecule) -> Molecule:
    """Assign alternating SINGLE and DOUBLE Kekule bond orders to aromatic subgraphs."""
    out = mol.copy()
    if not out.bonds:
        return out

    arom_atoms = [a.index for a in out.atoms if a.is_aromatic]
    if not arom_atoms:
        return out

    needs_double: Set[int] = set()
    for a_idx in arom_atoms:
        atom = out.atoms[a_idx]
        has_existing_double = any(
            b.order == BondOrder.DOUBLE and not b.is_aromatic for _, b in out.neighbors(a_idx)
        )
        if has_existing_double:
            continue
        if atom.symbol == "C" and atom.formal_charge == 0:
            needs_double.add(a_idx)
        elif atom.symbol == "N" and atom.formal_charge == 0 and atom.total_h == 0 and out.degree(a_idx) == 2:
            needs_double.add(a_idx)
        elif atom.symbol == "N" and atom.formal_charge == 1:
            needs_double.add(a_idx)

    arom_adj: Dict[int, List[Tuple[int, int]]] = {a_idx: [] for a_idx in needs_double}
    for bond in out.bonds:
        if bond.is_aromatic or bond.order == BondOrder.AROMATIC:
            bond.order = BondOrder.SINGLE
            if bond.begin_idx in needs_double and bond.end_idx in needs_double:
                arom_adj[bond.begin_idx].append((bond.end_idx, bond.index))
                arom_adj[bond.end_idx].append((bond.begin_idx, bond.index))

    if not needs_double:
        return out

    nodes = sorted(needs_double, key=lambda idx: len(arom_adj[idx]))
    matched: Dict[int, int] = {}
    matched_bonds: Set[int] = set()

    def backtrack(pos: int) -> bool:
        while pos < len(nodes) and nodes[pos] in matched:
            pos += 1
        if pos == len(nodes):
            return True
        u = nodes[pos]
        for v, b_idx in arom_adj[u]:
            if v not in matched:
                matched[u] = v
                matched[v] = u
                matched_bonds.add(b_idx)
                if backtrack(pos + 1):
                    return True
                del matched[u]
                del matched[v]
                matched_bonds.remove(b_idx)
        return False

    if not backtrack(0):
        raise KekulizationError(f"Failed to Kekulize aromatic system in '{mol.smiles}'")

    for b_idx in matched_bonds:
        out.bonds[b_idx].order = BondOrder.DOUBLE
    return out


def compute_gasteiger_charges(mol: Molecule, iterations: int = 8) -> List[float]:
    """Compute Gasteiger-Marsili iterative partial electronegativity equalization charges."""
    n = mol.num_atoms
    if n == 0:
        return []

    charges = [float(a.formal_charge) for a in mol.atoms]
    damping = 1.0
    chi_h_pos = 20.02

    for _ in range(iterations):
        damping *= 0.5
        chi = [0.0] * n
        for idx, atom in enumerate(mol.atoms):
            info = ELEMENT_TABLE[atom.symbol]
            q = charges[idx]
            chi[idx] = info.gasteiger_a + info.gasteiger_b * q + info.gasteiger_c * (q * q)

        delta_q = [0.0] * n
        for bond in mol.bonds:
            i, j = bond.begin_idx, bond.end_idx
            if chi[i] == chi[j]:
                continue
            if chi[j] > chi[i]:
                donor, acceptor = i, j
            else:
                donor, acceptor = j, i
            info_d = ELEMENT_TABLE[mol.atoms[donor].symbol]
            chi_d_plus = info_d.gasteiger_a + info_d.gasteiger_b + info_d.gasteiger_c
            denom = chi_d_plus if chi_d_plus > 1e-6 else chi_h_pos
            dq = damping * (chi[acceptor] - chi[donor]) / denom
            delta_q[donor] += dq
            delta_q[acceptor] -= dq

        for idx, atom in enumerate(mol.atoms):
            if atom.total_h > 0:
                chi_i = chi[idx]
                chi_h = 7.17
                if chi_i > chi_h:
                    dq_h = damping * (chi_i - chi_h) / chi_h_pos
                    delta_q[idx] -= 0.15 * atom.total_h * dq_h

        for idx in range(n):
            charges[idx] += delta_q[idx]

    target_total = float(sum(a.formal_charge for a in mol.atoms))
    drift = (sum(charges) - target_total) / n
    charges = [c - drift for c in charges]

    for idx, c in enumerate(charges):
        mol.atoms[idx].partial_charge = c
    return charges


def compute_conjugation(mol: Molecule) -> None:
    """Flag conjugated bonds (aromatic or alternating single/multiple bonds)."""
    atom_unsat = [
        a.is_aromatic
        or any(b.order in (BondOrder.DOUBLE, BondOrder.TRIPLE, BondOrder.AROMATIC) for _, b in mol.neighbors(a.index))
        for a in mol.atoms
    ]
    for bond in mol.bonds:
        if bond.is_aromatic or bond.order in (BondOrder.DOUBLE, BondOrder.TRIPLE, BondOrder.AROMATIC):
            bond.is_conjugated = True
        elif atom_unsat[bond.begin_idx] and atom_unsat[bond.end_idx]:
            bond.is_conjugated = True
        else:
            bond.is_conjugated = False


def compute_ertl_tpsa(mol: Molecule) -> float:
    """Estimate Ertl Topological Polar Surface Area (TPSA) in Angstrom^2 for N, O, S, P."""
    tpsa = 0.0
    for atom in mol.atoms:
        sym = atom.symbol
        if sym not in ("N", "O", "S", "P"):
            continue
        deg = mol.degree(atom.index)
        h = atom.total_h
        q = atom.formal_charge
        nbrs = mol.neighbors(atom.index)
        n_double = sum(1 for _, b in nbrs if b.order == BondOrder.DOUBLE)
        n_triple = sum(1 for _, b in nbrs if b.order == BondOrder.TRIPLE)

        if sym == "O":
            if q == -1:
                tpsa += 23.06
            elif atom.is_aromatic:
                tpsa += 13.14
            elif n_double == 1 and deg == 1:
                tpsa += 17.07
            elif h == 1 and deg == 1:
                tpsa += 20.23
            elif deg == 2 and h == 0:
                tpsa += 9.23
            else:
                tpsa += 20.23
        elif sym == "N":
            if atom.is_aromatic:
                tpsa += 15.79 if h >= 1 else 12.89
            elif q == 1:
                tpsa += 16.61 if h >= 2 else 4.44
            elif n_triple == 1:
                tpsa += 23.79
            elif n_double == 1 and deg == 2 and h == 0:
                tpsa += 12.36
            elif n_double == 1 and h == 1:
                tpsa += 23.85
            elif h == 2 and deg == 1:
                tpsa += 26.02
            elif h == 1 and deg == 2:
                tpsa += 12.03
            elif deg == 3 and h == 0:
                tpsa += 3.24
            else:
                tpsa += 12.03
        elif sym == "S":
            if n_double >= 1:
                tpsa += 17.07 if n_double == 1 else 8.38
            elif h == 1:
                tpsa += 38.80
            else:
                tpsa += 25.30
        elif sym == "P":
            tpsa += 9.81 if n_double >= 1 else 13.59
    return round(tpsa, 2)


def compute_crippen_logp(mol: Molecule) -> float:
    """Compute Wildman-Crippen atom-additive octanol-water partition coefficient logP."""
    logp = 0.0
    for atom in mol.atoms:
        info = ELEMENT_TABLE[atom.symbol]
        base = info.crippen_logp
        if atom.symbol == "C":
            if atom.is_aromatic:
                base = 0.158
            elif any(b.order == BondOrder.DOUBLE for _, b in mol.neighbors(atom.index)):
                base = 0.088
            elif any(mol.atoms[n].symbol in ("N", "O", "S") for n, _ in mol.neighbors(atom.index)):
                base = -0.055
            else:
                base = 0.144
        elif atom.symbol == "N":
            if atom.is_aromatic:
                base = -0.323
            elif any(
                b2.order == BondOrder.DOUBLE and mol.atoms[n2].symbol == "O"
                for n1, _ in mol.neighbors(atom.index)
                if mol.atoms[n1].symbol == "C"
                for n2, b2 in mol.neighbors(n1)
            ):
                base = -0.495
            elif atom.formal_charge > 0:
                base = -1.05
        elif atom.symbol == "O":
            if any(b.order == BondOrder.DOUBLE for _, b in mol.neighbors(atom.index)):
                base = -0.258
            elif atom.total_h > 0:
                base = -0.289
            else:
                base = -0.152
        logp += base + 0.123 * atom.total_h
    return round(logp, 3)


def count_rotatable_bonds(mol: Molecule) -> int:
    """Count non-ring single bonds connecting two non-terminal heavy atoms (excluding amide C-N)."""
    rot = 0
    for bond in mol.bonds:
        if bond.is_in_ring or bond.order != BondOrder.SINGLE:
            continue
        a1 = mol.atoms[bond.begin_idx]
        a2 = mol.atoms[bond.end_idx]
        if a1.symbol == "H" or a2.symbol == "H":
            continue
        if mol.degree(a1.index) <= 1 or mol.degree(a2.index) <= 1:
            continue
        is_amide = False
        for c_atom, n_atom in ((a1, a2), (a2, a1)):
            if c_atom.symbol == "C" and n_atom.symbol == "N":
                if any(
                    b.order == BondOrder.DOUBLE and mol.atoms[nbr].symbol == "O"
                    for nbr, b in mol.neighbors(c_atom.index)
                ):
                    is_amide = True
        if not is_amide:
            rot += 1
    return rot


def _ads_desirability(x: float, a: float, b: float, c: float, d: float, e: float, f: float, dmax: float) -> float:
    """Bickerton et al. (2012) Asymmetric Double Sigmoidal (ADS) desirability function."""
    term1 = 1.0 / (1.0 + math.exp(-(x - c + d / 2.0) / max(1e-9, e)))
    term2 = 1.0 - 1.0 / (1.0 + math.exp(-(x - c - d / 2.0) / max(1e-9, f)))
    val = a + b * term1 * term2
    return max(1e-6, min(1.0, val / max(1e-9, dmax)))


def compute_qed(
    mw: float,
    alogp: float,
    hbd: int,
    hba: int,
    psa: float,
    rotb: int,
    arom_rings: int,
    alerts: int = 0,
) -> float:
    """Compute Bickerton et al. (2012) Quantitative Estimate of Drug-likeness (QED)."""
    d_mw = _ads_desirability(
        mw, 2.817065, 392.575495, 290.748976, 2.419764, 49.223256, 65.370517, 104.980556
    )
    d_logp = _ads_desirability(
        alogp, 3.172690, 137.862475, 2.534937, 4.581456, 0.822739, 0.576295, 131.318660
    )
    d_hba = _ads_desirability(
        float(hba), 2.948620, 160.460597, 3.615295, 4.435986, 0.290141, 1.300669, 148.776304
    )
    d_hbd = _ads_desirability(
        float(hbd), 1.618662, 1010.051101, 0.985094, 0.000001, 0.713820, 0.920922, 258.163261
    )
    d_psa = _ads_desirability(
        psa, 1.876861, 125.223265, 62.907737, 87.833666, 12.019998, 28.513247, 104.568616
    )
    d_rotb = _ads_desirability(
        float(rotb), 0.010000, 272.412142, 2.558379, 1.565547, 1.271567, 2.758063, 105.442040
    )
    d_arom = _ads_desirability(
        float(arom_rings), 3.217788, 957.737410, 2.274627, 0.000001, 1.317690, 0.375760, 312.337261
    )
    d_alrt = _ads_desirability(
        float(alerts), 0.010000, 1199.994088, -0.090028, 0.000001, 0.185904, 0.875193, 417.725314
    )

    weights = (0.66, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95)
    vals = (d_mw, d_logp, d_hba, d_hbd, d_psa, d_rotb, d_arom, d_alrt)
    weighted_log = sum(w * math.log(max(1e-9, v)) for w, v in zip(weights, vals)) / sum(weights)
    return round(float(math.exp(weighted_log)), 4)


def _topological_distances(mol: Molecule) -> List[List[int]]:
    """Compute all-pairs shortest topological bond distances (BFS per atom)."""
    n = mol.num_atoms
    dist_mat = [[999999] * n for _ in range(n)]
    for src in range(n):
        dist_mat[src][src] = 0
        dq: deque[int] = deque([src])
        while dq:
            u = dq.popleft()
            d_u = dist_mat[src][u]
            for v, _ in mol.neighbors(u):
                if dist_mat[src][v] > d_u + 1:
                    dist_mat[src][v] = d_u + 1
                    dq.append(v)
    return dist_mat


def _compute_sa_score(mol: Molecule, rings: Sequence[RingSystem]) -> float:
    """Estimate Ertl-Schuffenhauer-inspired Synthetic Accessibility (1.0 easy to 10.0 hard)."""
    n = max(1, mol.num_atoms)
    ring_counts = [len(a.ring_sizes) for a in mol.atoms]
    spiro_or_fused = sum(1 for c in ring_counts if c >= 2)
    unusual_rings = sum(1 for r in rings if r.size not in (5, 6))
    chiral_centers = sum(1 for a in mol.atoms if a.chirality is not None)
    macrocycle = sum(1 for r in rings if r.size >= 8)

    rare_elements = sum(1 for a in mol.atoms if a.symbol in ("B", "Si", "Fe", "I"))
    size_penalty = max(0.0, (n - 20) * 0.08)
    complexity = (
        1.4
        + 0.35 * spiro_or_fused
        + 0.65 * unusual_rings
        + 1.15 * macrocycle
        + 0.55 * chiral_centers
        + 0.85 * rare_elements
        + size_penalty
    )
    return round(max(1.0, min(10.0, complexity)), 3)


def analyze_molecule(mol: Molecule, alert_count: int = 0) -> Tuple[List[RingSystem], MolecularDescriptors]:
    """Run full perception (SSSR, Huckel aromaticity, conjugation, Gasteiger) and compute descriptors."""
    rings = perceive_sssr(mol)
    compute_conjugation(mol)
    compute_gasteiger_charges(mol)

    counts: Dict[str, int] = {}
    mw = 0.0
    hbd = 0
    hba = 0
    f_chg = 0
    sp3_c = 0
    total_c = 0

    for atom in mol.atoms:
        info = ELEMENT_TABLE[atom.symbol]
        counts[atom.symbol] = counts.get(atom.symbol, 0) + 1
        if atom.total_h > 0:
            counts["H"] = counts.get("H", 0) + atom.total_h
        mw += info.atomic_weight + atom.total_h * ELEMENT_TABLE["H"].atomic_weight
        f_chg += atom.formal_charge
        if atom.symbol in ("N", "O", "S") and atom.total_h > 0:
            hbd += 1
        if atom.symbol in ("N", "O") and atom.formal_charge <= 0:
            hba += 1
        if atom.symbol == "C":
            total_c += 1
            if atom.hybridization == Hybridization.SP3:
                sp3_c += 1

    formula_parts: List[str] = []
    for key in ("C", "H"):
        if counts.get(key, 0) > 0:
            cnt = counts[key]
            formula_parts.append(key if cnt == 1 else f"{key}{cnt}")
    for key in sorted(k for k in counts if k not in ("C", "H")):
        cnt = counts[key]
        formula_parts.append(key if cnt == 1 else f"{key}{cnt}")
    formula = "".join(formula_parts)

    logp = compute_crippen_logp(mol)
    tpsa = compute_ertl_tpsa(mol)
    rotb = count_rotatable_bonds(mol)
    arom_rings = sum(1 for r in rings if r.is_aromatic)
    fsp3 = round(sp3_c / total_c, 4) if total_c > 0 else 0.0

    lipinski = 0
    if mw > 500.0:
        lipinski += 1
    if logp > 5.0:
        lipinski += 1
    if hbd > 5:
        lipinski += 1
    if hba > 10:
        lipinski += 1

    veber = 0
    if rotb > 10:
        veber += 1
    if tpsa > 140.0:
        veber += 1

    qed = compute_qed(mw, logp, hbd, hba, tpsa, rotb, arom_rings, alerts=alert_count)
    sa_score = _compute_sa_score(mol, rings)

    dist_mat = _topological_distances(mol)
    n = mol.num_atoms
    wiener = 0
    dist_sums = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            d_ij = dist_mat[i][j]
            if d_ij < 999999:
                wiener += d_ij
                dist_sums[i] += d_ij
                dist_sums[j] += d_ij

    degs = [mol.degree(i) for i in range(n)]
    zagreb_m1 = sum(d * d for d in degs)
    zagreb_m2 = sum(degs[b.begin_idx] * degs[b.end_idx] for b in mol.bonds)

    cyc_mu = max(0, mol.num_bonds - n + 1)
    balaban_sum = 0.0
    for b in mol.bonds:
        si = dist_sums[b.begin_idx]
        sj = dist_sums[b.end_idx]
        if si > 0 and sj > 0:
            balaban_sum += 1.0 / math.sqrt(float(si * sj))
    balaban_j = round((mol.num_bonds / float(cyc_mu + 1)) * balaban_sum, 4) if mol.num_bonds > 0 else 0.0

    descriptors = MolecularDescriptors(
        formula=formula,
        molecular_weight=round(mw, 3),
        exact_mass=round(mw, 4),
        heavy_atom_count=n,
        ring_count=len(rings),
        aromatic_ring_count=arom_rings,
        rotatable_bonds=rotb,
        hbd_count=hbd,
        hba_count=hba,
        formal_charge=f_chg,
        crippen_logp=logp,
        tpsa=tpsa,
        fsp3=fsp3,
        qed=qed,
        sa_score=sa_score,
        lipinski_violations=lipinski,
        veber_violations=veber,
        wiener_index=wiener,
        zagreb_m1=zagreb_m1,
        zagreb_m2=zagreb_m2,
        balaban_j=balaban_j,
    )
    return rings, descriptors
