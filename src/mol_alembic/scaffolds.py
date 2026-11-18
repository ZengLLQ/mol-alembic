"""Bemis-Murcko scaffold decomposition, Matched Molecular Pairs (MMP), and SALI activity cliffs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .chemistry import perceive_sssr
from .fingerprints import morgan_bit_vector, tanimoto_similarity
from .smiles import parse_smiles, to_smiles
from .types import Atom, Bond, BondOrder, Molecule


@dataclass(frozen=True)
class MatchedMolecularPair:
    """Single-cut Matched Molecular Pair sharing an identical conserved scaffold core."""

    mol_a_name: str
    mol_b_name: str
    core_smiles: str
    r_group_a: str
    r_group_b: str
    delta_activity: float
    tanimoto: float
    sali_score: float


def _subgraph_to_molecule(
    mol: Molecule,
    keep_atoms: Set[int],
    keep_bonds: Optional[Set[int]] = None,
    generic_skeleton: bool = False,
) -> Molecule:
    """Extract an induced molecular subgraph re-indexed from 0."""
    ordered = sorted(keep_atoms)
    old_to_new = {old: new for new, old in enumerate(ordered)}
    sub = Molecule(name=f"{mol.name}_sub", smiles="")

    for old_idx in ordered:
        a = mol.atoms[old_idx]
        sym = "C" if generic_skeleton else a.symbol
        z = 6 if generic_skeleton else a.atomic_number
        is_arom = False if generic_skeleton else a.is_aromatic
        sub.atoms.append(
            Atom(
                index=old_to_new[old_idx],
                symbol=sym,
                atomic_number=z,
                formal_charge=0 if generic_skeleton else a.formal_charge,
                explicit_h=0,
                implicit_h=0,
                is_aromatic=is_arom,
                is_in_ring=a.is_in_ring,
                ring_sizes=a.ring_sizes,
            )
        )

    for bond in mol.bonds:
        if bond.begin_idx in old_to_new and bond.end_idx in old_to_new:
            if keep_bonds is not None and bond.index not in keep_bonds:
                continue
            order = BondOrder.SINGLE if generic_skeleton else bond.order
            is_arom = False if generic_skeleton else bond.is_aromatic
            sub.bonds.append(
                Bond(
                    index=len(sub.bonds),
                    begin_idx=old_to_new[bond.begin_idx],
                    end_idx=old_to_new[bond.end_idx],
                    order=order,
                    is_aromatic=is_arom,
                    is_in_ring=bond.is_in_ring,
                )
            )
    sub.rebuild_adjacency()
    for atom in sub.atoms:
        if not atom.is_aromatic:
            val_map = {"C": 4, "N": 3, "O": 2, "S": 2, "P": 3, "B": 3}
            target = val_map.get(atom.symbol, 4)
            atom.implicit_h = max(0, int(round(target - sub.bond_order_sum(atom.index))))
    sub.smiles = to_smiles(sub, canonical=True)
    return sub


def extract_bemis_murcko_scaffold(
    mol: Molecule,
    generic_skeleton: bool = False,
) -> Tuple[str, Molecule]:
    """Extract the Bemis-Murcko ring + linker framework (or generic carbon skeleton)."""
    work = mol.copy()
    rings = perceive_sssr(work)
    if not rings:
        return "", Molecule(name=f"{mol.name}_acyclic", smiles="")

    active: Set[int] = {a.index for a in work.atoms}
    changed = True
    while changed:
        changed = False
        to_remove: List[int] = []
        for idx in list(active):
            if work.atoms[idx].is_in_ring:
                continue
            active_deg = sum(1 for nbr, _ in work.neighbors(idx) if nbr in active)
            if active_deg <= 1:
                to_remove.append(idx)
        if to_remove:
            for idx in to_remove:
                active.remove(idx)
            changed = True

    if not active:
        return "", Molecule(name=f"{mol.name}_empty", smiles="")

    scaffold_mol = _subgraph_to_molecule(work, active, generic_skeleton=generic_skeleton)
    return scaffold_mol.smiles, scaffold_mol


def _connected_components_without_bond(mol: Molecule, cut_bond_idx: int) -> List[Set[int]]:
    """Return connected atom components resulting from cutting cut_bond_idx."""
    n = mol.num_atoms
    visited = [False] * n
    comps: List[Set[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        comp: Set[int] = set()
        stack = [start]
        visited[start] = True
        while stack:
            u = stack.pop()
            comp.add(u)
            for v, bond in mol.neighbors(u):
                if bond.index == cut_bond_idx:
                    continue
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def single_cut_fragments(mol: Molecule, min_core_atoms: int = 5) -> List[Tuple[str, str]]:
    """Generate single-cut acyclic-bond (scaffold_fragment, r_group) canonical SMILES pairs."""
    work = mol.copy()
    perceive_sssr(work)
    pairs: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    for bond in work.bonds:
        if bond.is_in_ring or bond.order != BondOrder.SINGLE:
            continue
        comps = _connected_components_without_bond(work, bond.index)
        if len(comps) != 2:
            continue
        c1, c2 = comps[0], comps[1]
        if len(c1) >= min_core_atoms and len(c2) >= 1:
            core_s = _subgraph_to_molecule(work, c1).smiles
            rg_s = _subgraph_to_molecule(work, c2).smiles
            if core_s and rg_s and (core_s, rg_s) not in seen:
                seen.add((core_s, rg_s))
                pairs.append((core_s, rg_s))
        if len(c2) >= min_core_atoms and len(c1) >= 1:
            core_s = _subgraph_to_molecule(work, c2).smiles
            rg_s = _subgraph_to_molecule(work, c1).smiles
            if core_s and rg_s and (core_s, rg_s) not in seen:
                seen.add((core_s, rg_s))
                pairs.append((core_s, rg_s))
    return pairs


def find_matched_molecular_pairs(
    records: Sequence[Tuple[str, str, float]],
    min_core_atoms: int = 5,
    sali_threshold: float = 2.0,
) -> Tuple[List[MatchedMolecularPair], List[MatchedMolecularPair]]:
    """Index single-cut MMPs across a compound series and flag SALI activity cliffs."""
    mols = [parse_smiles(smi, name=name) for name, smi, _ in records]
    fps = [morgan_bit_vector(m, radius=2, nbits=1024) for m in mols]

    core_index: Dict[str, List[Tuple[int, str]]] = {}
    for idx, mol in enumerate(mols):
        for core_smi, rg_smi in single_cut_fragments(mol, min_core_atoms=min_core_atoms):
            core_index.setdefault(core_smi, []).append((idx, rg_smi))

    seen_mol_pairs: Set[Tuple[int, int, str]] = set()
    mmps: List[MatchedMolecularPair] = []

    for core_smi, entries in core_index.items():
        if len(entries) < 2:
            continue
        for pos_a in range(len(entries)):
            idx_a, rg_a = entries[pos_a]
            for pos_b in range(pos_a + 1, len(entries)):
                idx_b, rg_b = entries[pos_b]
                if idx_a == idx_b or rg_a == rg_b:
                    continue
                pair_key = (min(idx_a, idx_b), max(idx_a, idx_b), core_smi)
                if pair_key in seen_mol_pairs:
                    continue
                seen_mol_pairs.add(pair_key)
                sim = tanimoto_similarity(fps[idx_a], fps[idx_b])
                delta_act = abs(records[idx_a][2] - records[idx_b][2])
                sali = delta_act / max(0.02, 1.0 - sim)
                mmps.append(
                    MatchedMolecularPair(
                        mol_a_name=records[idx_a][0],
                        mol_b_name=records[idx_b][0],
                        core_smiles=core_smi,
                        r_group_a=rg_a,
                        r_group_b=rg_b,
                        delta_activity=round(delta_act, 4),
                        tanimoto=round(sim, 4),
                        sali_score=round(sali, 4),
                    )
                )

    mmps.sort(key=lambda m: -m.sali_score)
    cliffs = [m for m in mmps if m.sali_score >= sali_threshold]
    return mmps, cliffs
