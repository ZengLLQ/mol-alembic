"""De novo multi-objective molecular optimization via valence-safe graph mutations and NSGA-II."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from .adme import predict_adme, score_adme_desirability
from .chemistry import analyze_molecule
from .gnn import MolecularGraphNet
from .smarts import screen_structural_alerts
from .smiles import parse_smiles, to_smiles
from .types import (
    ELEMENT_TABLE,
    Atom,
    Bond,
    BondOrder,
    Molecule,
    OptimizationCandidate,
    ParetoFrontResult,
)


_DECORATION_GROUPS: List[Tuple[str, str]] = [
    ("fluoro", "F"),
    ("chloro", "Cl"),
    ("methyl", "C"),
    ("hydroxy", "O"),
    ("amino", "N"),
    ("methoxy", "OC"),
    ("trifluoromethyl", "C(F)(F)F"),
    ("nitrile", "C#N"),
    ("carboxamide", "C(=O)N"),
]


def _attach_fragment_to_atom(
    mol: Molecule, anchor_idx: int, frag_smiles: str, op_name: str
) -> Optional[Tuple[str, str]]:
    """Attach a small SMILES fragment via a single bond to anchor_idx if implicit H >= 1."""
    if mol.atoms[anchor_idx].implicit_h < 1:
        return None
    frag = parse_smiles(frag_smiles, name="frag")
    work = mol.copy()
    offset = work.num_atoms

    for a in frag.atoms:
        work.atoms.append(
            Atom(
                index=offset + a.index,
                symbol=a.symbol,
                atomic_number=a.atomic_number,
                formal_charge=a.formal_charge,
                explicit_h=a.explicit_h,
                implicit_h=a.implicit_h,
                is_aromatic=a.is_aromatic,
            )
        )
    for b in frag.bonds:
        work.bonds.append(
            Bond(
                index=len(work.bonds),
                begin_idx=offset + b.begin_idx,
                end_idx=offset + b.end_idx,
                order=b.order,
                is_aromatic=b.is_aromatic,
            )
        )
    work.bonds.append(
        Bond(
            index=len(work.bonds),
            begin_idx=anchor_idx,
            end_idx=offset,
            order=BondOrder.SINGLE,
        )
    )
    work.atoms[anchor_idx].implicit_h -= 1
    if work.atoms[offset].implicit_h > 0:
        work.atoms[offset].implicit_h -= 1
    work.rebuild_adjacency()

    try:
        smi = to_smiles(work, canonical=True)
        parse_smiles(smi)
        return smi, op_name
    except Exception:
        return None


def _mutate_aromatic_aza_swap(mol: Molecule, atom_idx: int) -> Optional[Tuple[str, str]]:
    """Swap an unsubstituted aromatic CH <-> aromatic N (bioisosteric pyridine/pyrimidine hop)."""
    atom = mol.atoms[atom_idx]
    if not atom.is_aromatic:
        return None
    work = mol.copy()
    if atom.symbol == "C" and atom.total_h == 1 and work.degree(atom_idx) == 2:
        work.atoms[atom_idx].symbol = "N"
        work.atoms[atom_idx].atomic_number = ELEMENT_TABLE["N"].atomic_number
        work.atoms[atom_idx].implicit_h = 0
        work.atoms[atom_idx].explicit_h = 0
        op = "aza_c_to_n"
    elif atom.symbol == "N" and atom.total_h == 0 and work.degree(atom_idx) == 2:
        work.atoms[atom_idx].symbol = "C"
        work.atoms[atom_idx].atomic_number = ELEMENT_TABLE["C"].atomic_number
        work.atoms[atom_idx].implicit_h = 1
        op = "aza_n_to_c"
    else:
        return None

    try:
        smi = to_smiles(work, canonical=True)
        parse_smiles(smi)
        return smi, op
    except Exception:
        return None


def _mutate_halogen_or_heteroatom(mol: Molecule, atom_idx: int) -> Optional[Tuple[str, str]]:
    """Swap terminal halogen (F <-> Cl) or exocyclic O <-> N."""
    atom = mol.atoms[atom_idx]
    if mol.degree(atom_idx) != 1 or atom.is_in_ring:
        return None
    work = mol.copy()
    if atom.symbol == "F":
        work.atoms[atom_idx].symbol = "Cl"
        work.atoms[atom_idx].atomic_number = ELEMENT_TABLE["Cl"].atomic_number
        op = "swap_F_to_Cl"
    elif atom.symbol == "Cl":
        work.atoms[atom_idx].symbol = "F"
        work.atoms[atom_idx].atomic_number = ELEMENT_TABLE["F"].atomic_number
        op = "swap_Cl_to_F"
    elif atom.symbol == "O" and atom.total_h == 1:
        work.atoms[atom_idx].symbol = "N"
        work.atoms[atom_idx].atomic_number = ELEMENT_TABLE["N"].atomic_number
        work.atoms[atom_idx].implicit_h = 2
        op = "swap_OH_to_NH2"
    elif atom.symbol == "N" and atom.total_h == 2:
        work.atoms[atom_idx].symbol = "O"
        work.atoms[atom_idx].atomic_number = ELEMENT_TABLE["O"].atomic_number
        work.atoms[atom_idx].implicit_h = 1
        op = "swap_NH2_to_OH"
    else:
        return None

    try:
        smi = to_smiles(work, canonical=True)
        parse_smiles(smi)
        return smi, op
    except Exception:
        return None


def _prune_terminal_substituent(mol: Molecule, atom_idx: int) -> Optional[Tuple[str, str]]:
    """Remove a non-ring degree-1 heavy atom substituent to control molecular weight."""
    if mol.num_atoms <= 7:
        return None
    atom = mol.atoms[atom_idx]
    if atom.is_in_ring or mol.degree(atom_idx) != 1:
        return None
    nbrs = mol.neighbors(atom_idx)
    if not nbrs:
        return None
    parent_idx, bond = nbrs[0]
    if bond.order != BondOrder.SINGLE:
        return None

    work = Molecule(name=mol.name, smiles="")
    idx_map: Dict[int, int] = {}
    for a in mol.atoms:
        if a.index == atom_idx:
            continue
        new_i = len(work.atoms)
        idx_map[a.index] = new_i
        inc_h = 1 if a.index == parent_idx else 0
        work.atoms.append(
            Atom(
                index=new_i,
                symbol=a.symbol,
                atomic_number=a.atomic_number,
                formal_charge=a.formal_charge,
                explicit_h=a.explicit_h,
                implicit_h=a.implicit_h + inc_h,
                is_aromatic=a.is_aromatic,
            )
        )
    for b in mol.bonds:
        if b.begin_idx == atom_idx or b.end_idx == atom_idx:
            continue
        work.bonds.append(
            Bond(
                index=len(work.bonds),
                begin_idx=idx_map[b.begin_idx],
                end_idx=idx_map[b.end_idx],
                order=b.order,
                is_aromatic=b.is_aromatic,
            )
        )
    work.rebuild_adjacency()
    try:
        smi = to_smiles(work, canonical=True)
        parse_smiles(smi)
        return smi, "prune_terminal"
    except Exception:
        return None


def generate_molecular_neighbors(
    smiles: str,
    rng: np.random.Generator,
    max_variants: int = 8,
) -> List[Tuple[str, str]]:
    """Generate up to max_variants chemically valid mutated SMILES neighbors."""
    mol = parse_smiles(smiles)
    analyze_molecule(mol)

    variants: List[Tuple[str, str]] = []
    seen: Set[str] = {to_smiles(mol, canonical=True)}
    atom_order = rng.permutation(mol.num_atoms).tolist()

    for idx in atom_order:
        if len(variants) >= max_variants:
            break
        a_idx = int(idx)
        atom = mol.atoms[a_idx]

        res = _mutate_aromatic_aza_swap(mol, a_idx)
        if res is not None and res[0] not in seen:
            seen.add(res[0])
            variants.append(res)
            continue

        res = _mutate_halogen_or_heteroatom(mol, a_idx)
        if res is not None and res[0] not in seen:
            seen.add(res[0])
            variants.append(res)
            continue

        if atom.implicit_h >= 1 and atom.symbol in ("C", "N") and mol.num_atoms < 36:
            g_name, g_smi = _DECORATION_GROUPS[int(rng.integers(0, len(_DECORATION_GROUPS)))]
            res = _attach_fragment_to_atom(mol, a_idx, g_smi, f"add_{g_name}")
            if res is not None and res[0] not in seen:
                seen.add(res[0])
                variants.append(res)
                continue

        if mol.num_atoms > 14:
            res = _prune_terminal_substituent(mol, a_idx)
            if res is not None and res[0] not in seen:
                seen.add(res[0])
                variants.append(res)

    return variants


def evaluate_candidate(
    smiles: str,
    gnn: MolecularGraphNet,
    generation: int,
    origin_op: str,
) -> OptimizationCandidate:
    """Evaluate a candidate SMILES across potency, uncertainty, QED, SA, ADME, and PAINS alerts."""
    mol = parse_smiles(smiles, name=f"gen{generation}")
    alerts = screen_structural_alerts(mol)
    _, desc = analyze_molecule(mol, alert_count=len(alerts))
    adme = predict_adme(mol)
    adme_s = score_adme_desirability(adme)
    pred_pic50, unc, _ = gnn.predict_smiles(smiles, mc_samples=6)

    pot_norm = max(0.0, min(1.0, (pred_pic50 - 4.5) / 5.0))
    sa_norm = max(0.0, min(1.0, (10.0 - desc.sa_score) / 9.0))
    alert_pen = math.exp(-0.7 * len(alerts))
    mpo = (0.38 * pot_norm + 0.25 * desc.qed + 0.22 * adme_s + 0.15 * sa_norm) * alert_pen

    return OptimizationCandidate(
        smiles=to_smiles(mol, canonical=True),
        generation=generation,
        origin_op=origin_op,
        predicted_pic50=round(pred_pic50, 4),
        uncertainty=round(unc, 4),
        qed=round(desc.qed, 4),
        sa_score=round(desc.sa_score, 4),
        adme_score=round(adme_s, 4),
        alert_count=len(alerts),
        mpo_score=round(mpo, 4),
    )


def _dominates(a: OptimizationCandidate, b: OptimizationCandidate) -> bool:
    """Return True if candidate a Pareto-dominates b (maximize pIC50, QED, ADME; minimize SA, alerts)."""
    objs_a = (a.predicted_pic50, a.qed, a.adme_score, -a.sa_score, -float(a.alert_count))
    objs_b = (b.predicted_pic50, b.qed, b.adme_score, -b.sa_score, -float(b.alert_count))
    ge_all = all(x >= y for x, y in zip(objs_a, objs_b))
    gt_one = any(x > y for x, y in zip(objs_a, objs_b))
    return ge_all and gt_one


def nsga2_rank_and_crowding(candidates: Sequence[OptimizationCandidate]) -> List[OptimizationCandidate]:
    """Perform NSGA-II fast non-dominated sorting and crowding distance assignment."""
    n = len(candidates)
    if n == 0:
        return []

    dominates_list: List[List[int]] = [[] for _ in range(n)]
    dom_count = [0] * n
    fronts: List[List[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(candidates[i], candidates[j]):
                dominates_list[i].append(j)
            elif _dominates(candidates[j], candidates[i]):
                dom_count[i] += 1
        if dom_count[i] == 0:
            fronts[0].append(i)

    cur = 0
    while cur < len(fronts) and fronts[cur]:
        nxt: List[int] = []
        for i in fronts[cur]:
            for j in dominates_list[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        cur += 1
        if nxt:
            fronts.append(nxt)

    ranks = [0] * n
    crowd = [0.0] * n
    for r_idx, front in enumerate(fronts):
        for i in front:
            ranks[i] = r_idx
        if len(front) <= 2:
            for i in front:
                crowd[i] = 1e6
            continue
        for getter in (
            lambda c: c.predicted_pic50,
            lambda c: c.qed,
            lambda c: c.adme_score,
            lambda c: -c.sa_score,
        ):
            sorted_f = sorted(front, key=lambda idx: getter(candidates[idx]))
            crowd[sorted_f[0]] = 1e6
            crowd[sorted_f[-1]] = 1e6
            lo = getter(candidates[sorted_f[0]])
            hi = getter(candidates[sorted_f[-1]])
            span = max(1e-9, hi - lo)
            for pos in range(1, len(sorted_f) - 1):
                prev_v = getter(candidates[sorted_f[pos - 1]])
                next_v = getter(candidates[sorted_f[pos + 1]])
                crowd[sorted_f[pos]] += (next_v - prev_v) / span

    ranked: List[OptimizationCandidate] = []
    for i, c in enumerate(candidates):
        ranked.append(
            OptimizationCandidate(
                smiles=c.smiles,
                generation=c.generation,
                origin_op=c.origin_op,
                predicted_pic50=c.predicted_pic50,
                uncertainty=c.uncertainty,
                qed=c.qed,
                sa_score=c.sa_score,
                adme_score=c.adme_score,
                alert_count=c.alert_count,
                mpo_score=c.mpo_score,
                pareto_rank=ranks[i],
                crowding_distance=round(crowd[i], 4),
            )
        )
    ranked.sort(key=lambda c: (c.pareto_rank, -c.mpo_score, -c.crowding_distance))
    return ranked


def optimize_molecules(
    seed_smiles: str,
    gnn: MolecularGraphNet,
    generations: int = 6,
    pop_size: int = 14,
    seed: int = 42,
) -> ParetoFrontResult:
    """Run evolutionary NSGA-II multi-objective de novo molecular optimization."""
    rng = np.random.default_rng(seed)
    seed_canon = to_smiles(parse_smiles(seed_smiles), canonical=True)
    archive: Dict[str, OptimizationCandidate] = {
        seed_canon: evaluate_candidate(seed_canon, gnn, generation=0, origin_op="seed")
    }
    population = [archive[seed_canon]]
    best_mpo_history: List[float] = [archive[seed_canon].mpo_score]
    eval_count = 1

    for gen in range(1, generations + 1):
        for parent in population[: max(4, pop_size // 2)]:
            neighbors = generate_molecular_neighbors(parent.smiles, rng, max_variants=4)
            for child_smi, op_name in neighbors:
                eval_count += 1
                if child_smi in archive:
                    continue
                cand = evaluate_candidate(child_smi, gnn, generation=gen, origin_op=op_name)
                archive[child_smi] = cand

        pool = list(archive.values())
        ranked = nsga2_rank_and_crowding(pool)
        population = ranked[:pop_size]
        best_mpo_history.append(max(c.mpo_score for c in population))

    final_ranked = nsga2_rank_and_crowding(list(archive.values()))
    pareto_front = [c for c in final_ranked if c.pareto_rank == 0]
    top_all = sorted(final_ranked, key=lambda c: -c.mpo_score)[:15]

    return ParetoFrontResult(
        seed_smiles=seed_canon,
        generations=generations,
        evaluated_count=eval_count,
        valid_unique_count=len(archive),
        pareto_front=pareto_front,
        all_top_candidates=top_all,
        best_mpo_history=best_mpo_history,
    )
