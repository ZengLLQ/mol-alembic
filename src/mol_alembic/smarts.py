"""SMARTS substructure compiler, VF2 subgraph isomorphism, PAINS/Brenk alerts, and pharmacophores."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .chemistry import perceive_sssr
from .types import (
    ATOMIC_NUM_TO_SYMBOL,
    Bond,
    BondOrder,
    Molecule,
    PharmacophoreFeature,
    PharmacophoreType,
    SmartsSyntaxError,
    StructuralAlertMatch,
)


@dataclass
class SmartsAtomExpr:
    """Boolean-combinable SMARTS atom predicate."""

    kind: str
    symbol: Optional[str] = None
    arom_flag: Optional[bool] = None
    num: Optional[int] = None
    children: Tuple["SmartsAtomExpr", ...] = ()

    def matches(self, mol: Molecule, atom_idx: int) -> bool:
        """Evaluate this predicate against mol.atoms[atom_idx]."""
        atom = mol.atoms[atom_idx]
        if self.kind == "any":
            return True
        if self.kind == "and":
            return all(c.matches(mol, atom_idx) for c in self.children)
        if self.kind == "or":
            return any(c.matches(mol, atom_idx) for c in self.children)
        if self.kind == "not":
            return not self.children[0].matches(mol, atom_idx)
        if self.kind == "atom":
            if atom.symbol != self.symbol:
                return False
            if self.arom_flag is None:
                return True
            return atom.is_aromatic == self.arom_flag
        if self.kind == "arom":
            return atom.is_aromatic
        if self.kind == "aliph":
            return not atom.is_aromatic
        if self.kind == "ring":
            if self.num == 0:
                return not atom.is_in_ring
            if self.num is None:
                return atom.is_in_ring
            return len(atom.ring_sizes) == self.num
        if self.kind == "ring_size":
            return self.num is not None and self.num in atom.ring_sizes
        if self.kind == "deg":
            return self.num is not None and mol.degree(atom_idx) == self.num
        if self.kind == "h":
            return self.num is not None and atom.total_h == self.num
        if self.kind == "val":
            return self.num is not None and int(round(mol.bond_order_sum(atom_idx) + atom.total_h)) == self.num
        if self.kind == "chg":
            return self.num is not None and atom.formal_charge == self.num
        if self.kind == "conn":
            return self.num is not None and (mol.degree(atom_idx) + atom.total_h) == self.num
        return False


@dataclass
class SmartsBondExpr:
    """SMARTS bond predicate."""

    allowed_orders: Optional[Tuple[BondOrder, ...]] = None
    require_ring: Optional[bool] = None
    any_bond: bool = False

    def matches(self, bond: Bond) -> bool:
        """Return True if bond satisfies the SMARTS bond expression."""
        if self.require_ring is not None and bond.is_in_ring != self.require_ring:
            return False
        if self.any_bond or self.allowed_orders is None:
            return True
        return bond.order in self.allowed_orders


@dataclass
class SmartsPattern:
    """Compiled SMARTS query graph."""

    raw: str
    atom_preds: List[SmartsAtomExpr] = field(default_factory=list)
    bonds: List[Tuple[int, int, SmartsBondExpr]] = field(default_factory=list)
    adj: Dict[int, List[Tuple[int, SmartsBondExpr]]] = field(default_factory=dict)


def _parse_atom_primitive(inner: str, pos: int) -> Tuple[SmartsAtomExpr, int]:
    """Parse one atomic primitive inside a [...] SMARTS bracket."""
    n = len(inner)
    ch = inner[pos]

    if ch == "*":
        return SmartsAtomExpr("any"), pos + 1
    if ch == "a":
        return SmartsAtomExpr("arom"), pos + 1
    if ch == "A":
        return SmartsAtomExpr("aliph"), pos + 1
    if ch == "#":
        pos += 1
        start = pos
        while pos < n and inner[pos].isdigit():
            pos += 1
        if pos == start:
            raise SmartsSyntaxError(f"Expected atomic number after '#' in '{inner}'")
        z = int(inner[start:pos])
        sym = ATOMIC_NUM_TO_SYMBOL.get(z)
        if sym is None:
            raise SmartsSyntaxError(f"Unsupported atomic number #{z} in SMARTS")
        return SmartsAtomExpr("atom", symbol=sym, arom_flag=None), pos

    if ch in ("R", "r", "D", "H", "v", "X"):
        pos += 1
        start = pos
        while pos < n and inner[pos].isdigit():
            pos += 1
        num: Optional[int] = int(inner[start:pos]) if pos > start else None
        if ch == "R":
            return SmartsAtomExpr("ring", num=num), pos
        if ch == "r":
            if num is None:
                return SmartsAtomExpr("ring", num=None), pos
            return SmartsAtomExpr("ring_size", num=num), pos
        if ch == "D":
            return SmartsAtomExpr("deg", num=1 if num is None else num), pos
        if ch == "H":
            return SmartsAtomExpr("h", num=1 if num is None else num), pos
        if ch == "v":
            return SmartsAtomExpr("val", num=1 if num is None else num), pos
        if ch == "X":
            return SmartsAtomExpr("conn", num=1 if num is None else num), pos

    if ch in ("+", "-"):
        sign = 1 if ch == "+" else -1
        pos += 1
        start = pos
        while pos < n and inner[pos].isdigit():
            pos += 1
        mag = int(inner[start:pos]) if pos > start else 1
        return SmartsAtomExpr("chg", num=sign * mag), pos

    if ch in ("c", "n", "o", "p", "s", "b"):
        return SmartsAtomExpr("atom", symbol=ch.upper(), arom_flag=True), pos + 1

    if ch.isupper():
        two = inner[pos : pos + 2]
        if two in ("Cl", "Br", "Si", "Fe"):
            return SmartsAtomExpr("atom", symbol=two, arom_flag=False), pos + 2
        if ch in ("B", "C", "N", "O", "P", "S", "F", "I"):
            return SmartsAtomExpr("atom", symbol=ch, arom_flag=False), pos + 1

    raise SmartsSyntaxError(f"Unsupported SMARTS atom primitive '{inner[pos:]}' in '[{inner}]'")


def _parse_smarts_bracket(inner: str) -> SmartsAtomExpr:
    """Parse logical SMARTS bracket expression supporting !, &, implicit-AND, ',', ';'."""
    if not inner:
        raise SmartsSyntaxError("Empty SMARTS bracket '[]'")

    def parse_semi(s: str) -> SmartsAtomExpr:
        parts = [p for p in s.split(";") if p]
        if len(parts) > 1:
            return SmartsAtomExpr("and", children=tuple(parse_or(p) for p in parts))
        return parse_or(s)

    def parse_or(s: str) -> SmartsAtomExpr:
        parts = [p for p in s.split(",") if p]
        if len(parts) > 1:
            return SmartsAtomExpr("or", children=tuple(parse_and(p) for p in parts))
        return parse_and(s)

    def parse_and(s: str) -> SmartsAtomExpr:
        terms: List[SmartsAtomExpr] = []
        pos = 0
        while pos < len(s):
            if s[pos] == "&":
                pos += 1
                continue
            negate = False
            if s[pos] == "!":
                negate = True
                pos += 1
            if pos >= len(s):
                raise SmartsSyntaxError(f"Dangling '!' in SMARTS '[{inner}]'")
            prim, pos = _parse_atom_primitive(s, pos)
            terms.append(SmartsAtomExpr("not", children=(prim,)) if negate else prim)
        if not terms:
            raise SmartsSyntaxError(f"Empty sub-expression in SMARTS '[{inner}]'")
        return terms[0] if len(terms) == 1 else SmartsAtomExpr("and", children=tuple(terms))

    return parse_semi(inner)


def compile_smarts(smarts: str) -> SmartsPattern:
    """Compile a SMARTS string into a SmartsPattern query graph."""
    text = smarts.strip()
    if not text:
        raise SmartsSyntaxError("Empty SMARTS string")

    pattern = SmartsPattern(raw=text)
    branch_stack: List[Optional[int]] = []
    ring_open: Dict[int, Tuple[int, Optional[SmartsBondExpr]]] = {}
    prev_idx: Optional[int] = None
    pending_bond: Optional[SmartsBondExpr] = None

    def add_bond(u: int, v: int, expr: SmartsBondExpr) -> None:
        pattern.bonds.append((u, v, expr))
        pattern.adj.setdefault(u, []).append((v, expr))
        pattern.adj.setdefault(v, []).append((u, expr))

    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "(":
            if prev_idx is None:
                raise SmartsSyntaxError("Branch '(' without preceding SMARTS atom")
            branch_stack.append(prev_idx)
            i += 1
            continue
        if ch == ")":
            if not branch_stack:
                raise SmartsSyntaxError("Unmatched ')' in SMARTS")
            prev_idx = branch_stack.pop()
            pending_bond = None
            i += 1
            continue
        if ch in ("-", "=", "#", ":", "~", "@"):
            if ch == "-":
                pending_bond = SmartsBondExpr(allowed_orders=(BondOrder.SINGLE,))
            elif ch == "=":
                pending_bond = SmartsBondExpr(allowed_orders=(BondOrder.DOUBLE,))
            elif ch == "#":
                pending_bond = SmartsBondExpr(allowed_orders=(BondOrder.TRIPLE,))
            elif ch == ":":
                pending_bond = SmartsBondExpr(allowed_orders=(BondOrder.AROMATIC,))
            elif ch == "~":
                pending_bond = SmartsBondExpr(any_bond=True)
            elif ch == "@":
                pending_bond = SmartsBondExpr(any_bond=True, require_ring=True)
            i += 1
            continue
        if ch.isdigit():
            if prev_idx is None:
                raise SmartsSyntaxError("Ring closure digit without atom in SMARTS")
            r_id = int(ch)
            i += 1
            if r_id not in ring_open:
                ring_open[r_id] = (prev_idx, pending_bond)
            else:
                other, open_b = ring_open.pop(r_id)
                b_expr = pending_bond or open_b or SmartsBondExpr(
                    allowed_orders=(BondOrder.SINGLE, BondOrder.AROMATIC)
                )
                add_bond(other, prev_idx, b_expr)
            pending_bond = None
            continue

        if ch == "[":
            end_br = text.find("]", i + 1)
            if end_br == -1:
                raise SmartsSyntaxError(f"Unclosed '[' in SMARTS '{text}'")
            atom_expr = _parse_smarts_bracket(text[i + 1 : end_br])
            i = end_br + 1
        elif ch == "*":
            atom_expr = SmartsAtomExpr("any")
            i += 1
        elif ch == "a":
            atom_expr = SmartsAtomExpr("arom")
            i += 1
        elif ch == "A":
            atom_expr = SmartsAtomExpr("aliph")
            i += 1
        elif ch in ("c", "n", "o", "p", "s", "b"):
            atom_expr = SmartsAtomExpr("atom", symbol=ch.upper(), arom_flag=True)
            i += 1
        elif ch.isupper():
            two = text[i : i + 2]
            if two in ("Cl", "Br", "Si", "Fe"):
                atom_expr = SmartsAtomExpr("atom", symbol=two, arom_flag=False)
                i += 2
            elif ch in ("B", "C", "N", "O", "P", "S", "F", "I"):
                atom_expr = SmartsAtomExpr("atom", symbol=ch, arom_flag=False)
                i += 1
            else:
                raise SmartsSyntaxError(f"Unsupported element '{ch}' in SMARTS '{text}'")
        else:
            raise SmartsSyntaxError(f"Unexpected character '{ch}' in SMARTS '{text}'")

        idx = len(pattern.atom_preds)
        pattern.atom_preds.append(atom_expr)
        pattern.adj.setdefault(idx, [])

        if prev_idx is not None:
            b_expr = pending_bond or SmartsBondExpr(
                allowed_orders=(BondOrder.SINGLE, BondOrder.AROMATIC)
            )
            add_bond(prev_idx, idx, b_expr)

        prev_idx = idx
        pending_bond = None

    if branch_stack or ring_open:
        raise SmartsSyntaxError(f"Unclosed branch or ring closure in SMARTS '{text}'")
    return pattern


def match_smarts(mol: Molecule, smarts: str, unique: bool = True) -> List[Tuple[int, ...]]:
    """Find subgraph isomorphism matches of a SMARTS pattern in mol via backtracking search."""
    if not any(a.is_in_ring for a in mol.atoms) and mol.num_bonds >= mol.num_atoms:
        perceive_sssr(mol)

    pattern = compile_smarts(smarts)
    q_count = len(pattern.atom_preds)
    if q_count == 0 or mol.num_atoms < q_count:
        return []

    candidates: List[List[int]] = []
    for q_idx in range(q_count):
        pred = pattern.atom_preds[q_idx]
        q_deg = len(pattern.adj.get(q_idx, []))
        cands = [
            a.index
            for a in mol.atoms
            if mol.degree(a.index) >= q_deg and pred.matches(mol, a.index)
        ]
        if not cands:
            return []
        candidates.append(cands)

    remaining = set(range(q_count))
    order: List[int] = []
    start_q = min(remaining, key=lambda q: (len(candidates[q]), -len(pattern.adj[q])))
    order.append(start_q)
    remaining.remove(start_q)
    while remaining:
        frontier = [q for q in remaining if any(nbr in order for nbr, _ in pattern.adj[q])]
        pool = frontier if frontier else list(remaining)
        nxt = min(pool, key=lambda q: (len(candidates[q]), -len(pattern.adj[q])))
        order.append(nxt)
        remaining.remove(nxt)

    mapping: Dict[int, int] = {}
    used_targets: Set[int] = set()
    matches: List[Tuple[int, ...]] = []
    seen_sets: Set[frozenset[int]] = set()

    def backtrack(step: int) -> None:
        if step == q_count:
            ordered_tuple = tuple(mapping[q] for q in range(q_count))
            if unique:
                fset = frozenset(ordered_tuple)
                if fset in seen_sets:
                    return
                seen_sets.add(fset)
            matches.append(ordered_tuple)
            return

        q_u = order[step]
        mapped_nbrs = [(q_v, b_expr) for q_v, b_expr in pattern.adj[q_u] if q_v in mapping]
        if mapped_nbrs:
            anchor_v, anchor_expr = mapped_nbrs[0]
            t_anchor = mapping[anchor_v]
            pool = [
                t_u
                for t_u, bond in mol.neighbors(t_anchor)
                if t_u not in used_targets
                and t_u in candidates[q_u]
                and anchor_expr.matches(bond)
            ]
        else:
            pool = [t_u for t_u in candidates[q_u] if t_u not in used_targets]

        for t_u in pool:
            ok = True
            for q_v, b_expr in mapped_nbrs[1:]:
                bond = mol.bond_between(t_u, mapping[q_v])
                if bond is None or not b_expr.matches(bond):
                    ok = False
                    break
            if not ok:
                continue
            mapping[q_u] = t_u
            used_targets.add(t_u)
            backtrack(step + 1)
            used_targets.remove(t_u)
            del mapping[q_u]

    backtrack(0)
    return matches


STRUCTURAL_ALERTS: List[Tuple[str, str, str, str]] = [
    ("Nitro_Group", "MEDIUM", "[#7+](=[#8])[#8-]", "Nitroaromatic/aliphatic mutagenic reduction risk"),
    ("Nitro_Double_Valence", "MEDIUM", "[#7](=[#8])=[#8]", "Hypervalent nitro group reduction risk"),
    ("Acyl_Halide", "HIGH", "[#6](=[#8])[F,Cl,Br,I]", "Hyper-reactive electrophilic acyl halide"),
    ("Michael_Acceptor", "MEDIUM", "[#6]=[#6]-[#6](=[#8])", "Alpha,beta-unsaturated carbonyl covalent trap"),
    ("Epoxide_Ring", "HIGH", "[#6;r3]1-[#8;r3]-[#6;r3]1", "Strained reactive oxirane alkylator"),
    ("Aldehyde", "MEDIUM", "[#6;H1](=[#8])[#6,#1]", "Reactive aldehyde Schiff-base former"),
    ("Hydrazine", "HIGH", "[#7;X3]-[#7;X3]", "Hepatotoxic / redox-active hydrazine motif"),
    ("Peroxide", "HIGH", "[#8]-[#8]", "Unstable radical peroxide linkage"),
    ("Thiourea", "LOW", "[#7]-[#6](=[#16])-[#7]", "Metabolic thyroid toxicity / frequent hitter"),
    ("Polyene_Triene", "LOW", "[#6]=[#6]-[#6]=[#6]-[#6]=[#6]", "Photoreactive conjugated triene"),
    ("Alkyl_Halide_Primary", "HIGH", "[#6;H2,H3;A][Cl,Br,I]", "Non-selective SN2 alkylating halide"),
    ("Acyl_Hydrazide", "MEDIUM", "[#6](=[#8])-[#7]-[#7]", "Reactive hydrazide PAINS motif"),
]


def screen_structural_alerts(mol: Molecule) -> List[StructuralAlertMatch]:
    """Screen a molecule against curated PAINS and Brenk medicinal chemistry alerts."""
    hits: List[StructuralAlertMatch] = []
    for name, severity, smarts, rationale in STRUCTURAL_ALERTS:
        for match in match_smarts(mol, smarts, unique=True):
            hits.append(
                StructuralAlertMatch(
                    alert_name=name,
                    severity=severity,
                    smarts=smarts,
                    matched_atoms=match,
                    rationale=rationale,
                )
            )
    return hits


def annotate_pharmacophores(mol: Molecule) -> List[PharmacophoreFeature]:
    """Annotate HBD, HBA, aromatic, hydrophobic, and ionizable pharmacophore centers."""
    rings = perceive_sssr(mol)
    features: List[PharmacophoreFeature] = []

    for atom in mol.atoms:
        idx = atom.index
        if atom.symbol in ("N", "O", "S") and atom.total_h > 0:
            features.append(PharmacophoreFeature(PharmacophoreType.HBD, (idx,), f"{atom.symbol}H"))
        if atom.symbol in ("N", "O") and atom.formal_charge <= 0:
            features.append(PharmacophoreFeature(PharmacophoreType.HBA, (idx,), f"{atom.symbol}_acc"))
        if atom.symbol == "N" and atom.formal_charge >= 0 and not atom.is_aromatic:
            is_amide = any(
                mol.atoms[n].symbol == "C"
                and any(b2.order == BondOrder.DOUBLE and mol.atoms[n2].symbol == "O" for n2, b2 in mol.neighbors(n))
                for n, _ in mol.neighbors(idx)
            )
            if not is_amide and atom.formal_charge >= 0:
                features.append(PharmacophoreFeature(PharmacophoreType.POS_IONIZABLE, (idx,), "basic_N"))
        if atom.symbol == "C":
            oxy_nbrs = [n for n, _b in mol.neighbors(idx) if mol.atoms[n].symbol == "O"]
            has_co2 = len(oxy_nbrs) >= 2 and any(
                b.order == BondOrder.DOUBLE and mol.atoms[n].symbol == "O" for n, b in mol.neighbors(idx)
            )
            if has_co2:
                features.append(
                    PharmacophoreFeature(PharmacophoreType.NEG_IONIZABLE, (idx, *oxy_nbrs), "carboxylate")
                )
            elif not atom.is_aromatic and all(
                mol.atoms[n].symbol in ("C", "F", "Cl", "Br", "I", "S") for n, _ in mol.neighbors(idx)
            ):
                features.append(PharmacophoreFeature(PharmacophoreType.HYDROPHOBIC, (idx,), "hydrophobic_C"))

    for ring in rings:
        if ring.is_aromatic:
            features.append(PharmacophoreFeature(PharmacophoreType.AROMATIC, ring.atom_indices, "arom_ring"))

    return features
