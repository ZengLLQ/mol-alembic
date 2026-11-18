"""SMILES lexer, recursive graph parser, valence completion, canonical serializer, and benchmark library."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .types import (
    AROMATIC_SYMBOLS,
    ELEMENT_TABLE,
    Atom,
    Bond,
    BondOrder,
    Hybridization,
    Molecule,
    SmilesParseError,
)


_ORGANIC_SUBSET = {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}
_BOND_CHAR_TO_ORDER: Dict[str, Tuple[BondOrder, bool]] = {
    "-": (BondOrder.SINGLE, False),
    "=": (BondOrder.DOUBLE, False),
    "#": (BondOrder.TRIPLE, False),
    ":": (BondOrder.AROMATIC, True),
}
_ORDER_TO_CHAR: Dict[BondOrder, str] = {
    BondOrder.SINGLE: "",
    BondOrder.DOUBLE: "=",
    BondOrder.TRIPLE: "#",
    BondOrder.AROMATIC: "",
}

# Curated medicinal chemistry, kinase inhibitor, CNS, oral drug, and PROTAC library.
BENCHMARK_MOLECULES: List[Tuple[str, str, float]] = [
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O", 5.82),
    ("Ibuprofen", "CC(C)Cc1ccc(C(C)C(=O)O)cc1", 6.15),
    ("Paracetamol", "CC(=O)Nc1ccc(O)cc1", 5.10),
    ("Caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", 5.45),
    ("Propranolol", "CC(C)NCC(O)COc1cccc2ccccc12", 7.42),
    ("Lidocaine", "CCN(CC)CC(=O)Nc1c(C)cccc1C", 6.28),
    ("Diazepam", "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21", 7.85),
    ("Fluoxetine", "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1", 7.64),
    ("Tamoxifen", "CC/C(=C(\\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1", 7.91),
    ("Gefitinib_Motif", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCC", 8.65),
    ("Erlotinib_Motif", "C#Cc1cccc(Nc2ncnc3cc(OCC)c(OCC)cc23)c1", 8.52),
    ("Imatinib_Motif", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1", 8.88),
    ("Warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", 7.35),
    ("Sildenafil_Motif", "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12", 8.32),
    ("Metformin", "CN(C)C(=N)N=C(N)N", 4.65),
    ("Atorvastatin_Lactone", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC1CC(O)CC(=O)O1", 8.70),
    ("Lenalidomide", "Nc1cccc2c1CN(C1CCC(=O)NC1=O)C2=O", 7.58),
    ("Pomalidomide", "Nc1cccc2c1C(=O)N(C1CCC(=O)NC1=O)C2=O", 7.62),
    ("VHL_Ligand_Fragment", "Cc1ncsc1-c1ccc(CNC(=O)C2CC(O)CN2C(=O)C(NC(=O)C)C(C)(C)C)cc1", 8.15),
    ("Sorafenib_Motif", "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1", 8.44),
    ("Celecoxib", "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1", 7.73),
    ("Ciprofloxacin_Motif", "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O", 7.40),
    ("Indomethacin_Motif", "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1", 7.21),
    ("Chlorpromazine", "CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21", 7.55),
]


def _parse_bracket_atom(token: str) -> Tuple[str, int, Optional[int], int, bool, Optional[str]]:
    """Parse a bracketed SMILES atom specification such as [nH+], [13CH3], or [C@@H]."""
    inner = token[1:-1]
    if not inner:
        raise SmilesParseError("Empty bracket atom '[]'")

    pos = 0
    n = len(inner)
    isotope: Optional[int] = None
    while pos < n and inner[pos].isdigit():
        pos += 1
    if pos > 0:
        isotope = int(inner[:pos])

    if pos >= n:
        raise SmilesParseError(f"Missing element symbol in bracket atom '{token}'")

    is_aromatic = False
    symbol = ""
    if inner[pos] in AROMATIC_SYMBOLS:
        is_aromatic = True
        symbol = AROMATIC_SYMBOLS[inner[pos]]
        pos += 1
    elif inner[pos].isupper():
        if pos + 1 < n and inner[pos + 1].islower():
            two_char = inner[pos : pos + 2]
            if two_char in ELEMENT_TABLE:
                symbol = two_char
                pos += 2
        if not symbol:
            one_char = inner[pos]
            if one_char in ELEMENT_TABLE:
                symbol = one_char
                pos += 1
            else:
                raise SmilesParseError(f"Unsupported element in bracket atom '{token}'")
    else:
        raise SmilesParseError(f"Invalid element start in bracket atom '{token}'")

    chirality: Optional[str] = None
    if pos < n and inner[pos] == "@":
        if pos + 1 < n and inner[pos + 1] == "@":
            chirality = "@@"
            pos += 2
        else:
            chirality = "@"
            pos += 1

    explicit_h = 0
    if pos < n and inner[pos] == "H":
        pos += 1
        h_start = pos
        while pos < n and inner[pos].isdigit():
            pos += 1
        explicit_h = int(inner[h_start:pos]) if pos > h_start else 1

    formal_charge = 0
    if pos < n and inner[pos] in ("+", "-"):
        sign = 1 if inner[pos] == "+" else -1
        pos += 1
        if pos < n and inner[pos].isdigit():
            c_start = pos
            while pos < n and inner[pos].isdigit():
                pos += 1
            formal_charge = sign * int(inner[c_start:pos])
        else:
            count = 1
            while pos < n and inner[pos] == ("+" if sign > 0 else "-"):
                count += 1
                pos += 1
            formal_charge = sign * count

    if pos < n and inner[pos] == ":":
        pos += 1
        while pos < n and inner[pos].isdigit():
            pos += 1

    if pos != n:
        raise SmilesParseError(f"Unrecognized trailing tokens in bracket atom '{token}'")

    return symbol, formal_charge, isotope, explicit_h, is_aromatic, chirality


def _assign_implicit_hydrogens(mol: Molecule, explicit_bracket_flags: List[bool]) -> None:
    """Compute implicit hydrogen counts and preliminary atomic hybridization."""
    for atom in mol.atoms:
        info = ELEMENT_TABLE[atom.symbol]
        bond_sum = mol.bond_order_sum(atom.index)
        deg = mol.degree(atom.index)
        has_double = any(b.order == BondOrder.DOUBLE for _, b in mol.neighbors(atom.index))
        has_triple = any(b.order == BondOrder.TRIPLE for _, b in mol.neighbors(atom.index))

        if explicit_bracket_flags[atom.index]:
            atom.implicit_h = 0
        else:
            effective_valence = bond_sum
            if atom.is_aromatic and atom.symbol in ("C", "N", "O", "S", "P"):
                target_valence = info.default_valences[0]
                if atom.symbol == "N" and deg == 2:
                    atom.implicit_h = 0
                elif atom.symbol in ("O", "S") and deg == 2:
                    atom.implicit_h = 0
                else:
                    rem = int(round(target_valence - effective_valence))
                    atom.implicit_h = max(0, rem)
            else:
                chosen = info.default_valences[-1]
                for val in info.default_valences:
                    if val >= effective_valence - 1e-6:
                        chosen = val
                        break
                atom.implicit_h = max(0, int(round(chosen - effective_valence)))

        if atom.symbol == "H":
            atom.hybridization = Hybridization.S
        elif atom.is_aromatic or has_double:
            atom.hybridization = Hybridization.SP2
        elif has_triple:
            atom.hybridization = Hybridization.SP
        elif atom.symbol in ("C", "N", "O", "P", "S", "Si"):
            atom.hybridization = Hybridization.SP3
        else:
            atom.hybridization = Hybridization.OTHER


def parse_smiles(smiles: str, name: str = "ligand") -> Molecule:
    """Parse a SMILES string into a validated Molecule graph with implicit hydrogens."""
    text = smiles.strip()
    if not text:
        raise SmilesParseError("SMILES string is empty")

    mol = Molecule(name=name, smiles=text)
    explicit_flags: List[bool] = []
    branch_stack: List[Optional[int]] = []
    ring_closures: Dict[int, Tuple[int, Optional[Tuple[BondOrder, bool]], Optional[str]]] = {}

    prev_atom: Optional[int] = None
    pending_bond: Optional[Tuple[BondOrder, bool]] = None
    pending_stereo: Optional[str] = None

    i = 0
    n = len(text)
    while i < n:
        ch = text[i]

        if ch in (" ", "\t", "\n"):
            i += 1
            continue

        if ch == "(":
            if prev_atom is None:
                raise SmilesParseError("Branch '(' cannot appear without preceding atom")
            branch_stack.append(prev_atom)
            i += 1
            continue

        if ch == ")":
            if not branch_stack:
                raise SmilesParseError("Unmatched closing parenthesis ')' in SMILES")
            prev_atom = branch_stack.pop()
            pending_bond = None
            pending_stereo = None
            i += 1
            continue

        if ch == ".":
            prev_atom = None
            pending_bond = None
            pending_stereo = None
            i += 1
            continue

        if ch in _BOND_CHAR_TO_ORDER:
            if pending_bond is not None:
                raise SmilesParseError(f"Consecutive bond tokens near index {i} in '{text}'")
            pending_bond = _BOND_CHAR_TO_ORDER[ch]
            i += 1
            continue

        if ch in ("/", "\\"):
            pending_stereo = ch
            i += 1
            continue

        if ch == "%" or ch.isdigit():
            if prev_atom is None:
                raise SmilesParseError(f"Ring closure digit at index {i} has no preceding atom")
            if ch == "%":
                if i + 2 >= n or not (text[i + 1].isdigit() and text[i + 2].isdigit()):
                    raise SmilesParseError(f"Malformed two-digit ring closure '%nn' near index {i}")
                ring_id = int(text[i + 1 : i + 3])
                i += 3
            else:
                ring_id = int(ch)
                i += 1

            if ring_id not in ring_closures:
                ring_closures[ring_id] = (prev_atom, pending_bond, pending_stereo)
            else:
                other_atom, open_bond, open_stereo = ring_closures.pop(ring_id)
                if other_atom == prev_atom:
                    raise SmilesParseError(f"Self-loop ring closure {ring_id} on atom {prev_atom}")
                chosen_bond = pending_bond or open_bond
                a1 = mol.atoms[other_atom]
                a2 = mol.atoms[prev_atom]
                if chosen_bond is None:
                    if a1.is_aromatic and a2.is_aromatic:
                        order, is_arom = BondOrder.AROMATIC, True
                    else:
                        order, is_arom = BondOrder.SINGLE, False
                else:
                    order, is_arom = chosen_bond
                mol.bonds.append(
                    Bond(
                        index=len(mol.bonds),
                        begin_idx=other_atom,
                        end_idx=prev_atom,
                        order=order,
                        is_aromatic=is_arom,
                        stereo=pending_stereo or open_stereo,
                    )
                )
            pending_bond = None
            pending_stereo = None
            continue

        is_bracket = False
        if ch == "[":
            end_br = text.find("]", i + 1)
            if end_br == -1:
                raise SmilesParseError(f"Unclosed bracket atom starting at index {i}")
            sym, f_chg, iso, exp_h, is_arom, chiral = _parse_bracket_atom(text[i : end_br + 1])
            is_bracket = True
            i = end_br + 1
        elif ch in AROMATIC_SYMBOLS:
            sym = AROMATIC_SYMBOLS[ch]
            f_chg, iso, exp_h, is_arom, chiral = 0, None, 0, True, None
            i += 1
        elif ch.isupper():
            two_char = text[i : i + 2]
            if len(two_char) == 2 and two_char in _ORGANIC_SUBSET:
                sym = two_char
                i += 2
            elif ch in _ORGANIC_SUBSET:
                sym = ch
                i += 1
            else:
                raise SmilesParseError(f"Unsupported bare element '{ch}' at index {i} in '{text}'")
            f_chg, iso, exp_h, is_arom, chiral = 0, None, 0, False, None
        else:
            raise SmilesParseError(f"Unexpected token '{ch}' at index {i} in '{text}'")

        info = ELEMENT_TABLE[sym]
        atom_idx = len(mol.atoms)
        atom = Atom(
            index=atom_idx,
            symbol=sym,
            atomic_number=info.atomic_number,
            formal_charge=f_chg,
            isotope=iso,
            explicit_h=exp_h,
            is_aromatic=is_arom,
            chirality=chiral,
        )
        mol.atoms.append(atom)
        explicit_flags.append(is_bracket)

        if prev_atom is not None:
            if pending_bond is None:
                if mol.atoms[prev_atom].is_aromatic and atom.is_aromatic:
                    order, bond_arom = BondOrder.AROMATIC, True
                else:
                    order, bond_arom = BondOrder.SINGLE, False
            else:
                order, bond_arom = pending_bond
            mol.bonds.append(
                Bond(
                    index=len(mol.bonds),
                    begin_idx=prev_atom,
                    end_idx=atom_idx,
                    order=order,
                    is_aromatic=bond_arom,
                    stereo=pending_stereo,
                )
            )

        prev_atom = atom_idx
        pending_bond = None
        pending_stereo = None

    if branch_stack:
        raise SmilesParseError("Unclosed branch parenthesis '(' in SMILES")
    if ring_closures:
        unclosed = ", ".join(str(k) for k in sorted(ring_closures))
        raise SmilesParseError(f"Unclosed ring closure(s): {unclosed}")
    if not mol.atoms:
        raise SmilesParseError("SMILES produced zero atoms")

    mol.rebuild_adjacency()
    _assign_implicit_hydrogens(mol, explicit_flags)
    return mol


def _format_atom_smiles(atom: Atom) -> str:
    """Serialize a single Atom into SMILES notation."""
    needs_bracket = (
        atom.formal_charge != 0
        or atom.isotope is not None
        or atom.symbol not in _ORGANIC_SUBSET
        or (atom.is_aromatic and atom.symbol in ("N", "O", "S") and atom.total_h > 0)
    )
    sym = atom.symbol.lower() if atom.is_aromatic and atom.symbol in ("B", "C", "N", "O", "P", "S") else atom.symbol
    if not needs_bracket:
        return sym

    parts: List[str] = ["["]
    if atom.isotope is not None:
        parts.append(str(atom.isotope))
    parts.append(sym)
    if atom.total_h > 0:
        parts.append("H" if atom.total_h == 1 else f"H{atom.total_h}")
    if atom.formal_charge > 0:
        parts.append("+" if atom.formal_charge == 1 else f"+{atom.formal_charge}")
    elif atom.formal_charge < 0:
        parts.append("-" if atom.formal_charge == -1 else f"{atom.formal_charge}")
    parts.append("]")
    return "".join(parts)


def to_smiles(mol: Molecule, canonical: bool = True) -> str:
    """Serialize a Molecule graph into a canonical or traversal-order SMILES string."""
    if not mol.atoms:
        return ""

    n = mol.num_atoms
    ranks = [
        (a.atomic_number * 1000 + mol.degree(a.index) * 50 + a.total_h * 10 + (1 if a.is_aromatic else 0))
        for a in mol.atoms
    ]
    if canonical:
        for _ in range(4):
            new_ranks: List[int] = []
            for idx in range(n):
                nbr_sig = sum(ranks[nbr] * (int(round(b.order.value * 2)) + 1) for nbr, b in mol.neighbors(idx))
                new_ranks.append(ranks[idx] * 31 + nbr_sig)
            ranks = new_ranks

    visited = [False] * n
    dfs_children: Dict[int, List[Tuple[int, Bond]]] = {i: [] for i in range(n)}
    ring_edges: List[Tuple[int, int, Bond]] = []
    seen_bonds = set()

    def dfs(u: int, p: Optional[int]) -> None:
        visited[u] = True
        nbrs = sorted(mol.neighbors(u), key=lambda pair: (ranks[pair[0]], pair[0]))
        for v, bond in nbrs:
            if bond.index in seen_bonds:
                continue
            if not visited[v]:
                seen_bonds.add(bond.index)
                dfs_children[u].append((v, bond))
                dfs(v, u)
            elif v != p:
                seen_bonds.add(bond.index)
                ring_edges.append((u, v, bond))

    roots: List[int] = []
    order_nodes = sorted(range(n), key=lambda idx: (ranks[idx], idx))
    for start in order_nodes:
        if not visited[start]:
            roots.append(start)
            dfs(start, None)

    ring_labels_at_atom: Dict[int, List[Tuple[int, str]]] = {i: [] for i in range(n)}
    for r_num, (u, v, bond) in enumerate(ring_edges, start=1):
        b_sym = _ORDER_TO_CHAR[bond.order]
        if bond.order == BondOrder.SINGLE and mol.atoms[u].is_aromatic and mol.atoms[v].is_aromatic:
            b_sym = "-"
        ring_labels_at_atom[u].append((r_num, b_sym))
        ring_labels_at_atom[v].append((r_num, ""))

    def emit(u: int) -> str:
        chunks = [_format_atom_smiles(mol.atoms[u])]
        for r_num, b_sym in ring_labels_at_atom[u]:
            r_tok = f"{r_num}" if r_num < 10 else f"%{r_num}"
            chunks.append(f"{b_sym}{r_tok}")

        children = dfs_children[u]
        if not children:
            return "".join(chunks)

        for idx, (v, bond) in enumerate(children):
            b_sym = _ORDER_TO_CHAR[bond.order]
            if bond.order == BondOrder.SINGLE and mol.atoms[u].is_aromatic and mol.atoms[v].is_aromatic:
                b_sym = "-"
            sub = b_sym + emit(v)
            if idx < len(children) - 1:
                chunks.append(f"({sub})")
            else:
                chunks.append(sub)
        return "".join(chunks)

    return ".".join(emit(r) for r in roots)


def validate_smiles_roundtrip(smiles: str) -> Tuple[bool, str]:
    """Parse a SMILES string and verify canonical roundtrip re-parsing preserves atom/bond counts."""
    mol = parse_smiles(smiles)
    canon = to_smiles(mol, canonical=True)
    reparsed = parse_smiles(canon)
    ok = mol.num_atoms == reparsed.num_atoms and mol.num_bonds == reparsed.num_bonds
    return ok, canon
