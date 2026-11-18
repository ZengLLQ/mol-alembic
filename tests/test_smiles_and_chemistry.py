"""Tests for SMILES parsing, SSSR ring perception, Kekulization, Gasteiger charges, and descriptors."""

import math
import pytest

from mol_alembic.chemistry import (
    analyze_molecule,
    compute_gasteiger_charges,
    kekulize_molecule,
    perceive_sssr,
)
from mol_alembic.smiles import BENCHMARK_MOLECULES, parse_smiles, to_smiles, validate_smiles_roundtrip
from mol_alembic.types import BondOrder, KekulizationError, SmilesParseError


def test_benchmark_smiles_roundtrip_all() -> None:
    for name, smi, _ in BENCHMARK_MOLECULES:
        ok, canon = validate_smiles_roundtrip(smi)
        assert ok, f"Roundtrip failed for {name}: {smi} -> {canon}"
        assert len(canon) > 0


def test_bracket_atoms_multidigit_rings_and_salts() -> None:
    mol = parse_smiles("C%10CCCCC%10.[NH4+].[O-]C(=O)C", name="salt_test")
    assert mol.num_atoms == 11
    assert mol.atoms[6].formal_charge == 1
    assert mol.atoms[7].formal_charge == -1
    assert sum(a.formal_charge for a in mol.atoms) == 0


@pytest.mark.parametrize(
    "bad_smiles",
    ["", "C(C", "CC)C", "C1CC", "C11", "[]", "[Xz]", "C==C"],
)
def test_malformed_smiles_raise_error(bad_smiles: str) -> None:
    with pytest.raises(SmilesParseError):
        parse_smiles(bad_smiles)


@pytest.mark.parametrize(
    "smiles,expected_rings,expected_arom_rings",
    [
        ("CCCCCC", 0, 0),
        ("C1CCCCC1", 1, 0),
        ("c1ccccc1", 1, 1),
        ("c1ccc2ccccc2c1", 2, 2),
        ("c1ccc2[nH]ccc2c1", 2, 2),
        ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", 2, 2),
        ("C1CC2CCC1C2", 2, 0),
    ],
)
def test_sssr_cyclomatic_dimension_and_aromaticity(
    smiles: str, expected_rings: int, expected_arom_rings: int
) -> None:
    mol = parse_smiles(smiles)
    rings = perceive_sssr(mol)
    assert len(rings) == expected_rings
    assert sum(1 for r in rings if r.is_aromatic) == expected_arom_rings


def test_kekulization_preserves_carbon_valence() -> None:
    for smi in ("c1ccccc1", "c1ccc2ccccc2c1", "c1ccncc1", "c1ccc2[nH]ccc2c1"):
        mol = parse_smiles(smi)
        perceive_sssr(mol)
        kek = kekulize_molecule(mol)
        assert not any(b.order == BondOrder.AROMATIC for b in kek.bonds)
        for atom in kek.atoms:
            if atom.symbol == "C":
                total_val = kek.bond_order_sum(atom.index) + atom.total_h
                assert math.isclose(total_val, 4.0, abs_tol=1e-6)


def test_kekulization_rejects_impossible_odd_radical_ring() -> None:
    mol = parse_smiles("c1cccc1")
    with pytest.raises(KekulizationError):
        kekulize_molecule(mol)


def test_gasteiger_exact_charge_conservation() -> None:
    for smi, expected_q in [
        ("CC(=O)Oc1ccccc1C(=O)O", 0.0),
        ("[NH3+]CC(=O)[O-]", 0.0),
        ("CC(=O)[O-]", -1.0),
        ("C[NH3+]", 1.0),
    ]:
        mol = parse_smiles(smi)
        charges = compute_gasteiger_charges(mol, iterations=8)
        assert math.isclose(sum(charges), expected_q, abs_tol=1e-11)
        if smi == "CC(=O)[O-]":
            assert charges[2] < charges[1]


def test_descriptors_qed_lipinski_and_topological_indices() -> None:
    aspirin = parse_smiles("CC(=O)Oc1ccccc1C(=O)O", name="Aspirin")
    _, desc = analyze_molecule(aspirin)
    assert desc.formula == "C9H8O4"
    assert 179.0 < desc.molecular_weight < 181.5
    assert 0.0 < desc.qed <= 1.0
    assert desc.lipinski_violations == 0
    assert desc.veber_violations == 0
    assert desc.wiener_index > 0
    assert desc.zagreb_m1 > 0 and desc.zagreb_m2 > 0
    assert desc.balaban_j > 0.0
    assert to_smiles(aspirin, canonical=True) != ""
