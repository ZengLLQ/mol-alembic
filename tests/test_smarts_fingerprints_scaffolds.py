"""Tests for SMARTS substructure matching, PAINS alerts, Morgan fingerprints, clustering, and MMP/SALI cliffs."""

import math
import numpy as np
import pytest

from mol_alembic.fingerprints import (
    butina_clustering,
    dice_similarity,
    maxmin_diversity_selection,
    morgan_bit_vector,
    morgan_fingerprint_counts,
    pairwise_tanimoto_matrix,
    tanimoto_similarity,
    topological_torsion_vector,
    tversky_similarity,
)
from mol_alembic.scaffolds import (
    extract_bemis_murcko_scaffold,
    find_matched_molecular_pairs,
    single_cut_fragments,
)
from mol_alembic.smarts import (
    annotate_pharmacophores,
    match_smarts,
    screen_structural_alerts,
)
from mol_alembic.smiles import BENCHMARK_MOLECULES, parse_smiles
from mol_alembic.types import PharmacophoreType, SmartsSyntaxError


def test_smarts_isomorphism_carboxylic_acid_and_amide() -> None:
    ibu = parse_smiles("CC(C)Cc1ccc(C(C)C(=O)O)cc1")
    acid_hits = match_smarts(ibu, "[#6](=[#8])[#8;H1]")
    assert len(acid_hits) == 1

    paracetamol = parse_smiles("CC(=O)Nc1ccc(O)cc1")
    amide_hits = match_smarts(paracetamol, "[#6](=[#8])-[#7]")
    assert len(amide_hits) == 1

    with pytest.raises(SmartsSyntaxError):
        match_smarts(ibu, "[#999]")


def test_structural_alerts_detect_pains_and_clean_drugs() -> None:
    aspirin = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
    assert len(screen_structural_alerts(aspirin)) == 0

    michael = parse_smiles("C=CC(=O)c1ccccc1")
    hits = screen_structural_alerts(michael)
    assert any(h.alert_name == "Michael_Acceptor" for h in hits)

    epoxide = parse_smiles("C1OC1c1ccccc1")
    epox_hits = screen_structural_alerts(epoxide)
    assert any(h.alert_name == "Epoxide_Ring" for h in epox_hits)


def test_pharmacophore_annotation() -> None:
    propranolol = parse_smiles("CC(C)NCC(O)COc1cccc2ccccc12")
    pharm = annotate_pharmacophores(propranolol)
    types = {p.feature_type for p in pharm}
    assert PharmacophoreType.HBD in types
    assert PharmacophoreType.HBA in types
    assert PharmacophoreType.AROMATIC in types
    assert PharmacophoreType.POS_IONIZABLE in types


def test_morgan_fingerprint_and_similarity_metrics() -> None:
    m1 = parse_smiles("Fc1ccc(Nc2ncnc3ccccc23)cc1")
    m2 = parse_smiles("Clc1ccc(Nc2ncnc3ccccc23)cc1")
    m3 = parse_smiles("CCCCCCCCCC")

    counts1, prov1 = morgan_fingerprint_counts(m1, radius=2)
    assert len(counts1) > 10
    assert all(len(v) >= 1 for v in prov1.values())

    fp1 = morgan_bit_vector(m1, radius=2, nbits=1024)
    fp2 = morgan_bit_vector(m2, radius=2, nbits=1024)
    fp3 = morgan_bit_vector(m3, radius=2, nbits=1024)

    assert math.isclose(tanimoto_similarity(fp1, fp1), 1.0, abs_tol=1e-9)
    assert tanimoto_similarity(fp1, fp2) > tanimoto_similarity(fp1, fp3)
    assert dice_similarity(fp1, fp2) >= tanimoto_similarity(fp1, fp2)
    assert 0.0 <= tversky_similarity(fp1, fp2, alpha=0.8, beta=0.2) <= 1.0
    assert np.sum(topological_torsion_vector(m1)) > 0.0


def test_butina_clustering_and_maxmin_diversity() -> None:
    fps = np.vstack([morgan_bit_vector(parse_smiles(s), radius=2, nbits=1024) for _, s, _ in BENCHMARK_MOLECULES[:12]])
    sim = pairwise_tanimoto_matrix(fps)
    assert sim.shape == (12, 12)
    assert np.allclose(sim, sim.T, atol=1e-12)

    clusters = butina_clustering(fps, similarity_threshold=0.25)
    flattened = [idx for c in clusters for idx in c]
    assert sorted(flattened) == list(range(12))

    diverse = maxmin_diversity_selection(fps, k=5, seed_idx=0)
    assert len(diverse) == 5
    assert len(set(diverse)) == 5


def test_bemis_murcko_and_mmp_activity_cliffs() -> None:
    gefitinib = parse_smiles("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCC")
    scaf, _ = extract_bemis_murcko_scaffold(gefitinib, generic_skeleton=False)
    gen, _ = extract_bemis_murcko_scaffold(gefitinib, generic_skeleton=True)
    assert len(scaf) > 5 and len(gen) > 5
    assert "F" not in scaf and "Cl" not in scaf

    frags = single_cut_fragments(gefitinib, min_core_atoms=5)
    assert len(frags) >= 2

    series = [
        ("Analogue_H", "c1ccc(Nc2ncnc3ccccc23)cc1", 6.2),
        ("Analogue_F", "Fc1ccc(Nc2ncnc3ccccc23)cc1", 8.6),
    ]
    mmps, cliffs = find_matched_molecular_pairs(series, min_core_atoms=6, sali_threshold=2.0)
    assert len(mmps) >= 1
    assert len(cliffs) >= 1
    assert cliffs[0].sali_score >= 2.0
