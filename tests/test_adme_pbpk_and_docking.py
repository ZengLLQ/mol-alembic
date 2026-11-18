"""Tests for ADME physicochemical models, well-stirred clearance, PBPK ODE mass balance, and 3D docking."""

import math
import numpy as np

from mol_alembic.adme import predict_adme, score_adme_desirability, simulate_pbpk_2comp
from mol_alembic.docking import dock_ligand, embed_3d_conformer, score_ligand_pose
from mol_alembic.smiles import parse_smiles


def test_adme_physical_trends_and_well_stirred_bounds() -> None:
    met = parse_smiles("CN(C)C(=N)N=C(N)N", name="Metformin")
    gef = parse_smiles("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCC", name="Gefitinib")

    adme_met = predict_adme(met)
    adme_gef = predict_adme(gef)

    assert adme_met.log_s_mol_l > adme_gef.log_s_mol_l
    assert adme_met.fu_plasma > adme_gef.fu_plasma
    assert 0.0 < adme_met.cl_hepatic_ml_min_kg < 20.7
    assert 0.0 < adme_gef.cl_hepatic_ml_min_kg < 20.7
    assert 0.0 <= score_adme_desirability(adme_gef) <= 1.0


def test_pbpk_2comp_mass_conservation_oral_and_iv() -> None:
    mol = parse_smiles("CC(=O)Oc1ccccc1C(=O)O", name="Aspirin")
    pk_oral = simulate_pbpk_2comp(mol, dose_mg=100.0, route="oral", duration_h=24.0)
    assert pk_oral.mass_balance_error < 1e-6
    assert pk_oral.cmax_mg_l > 0.0
    assert 0.0 < pk_oral.tmax_h < 12.0
    assert pk_oral.auc_0_inf_mg_h_l > 0.0
    assert np.all(pk_oral.plasma_conc_mg_l >= -1e-12)

    pk_iv = simulate_pbpk_2comp(mol, dose_mg=100.0, route="iv", duration_h=24.0)
    assert pk_iv.mass_balance_error < 1e-6
    assert pk_iv.tmax_h == 0.0
    assert pk_iv.auc_0_inf_mg_h_l >= pk_oral.auc_0_inf_mg_h_l


def test_3d_distance_geometry_conformer_and_docking() -> None:
    mol = parse_smiles("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCC", name="Gefitinib")
    coords, strain = embed_3d_conformer(mol, seed=11)
    assert coords.shape == (mol.num_atoms, 3)
    assert np.allclose(np.mean(coords, axis=0), np.zeros(3), atol=1e-6)
    assert math.isfinite(strain)

    for bond in mol.bonds:
        d = float(np.linalg.norm(coords[bond.begin_idx] - coords[bond.end_idx]))
        assert 1.0 < d < 2.0

    unopt = score_ligand_pose(mol, coords, internal_strain=strain)
    opt = dock_ligand(mol, seed=11, n_restarts=4)
    assert opt.total_score_kcal_mol <= unopt.total_score_kcal_mol + 1e-6
    assert opt.pocket_contacts >= 1
    assert math.isfinite(opt.ligand_efficiency)
