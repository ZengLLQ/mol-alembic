"""Tests for PROTAC ternary equilibrium, Sobol sensitivity, NSGA-II optimizer, CLI subcommands, and 7-bit ASCII."""

import math
import numpy as np

from mol_alembic.calibration import build_sar_benchmark_dataset, calibrate_conformal_qsar, protac_sobol_audit
from mol_alembic.cli import main as cli_main
from mol_alembic.optimizer import generate_molecular_neighbors, nsga2_rank_and_crowding, optimize_molecules
from mol_alembic.protac import (
    estimate_linker_cooperativity,
    solve_ternary_equilibrium,
    sweep_protac_hook_effect,
)
from mol_alembic.smiles import parse_smiles
from mol_alembic.types import OptimizationCandidate
from mol_alembic.visualizer import assert_ascii


def test_protac_douglass_miller_mass_conservation_and_hook_effect() -> None:
    p0, e0, kd_p, kd_e = 100.0, 50.0, 10.0, 30.0
    for l0 in (0.1, 5.0, 20.0, 200.0, 5000.0):
        pf, ef, lf, pl, le, ple = solve_ternary_equilibrium(p0, e0, l0, kd_p, kd_e, alpha=6.0)
        assert math.isclose(pf + pl + ple, p0, abs_tol=1e-9)
        assert math.isclose(ef + le + ple, e0, abs_tol=1e-9)
        assert math.isclose(lf + pl + le + ple, l0, abs_tol=1e-9)
        assert min(pf, ef, lf, pl, le, ple) >= 0.0

    sweep_coop = sweep_protac_hook_effect(kd_poi_nm=10.0, kd_e3_nm=30.0, alpha=8.0, n_points=35)
    sweep_non = sweep_protac_hook_effect(kd_poi_nm=10.0, kd_e3_nm=30.0, alpha=1.0, n_points=35)
    assert sweep_coop.max_mass_balance_error < 1e-8
    assert sweep_coop.peak_ple_nm > sweep_non.peak_ple_nm
    assert sweep_coop.peak_ple_nm > float(sweep_coop.ternary_ple_nm[0])
    assert sweep_coop.peak_ple_nm > float(sweep_coop.ternary_ple_nm[-1])
    assert sweep_coop.dc50_nm is not None and sweep_coop.dmax > 0.5


def test_linker_cooperativity_and_sobol_sensitivity() -> None:
    alpha_short = estimate_linker_cooperativity(linker_length_atoms=2, optimal_length=8)
    alpha_opt = estimate_linker_cooperativity(linker_length_atoms=8, optimal_length=8)
    assert alpha_opt > alpha_short

    sobol = protac_sobol_audit(n_samples=32, seed=42)
    for k in sobol.param_names:
        assert 0.0 <= sobol.first_order[k] <= 1.0
        assert 0.0 <= sobol.total_order[k] <= 1.0


def test_molecular_mutation_and_nsga2_pareto_sorting() -> None:
    rng = np.random.default_rng(7)
    seed_smi = "c1ccc(Nc2ncnc3ccccc23)cc1"
    variants = generate_molecular_neighbors(seed_smi, rng, max_variants=6)
    assert len(variants) >= 3
    for smi, _op in variants:
        mol = parse_smiles(smi)
        assert mol.num_atoms >= 6

    cands = [
        OptimizationCandidate("A", 0, "seed", 8.2, 0.1, 0.80, 2.0, 0.85, 0, 0.82),
        OptimizationCandidate("B", 1, "mut", 7.0, 0.1, 0.60, 3.0, 0.60, 0, 0.60),
        OptimizationCandidate("C", 1, "mut", 8.9, 0.1, 0.62, 2.5, 0.72, 0, 0.79),
    ]
    ranked = nsga2_rank_and_crowding(cands)
    rank_map = {c.smiles: c.pareto_rank for c in ranked}
    assert rank_map["A"] == 0
    assert rank_map["C"] == 0
    assert rank_map["B"] >= 1


def test_denovo_optimize_pipeline() -> None:
    _, smiles_list, activities = build_sar_benchmark_dataset(seed=42)
    gnn, _ = calibrate_conformal_qsar(smiles_list, activities, epochs=14, seed=42)
    res = optimize_molecules("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCC", gnn, generations=3, pop_size=8, seed=42)
    assert res.valid_unique_count >= 5
    assert len(res.pareto_front) >= 1
    assert res.best_mpo_history[-1] >= res.best_mpo_history[0] - 1e-9


def test_cli_subcommands_and_strict_ascii_output(capsys: object) -> None:
    commands = [
        ["demo", "--seed", "42"],
        ["screen", "--epochs", "12", "--seed", "7"],
        ["adme", "--smiles", "CC(=O)Oc1ccccc1C(=O)O", "--dose-mg", "100"],
        ["dock", "--smiles", "c1ccc(Nc2ncnc3ccccc23)cc1", "--seed", "5"],
        ["protac", "--kd-poi-nm", "12.0", "--kd-e3-nm", "35.0", "--alpha", "8.5"],
        ["optimize", "--generations", "2", "--pop-size", "6", "--seed", "9"],
    ]
    for cmd in commands:
        rc = cli_main(cmd)
        assert rc == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert_ascii(captured.out)
    assert "MOLALEMBIC" in captured.out
