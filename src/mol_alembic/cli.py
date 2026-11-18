"""Command-line interface for MolAlembic computational drug discovery workflows."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .pipeline import (
    run_adme_pk_campaign,
    run_denovo_campaign,
    run_docking_campaign,
    run_full_discovery_demo,
    run_protac_campaign,
    run_qsar_screen,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mol-alembic",
        description="MolAlembic: Computational Drug Discovery, Cheminformatics, GAT-MPNN QSAR & De Novo Optimization",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Run integrated end-to-end AI drug discovery pipeline")
    p_demo.add_argument("--seed", type=int, default=42)

    p_screen = sub.add_parser("screen", help="Run QSAR GAT-MPNN conformal screening, clustering, and MMP analysis")
    p_screen.add_argument("--epochs", type=int, default=24)
    p_screen.add_argument("--sim-cutoff", type=float, default=0.30)
    p_screen.add_argument("--seed", type=int, default=42)

    p_adme = sub.add_parser("adme", help="Predict ADME properties and simulate 2-compartment PBPK PK curve")
    p_adme.add_argument("--smiles", type=str, required=True)
    p_adme.add_argument("--name", type=str, default="Candidate")
    p_adme.add_argument("--dose-mg", type=float, default=100.0)
    p_adme.add_argument("--duration-h", type=float, default=24.0)

    p_dock = sub.add_parser("dock", help="Embed 3D conformer and optimize empirical Vina/GBSA docking pose")
    p_dock.add_argument("--smiles", type=str, required=True)
    p_dock.add_argument("--name", type=str, default="DockLigand")
    p_dock.add_argument("--seed", type=int, default=42)

    p_protac = sub.add_parser("protac", help="Sweep PROTAC ternary equilibrium Hook Effect and Sobol sensitivity")
    p_protac.add_argument("--kd-poi-nm", type=float, default=12.0)
    p_protac.add_argument("--kd-e3-nm", type=float, default=35.0)
    p_protac.add_argument("--alpha", type=float, default=8.5)
    p_protac.add_argument("--poi-nm", type=float, default=100.0)
    p_protac.add_argument("--e3-nm", type=float, default=50.0)

    p_opt = sub.add_parser("optimize", help="Run NSGA-II multi-objective de novo molecular optimization")
    p_opt.add_argument("--smiles", type=str, default="COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCC")
    p_opt.add_argument("--generations", type=int, default=5)
    p_opt.add_argument("--pop-size", type=int, default=12)
    p_opt.add_argument("--seed", type=int, default=42)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point returning POSIX exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        print(run_full_discovery_demo(seed=args.seed))
        return 0

    if args.command == "screen":
        rep = run_qsar_screen(epochs=args.epochs, seed=args.seed, sim_cutoff=args.sim_cutoff)
        cm = rep.conformal_metrics
        print(
            f"QSAR Screening Complete | GAT-MPNN RMSE={cm.rmse:.3f} MAE={cm.mae:.3f} R2={cm.r2:.3f} "
            f"Rho={cm.spearman_rho:.3f} | 90% Coverage={cm.empirical_coverage_90:.2f} "
            f"(q={cm.conformal_quantile_90:.2f})\n"
            f"Ridge ECFP4 Baseline: R2={rep.ridge_r2:.3f} RMSE={rep.ridge_rmse:.3f} | "
            f"Butina Clusters={rep.cluster_count} (max={rep.largest_cluster_size}) | "
            f"MMPs={rep.mmp_count} Cliffs={len(rep.activity_cliffs)}\n\n"
            f"{rep.heatmap_ascii}"
        )
        return 0

    if args.command == "adme":
        _, pk, report = run_adme_pk_campaign(
            smiles=args.smiles,
            name=args.name,
            dose_mg=args.dose_mg,
            duration_h=args.duration_h,
        )
        print(report)
        print(
            f"\nPK Metrics: AUC_0-inf={pk.auc_0_inf_mg_h_l:.3f} mg*h/L | "
            f"t1/2_term={pk.half_life_terminal_h:.2f} h | "
            f"Time>IC50={pk.time_above_ic50_h:.2f} h | MassBalanceErr={pk.mass_balance_error:.2e} mg"
        )
        return 0

    if args.command == "dock":
        _, report = run_docking_campaign(smiles=args.smiles, name=args.name, seed=args.seed)
        print(report)
        return 0

    if args.command == "protac":
        sweep, _, report = run_protac_campaign(
            kd_poi_nm=args.kd_poi_nm,
            kd_e3_nm=args.kd_e3_nm,
            alpha=args.alpha,
            poi_conc_nm=args.poi_nm,
            e3_conc_nm=args.e3_nm,
        )
        print(report)
        print(f"\nMax Ternary Mass Balance Error: {sweep.max_mass_balance_error:.2e} nM | Dmax={sweep.dmax:.3f}")
        return 0

    if args.command == "optimize":
        res, report = run_denovo_campaign(
            seed_smiles=args.smiles,
            generations=args.generations,
            pop_size=args.pop_size,
            seed=args.seed,
        )
        print(report)
        print(
            f"\nEvaluated={res.evaluated_count} UniqueValid={res.valid_unique_count} "
            f"ParetoSize={len(res.pareto_front)} MPO_History={[round(x, 3) for x in res.best_mpo_history]}"
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
