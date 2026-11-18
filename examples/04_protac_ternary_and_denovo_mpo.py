"""Example 04: PROTAC ternary complex Hook effect, Sobol sensitivity, and NSGA-II de novo optimization."""

from mol_alembic.calibration import build_sar_benchmark_dataset, calibrate_conformal_qsar, protac_sobol_audit
from mol_alembic.optimizer import optimize_molecules
from mol_alembic.protac import estimate_linker_cooperativity, sweep_protac_hook_effect
from mol_alembic.visualizer import render_pareto_scatter, render_protac_hook_curve


def main() -> None:
    print("=== Example 04: PROTAC Ternary Cooperativity & De Novo NSGA-II Optimization ===")
    alpha = estimate_linker_cooperativity(linker_length_atoms=8, optimal_length=8, rigid_ring_fraction=0.40)
    sweep = sweep_protac_hook_effect(kd_poi_nm=10.0, kd_e3_nm=30.0, alpha=alpha, poi_conc_nm=100.0, e3_conc_nm=50.0)
    print(render_protac_hook_curve(sweep))
    print(
        f"Cooperativity alpha={alpha:.3f} | Peak [PLE]={sweep.peak_ple_nm:.2f} nM @ {sweep.hook_peak_conc_nm:.2f} nM | "
        f"DC50={sweep.dc50_nm} nM | Dmax={sweep.dmax:.3f} | MaxMassErr={sweep.max_mass_balance_error:.2e} nM"
    )

    sobol = protac_sobol_audit(n_samples=64, seed=42)
    print("\nSobol Sensitivity on [PLE] (S1 / ST):")
    for k in sobol.param_names:
        print(f"  {k:<12s}: S1={sobol.first_order[k]:.4f}  ST={sobol.total_order[k]:.4f}")

    _, smiles_list, activities = build_sar_benchmark_dataset(seed=42)
    gnn, _ = calibrate_conformal_qsar(smiles_list, activities, epochs=18, seed=42)
    opt = optimize_molecules("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCC", gnn, generations=4, pop_size=12, seed=42)
    print()
    print(render_pareto_scatter(opt.all_top_candidates))
    print(f"Pareto Front Size={len(opt.pareto_front)} | Unique Valid={opt.valid_unique_count}")


if __name__ == "__main__":
    main()
