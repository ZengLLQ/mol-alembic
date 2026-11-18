"""Example 02: GAT-MPNN QSAR training, Ridge ECFP4 baseline, split conformal intervals, and atom saliency."""

from mol_alembic.calibration import (
    build_sar_benchmark_dataset,
    calibrate_conformal_qsar,
    evaluate_ridge_ecfp_baseline,
)
from mol_alembic.smiles import parse_smiles
from mol_alembic.visualizer import render_gat_saliency_bars


def main() -> None:
    print("=== Example 02: GAT-MPNN QSAR & Split Conformal Calibration ===")
    names, smiles_list, activities = build_sar_benchmark_dataset(seed=42)
    n = len(smiles_list)
    train_idx = list(range(0, int(0.75 * n)))
    test_idx = list(range(int(0.75 * n), n))

    ridge_r2, ridge_rmse, _ = evaluate_ridge_ecfp_baseline(smiles_list, activities, train_idx, test_idx)
    print(f"ECFP4 Ridge Baseline -> R2={ridge_r2:.4f}, RMSE={ridge_rmse:.4f}")

    gnn, conf = calibrate_conformal_qsar(smiles_list, activities, epochs=24, seed=42)
    print(
        f"GAT-MPNN Conformal (90%) -> R2={conf.r2:.4f}, RMSE={conf.rmse:.4f}, "
        f"MAE={conf.mae:.4f}, Spearman={conf.spearman_rho:.4f}, "
        f"Coverage={conf.empirical_coverage_90:.2f}, MeanWidth={conf.mean_interval_width:.3f}"
    )

    probe_name, probe_smi = names[9], smiles_list[9]
    mol = parse_smiles(probe_smi, name=probe_name)
    pred, unc, saliency = gnn.predict_smiles(probe_smi, mc_samples=10)
    print(f"\nProbe Molecule: {probe_name} | Predicted pIC50={pred:.3f} +/- {unc:.3f} (Observed={activities[9]:.3f})")
    print(render_gat_saliency_bars(mol, saliency, top_k=10))


if __name__ == "__main__":
    main()
