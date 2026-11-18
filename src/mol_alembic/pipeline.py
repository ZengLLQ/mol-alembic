"""High-level orchestration pipelines combining cheminformatics, GAT-MPNN QSAR, ADME/PBPK, docking, and NSGA-II."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .adme import predict_adme, simulate_pbpk_2comp
from .calibration import (
    SobolResult,
    build_sar_benchmark_dataset,
    calibrate_conformal_qsar,
    evaluate_ridge_ecfp_baseline,
    protac_sobol_audit,
)
from .chemistry import analyze_molecule
from .docking import dock_ligand
from .fingerprints import butina_clustering, morgan_bit_vector, pairwise_tanimoto_matrix
from .gnn import MolecularGraphNet
from .optimizer import optimize_molecules
from .protac import sweep_protac_hook_effect
from .scaffolds import MatchedMolecularPair, extract_bemis_murcko_scaffold, find_matched_molecular_pairs
from .smarts import annotate_pharmacophores, screen_structural_alerts
from .smiles import parse_smiles
from .types import (
    ADMEProfile,
    ConformalPredictionResult,
    DockingPoseResult,
    PKSimulationResult,
    ParetoFrontResult,
    ProtacSweepResult,
)
from .visualizer import (
    render_gat_saliency_bars,
    render_molecule_card,
    render_pareto_scatter,
    render_pbpk_curve,
    render_protac_hook_curve,
    render_similarity_heatmap,
)


@dataclass
class ScreeningReport:
    """Summary of library QSAR conformal training, scaffold analysis, and clustering."""

    gnn: MolecularGraphNet
    conformal_metrics: ConformalPredictionResult
    ridge_r2: float
    ridge_rmse: float
    cluster_count: int
    largest_cluster_size: int
    mmp_count: int
    activity_cliffs: List[MatchedMolecularPair]
    heatmap_ascii: str


def run_qsar_screen(epochs: int = 26, seed: int = 42, sim_cutoff: float = 0.30) -> ScreeningReport:
    """Execute end-to-end SAR screening: ECFP4 clustering, MMP/SALI cliffs, Ridge baseline, and GAT-MPNN conformal."""
    names, smiles_list, activities = build_sar_benchmark_dataset(seed=seed)
    mols = [parse_smiles(s, name=n) for n, s in zip(names, smiles_list)]
    fps = np.vstack([morgan_bit_vector(m, radius=2, nbits=1024) for m in mols])
    sim_mat = pairwise_tanimoto_matrix(fps)
    clusters = butina_clustering(fps, similarity_threshold=sim_cutoff)

    records = list(zip(names, smiles_list, activities))
    mmps, cliffs = find_matched_molecular_pairs(records, min_core_atoms=5, sali_threshold=1.5)

    n = len(smiles_list)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).tolist()
    n_train = max(8, int(0.75 * n))
    ridge_r2, ridge_rmse, _ = evaluate_ridge_ecfp_baseline(
        smiles_list, activities, perm[:n_train], perm[n_train:]
    )

    gnn, conf_res = calibrate_conformal_qsar(smiles_list, activities, epochs=epochs, seed=seed)
    heatmap = render_similarity_heatmap(names, sim_mat, max_items=10)

    return ScreeningReport(
        gnn=gnn,
        conformal_metrics=conf_res,
        ridge_r2=round(ridge_r2, 4),
        ridge_rmse=round(ridge_rmse, 4),
        cluster_count=len(clusters),
        largest_cluster_size=len(clusters[0]) if clusters else 0,
        mmp_count=len(mmps),
        activity_cliffs=cliffs,
        heatmap_ascii=heatmap,
    )


def run_adme_pk_campaign(
    smiles: str,
    name: str = "Candidate",
    dose_mg: float = 100.0,
    duration_h: float = 24.0,
) -> Tuple[ADMEProfile, PKSimulationResult, str]:
    """Run physicochemical perception, ADME prediction, 2-compartment PBPK ODE, and ASCII telemetry."""
    mol = parse_smiles(smiles, name=name)
    alerts = screen_structural_alerts(mol)
    _, desc = analyze_molecule(mol, alert_count=len(alerts))
    adme = predict_adme(mol)
    pk = simulate_pbpk_2comp(mol, adme=adme, dose_mg=dose_mg, duration_h=duration_h)
    card = render_molecule_card(mol, desc, adme, alert_count=len(alerts))
    curve = render_pbpk_curve(pk)
    return adme, pk, f"{card}\n\n{curve}"


def run_docking_campaign(smiles: str, name: str = "DockLigand", seed: int = 42) -> Tuple[DockingPoseResult, str]:
    """Embed 3D conformer, dock into receptor cavity, and summarize pharmacophores."""
    mol = parse_smiles(smiles, name=name)
    pharm = annotate_pharmacophores(mol)
    scaffold_smi, _ = extract_bemis_murcko_scaffold(mol)
    pose = dock_ligand(mol, seed=seed, n_restarts=5)
    summary = (
        f"Ligand: {name} | Scaffold: {scaffold_smi or 'acyclic'}\n"
        f"Pharmacophores detected: {len(pharm)} | Pocket contacts: {pose.pocket_contacts}\n"
        f"Total Docking Score: {pose.total_score_kcal_mol:.3f} kcal/mol (LE={pose.ligand_efficiency:.3f})\n"
        f"  Gauss1={pose.gauss1_term:.3f}  Gauss2={pose.gauss2_term:.3f}  Repulsion={pose.repulsion_term:.3f}\n"
        f"  Hydrophobic={pose.hydrophobic_term:.3f}  HBond={pose.hbond_term:.3f}  GBSA={pose.gbsa_desolvation:.3f}\n"
        f"  TorsionPen={pose.torsion_penalty:.3f}  InternalStrain={pose.internal_strain:.3f}"
    )
    return pose, summary


def run_protac_campaign(
    kd_poi_nm: float = 12.0,
    kd_e3_nm: float = 35.0,
    alpha: float = 8.5,
    poi_conc_nm: float = 100.0,
    e3_conc_nm: float = 50.0,
) -> Tuple[ProtacSweepResult, SobolResult, str]:
    """Run PROTAC Hook-effect ternary sweep + Sobol global sensitivity audit."""
    sweep = sweep_protac_hook_effect(
        kd_poi_nm=kd_poi_nm,
        kd_e3_nm=kd_e3_nm,
        alpha=alpha,
        poi_conc_nm=poi_conc_nm,
        e3_conc_nm=e3_conc_nm,
    )
    sobol = protac_sobol_audit(n_samples=64, seed=42)
    curve = render_protac_hook_curve(sweep)
    sobol_txt = "PROTAC Sobol Sensitivity on [PLE] @ 50 nM:\n" + "\n".join(
        f"  {k:<12s}: S1={sobol.first_order[k]:.4f}  ST={sobol.total_order[k]:.4f}"
        for k in sobol.param_names
    )
    return sweep, sobol, f"{curve}\n\n{sobol_txt}"


def run_denovo_campaign(
    seed_smiles: str = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCC",
    generations: int = 5,
    pop_size: int = 12,
    seed: int = 42,
) -> Tuple[ParetoFrontResult, str]:
    """Train QSAR GAT-MPNN and evolve seed_smiles along the NSGA-II multi-objective Pareto front."""
    _, smiles_list, activities = build_sar_benchmark_dataset(seed=seed)
    gnn, _ = calibrate_conformal_qsar(smiles_list, activities, epochs=20, seed=seed)
    res = optimize_molecules(seed_smiles, gnn, generations=generations, pop_size=pop_size, seed=seed)
    scatter = render_pareto_scatter(res.all_top_candidates)
    rows = ["Top Pareto Rank-0 Candidates:"]
    for c in res.pareto_front[:6]:
        rows.append(
            f"  [Gen {c.generation} | {c.origin_op:<14s}] pIC50={c.predicted_pic50:.2f}+/-{c.uncertainty:.2f} "
            f"QED={c.qed:.3f} ADME={c.adme_score:.3f} SA={c.sa_score:.2f} MPO={c.mpo_score:.3f} | {c.smiles}"
        )
    return res, f"{scatter}\n\n" + "\n".join(rows)


def run_full_discovery_demo(seed: int = 42) -> str:
    """Run an integrated end-to-end AI drug discovery showcase across all MolAlembic engines."""
    screen_rep = run_qsar_screen(epochs=20, seed=seed)
    lead_smiles = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCC"
    lead_mol = parse_smiles(lead_smiles, name="Gefitinib_Motif")
    pred_pic50, unc, saliency = screen_rep.gnn.predict_smiles(lead_smiles, mc_samples=8)
    saliency_txt = render_gat_saliency_bars(lead_mol, saliency, top_k=8)

    _, _, adme_txt = run_adme_pk_campaign(lead_smiles, name="Gefitinib_Motif", dose_mg=250.0)
    _, dock_txt = run_docking_campaign(lead_smiles, name="Gefitinib_Motif", seed=seed)
    _, _, protac_txt = run_protac_campaign(kd_poi_nm=12.0, kd_e3_nm=35.0, alpha=8.5)
    _, opt_txt = run_denovo_campaign(lead_smiles, generations=4, pop_size=10, seed=seed)

    cm = screen_rep.conformal_metrics
    sections = [
        "=== MOLALEMBIC END-TO-END COMPUTATIONAL DRUG DISCOVERY DEMO ===",
        (
            f"[1] QSAR & Screening: Conformal R2={cm.r2:.3f}, RMSE={cm.rmse:.3f}, "
            f"90% Coverage={cm.empirical_coverage_90:.2f}, Clusters={screen_rep.cluster_count}, "
            f"MMPs={screen_rep.mmp_count}, Lead pIC50={pred_pic50:.2f}+/-{unc:.2f}"
        ),
        screen_rep.heatmap_ascii,
        saliency_txt,
        "[2] ADME & 2-Compartment PBPK Telemetry:\n" + adme_txt,
        "[3] 3D Distance-Geometry & Empirical Vina/GBSA Docking:\n" + dock_txt,
        "[4] PROTAC Ternary Cooperativity & Sobol Sensitivity:\n" + protac_txt,
        "[5] De Novo NSGA-II Multi-Objective Optimization:\n" + opt_txt,
    ]
    return "\n\n".join(sections)
