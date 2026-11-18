"""MolAlembic: Computational Drug Discovery, Cheminformatics, GAT-MPNN QSAR & De Novo Optimization."""

from .adme import predict_adme, predict_adme_from_smiles, score_adme_desirability, simulate_pbpk_2comp
from .calibration import (
    build_sar_benchmark_dataset,
    calibrate_conformal_qsar,
    compute_sobol_sensitivity,
    protac_sobol_audit,
)
from .chemistry import (
    analyze_molecule,
    compute_crippen_logp,
    compute_ertl_tpsa,
    compute_gasteiger_charges,
    compute_qed,
    kekulize_molecule,
    perceive_sssr,
)
from .docking import dock_ligand, embed_3d_conformer, score_ligand_pose
from .featurizer import collate_graph_batch, featurize_molecule
from .fingerprints import (
    butina_clustering,
    maxmin_diversity_selection,
    morgan_bit_vector,
    morgan_fingerprint_counts,
    pairwise_tanimoto_matrix,
    tanimoto_similarity,
)
from .gnn import GNNConfig, MolecularGraphNet
from .optimizer import generate_molecular_neighbors, nsga2_rank_and_crowding, optimize_molecules
from .pipeline import (
    run_adme_pk_campaign,
    run_denovo_campaign,
    run_docking_campaign,
    run_full_discovery_demo,
    run_protac_campaign,
    run_qsar_screen,
)
from .protac import (
    estimate_linker_cooperativity,
    solve_ternary_equilibrium,
    sweep_protac_hook_effect,
)
from .scaffolds import extract_bemis_murcko_scaffold, find_matched_molecular_pairs, single_cut_fragments
from .smarts import annotate_pharmacophores, match_smarts, screen_structural_alerts
from .smiles import BENCHMARK_MOLECULES, parse_smiles, to_smiles, validate_smiles_roundtrip

__version__ = "0.1.0"

__all__ = [
    "BENCHMARK_MOLECULES",
    "GNNConfig",
    "MolecularGraphNet",
    "__version__",
    "analyze_molecule",
    "annotate_pharmacophores",
    "build_sar_benchmark_dataset",
    "butina_clustering",
    "calibrate_conformal_qsar",
    "collate_graph_batch",
    "compute_crippen_logp",
    "compute_ertl_tpsa",
    "compute_gasteiger_charges",
    "compute_qed",
    "compute_sobol_sensitivity",
    "dock_ligand",
    "embed_3d_conformer",
    "estimate_linker_cooperativity",
    "extract_bemis_murcko_scaffold",
    "featurize_molecule",
    "find_matched_molecular_pairs",
    "generate_molecular_neighbors",
    "kekulize_molecule",
    "match_smarts",
    "maxmin_diversity_selection",
    "morgan_bit_vector",
    "morgan_fingerprint_counts",
    "nsga2_rank_and_crowding",
    "optimize_molecules",
    "pairwise_tanimoto_matrix",
    "parse_smiles",
    "perceive_sssr",
    "predict_adme",
    "predict_adme_from_smiles",
    "protac_sobol_audit",
    "run_adme_pk_campaign",
    "run_denovo_campaign",
    "run_docking_campaign",
    "run_full_discovery_demo",
    "run_protac_campaign",
    "run_qsar_screen",
    "score_adme_desirability",
    "score_ligand_pose",
    "screen_structural_alerts",
    "simulate_pbpk_2comp",
    "single_cut_fragments",
    "solve_ternary_equilibrium",
    "sweep_protac_hook_effect",
    "tanimoto_similarity",
    "to_smiles",
    "validate_smiles_roundtrip",
]
