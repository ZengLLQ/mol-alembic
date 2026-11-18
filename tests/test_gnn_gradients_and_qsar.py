"""Tests for graph featurization, exact analytical vs finite-difference GNN backpropagation, and conformal QSAR."""

import math
import numpy as np

from mol_alembic.calibration import (
    build_sar_benchmark_dataset,
    calibrate_conformal_qsar,
    evaluate_ridge_ecfp_baseline,
)
from mol_alembic.featurizer import ATOM_FEATURE_DIM, EDGE_FEATURE_DIM, collate_graph_batch, featurize_molecule
from mol_alembic.gnn import GNNConfig, MolecularGraphNet
from mol_alembic.smiles import parse_smiles


def test_graph_featurizer_shapes_and_laplacian_spectrum() -> None:
    mol = parse_smiles("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCC")
    g = featurize_molecule(mol)
    n = mol.num_atoms
    assert g.node_features.shape == (n, ATOM_FEATURE_DIM)
    assert g.edge_index.shape == (2, 2 * mol.num_bonds)
    assert g.edge_features.shape == (2 * mol.num_bonds, EDGE_FEATURE_DIM)
    assert g.shortest_paths.shape == (n, n)
    assert np.all(g.shortest_paths == g.shortest_paths.T)

    eigvals = np.linalg.eigvalsh(g.norm_adjacency)
    assert float(np.min(eigvals)) >= -1.0 - 1e-9
    assert float(np.max(eigvals)) <= 1.0 + 1e-9

    batch = collate_graph_batch([g, featurize_molecule(parse_smiles("c1ccccc1"))])
    assert batch["node_features"].shape[0] == n + 6


def test_gnn_analytical_gradients_match_finite_differences() -> None:
    mol = parse_smiles("CC(=O)Nc1ccc(O)cc1")
    graph = featurize_molecule(mol)
    gnn = MolecularGraphNet(
        GNNConfig(hidden_dim=8, num_layers=2, dropout_rate=0.0, seed=7)
    )
    target = 1.25
    cache = gnn._forward_single(graph, training=False)
    _, analytical_grads = gnn._backward_single(cache, target_norm=target)

    eps = 1e-6
    checked_params = ["W_in", "W_self_0", "W_nbr_0", "W_edge_0", "a_attn_0", "W_gate", "W_head1", "W_head2"]
    for pname in checked_params:
        param = gnn.params[pname]
        grad = analytical_grads[pname]
        coords = [(0, 0), (param.shape[0] - 1, param.shape[1] - 1)]
        for r, c in coords:
            orig = float(param[r, c])
            param[r, c] = orig + eps
            lp = 0.5 * (gnn._forward_single(graph, training=False).pred - target) ** 2
            param[r, c] = orig - eps
            lm = 0.5 * (gnn._forward_single(graph, training=False).pred - target) ** 2
            param[r, c] = orig
            fd = (lp - lm) / (2.0 * eps)
            ana = float(grad[r, c])
            assert math.isclose(ana, fd, rel_tol=1e-4, abs_tol=1e-6), (
                f"Gradient mismatch at {pname}[{r},{c}]: ana={ana} vs fd={fd}"
            )


def test_gnn_training_convergence_and_saliency_simplex() -> None:
    _names, smiles_list, activities = build_sar_benchmark_dataset(seed=13)
    gnn = MolecularGraphNet(GNNConfig(hidden_dim=20, num_layers=2, learning_rate=0.012, seed=13))
    hist = gnn.fit(smiles_list[:14], activities[:14], epochs=22, batch_size=7)
    assert len(hist) == 22
    assert hist[-1] < hist[0]

    pred, unc, saliency = gnn.predict_smiles(smiles_list[0], mc_samples=8)
    assert math.isfinite(pred)
    assert unc > 0.0
    assert math.isclose(float(np.sum(saliency)), 1.0, abs_tol=1e-6)
    assert np.all(saliency >= 0.0)


def test_conformal_qsar_and_ridge_baseline() -> None:
    _, smiles_list, activities = build_sar_benchmark_dataset(seed=42)
    ridge_r2, ridge_rmse, preds = evaluate_ridge_ecfp_baseline(
        smiles_list, activities, list(range(18)), list(range(18, len(smiles_list)))
    )
    assert math.isfinite(ridge_r2) and ridge_rmse > 0.0
    assert len(preds) == len(smiles_list) - 18

    _, conf = calibrate_conformal_qsar(smiles_list, activities, epochs=18, seed=42)
    assert 0.0 <= conf.empirical_coverage_90 <= 1.0
    assert conf.conformal_quantile_90 >= 0.0
    assert conf.mean_interval_width > 0.0
    assert conf.rmse > 0.0
