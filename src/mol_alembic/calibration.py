"""Split conformal QSAR uncertainty calibration, SAR benchmarking, and Saltelli Sobol sensitivity."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, List, Sequence, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

from .chemistry import analyze_molecule
from .fingerprints import morgan_bit_vector
from .gnn import GNNConfig, MolecularGraphNet
from .protac import solve_ternary_equilibrium
from .smiles import BENCHMARK_MOLECULES, parse_smiles
from .types import ConformalPredictionResult


@dataclass(frozen=True)
class SobolResult:
    """First-order and total-order Saltelli Sobol sensitivity indices."""

    param_names: Tuple[str, ...]
    first_order: Dict[str, float]
    total_order: Dict[str, float]
    output_mean: float
    output_std: float


def build_sar_benchmark_dataset(
    seed: int = 42,
) -> Tuple[List[str], List[str], List[float]]:
    """Return curated medicinal chemistry (names, smiles, pIC50) dataset with biophysical noise."""
    rng = np.random.default_rng(seed)
    names: List[str] = []
    smiles_list: List[str] = []
    activities: List[float] = []

    for name, smi, base_pic50 in BENCHMARK_MOLECULES:
        mol = parse_smiles(smi, name=name)
        _, desc = analyze_molecule(mol)
        noise = float(rng.normal(0.0, 0.08))
        pic50 = base_pic50 + 0.04 * (desc.aromatic_ring_count - 1) + noise
        names.append(name)
        smiles_list.append(smi)
        activities.append(round(pic50, 4))

    return names, smiles_list, activities


def evaluate_ridge_ecfp_baseline(
    smiles_list: Sequence[str],
    targets: Sequence[float],
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    alpha_reg: float = 5.0,
) -> Tuple[float, float, NDArray[np.float64]]:
    """Fit closed-form L2-regularized Ridge regression on 512-bit ECFP4 vectors."""
    fps = np.vstack([morgan_bit_vector(parse_smiles(s), radius=2, nbits=512) for s in smiles_list])
    y = np.asarray(targets, dtype=np.float64)

    x_tr = fps[list(train_idx)]
    y_tr = y[list(train_idx)]
    x_te = fps[list(test_idx)]
    y_te = y[list(test_idx)]

    y_mean = float(np.mean(y_tr))
    yc = y_tr - y_mean
    gram = x_tr @ x_tr.T + alpha_reg * np.eye(len(x_tr))
    dual = np.linalg.solve(gram, yc)
    preds = y_mean + (x_te @ x_tr.T) @ dual

    rmse = float(np.sqrt(np.mean((preds - y_te) ** 2)))
    ss_tot = float(np.sum((y_te - np.mean(y_te)) ** 2))
    r2 = 1.0 - float(np.sum((preds - y_te) ** 2)) / max(1e-12, ss_tot)
    return r2, rmse, preds


def calibrate_conformal_qsar(
    smiles_list: Sequence[str],
    targets: Sequence[float],
    epochs: int = 28,
    seed: int = 42,
    alpha_miscoverage: float = 0.10,
) -> Tuple[MolecularGraphNet, ConformalPredictionResult]:
    """Train GAT-MPNN on train split and compute split-conformal 90% prediction intervals."""
    n = len(smiles_list)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).tolist()

    n_train = max(8, int(0.60 * n))
    n_cal = max(3, int(0.20 * n))
    train_ids = perm[:n_train]
    cal_ids = perm[n_train : n_train + n_cal]
    test_ids = perm[n_train + n_cal :]
    if not test_ids:
        test_ids = cal_ids

    gnn = MolecularGraphNet(GNNConfig(hidden_dim=28, num_layers=2, learning_rate=0.01, seed=seed))
    gnn.fit([smiles_list[i] for i in train_ids], [targets[i] for i in train_ids], epochs=epochs, batch_size=6)

    cal_scores: List[float] = []
    for idx in cal_ids:
        mu, sigma, _ = gnn.predict_smiles(smiles_list[idx], mc_samples=6)
        cal_scores.append(abs(targets[idx] - mu) / max(0.05, sigma))

    cal_arr = np.sort(np.asarray(cal_scores, dtype=np.float64))
    q_level = min(1.0, math.ceil((1.0 - alpha_miscoverage) * (len(cal_arr) + 1)) / max(1, len(cal_arr)))
    q_idx = min(len(cal_arr) - 1, max(0, int(math.ceil(q_level * len(cal_arr))) - 1))
    q_hat = float(cal_arr[q_idx])

    preds: List[float] = []
    truths: List[float] = []
    covered = 0
    widths: List[float] = []
    for idx in test_ids:
        mu, sigma, _ = gnn.predict_smiles(smiles_list[idx], mc_samples=6)
        half_w = q_hat * max(0.05, sigma)
        preds.append(mu)
        truths.append(targets[idx])
        widths.append(2.0 * half_w)
        if abs(targets[idx] - mu) <= half_w:
            covered += 1

    p_arr = np.asarray(preds, dtype=np.float64)
    t_arr = np.asarray(truths, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((p_arr - t_arr) ** 2)))
    mae = float(np.mean(np.abs(p_arr - t_arr)))
    ss_tot = float(np.sum((t_arr - np.mean(t_arr)) ** 2))
    r2 = 1.0 - float(np.sum((p_arr - t_arr) ** 2)) / max(1e-12, ss_tot)
    rho_val = float(spearmanr(t_arr, p_arr).statistic) if len(t_arr) >= 3 else 0.0
    if math.isnan(rho_val):
        rho_val = 0.0

    k_top = max(1, int(math.ceil(0.2 * len(t_arr))))
    active_thresh = float(np.median(t_arr))
    true_active_rate = float(np.mean(t_arr >= active_thresh))
    top_pred_ids = np.argsort(p_arr)[::-1][:k_top]
    top_active_rate = float(np.mean(t_arr[top_pred_ids] >= active_thresh))
    ef = top_active_rate / max(1e-6, true_active_rate)

    res = ConformalPredictionResult(
        r2=round(r2, 4),
        rmse=round(rmse, 4),
        mae=round(mae, 4),
        spearman_rho=round(rho_val, 4),
        enrichment_5pct=round(ef, 4),
        conformal_quantile_90=round(q_hat, 4),
        empirical_coverage_90=round(covered / max(1, len(test_ids)), 4),
        mean_interval_width=round(float(np.mean(widths)), 4),
    )
    return gnn, res


def compute_sobol_sensitivity(
    fn: Callable[[NDArray[np.float64]], float],
    bounds: Sequence[Tuple[float, float]],
    param_names: Sequence[str],
    n_samples: int = 96,
    seed: int = 42,
) -> SobolResult:
    """Compute Saltelli-Jansen Sobol first-order (S1) and total-order (ST) sensitivity indices."""
    d = len(bounds)
    rng = np.random.default_rng(seed)

    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    span = hi - lo

    mat_a = lo + rng.random((n_samples, d)) * span
    mat_b = lo + rng.random((n_samples, d)) * span

    y_a = np.array([fn(mat_a[i]) for i in range(n_samples)], dtype=np.float64)
    y_b = np.array([fn(mat_b[i]) for i in range(n_samples)], dtype=np.float64)

    y_all = np.concatenate([y_a, y_b])
    var_y = float(np.var(y_all))
    if var_y <= 1e-14:
        var_y = 1e-14

    s1_map: Dict[str, float] = {}
    st_map: Dict[str, float] = {}

    for j in range(d):
        mat_ab_j = mat_a.copy()
        mat_ab_j[:, j] = mat_b[:, j]
        y_ab_j = np.array([fn(mat_ab_j[i]) for i in range(n_samples)], dtype=np.float64)

        s1_raw = float(np.mean(y_b * (y_ab_j - y_a))) / var_y
        st_raw = 0.5 * float(np.mean((y_a - y_ab_j) ** 2)) / var_y

        s1_map[param_names[j]] = round(max(0.0, min(1.0, s1_raw)), 4)
        st_map[param_names[j]] = round(max(0.0, min(1.0, st_raw)), 4)

    return SobolResult(
        param_names=tuple(param_names),
        first_order=s1_map,
        total_order=st_map,
        output_mean=round(float(np.mean(y_all)), 4),
        output_std=round(float(np.std(y_all)), 4),
    )


def protac_sobol_audit(n_samples: int = 80, seed: int = 42) -> SobolResult:
    """Run Sobol global sensitivity over PROTAC ternary complex formation [PLE] at 50 nM dose."""
    names = ("kd_poi_nm", "kd_e3_nm", "alpha", "e3_conc_nm")
    bounds = ((2.0, 60.0), (5.0, 120.0), (0.5, 18.0), (15.0, 120.0))

    def eval_fn(x: NDArray[np.float64]) -> float:
        _, _, _, _, _, ple = solve_ternary_equilibrium(
            p_tot_nm=100.0,
            e_tot_nm=float(x[3]),
            l_tot_nm=50.0,
            kd_p_nm=float(x[0]),
            kd_e_nm=float(x[1]),
            alpha=float(x[2]),
        )
        return ple

    return compute_sobol_sensitivity(eval_fn, bounds, names, n_samples=n_samples, seed=seed)
