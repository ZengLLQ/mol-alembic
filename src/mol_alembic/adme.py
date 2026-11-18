"""Multitask ADME predictors, well-stirred hepatic clearance, and 2-compartment PBPK ODE simulator."""

from __future__ import annotations

import math
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .chemistry import analyze_molecule
from .smarts import screen_structural_alerts
from .smiles import parse_smiles
from .types import ADMEProfile, Molecule, PKSimulationResult


def predict_adme(mol: Molecule) -> ADMEProfile:
    """Predict ADME physicochemical and pharmacokinetic parameters from molecular structure."""
    alerts = screen_structural_alerts(mol)
    _rings, desc = analyze_molecule(mol, alert_count=len(alerts))

    tm_est_c = 55.0 + 18.0 * desc.aromatic_ring_count + 0.15 * desc.molecular_weight - 4.0 * desc.rotatable_bonds
    tm_est_c = max(25.0, min(280.0, tm_est_c))
    log_s = 0.5 - desc.crippen_logp - 0.01 * (tm_est_c - 25.0)
    log_s = max(-9.5, min(1.5, log_s))

    log_papp = (
        2.45
        - 0.014 * desc.tpsa
        - 0.18 * desc.hbd_count
        + 0.16 * min(4.0, desc.crippen_logp)
        - 0.0018 * max(0.0, desc.molecular_weight - 320.0)
    )
    papp = float(10.0 ** max(-1.5, min(3.2, log_papp)))
    fa = 1.0 / (1.0 + math.exp(-(log_papp - 1.15) * 1.9))
    sol_penalty = 1.0 / (1.0 + math.exp(-(log_s + 5.2) * 1.4))
    fa = max(0.03, min(0.99, fa * sol_penalty))

    log_bb = 0.152 * desc.crippen_logp - 0.0148 * desc.tpsa + 0.139
    log_bb = max(-2.5, min(1.5, log_bb))

    pgp_logit = (
        -2.2
        + 0.0075 * max(0.0, desc.molecular_weight - 300.0)
        + 0.25 * max(0, desc.hba_count - 4)
        + 0.015 * max(0.0, desc.tpsa - 60.0)
    )
    pgp_risk = 1.0 / (1.0 + math.exp(-pgp_logit))

    has_acid = any(
        a.symbol == "C"
        and sum(1 for n, _ in mol.neighbors(a.index) if mol.atoms[n].symbol == "O") >= 2
        for a in mol.atoms
    )
    log_bound_ratio = 0.58 * desc.crippen_logp - 0.65 + (0.65 if has_acid else 0.0)
    fu = 1.0 / (1.0 + 10.0 ** max(-2.0, min(2.6, log_bound_ratio)))
    fu = max(0.002, min(0.98, fu))

    has_halogen = any(a.symbol in ("F", "Cl", "Br") for a in mol.atoms)
    halogen_shield = 0.68 if has_halogen else 1.0
    cl_int = (
        12.0 * math.exp(0.42 * max(-1.0, desc.crippen_logp)) + 0.45 * desc.rotatable_bonds
    ) * halogen_shield
    cl_int = max(1.5, min(450.0, cl_int))

    qh = 20.7
    fu_cl = fu * cl_int
    cl_hepatic = qh * fu_cl / (qh + fu_cl)

    has_basic_n = any(
        a.symbol == "N" and not a.is_aromatic and a.formal_charge >= 0
        for a in mol.atoms
    )
    vd = 0.45 + 0.42 * max(0.0, desc.crippen_logp) + (2.4 if has_basic_n else 0.0) - (0.25 if has_acid else 0.0)
    vd = max(0.15, min(18.0, vd))

    cl_l_h_kg = max(1e-4, cl_hepatic * 0.06)
    t_half = 0.693 * vd / cl_l_h_kg

    cyp3a4_logit = -2.0 + 0.48 * desc.crippen_logp + 0.004 * max(0.0, desc.molecular_weight - 300.0)
    cyp2d6_logit = -2.4 + 0.35 * desc.crippen_logp + (1.35 if has_basic_n else 0.0)
    cyp3a4 = 1.0 / (1.0 + math.exp(-cyp3a4_logit))
    cyp2d6 = 1.0 / (1.0 + math.exp(-cyp2d6_logit))

    return ADMEProfile(
        smiles=mol.smiles,
        log_s_mol_l=round(log_s, 3),
        papp_caco2_nm_s=round(papp, 2),
        fraction_absorbed=round(fa, 4),
        log_bb=round(log_bb, 3),
        pgp_efflux_risk=round(pgp_risk, 4),
        fu_plasma=round(fu, 4),
        cl_int_ml_min_kg=round(cl_int, 2),
        cl_hepatic_ml_min_kg=round(cl_hepatic, 3),
        vd_l_kg=round(vd, 3),
        cyp3a4_hazard=round(cyp3a4, 4),
        cyp2d6_hazard=round(cyp2d6, 4),
        half_life_h=round(t_half, 2),
    )


def simulate_pbpk_2comp(
    mol: Molecule,
    adme: Optional[ADMEProfile] = None,
    dose_mg: float = 100.0,
    body_weight_kg: float = 70.0,
    route: str = "oral",
    duration_h: float = 24.0,
    n_points: int = 121,
    target_ic50_nm: float = 50.0,
) -> PKSimulationResult:
    """Simulate a 2-compartment oral or IV PBPK ODE system using adaptive RK45 integration."""
    profile = adme or predict_adme(mol)
    _, desc = analyze_molecule(mol)
    mw = max(50.0, desc.molecular_weight)

    v1 = max(2.0, 0.45 * profile.vd_l_kg * body_weight_kg)
    v2 = max(2.0, 0.55 * profile.vd_l_kg * body_weight_kg)
    cl_l_h = max(0.05, profile.cl_hepatic_ml_min_kg * 0.06 * body_weight_kg)
    q_dist_l_h = 0.35 * cl_l_h + 1.5

    k10 = cl_l_h / v1
    k12 = q_dist_l_h / v1
    k21 = q_dist_l_h / v2
    ka = 0.6 + 0.008 * min(200.0, profile.papp_caco2_nm_s)

    if route.lower() == "iv":
        y0 = np.array([0.0, dose_mg, 0.0, 0.0], dtype=np.float64)
        unabsorbed_const = 0.0
    else:
        bioavail_gut = dose_mg * profile.fraction_absorbed
        unabsorbed_const = dose_mg - bioavail_gut
        y0 = np.array([bioavail_gut, 0.0, 0.0, 0.0], dtype=np.float64)

    def rhs(_t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        a_g, a_c, a_p, _a_e = y
        d_ag = -ka * a_g
        d_ac = ka * a_g - (k10 + k12) * a_c + k21 * a_p
        d_ap = k12 * a_c - k21 * a_p
        d_ae = k10 * a_c
        return np.array([d_ag, d_ac, d_ap, d_ae], dtype=np.float64)

    t_eval = np.linspace(0.0, duration_h, n_points)
    sol = solve_ivp(rhs, (0.0, duration_h), y0, method="RK45", t_eval=t_eval, rtol=1e-8, atol=1e-10)

    a_gut = sol.y[0]
    a_cent = sol.y[1]
    a_periph = sol.y[2]
    a_elim = sol.y[3]

    initial_mass = float(np.sum(y0))
    mass_err = float(np.max(np.abs((a_gut + a_cent + a_periph + a_elim) - initial_mass)))

    c_plasma = a_cent / v1
    c_periph = a_periph / v2
    c_unbound_nm = (c_plasma * 1e6 / mw) * profile.fu_plasma

    cmax_idx = int(np.argmax(c_plasma))
    cmax = float(c_plasma[cmax_idx])
    tmax = float(t_eval[cmax_idx])

    auc_grid = float(np.trapezoid(c_plasma, t_eval))
    disc = (k10 + k12 + k21) ** 2 - 4.0 * k10 * k21
    beta = 0.5 * ((k10 + k12 + k21) - math.sqrt(max(0.0, disc)))
    beta = max(1e-4, beta)
    auc_inf = auc_grid + float(c_plasma[-1]) / beta
    t_half_term = 0.693147 / beta

    above_mask = c_unbound_nm >= target_ic50_nm
    dt = duration_h / max(1, n_points - 1)
    time_above = float(np.sum(above_mask)) * dt

    return PKSimulationResult(
        times_h=t_eval,
        gut_amount_mg=a_gut + unabsorbed_const,
        plasma_conc_mg_l=c_plasma,
        peripheral_conc_mg_l=c_periph,
        unbound_plasma_nm=c_unbound_nm,
        cmax_mg_l=round(cmax, 4),
        tmax_h=round(tmax, 3),
        auc_0_inf_mg_h_l=round(auc_inf, 4),
        half_life_terminal_h=round(t_half_term, 3),
        time_above_ic50_h=round(time_above, 3),
        mass_balance_error=mass_err,
    )


def score_adme_desirability(adme: ADMEProfile) -> float:
    """Aggregate normalized [0, 1] ADME/PK desirability score for multi-objective optimization."""
    s_sol = 1.0 / (1.0 + math.exp(-(adme.log_s_mol_l + 4.5) * 1.4))
    s_perm = adme.fraction_absorbed
    s_cl = math.exp(-adme.cl_hepatic_ml_min_kg / 20.0)
    s_pgp = 1.0 - 0.65 * adme.pgp_efflux_risk
    s_cyp = 1.0 - 0.5 * max(adme.cyp3a4_hazard, adme.cyp2d6_hazard)
    score = 0.22 * s_sol + 0.25 * s_perm + 0.23 * s_cl + 0.15 * s_pgp + 0.15 * s_cyp
    return round(max(0.0, min(1.0, score)), 4)


def predict_adme_from_smiles(smiles: str) -> ADMEProfile:
    """Convenience wrapper parsing SMILES and predicting ADME profile."""
    return predict_adme(parse_smiles(smiles))
