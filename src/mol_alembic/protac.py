"""Douglass-Miller PROTAC ternary complex equilibrium, cooperativity, Hook effect, and degradation ODEs."""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .types import ProtacSweepResult


def estimate_linker_cooperativity(
    linker_length_atoms: int,
    optimal_length: int = 8,
    rigid_ring_fraction: float = 0.25,
    base_alpha: float = 6.0,
) -> float:
    """Estimate ternary PPI cooperativity alpha from linker length and conformational rigidity."""
    mismatch = float(linker_length_atoms - optimal_length)
    if linker_length_atoms < max(2, optimal_length - 4):
        steric = math.exp(-0.9 * (optimal_length - 4 - linker_length_atoms))
        return round(max(0.05, 0.4 * steric), 4)

    gaussian_fit = math.exp(-0.08 * (mismatch ** 2))
    rigidity_bonus = 1.0 + 0.85 * max(0.0, min(1.0, rigid_ring_fraction))
    alpha = max(0.1, base_alpha * gaussian_fit * rigidity_bonus)
    return round(alpha, 4)


def solve_ternary_equilibrium(
    p_tot_nm: float,
    e_tot_nm: float,
    l_tot_nm: float,
    kd_p_nm: float,
    kd_e_nm: float,
    alpha: float = 1.0,
) -> Tuple[float, float, float, float, float, float]:
    """Solve exact Douglass-Miller three-body mass-action equilibrium in nM."""
    if p_tot_nm <= 0.0 or e_tot_nm <= 0.0 or l_tot_nm <= 0.0:
        return p_tot_nm, e_tot_nm, l_tot_nm, 0.0, 0.0, 0.0

    kp = max(1e-6, kd_p_nm)
    ke = max(1e-6, kd_e_nm)
    al = max(1e-6, alpha)

    def _free_pe_given_l(l_free: float) -> Tuple[float, float, float]:
        a_p = 1.0 + l_free / kp
        a_e = 1.0 + l_free / ke
        c = al * l_free / (kp * ke)

        if c < 1e-18:
            p_f = p_tot_nm / a_p
            e_f = e_tot_nm / a_e
        else:
            qa = c * a_p
            qb = a_p * a_e + c * (e_tot_nm - p_tot_nm)
            qc = -p_tot_nm * a_e
            disc = max(0.0, qb * qb - 4.0 * qa * qc)
            p_f = (-qb + math.sqrt(disc)) / (2.0 * qa)
            e_f = e_tot_nm / (a_e + c * p_f)

        pl = p_f * l_free / kp
        le = l_free * e_f / ke
        ple = c * p_f * e_f
        l_calc = l_free + pl + le + ple
        return p_f, e_f, l_calc - l_tot_nm

    l_opt = float(
        brentq(lambda x: _free_pe_given_l(x)[2], 0.0, l_tot_nm, xtol=1e-13, rtol=1e-13, maxiter=120)
    )
    p_free, e_free, _ = _free_pe_given_l(l_opt)
    pl = p_free * l_opt / kp
    le = l_opt * e_free / ke
    ple = al * p_free * l_opt * e_free / (kp * ke)
    return p_free, e_free, l_opt, pl, le, ple


def simulate_protac_degradation_kinetics(
    ple_nm: float,
    p_tot_nm: float = 100.0,
    k_syn_nm_h: float = 5.0,
    k_deg_base_h: float = 0.05,
    k_cat_ub_h: float = 0.85,
    duration_h: float = 24.0,
) -> float:
    """Integrate catalytic UPS target degradation ODE over duration_h and return fractional D."""
    frac_ternary = max(0.0, min(1.0, ple_nm / max(1e-9, p_tot_nm)))

    def rhs(_t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        p_cur = max(0.0, float(y[0]))
        dp = k_syn_nm_h - (k_deg_base_h + k_cat_ub_h * frac_ternary) * p_cur
        return np.array([dp], dtype=np.float64)

    sol = solve_ivp(rhs, (0.0, duration_h), np.array([p_tot_nm], dtype=np.float64), rtol=1e-7, atol=1e-9)
    p_end = float(sol.y[0, -1])
    deg_frac = max(0.0, min(1.0, 1.0 - p_end / max(1e-9, p_tot_nm)))
    return deg_frac


def sweep_protac_hook_effect(
    kd_poi_nm: float = 12.0,
    kd_e3_nm: float = 35.0,
    alpha: float = 8.5,
    poi_conc_nm: float = 100.0,
    e3_conc_nm: float = 50.0,
    min_log_nm: float = -2.0,
    max_log_nm: float = 5.0,
    n_points: int = 50,
    duration_h: float = 24.0,
) -> ProtacSweepResult:
    """Sweep PROTAC concentration across decades to resolve the ternary Hook Effect and DC50/Dmax."""
    conc = np.logspace(min_log_nm, max_log_nm, n_points)
    ple_arr = np.zeros(n_points, dtype=np.float64)
    pl_arr = np.zeros(n_points, dtype=np.float64)
    le_arr = np.zeros(n_points, dtype=np.float64)
    frac_poi = np.zeros(n_points, dtype=np.float64)
    deg_arr = np.zeros(n_points, dtype=np.float64)

    max_err = 0.0
    for idx, l_tot in enumerate(conc):
        pf, ef, lf, pl, le, ple = solve_ternary_equilibrium(
            poi_conc_nm, e3_conc_nm, float(l_tot), kd_poi_nm, kd_e3_nm, alpha=alpha
        )
        err_p = abs((pf + pl + ple) - poi_conc_nm)
        err_e = abs((ef + le + ple) - e3_conc_nm)
        err_l = abs((lf + pl + le + ple) - float(l_tot))
        max_err = max(max_err, err_p, err_e, err_l)

        ple_arr[idx] = ple
        pl_arr[idx] = pl
        le_arr[idx] = le
        frac_poi[idx] = ple / max(1e-9, poi_conc_nm)
        deg_arr[idx] = simulate_protac_degradation_kinetics(ple, p_tot_nm=poi_conc_nm, duration_h=duration_h)

    peak_idx = int(np.argmax(ple_arr))
    peak_ple = float(ple_arr[peak_idx])
    hook_peak = float(conc[peak_idx])
    dmax = float(np.max(deg_arr))

    dc50: Optional[float] = None
    for idx in range(peak_idx + 1):
        if deg_arr[idx] >= 0.50:
            if idx == 0:
                dc50 = float(conc[0])
            else:
                d0, d1 = float(deg_arr[idx - 1]), float(deg_arr[idx])
                c0, c1 = math.log10(float(conc[idx - 1])), math.log10(float(conc[idx]))
                t = (0.50 - d0) / max(1e-9, d1 - d0)
                dc50 = round(float(10.0 ** (c0 + t * (c1 - c0))), 4)
            break

    return ProtacSweepResult(
        conc_nm=conc,
        ternary_ple_nm=ple_arr,
        binary_pl_nm=pl_arr,
        binary_le_nm=le_arr,
        fraction_bound_poi=frac_poi,
        degradation_fraction=deg_arr,
        peak_ple_nm=round(peak_ple, 4),
        hook_peak_conc_nm=round(hook_peak, 4),
        dc50_nm=dc50,
        dmax=round(dmax, 4),
        cooperativity_alpha=alpha,
        max_mass_balance_error=max_err,
    )
