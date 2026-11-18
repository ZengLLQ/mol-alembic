"""Strict 7-bit ASCII terminal visualizers for molecular cards, QSAR saliency, PBPK, PROTAC, and Pareto."""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np
from numpy.typing import NDArray

from .types import (
    ADMEProfile,
    MolecularDescriptors,
    Molecule,
    OptimizationCandidate,
    PKSimulationResult,
    ProtacSweepResult,
)


def assert_ascii(text: str) -> str:
    """Verify every character in text lies strictly in 7-bit ASCII (0..127)."""
    for idx, ch in enumerate(text):
        if ord(ch) > 127:
            raise ValueError(f"Non-ASCII character ord={ord(ch)} at index {idx}")
    return text


def render_molecule_card(
    mol: Molecule,
    desc: MolecularDescriptors,
    adme: ADMEProfile,
    alert_count: int,
) -> str:
    """Render an ASCII physicochemical and ADME telemetry card for a ligand."""
    lines = [
        "+" + "-" * 80 + "+",
        f"| MolAlembic Card: {mol.name:<22} Formula: {desc.formula:<28} |",
        f"| SMILES: {mol.smiles[:70]:<70} |",
        "+" + "-" * 80 + "+",
        (
            f"| MW: {desc.molecular_weight:>7.2f} | LogP: {desc.crippen_logp:>5.2f} | "
            f"TPSA: {desc.tpsa:>6.2f} A^2 | QED: {desc.qed:>5.3f} | SA: {desc.sa_score:>4.2f} |"
        ),
        (
            f"| HBD: {desc.hbd_count:>2d} | HBA: {desc.hba_count:>2d} | RotB: {desc.rotatable_bonds:>2d} | "
            f"Rings: {desc.ring_count:>2d} (Arom:{desc.aromatic_ring_count:>2d}) | Wiener: {desc.wiener_index:<10d} |"
        ),
        (
            f"| Ro5 Viol: {desc.lipinski_violations} | Veber Viol: {desc.veber_violations} | "
            f"PAINS Alerts: {alert_count:<2d} | Balaban J: {desc.balaban_j:>7.3f}      |"
        ),
        "+" + "-" * 80 + "+",
        (
            f"| ADME logS: {adme.log_s_mol_l:>6.2f} | Papp: {adme.papp_caco2_nm_s:>6.1f} nm/s | "
            f"Fa: {adme.fraction_absorbed:>5.3f} | logBB: {adme.log_bb:>6.2f}         |"
        ),
        (
            f"| fu,p: {adme.fu_plasma:>5.3f} | CLh: {adme.cl_hepatic_ml_min_kg:>6.2f} mL/min/kg | "
            f"Vd: {adme.vd_l_kg:>5.2f} L/kg | t1/2: {adme.half_life_h:>6.2f} h     |"
        ),
        "+" + "-" * 80 + "+",
    ]
    return assert_ascii("\n".join(lines))


def render_gat_saliency_bars(mol: Molecule, saliency: NDArray[np.float64], top_k: int = 10) -> str:
    """Render horizontal 7-bit ASCII bar chart of GAT atom attention attributions."""
    if all(abs(a.partial_charge) < 1e-12 for a in mol.atoms):
        from .chemistry import compute_gasteiger_charges

        compute_gasteiger_charges(mol)
    order = np.argsort(saliency)[::-1][:top_k]
    max_val = max(1e-9, float(np.max(saliency)))
    lines = ["GAT Atom-Attention Attribution (Top Atoms):"]
    for idx in order:
        i = int(idx)
        atom = mol.atoms[i]
        w = float(saliency[i])
        bar_len = int(round((w / max_val) * 28))
        bar = "#" * bar_len + "." * (28 - bar_len)
        arom = "*" if atom.is_aromatic else " "
        lines.append(f"  atom[{i:>2d}] {atom.symbol:>2s}{arom} (q={atom.partial_charge:+.3f}) |{bar}| {w:.4f}")
    return assert_ascii("\n".join(lines))


def render_similarity_heatmap(names: Sequence[str], sim_mat: NDArray[np.float64], max_items: int = 10) -> str:
    """Render a compact 7-bit ASCII Tanimoto similarity matrix using density glyphs."""
    m = min(max_items, len(names))
    ramp = " .:-=+*#%@"
    lines = ["Tanimoto Similarity Heatmap (.:<0.2, -:<0.4, +:<0.6, *:<0.8, @>=0.9):"]
    header = "                " + " ".join(f"{i:>2d}" for i in range(m))
    lines.append(header)
    for i in range(m):
        cells = []
        for j in range(m):
            v = max(0.0, min(1.0, float(sim_mat[i, j])))
            glyph = ramp[min(len(ramp) - 1, int(v * (len(ramp) - 1)))]
            cells.append(f" {glyph}")
        label = names[i][:8]
        lines.append(f"  [{i:>2d}] {label:<8s} " + " ".join(cells))
    return assert_ascii("\n".join(lines))


def render_pbpk_curve(pk: PKSimulationResult, width: int = 58, height: int = 12) -> str:
    """Render 7-bit ASCII plasma concentration-time profile Cp(t) and peripheral curve."""
    t = pk.times_h
    c_p = pk.plasma_conc_mg_l
    c_t = pk.peripheral_conc_mg_l
    max_c = max(1e-9, float(np.max(c_p)), float(np.max(c_t)))

    grid = [[" " for _ in range(width)] for _ in range(height)]
    for col in range(width):
        idx = int(round(col * (len(t) - 1) / max(1, width - 1)))
        row_p = height - 1 - int(round((float(c_p[idx]) / max_c) * (height - 1)))
        row_t = height - 1 - int(round((float(c_t[idx]) / max_c) * (height - 1)))
        row_p = max(0, min(height - 1, row_p))
        row_t = max(0, min(height - 1, row_t))
        grid[row_t][col] = ":"
        grid[row_p][col] = "*" if row_p != row_t else "#"

    hdr = (
        f"PBPK 2-Comp Profile (*=Plasma Cp, :=Periph Ct, #=Overlap) "
        f"Cmax={pk.cmax_mg_l:.3f} mg/L Tmax={pk.tmax_h:.2f} h"
    )
    lines = [hdr]
    for r_idx, row in enumerate(grid):
        val = max_c * (height - 1 - r_idx) / max(1, height - 1)
        lines.append(f"  {val:>6.3f} |" + "".join(row) + "|")
    lines.append("         +" + "-" * width + "+")
    lines.append(f"         0.0h{' ' * (width - 10)}{t[-1]:.1f}h")
    return assert_ascii("\n".join(lines))


def render_protac_hook_curve(sweep: ProtacSweepResult, width: int = 58, height: int = 12) -> str:
    """Render 7-bit ASCII PROTAC ternary complex [PLE] bell-shaped Hook Effect curve."""
    ple = sweep.ternary_ple_nm
    deg = sweep.degradation_fraction * max(1e-6, sweep.peak_ple_nm)
    max_y = max(1e-6, float(np.max(ple)))

    grid = [[" " for _ in range(width)] for _ in range(height)]
    for col in range(width):
        idx = int(round(col * (len(ple) - 1) / max(1, width - 1)))
        r_ple = height - 1 - int(round((float(ple[idx]) / max_y) * (height - 1)))
        r_deg = height - 1 - int(round((float(deg[idx]) / max_y) * (height - 1)))
        r_ple = max(0, min(height - 1, r_ple))
        r_deg = max(0, min(height - 1, r_deg))
        grid[r_deg][col] = "."
        grid[r_ple][col] = "*"

    dc50_txt = f"{sweep.dc50_nm:.2f} nM" if sweep.dc50_nm is not None else "N/A"
    hdr = (
        f"PROTAC Hook Effect (*=[PLE] nM, .=Degrad) Peak={sweep.peak_ple_nm:.2f} nM "
        f"@ {sweep.hook_peak_conc_nm:.1f} nM | DC50={dc50_txt}"
    )
    lines = [hdr]
    for r_idx, row in enumerate(grid):
        val = max_y * (height - 1 - r_idx) / max(1, height - 1)
        lines.append(f"  {val:>6.2f} |" + "".join(row) + "|")
    lines.append("         +" + "-" * width + "+")
    lo_exp = math.log10(float(sweep.conc_nm[0]))
    hi_exp = math.log10(float(sweep.conc_nm[-1]))
    lines.append(f"         10^{lo_exp:+.1f} nM{' ' * (width - 18)}10^{hi_exp:+.1f} nM")
    return assert_ascii("\n".join(lines))


def render_pareto_scatter(
    candidates: Sequence[OptimizationCandidate],
    width: int = 54,
    height: int = 12,
) -> str:
    """Render 7-bit ASCII 2D Pareto frontier plot of QED (x) vs Predicted pIC50 (y)."""
    if not candidates:
        return "Pareto Frontier: (no candidates)"

    qeds = [c.qed for c in candidates]
    pics = [c.predicted_pic50 for c in candidates]
    x_min, x_max = min(qeds), max(qeds)
    y_min, y_max = min(pics), max(pics)
    x_span = max(0.05, x_max - x_min)
    y_span = max(0.2, y_max - y_min)

    grid = [[" " for _ in range(width)] for _ in range(height)]
    for c in sorted(candidates, key=lambda cand: -cand.pareto_rank):
        col = int(round(((c.qed - x_min) / x_span) * (width - 1)))
        row = height - 1 - int(round(((c.predicted_pic50 - y_min) / y_span) * (height - 1)))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        grid[row][col] = "*" if c.pareto_rank == 0 else "o"

    lines = ["Multi-Objective Pareto Frontier (*=Rank-0 Pareto Front, o=Archive):"]
    for r_idx, grid_row in enumerate(grid):
        y_val = y_max - (y_span * r_idx / max(1, height - 1))
        lines.append(f"  {y_val:>5.2f} |" + "".join(grid_row) + "|")
    lines.append("        +" + "-" * width + "+")
    lines.append(f"        QED={x_min:.2f}{' ' * (width - 16)}QED={x_max:.2f}")
    return assert_ascii("\n".join(lines))
