"""Example 01: SMILES graph perception, SSSR rings, Kekulization, Gasteiger charges, Murcko scaffolds, and MMPs."""

import numpy as np

from mol_alembic.chemistry import analyze_molecule, kekulize_molecule
from mol_alembic.fingerprints import butina_clustering, morgan_bit_vector, pairwise_tanimoto_matrix
from mol_alembic.scaffolds import extract_bemis_murcko_scaffold, find_matched_molecular_pairs
from mol_alembic.smarts import annotate_pharmacophores, screen_structural_alerts
from mol_alembic.smiles import BENCHMARK_MOLECULES, parse_smiles, validate_smiles_roundtrip
from mol_alembic.visualizer import render_similarity_heatmap


def main() -> None:
    print("=== Example 01: SMILES Perception, Scaffolds & Structural Alerts ===")
    for name, smi, _ in BENCHMARK_MOLECULES[:6]:
        ok, _canon = validate_smiles_roundtrip(smi)
        mol = parse_smiles(smi, name=name)
        rings, desc = analyze_molecule(mol)
        kek = kekulize_molecule(mol)
        scaf, _ = extract_bemis_murcko_scaffold(mol)
        pharm = annotate_pharmacophores(mol)
        alerts = screen_structural_alerts(mol)
        total_q = sum(a.partial_charge for a in mol.atoms)
        print(
            f"{name:<14s} | roundtrip={ok} | {desc.formula:<10s} MW={desc.molecular_weight:>6.2f} "
            f"LogP={desc.crippen_logp:>5.2f} TPSA={desc.tpsa:>5.1f} QED={desc.qed:.3f} "
            f"Rings={len(rings)} Pharm={len(pharm)} Alerts={len(alerts)} sum(q)={total_q:+.2e} "
            f"KekBonds={kek.num_bonds} | Murcko={scaf or 'acyclic'}"
        )

    series = [
        ("Quinazoline_H", "c1ccc(Nc2ncnc3ccccc23)cc1", 6.45),
        ("Quinazoline_F", "Fc1ccc(Nc2ncnc3ccccc23)cc1", 7.80),
        ("Quinazoline_Cl", "Clc1ccc(Nc2ncnc3ccccc23)cc1", 8.35),
        ("Quinazoline_OH", "Oc1ccc(Nc2ncnc3ccccc23)cc1", 6.10),
    ]
    mmps, cliffs = find_matched_molecular_pairs(series, min_core_atoms=6, sali_threshold=3.0)
    print(f"\nDetected {len(mmps)} MMPs and {len(cliffs)} SALI activity cliffs:")
    for mmp in mmps[:4]:
        print(
            f"  {mmp.mol_a_name} <-> {mmp.mol_b_name}: core={mmp.core_smiles} | "
            f"dAct={mmp.delta_activity:.2f} | Tanimoto={mmp.tanimoto:.3f} | SALI={mmp.sali_score:.2f}"
        )

    fps = np.vstack([morgan_bit_vector(parse_smiles(s), radius=2, nbits=1024) for _, s, _ in BENCHMARK_MOLECULES[:10]])
    sim = pairwise_tanimoto_matrix(fps)
    clusters = butina_clustering(fps, similarity_threshold=0.28)
    print(f"\nButina Clusters (cutoff=0.28): {clusters}")
    print(render_similarity_heatmap([n for n, _, _ in BENCHMARK_MOLECULES[:10]], sim, max_items=10))


if __name__ == "__main__":
    main()
