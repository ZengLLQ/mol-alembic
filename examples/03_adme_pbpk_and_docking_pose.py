"""Example 03: ADME prediction, 2-compartment PBPK ODE simulation, and 3D Vina/GBSA ligand docking."""

from mol_alembic.adme import predict_adme, simulate_pbpk_2comp
from mol_alembic.chemistry import analyze_molecule
from mol_alembic.docking import dock_ligand
from mol_alembic.smarts import screen_structural_alerts
from mol_alembic.smiles import parse_smiles
from mol_alembic.visualizer import render_molecule_card, render_pbpk_curve


def main() -> None:
    print("=== Example 03: ADME/PBPK ODE Simulation & 3D Docking ===")
    smiles = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCC"
    mol = parse_smiles(smiles, name="Gefitinib_Motif")
    alerts = screen_structural_alerts(mol)
    _, desc = analyze_molecule(mol, alert_count=len(alerts))
    adme = predict_adme(mol)
    pk = simulate_pbpk_2comp(mol, adme=adme, dose_mg=250.0, duration_h=24.0, target_ic50_nm=40.0)

    print(render_molecule_card(mol, desc, adme, alert_count=len(alerts)))
    print()
    print(render_pbpk_curve(pk))
    print(
        f"PK Summary: AUC_0-inf={pk.auc_0_inf_mg_h_l:.3f} mg*h/L | t1/2_term={pk.half_life_terminal_h:.2f} h | "
        f"Time>IC50={pk.time_above_ic50_h:.2f} h | MassBalanceErr={pk.mass_balance_error:.2e} mg"
    )

    pose = dock_ligand(mol, seed=11, n_restarts=5)
    print(
        f"\n3D Docking Score: {pose.total_score_kcal_mol:.3f} kcal/mol (LE={pose.ligand_efficiency:.3f}) | "
        f"Contacts={pose.pocket_contacts} | HBond={pose.hbond_term:.3f} | "
        f"Hydrophobic={pose.hydrophobic_term:.3f} | GBSA={pose.gbsa_desolvation:.3f}"
    )


if __name__ == "__main__":
    main()
