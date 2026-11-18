"""Typed chemical data structures, periodic constants, and domain exceptions for MolAlembic."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray


class SmilesParseError(ValueError):
    """Raised when a SMILES string violates lexical, ring-closure, or bracket syntax."""


class ValenceError(ValueError):
    """Raised when an atom exceeds chemically permissible covalent valence."""


class KekulizationError(ValueError):
    """Raised when an aromatic ring system cannot be assigned a valid Kekule bond order pattern."""


class SmartsSyntaxError(ValueError):
    """Raised when a SMARTS substructure query contains invalid atom or bond primitives."""


class GraphShapeError(ValueError):
    """Raised when molecular graph tensors or batch matrices have incompatible shapes."""


class BondOrder(float, Enum):
    """Supported covalent bond orders in molecular graphs."""

    SINGLE = 1.0
    DOUBLE = 2.0
    TRIPLE = 3.0
    AROMATIC = 1.5


class Hybridization(str, Enum):
    """Atomic orbital hybridization states."""

    SP = "SP"
    SP2 = "SP2"
    SP3 = "SP3"
    S = "S"
    OTHER = "OTHER"


class PharmacophoreType(str, Enum):
    """Canonical 3D/topological pharmacophore feature classes."""

    HBD = "HBD"
    HBA = "HBA"
    HYDROPHOBIC = "HYDROPHOBIC"
    AROMATIC = "AROMATIC"
    POS_IONIZABLE = "POS_IONIZABLE"
    NEG_IONIZABLE = "NEG_IONIZABLE"


@dataclass(frozen=True)
class ElementInfo:
    """Periodic table physical and electronic parameters for a chemical element."""

    symbol: str
    atomic_number: int
    atomic_weight: float
    covalent_radius: float
    vdw_radius: float
    electronegativity: float
    default_valences: Tuple[int, ...]
    crippen_logp: float
    gasteiger_a: float
    gasteiger_b: float
    gasteiger_c: float


# Physical parameters sourced from IUPAC atomic weights, Pauling electronegativities,
# Alvarez covalent radii, Bondi vdW radii, and Gasteiger-Marsili (1980) orbital constants.
ELEMENT_TABLE: Dict[str, ElementInfo] = {
    "H": ElementInfo("H", 1, 1.008, 0.31, 1.20, 2.20, (1,), 0.123, 7.17, 6.24, -0.56),
    "B": ElementInfo("B", 5, 10.81, 0.84, 1.92, 2.04, (3,), -0.250, 5.98, 6.82, 1.60),
    "C": ElementInfo("C", 6, 12.011, 0.76, 1.70, 2.55, (4,), 0.144, 7.98, 9.18, 1.88),
    "N": ElementInfo("N", 7, 14.007, 0.71, 1.55, 3.04, (3, 5), -0.480, 11.54, 10.82, 1.36),
    "O": ElementInfo("O", 8, 15.999, 0.66, 1.52, 3.44, (2,), -0.289, 14.18, 12.92, 1.39),
    "F": ElementInfo("F", 9, 18.998, 0.57, 1.47, 3.98, (1,), 0.412, 14.66, 13.85, 2.31),
    "P": ElementInfo("P", 15, 30.974, 1.07, 1.80, 2.19, (3, 5), 0.320, 8.90, 8.24, 0.96),
    "S": ElementInfo("S", 16, 32.06, 1.05, 1.80, 2.58, (2, 4, 6), 0.648, 10.14, 9.13, 1.38),
    "Cl": ElementInfo("Cl", 17, 35.45, 1.02, 1.75, 3.16, (1,), 0.689, 11.00, 9.69, 1.35),
    "Br": ElementInfo("Br", 35, 79.904, 1.20, 1.85, 2.96, (1,), 0.846, 10.08, 8.47, 1.16),
    "I": ElementInfo("I", 53, 126.904, 1.39, 1.98, 2.66, (1,), 1.050, 9.90, 7.96, 0.96),
    "Si": ElementInfo("Si", 14, 28.085, 1.11, 2.10, 1.90, (4,), 0.300, 7.30, 6.57, 0.66),
    "Fe": ElementInfo("Fe", 26, 55.845, 1.32, 2.00, 1.83, (2, 3), -0.300, 4.50, 5.20, 0.50),
}

ATOMIC_NUM_TO_SYMBOL: Dict[int, str] = {info.atomic_number: sym for sym, info in ELEMENT_TABLE.items()}
AROMATIC_SYMBOLS: Dict[str, str] = {"b": "B", "c": "C", "n": "N", "o": "O", "p": "P", "s": "S"}


@dataclass
class Atom:
    """Single heavy or explicit hydrogen atom node in a molecular graph."""

    index: int
    symbol: str
    atomic_number: int
    formal_charge: int = 0
    isotope: Optional[int] = None
    explicit_h: int = 0
    implicit_h: int = 0
    is_aromatic: bool = False
    is_in_ring: bool = False
    ring_sizes: Tuple[int, ...] = ()
    chirality: Optional[str] = None
    hybridization: Hybridization = Hybridization.OTHER
    partial_charge: float = 0.0
    coords: Optional[Tuple[float, float, float]] = None

    @property
    def total_h(self) -> int:
        """Total attached hydrogens (explicit plus implicit)."""
        return self.explicit_h + self.implicit_h


@dataclass
class Bond:
    """Undirected covalent bond connecting two atom indices in a molecular graph."""

    index: int
    begin_idx: int
    end_idx: int
    order: BondOrder = BondOrder.SINGLE
    is_aromatic: bool = False
    is_in_ring: bool = False
    is_conjugated: bool = False
    stereo: Optional[str] = None

    def other(self, atom_idx: int) -> int:
        """Return the neighbor atom index across this bond."""
        if atom_idx == self.begin_idx:
            return self.end_idx
        if atom_idx == self.end_idx:
            return self.begin_idx
        raise GraphShapeError(f"Atom {atom_idx} is not an endpoint of bond {self.index}")


@dataclass
class Molecule:
    """Mutable graph representation of a chemical molecule with adjacency cache."""

    name: str
    smiles: str
    atoms: List[Atom] = field(default_factory=list)
    bonds: List[Bond] = field(default_factory=list)
    _adj: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict, repr=False)

    def rebuild_adjacency(self) -> None:
        """Rebuild atom -> [(neighbor_idx, bond_idx)] adjacency map."""
        self._adj = {atom.index: [] for atom in self.atoms}
        for bond in self.bonds:
            self._adj[bond.begin_idx].append((bond.end_idx, bond.index))
            self._adj[bond.end_idx].append((bond.begin_idx, bond.index))

    def neighbors(self, atom_idx: int) -> List[Tuple[int, Bond]]:
        """Return neighboring atom indices and connecting Bond instances."""
        if not self._adj and self.atoms:
            self.rebuild_adjacency()
        return [(nbr_idx, self.bonds[b_idx]) for nbr_idx, b_idx in self._adj.get(atom_idx, [])]

    def degree(self, atom_idx: int) -> int:
        """Heavy-atom graph degree of atom_idx."""
        if not self._adj and self.atoms:
            self.rebuild_adjacency()
        return len(self._adj.get(atom_idx, []))

    def bond_between(self, idx_a: int, idx_b: int) -> Optional[Bond]:
        """Return the Bond connecting idx_a and idx_b, or None if unbonded."""
        for nbr_idx, bond in self.neighbors(idx_a):
            if nbr_idx == idx_b:
                return bond
        return None

    def bond_order_sum(self, atom_idx: int) -> float:
        """Sum of covalent bond orders attached to atom_idx excluding implicit H."""
        total = 0.0
        for _, bond in self.neighbors(atom_idx):
            total += float(bond.order.value)
        return total

    @property
    def num_atoms(self) -> int:
        """Number of heavy/explicit atoms in the graph."""
        return len(self.atoms)

    @property
    def num_bonds(self) -> int:
        """Number of covalent bonds in the graph."""
        return len(self.bonds)

    def copy(self, new_name: Optional[str] = None) -> Molecule:
        """Deep copy of the molecular graph."""
        copied_atoms = [
            Atom(
                index=a.index,
                symbol=a.symbol,
                atomic_number=a.atomic_number,
                formal_charge=a.formal_charge,
                isotope=a.isotope,
                explicit_h=a.explicit_h,
                implicit_h=a.implicit_h,
                is_aromatic=a.is_aromatic,
                is_in_ring=a.is_in_ring,
                ring_sizes=a.ring_sizes,
                chirality=a.chirality,
                hybridization=a.hybridization,
                partial_charge=a.partial_charge,
                coords=a.coords,
            )
            for a in self.atoms
        ]
        copied_bonds = [
            Bond(
                index=b.index,
                begin_idx=b.begin_idx,
                end_idx=b.end_idx,
                order=b.order,
                is_aromatic=b.is_aromatic,
                is_in_ring=b.is_in_ring,
                is_conjugated=b.is_conjugated,
                stereo=b.stereo,
            )
            for b in self.bonds
        ]
        mol = Molecule(name=new_name or self.name, smiles=self.smiles, atoms=copied_atoms, bonds=copied_bonds)
        mol.rebuild_adjacency()
        return mol


@dataclass(frozen=True)
class RingSystem:
    """Single ring from a Smallest Set of Smallest Rings (SSSR) cycle basis."""

    atom_indices: Tuple[int, ...]
    bond_indices: Tuple[int, ...]
    size: int
    is_aromatic: bool
    pi_electrons: int


@dataclass(frozen=True)
class MolecularDescriptors:
    """Physicochemical, topological, and drug-likeness descriptor bundle."""

    formula: str
    molecular_weight: float
    exact_mass: float
    heavy_atom_count: int
    ring_count: int
    aromatic_ring_count: int
    rotatable_bonds: int
    hbd_count: int
    hba_count: int
    formal_charge: int
    crippen_logp: float
    tpsa: float
    fsp3: float
    qed: float
    sa_score: float
    lipinski_violations: int
    veber_violations: int
    wiener_index: int
    zagreb_m1: int
    zagreb_m2: int
    balaban_j: float


@dataclass(frozen=True)
class PharmacophoreFeature:
    """Annotated pharmacophore center mapped to atom indices."""

    feature_type: PharmacophoreType
    atom_indices: Tuple[int, ...]
    label: str


@dataclass(frozen=True)
class StructuralAlertMatch:
    """PAINS or Brenk medicinal chemistry structural alert hit."""

    alert_name: str
    severity: str
    smarts: str
    matched_atoms: Tuple[int, ...]
    rationale: str


@dataclass
class GraphTensors:
    """Standardized matrix representation of a molecular graph for GNN inference."""

    node_features: NDArray[np.float64]
    edge_index: NDArray[np.int64]
    edge_features: NDArray[np.float64]
    adjacency: NDArray[np.float64]
    norm_adjacency: NDArray[np.float64]
    shortest_paths: NDArray[np.int64]
    smiles: str


@dataclass(frozen=True)
class ADMEProfile:
    """Predicted absorption, distribution, metabolism, and excretion parameters."""

    smiles: str
    log_s_mol_l: float
    papp_caco2_nm_s: float
    fraction_absorbed: float
    log_bb: float
    pgp_efflux_risk: float
    fu_plasma: float
    cl_int_ml_min_kg: float
    cl_hepatic_ml_min_kg: float
    vd_l_kg: float
    cyp3a4_hazard: float
    cyp2d6_hazard: float
    half_life_h: float


@dataclass
class PKSimulationResult:
    """Two-compartment oral/IV pharmacokinetic ODE trajectory and exposure metrics."""

    times_h: NDArray[np.float64]
    gut_amount_mg: NDArray[np.float64]
    plasma_conc_mg_l: NDArray[np.float64]
    peripheral_conc_mg_l: NDArray[np.float64]
    unbound_plasma_nm: NDArray[np.float64]
    cmax_mg_l: float
    tmax_h: float
    auc_0_inf_mg_h_l: float
    half_life_terminal_h: float
    time_above_ic50_h: float
    mass_balance_error: float


@dataclass
class DockingPoseResult:
    """3D conformer embedding and receptor-ligand empirical docking score."""

    smiles: str
    coords: NDArray[np.float64]
    total_score_kcal_mol: float
    gauss1_term: float
    gauss2_term: float
    repulsion_term: float
    hydrophobic_term: float
    hbond_term: float
    torsion_penalty: float
    gbsa_desolvation: float
    internal_strain: float
    pocket_contacts: int
    ligand_efficiency: float


@dataclass
class ProtacSweepResult:
    """Douglass-Miller ternary complex equilibrium sweep and Hook-effect summary."""

    conc_nm: NDArray[np.float64]
    ternary_ple_nm: NDArray[np.float64]
    binary_pl_nm: NDArray[np.float64]
    binary_le_nm: NDArray[np.float64]
    fraction_bound_poi: NDArray[np.float64]
    degradation_fraction: NDArray[np.float64]
    peak_ple_nm: float
    hook_peak_conc_nm: float
    dc50_nm: Optional[float]
    dmax: float
    cooperativity_alpha: float
    max_mass_balance_error: float


@dataclass(frozen=True)
class OptimizationCandidate:
    """Evaluated candidate molecule inside multi-objective de novo optimization."""

    smiles: str
    generation: int
    origin_op: str
    predicted_pic50: float
    uncertainty: float
    qed: float
    sa_score: float
    adme_score: float
    alert_count: int
    mpo_score: float
    pareto_rank: int = 0
    crowding_distance: float = 0.0


@dataclass
class ParetoFrontResult:
    """Result bundle from NSGA-II / evolutionary multi-objective optimization."""

    seed_smiles: str
    generations: int
    evaluated_count: int
    valid_unique_count: int
    pareto_front: List[OptimizationCandidate]
    all_top_candidates: List[OptimizationCandidate]
    best_mpo_history: List[float]


@dataclass
class ConformalPredictionResult:
    """Split conformal QSAR evaluation metrics and calibrated intervals."""

    r2: float
    rmse: float
    mae: float
    spearman_rho: float
    enrichment_5pct: float
    conformal_quantile_90: float
    empirical_coverage_90: float
    mean_interval_width: float
