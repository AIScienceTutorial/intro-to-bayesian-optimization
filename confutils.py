"""Utilities for conformer optimization

Adapted from: https://github.com/globus-labs/conformer-optimization"""

from typing import Tuple, Set, Dict, List, Union
from dataclasses import dataclass
from io import StringIO
import logging
import os

from ase.calculators.calculator import Calculator
from ase.constraints import FixInternals
from ase.io.xyz import read_xyz
from ase.optimize import BFGS
from ase import Atoms
from openbabel import OBMolBondIter
import networkx as nx
import numpy as np
import pybel

logger = logging.getLogger(__name__)


def get_initial_structure(smiles: str) -> Tuple[Atoms, pybel.Molecule]:
    """Generate an initial guess for a molecular structure
    
    Args:
        smiles: SMILES string
    Returns: 
        Generate an Atoms object
    """

    # Make the 3D structure
    mol = pybel.readstring("smi", smiles)
    mol.make3D()

    # Convert it to ASE
    atoms = next(read_xyz(StringIO(mol.write('xyz')), slice(None)))
    atoms.charge = mol.charge
    atoms.set_initial_charges([a.formalcharge for a in mol.atoms])
        
    return atoms, mol


@dataclass()
class DihedralInfo:
    """Describes a dihedral angle within a molecule"""

    chain: Tuple[int, int, int, int] = None
    """Atoms that form the dihedral. ASE rotates the last atom when setting a dihedral angle"""
    group: Set[int] = None
    """List of atoms that should rotate along with this dihedral"""
    type: str = None

    def get_angle(self, atoms: Atoms) -> float:
        """Get the value of the specified dihedral angle

        Args:
            atoms: Structure to assess
        """
        return atoms.get_dihedral(*self.chain)


def detect_dihedrals(mol: pybel.Molecule) -> List[DihedralInfo]:
    """Detect the bonds to be treated as rotors.
    
    We use the more generous definition from RDKit: 
    https://github.com/rdkit/rdkit/blob/1bf6ef3d65f5c7b06b56862b3fb9116a3839b229/rdkit/Chem/Lipinski.py#L47%3E
    
    It matches pairs of atoms that are connected by a single bond,
    both bonds have at least one other bond that is not a triple bond
    and they are not part of the same ring.
    
    Args:
        mol: Molecule to assess
    Returns:
        List of dihedral angles. Most are defined 
    """
    dihedrals = []

    # Compute the bonding graph
    g = get_bonding_graph(mol)

    # Get the indices of backbond atoms
    backbone = set(i for i, d in g.nodes(data=True) if d['z'] > 1)

    # Step 1: Get the bonds from a simple matching
    smarts = pybel.Smarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
    for i, j in smarts.findall(mol):
        dihedrals.append(get_dihedral_info(g, (i - 1, j - 1), backbone))
    return dihedrals


def get_bonding_graph(mol: pybel.Molecule) -> nx.Graph:
    """Generate a bonding graph from a molecule
    
    Args:
        mol: Molecule to be assessed
    Returns: 
        Graph describing the connectivity
    """

    # Get the bonding graph
    g = nx.Graph()
    g.add_nodes_from([
        (i, dict(z=a.atomicnum))
        for i, a in enumerate(mol.atoms)
    ])
    for bond in OBMolBondIter(mol.OBMol):
        g.add_edge(bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1,
                   data={"rotor": bond.IsRotor(), "ring": bond.IsInRing()})
    return g


def get_dihedral_info(graph: nx.Graph, bond: Tuple[int, int], backbone_atoms: Set[int]) -> DihedralInfo:
    """For a rotatable bond in a model, get the atoms which define the dihedral angle
    and the atoms that should rotate along with the right half of the molecule
    
    Args:
        graph: Bond graph of the molecule
        bond: Left and right indicies of the bond, respectively
        backbone_atoms: List of atoms defined as part of the backbone
    Returns:
        - Atom indices defining the dihedral. Last atom is the one that will be moved 
          by ase's "set_dihedral" function
        - List of atoms being rotated along with set_dihedral
    """

    # Pick the atoms to use in the dihedral, starting with the left
    points = list(bond)
    choices = set(graph[bond[0]]).difference(bond)
    bb_choices = choices.intersection(backbone_atoms)
    if len(bb_choices) > 0:  # Pick a backbone if available
        choices = bb_choices
    points.insert(0, min(choices))

    # Then the right
    choices = set(graph[bond[1]]).difference(bond)
    bb_choices = choices.intersection(backbone_atoms)
    if len(bb_choices) > 0:  # Pick a backbone if available
        choices = bb_choices
    points.append(min(choices))

    # Get the points that will rotate along with the bond
    h = graph.copy()
    h.remove_edge(*bond)
    a, b = nx.connected_components(h)
    if bond[1] in a:
        return DihedralInfo(chain=points, group=a, type='backbone')
    else:
        return DihedralInfo(chain=points, group=b, type='backbone')


def evaluate_energy(angles: Union[List[float], np.ndarray], atoms: Atoms,
                    dihedrals: List[DihedralInfo], calc: Calculator,
                    relax: bool = True) -> float:
    """Compute the energy of a cysteine molecule given dihedral angles
    Args:
        angles: List of dihedral angles
        atoms: Structure to optimize
        dihedrals: Description of the dihedral angles
        calc: Calculator used to compute energy/gradients
        relax: Whether to relax the non-dihedral degrees of freedom
    Returns:
        - (float) energy of the structure
    """
    # Make a copy of the input
    atoms = atoms.copy()

    # Set the dihedral angles to desired settings
    dih_cnsts = []
    for a, di in zip(angles, dihedrals):
        atoms.set_dihedral(*di.chain, a, indices=di.group)

        # Define the constraints
        dih_cnsts.append((a, di.chain))
        
    # If not relaxed, just compute the energy
    if not relax:
        return calc.get_potential_energy(atoms)
        
    atoms.set_constraint()
    atoms.set_constraint(FixInternals(dihedrals_deg=dih_cnsts))

    return relax_structure(atoms, calc)[0]


def relax_structure(atoms: Atoms, calc: Calculator) -> Tuple[float, Atoms]:
    """Relax and return the energy of the ground state
    
    Args:
        atoms: Atoms object to be optimized
        calc: Calculator used to compute energy/gradients
    Returns:
        Energy of the minimized structure
    """

    atoms.set_calculator(calc)

    dyn = BFGS(atoms, logfile=os.devnull)
    dyn.run(fmax=1e-3)

    return atoms.get_potential_energy(), atoms


def set_dihedrals(atoms: Atoms, angles: List[float], dihedrals: List[DihedralInfo]) -> Atoms:
    """Set the dihedral angles to a certain value
    
    Args:
        atoms: Base structure to modify
        angles: Desired dihedral angles
        dihedrals: Description of the dihedral angles
    Returns:
        Structure with the angles set as desired
    """
    
    output = atoms.copy()
    for a, di in zip(angles, dihedrals):
        output.set_dihedral(*di.chain, a, indices=di.group)
    return output
    