"""
High Entropy Crystal Optimization Module.

This module extends the CHGNet framework to optimize high-entropy crystal systems
by incorporating configurational entropy into the optimization target and 
implementing Particle Swarm Optimization (PSO) for global structure search.
"""

from __future__ import annotations

import csv
import os
import time
import warnings
import copy
import random
import logging
from datetime import datetime
from itertools import combinations
import numpy as np
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Literal, Dict, Any, List, Optional, Union, Tuple, Set, Callable, Type

import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes, all_properties
from ase.optimize.fire import FIRE
from ase.optimize.optimize import Optimizer, Dynamics
from pymatgen.core.structure import Structure, Lattice
from pymatgen.core.sites import PeriodicSite
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.spatial.distance import pdist, squareform

from chgnet.model.dynamics import (
    StructOptimizer, CHGNetCalculator, TrajectoryObserver,
    OPTIMIZERS, CrystalFeasObserver
)
from chgnet.model.model import CHGNet

if TYPE_CHECKING:
    from typing_extensions import Self

__author__ = "Custom CHGNet Development Team"
__version__ = "0.1.0"

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)


class ConfigEntropy:
    """Class for calculating configurational entropy of high-entropy crystal systems.

    This class implements the standard configurational entropy calculation
    for high-entropy crystals based on the formula:
    S_config = -k_B * Σ[p_i * ln(p_i)]
    where p_i is the atomic fraction of element i.
    """

    def __init__(self, kb: float = 8.617333262e-5):
        """Initialize the configurational entropy calculator.

        Args:
            kb (float): Boltzmann constant in eV/K. Default = 8.617333262e-5 eV/K
        """
        self.kb = kb

    def calculate(self, structure: Structure) -> float:
        """Calculate the configurational entropy for a given structure.

        Args:
            structure (Structure): pymatgen Structure object

        Returns:
            float: configurational entropy in eV/atom
        """
        # Get the composition as a dictionary of element:count
        elements = [site.species_string for site in structure.sites]
        composition = Counter(elements)

        # Calculate atomic fractions
        total_atoms = sum(composition.values())
        fractions = {elem: count / total_atoms for elem, count in composition.items()}

        # Calculate configurational entropy S_config = -k_B * Σ[p_i * ln(p_i)]
        entropy = 0.0
        for fraction in fractions.values():
            if fraction > 0:  # Avoid log(0)
                entropy -= fraction * np.log(fraction)

        # Convert to eV/atom using Boltzmann constant
        entropy_ev = entropy * self.kb

        return entropy_ev

    def calculate_detailed(self, structure: Structure) -> Dict[str, Any]:
        """Calculate detailed configurational entropy information.

        Args:
            structure (Structure): pymatgen Structure object

        Returns:
            Dict: Detailed entropy information including:
                - entropy: total entropy value in eV/atom
                - fractions: dictionary of element:fraction
                - composition: dictionary of element:count
                - formula: chemical formula of the structure
        """
        # Get the composition
        elements = [site.species_string for site in structure.sites]
        composition = Counter(elements)

        # Calculate atomic fractions
        total_atoms = sum(composition.values())
        fractions = {elem: count / total_atoms for elem, count in composition.items()}

        # Calculate configurational entropy
        entropy = 0.0
        for fraction in fractions.values():
            if fraction > 0:
                entropy -= fraction * np.log(fraction)

        entropy_ev = entropy * self.kb

        # Return detailed information
        return {
            "entropy": entropy_ev,
            "fractions": fractions,
            "composition": dict(composition),
            "formula": structure.formula,
            "entropy_kb": entropy,  # Raw entropy in units of kB
        }


class AtomicRadiusLoader:
    """Utility class to load and manage atomic radii data."""

    def __init__(self, default_radius: float = 100.0):
        """Initialize the atomic radius loader.

        Args:
            default_radius (float): Default radius in pm for elements not in the table.
                Default = 100.0
        """
        self.default_radius = default_radius
        self.radii = {}
        self.loaded = False
        self._missing_elements = set()
        self._site_preferences = {
            # Elements that typically occupy A site in ABX3 perovskites
            "A_site": {"Cs", "Rb", "Tl", "Ra", "K", "Ba", "Pb", "Sr", "Na", "Bi",
                       "La", "Ce", "Ca", "Cd", "Ag", "Nd", "Sm", "Th", "Pr", "Pm",
                       "Eu", "Ho", "Gd", "Tb", "Dy", "Y", "Er", "Li", "Tm", "Yb"},

            # Elements that typically occupy B site in ABX3 perovskites
            "B_site": {"Tl", "Bi", "Ce", "Nd", "Sm", "Th", "Pr", "Pm", "Eu", "Ho",
                       "Gd", "Tb", "Dy", "Y", "Er", "Tm", "Yb", "Ac", "Hg", "Cf",
                       "Pa", "U", "Pu", "Np", "Lu", "Au", "Am", "Cm", "Bk", "In",
                       "Sc", "Zn", "Cu", "Zr", "Mg", "Hf", "Sn", "Nb", "Ru", "Ir",
                       "Po", "Fe", "Mn", "Tc", "Ta", "Re", "Os", "Pt", "Cr", "Ga",
                       "Pd", "At", "Co", "Ti", "Ni", "W", "Sb", "Rh", "Mo", "V",
                       "As", "Al", "Ge", "Be", "P", "Si", "B", "C"},

            # Elements that typically occupy X site (e.g., O in oxides)
            "X_site": {"Te", "I", "Se", "Br", "S", "Cl", "N", "O", "F"}
        }

    def load_radii(self, filepath: Optional[str] = None):
        """Load atomic radii from CSV file.

        Args:
            filepath (str, optional): Path to the CSV file. If None, uses default path.
        """
        if filepath is None:
            # Try to find the atomic radius table in the package
            package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            filepath = os.path.join(package_dir, 'data', 'Atomic_radius_table.csv')

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    atom = row['Atom']
                    # Store radii for all available sites
                    self.radii[atom] = {
                        'A': float(row['Ion radius at site A']) if row['Ion radius at site A'] else None,
                        'B': float(row['Ion radius at site B']) if row['Ion radius at site B'] else None,
                        'O': float(row['Ion radius at site O']) if row['Ion radius at site O'] else None,
                    }
            logger.info(f"Successfully loaded atomic radius data from {filepath}")
            self.loaded = True
        except (FileNotFoundError, KeyError) as e:
            logger.warning(f"Failed to load atomic radius data: {e}. Using default radius for all atoms.")
            self.loaded = False

    def guess_site_type(self, element: str) -> str:
        """Guess the most likely site type for an element.

        Args:
            element (str): Element symbol

        Returns:
            str: Site type ('A', 'B', or 'O')
        """
        if element in self._site_preferences["A_site"]:
            return 'A'
        elif element in self._site_preferences["B_site"]:
            return 'B'
        elif element in self._site_preferences["X_site"]:
            return 'O'
        else:
            # Default to B site for most transition metals and smaller atoms
            return 'B'

    def get_radius(self, element: str, site: str | None = None) -> float:
        """Get the radius of an element at a specific site.

        Args:
            element (str): Chemical element symbol
            site (str, optional): Site type ('A', 'B', or 'O'). If None, will try to guess.

        Returns:
            float: Atomic radius in pm
        """
        if not self.loaded:
            self.load_radii()

        # If site not specified, try to guess it
        if site is None:
            site = self.guess_site_type(element)

        # Check if we have the radius for the requested site
        if element in self.radii and self.radii[element][site] is not None:
            return self.radii[element][site]

        # Try other sites if the requested site doesn't have data
        if element in self.radii:
            for site_type in ['A', 'B', 'O']:
                if self.radii[element][site_type] is not None:
                    logger.warning(f"No radius data for {element} at site {site}, using site {site_type} instead.")
                    return self.radii[element][site_type]

        # Use default radius if element not found
        if element not in self._missing_elements:
            logger.warning(f"No radius data for element {element}, using default value of {self.default_radius} pm.")
            self._missing_elements.add(element)  # Track to avoid repeated warnings

        return self.default_radius

    def get_missing_elements(self) -> Set[str]:
        """Get the list of elements that were requested but not found in the data.

        Returns:
            Set[str]: Set of missing element symbols
        """
        return self._missing_elements


class EntropyTrajectoryObserver(TrajectoryObserver):
    """Extended trajectory observer that also tracks entropy and free energy."""

    def __init__(self, atoms: Atoms, config_entropy: ConfigEntropy, temperature: float) -> None:
        """Create an EntropyTrajectoryObserver from an Atoms object.

        Args:
            atoms (Atoms): the structure to observe
            config_entropy (ConfigEntropy): configurational entropy calculator
            temperature (float): temperature in K for free energy calculation
        """
        super().__init__(atoms)
        self.config_entropy = config_entropy
        self.temperature = temperature
        self.entropies: list[float] = []
        self.free_energies: list[float] = []

    def __call__(self) -> None:
        """Record properties including entropy and free energy."""
        super().__call__()

        # Calculate configurational entropy
        structure = AseAtomsAdaptor.get_structure(self.atoms)
        entropy = self.config_entropy.calculate(structure)
        self.entropies.append(entropy)

        # Calculate free energy F = E - TS
        energy = self.energies[-1]
        free_energy = energy - self.temperature * entropy
        self.free_energies.append(free_energy)

    def save(self, filename: str) -> None:
        """Save the trajectory including entropy data to file.

        Args:
            filename (str): filename to save the trajectory
        """
        out_pkl = {
            "energy": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "magmoms": self.magmoms,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atoms.get_atomic_numbers(),
            "entropy": self.entropies,
            "free_energy": self.free_energies,
            "temperature": self.temperature,
        }
        with open(filename, "wb") as file:
            import pickle
            pickle.dump(out_pkl, file)


class HighEntropyCHGNetCalculator(CHGNetCalculator):
    """Extended CHGNet Calculator that includes configurational entropy in free energy."""

    def __init__(
            self,
            model: CHGNet | None = None,
            *,
            temperature: float = 300.0,
            config_entropy: Optional[ConfigEntropy] = None,
            use_device: str | None = None,
            check_cuda_mem: bool = False,
            stress_weight: float = 1 / 160.21,
            on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
            return_site_energies: bool = False,
            **kwargs,
    ) -> None:
        """Initialize the high entropy CHGNet calculator.

        Args:
            model (CHGNet): instance of a chgnet model. If set to None,
                the pretrained CHGNet is loaded.
            temperature (float): temperature in K for free energy calculation
            config_entropy (ConfigEntropy, optional): entropy calculator instance
            use_device (str, optional): device for calculations
            check_cuda_mem (bool): whether to check available CUDA memory
            stress_weight (float): stress conversion factor
            on_isolated_atoms (str): how to handle isolated atoms
            return_site_energies (bool): whether to return site energies
            **kwargs: additional arguments for Calculator
        """
        super().__init__(
            model=model,
            use_device=use_device,
            check_cuda_mem=check_cuda_mem,
            stress_weight=stress_weight,
            on_isolated_atoms=on_isolated_atoms,
            return_site_energies=return_site_energies,
            **kwargs,
        )

        self.temperature = temperature
        self.config_entropy = config_entropy or ConfigEntropy()

    def calculate(
            self,
            atoms: Atoms | None = None,
            properties: list | None = None,
            system_changes: list | None = None,
            task: str = "efsm",
    ) -> None:
        """Calculate properties including configurational entropy contribution.

        Args:
            atoms (Atoms): atoms to calculate properties for
            properties (list): properties to calculate
            system_changes (list): system changes
            task (str): task to perform
        """
        # Call the parent calculate method to get energy, forces, etc.
        super().calculate(
            atoms=atoms,
            properties=properties,
            system_changes=system_changes,
            task=task,
        )

        # Calculate configurational entropy and free energy
        structure = AseAtomsAdaptor.get_structure(atoms)
        entropy = self.config_entropy.calculate(structure)

        # Store the entropy and update free energy
        self.results["entropy"] = entropy
        self.results["config_entropy"] = entropy  # Also store as config_entropy for clarity
        self.results["free_energy"] = self.results["energy"] - self.temperature * entropy


class ParticleEncoder:
    """Handles encoding and decoding of crystal structure particles for PSO."""

    def __init__(self, radius_loader: AtomicRadiusLoader,
                 max_lattice_strain: float = 0.1,
                 max_atomic_displacement: float = 0.3):
        """Initialize the particle encoder.

        Args:
            radius_loader (AtomicRadiusLoader): For atomic radius data
            max_lattice_strain (float): Maximum lattice strain allowed
            max_atomic_displacement (float): Maximum atomic displacement as a fraction of radius
                Default is 0.3 (30%) as specified in requirements
        """
        self.radius_loader = radius_loader
        self.max_lattice_strain = max_lattice_strain
        self.max_atomic_displacement = max_atomic_displacement
        self._encoding_info = {}

        # For statistics and diagnostics
        self.constraints_applied = 0
        self.constraint_violations = {
            "lattice_params": 0,
            "angles": 0,
            "atomic_positions": 0,
            "proximity": 0
        }

    def encode(self, structure: Structure) -> np.ndarray:
        """Encode a crystal structure into a particle vector.

        The encoding has three parts:
        1. Lattice parameters (6 values: a, b, c, alpha, beta, gamma)
        2. Atom positions (3 * n_atoms values)
        3. Atom swaps encoding (one-hot encoding of possible swaps)

        Args:
            structure (Structure): Crystal structure to encode

        Returns:
            np.ndarray: Encoded particle
        """
        # Store information about encoding for later use in decoding
        n_atoms = len(structure.sites)

        # Part 1: Encode lattice parameters
        lattice = structure.lattice
        a, b, c = lattice.abc
        alpha, beta, gamma = lattice.angles
        lattice_encoding = np.array([a, b, c, alpha, beta, gamma])

        # Part 2: Encode atomic positions (flattened)
        positions = np.array([site.coords for site in structure.sites]).flatten()

        # Part 3: Encode atom swap possibilities
        # Group atoms by element type
        atoms_by_element = defaultdict(list)
        for i, site in enumerate(structure.sites):
            atoms_by_element[site.species_string].append(i)

        # Create swap encoding for atoms of the same element
        swap_encoding = []
        swap_pairs = []
        for element, indices in atoms_by_element.items():
            if len(indices) > 1:
                for i, j in combinations(indices, 2):
                    swap_pairs.append((i, j, element))
                    # Add a binary flag for each possible swap
                    swap_encoding.append(0.0)  # 0.0 means no swap, will be set to 1.0 to perform swap

        # Store encoding info for decoding
        self._encoding_info = {
            "n_atoms": n_atoms,
            "elements": [site.species_string for site in structure.sites],
            "lattice_slice": slice(0, 6),
            "positions_slice": slice(6, 6 + 3 * n_atoms),
            "swaps_slice": slice(6 + 3 * n_atoms, None),
            "swap_pairs": swap_pairs
        }

        # Combine all parts
        particle = np.concatenate([lattice_encoding, positions, np.array(swap_encoding)])

        return particle

    def decode(self, particle: np.ndarray, template_structure: Structure) -> Structure:
        """Decode a particle vector back into a crystal structure.

        Args:
            particle (np.ndarray): Encoded particle
            template_structure (Structure): Template structure for reference

        Returns:
            Structure: Decoded crystal structure
        """
        # Use the stored encoding info or encode the template to get it
        if not self._encoding_info:
            _ = self.encode(template_structure)

        # Extract parts from the particle
        n_atoms = self._encoding_info["n_atoms"]

        # Part 1: Decode lattice parameters
        a, b, c, alpha, beta, gamma = particle[self._encoding_info["lattice_slice"]]
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

        # Part 2: Decode atomic positions
        positions = particle[self._encoding_info["positions_slice"]].reshape(-1, 3)

        # Create new structure with decoded lattice and positions
        new_structure = Structure(lattice, [], [])
        for i, site in enumerate(template_structure.sites):
            new_structure.append(
                species=site.species,
                coords=positions[i],
                coords_are_cartesian=True,
                properties=site.properties
            )

        # Part 3: Decode and apply atom swaps
        swap_encoding = particle[self._encoding_info["swaps_slice"]]

        for i, (idx1, idx2, element) in enumerate(self._encoding_info["swap_pairs"]):
            if i < len(swap_encoding) and swap_encoding[i] > 0.5:
                # Swap these atoms
                tmp_species = new_structure[idx1].species
                tmp_props = new_structure[idx1].properties

                new_structure[idx1].species = new_structure[idx2].species
                new_structure[idx1].properties = new_structure[idx2].properties

                new_structure[idx2].species = tmp_species
                new_structure[idx2].properties = tmp_props

        return new_structure

    def _check_atom_proximity(self, positions: np.ndarray, elements: List[str]) -> List[Tuple[int, int]]:
        """Check if any atoms are too close based on their atomic radii.

        Args:
            positions (np.ndarray): Atom positions
            elements (List[str]): Element symbols for each atom

        Returns:
            List[Tuple[int, int]]: List of (atom1, atom2) pairs that are too close
        """
        # Calculate all pairwise distances
        distances = squareform(pdist(positions))
        n_atoms = len(elements)

        # Check for pairs that are too close
        violations = []
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # Get radii for both atoms (convert from pm to Angstrom)
                radius_i = self.radius_loader.get_radius(elements[i]) / 100.0
                radius_j = self.radius_loader.get_radius(elements[j]) / 100.0

                # Calculate minimum allowed distance (allow 30% overlap)
                min_allowed = (radius_i + radius_j) * 0.7

                if distances[i, j] < min_allowed:
                    violations.append((i, j))

        return violations

    def apply_constraints(self, particle: np.ndarray, template_structure: Structure) -> np.ndarray:
        """Apply constraints to ensure the particle represents a physically reasonable structure.

        Constraints:
        1. Lattice parameters are positive and within strain limits
        2. Angles are between 30 and 150 degrees
        3. Atoms are not too close (using atomic radii)
        4. Atomic displacements are within allowed range (30% of radius)

        Args:
            particle (np.ndarray): Particle to constrain
            template_structure (Structure): Template structure

        Returns:
            np.ndarray: Constrained particle
        """
        self.constraints_applied += 1
        if not self._encoding_info:
            _ = self.encode(template_structure)

        constrained = particle.copy()

        # 1. Lattice parameter constraints
        # Ensure a, b, c are positive and within allowed strain range
        for i in range(3):
            template_length = template_structure.lattice.abc[i]
            min_allowed = template_length * (1 - self.max_lattice_strain)
            max_allowed = template_length * (1 + self.max_lattice_strain)

            if not min_allowed <= constrained[i] <= max_allowed:
                self.constraint_violations["lattice_params"] += 1
                constrained[i] = np.clip(constrained[i], min_allowed, max_allowed)

        # 2. Angle constraints (30 to 150 degrees)
        for i in range(3, 6):
            if not 30.0 <= constrained[i] <= 150.0:
                self.constraint_violations["angles"] += 1
                constrained[i] = np.clip(constrained[i], 30.0, 150.0)

        # 3 & 4. Atomic position constraints
        n_atoms = self._encoding_info["n_atoms"]
        position_start = 6
        position_end = position_start + 3 * n_atoms

        # Reshape positions for easier handling
        positions = constrained[position_start:position_end].reshape(-1, 3)
        template_positions = np.array([site.coords for site in template_structure.sites])
        elements = self._encoding_info["elements"]

        # First, limit displacements based on atomic radii (constraint 4)
        for i in range(n_atoms):
            element = elements[i]
            radius_pm = self.radius_loader.get_radius(element)
            radius_ang = radius_pm / 100.0  # Convert pm to Angstrom

            # Calculate max allowed displacement (30% of radius)
            max_displacement = radius_ang * self.max_atomic_displacement

            # Get displacement vector
            displacement = positions[i] - template_positions[i]
            displacement_magnitude = np.linalg.norm(displacement)

            # If displacement is too large, scale it down
            if displacement_magnitude > max_displacement:
                self.constraint_violations["atomic_positions"] += 1
                scaling_factor = max_displacement / displacement_magnitude
                displacement *= scaling_factor
                positions[i] = template_positions[i] + displacement

        # 5. Check for atomic proximity violations (constraint 3)
        proximity_violations = self._check_atom_proximity(positions, elements)

        # Fix proximity violations by adjusting positions
        if proximity_violations:
            self.constraint_violations["proximity"] += len(proximity_violations)

            # Iteratively adjust positions to resolve proximity violations
            for atom1, atom2 in proximity_violations:
                # Get positions and displacement vector
                pos1, pos2 = positions[atom1], positions[atom2]
                distance = np.linalg.norm(pos1 - pos2)
                if distance < 1e-8:  # Avoid division by zero
                    # If atoms are at the same position, move one slightly
                    direction = np.array([1.0, 0.0, 0.0])
                    positions[atom1] += direction * 0.1  # Move 0.1 Å in x direction
                    continue

                direction = (pos1 - pos2) / distance

                # Get radii
                radius1 = self.radius_loader.get_radius(elements[atom1]) / 100.0
                radius2 = self.radius_loader.get_radius(elements[atom2]) / 100.0

                # Calculate target distance
                target_distance = (radius1 + radius2) * 0.7  # Allow 30% overlap

                # Calculate adjustment
                if distance < target_distance:
                    adjustment = (target_distance - distance) * 0.5  # Move each atom half the distance
                    positions[atom1] += direction * adjustment
                    positions[atom2] -= direction * adjustment

        # Update the constrained particle with adjusted positions
        constrained[position_start:position_end] = positions.flatten()

        # 6. Ensure swap encoding values are binary
        for i in range(position_end, len(constrained)):
            constrained[i] = 1.0 if constrained[i] > 0.5 else 0.0

        return constrained


class PSOOptimizer(Dynamics):
    """PSO (Particle Swarm Optimization) optimizer compatible with ASE interface."""

    def __init__(
            self,
            atoms: Atoms,
            n_particles: int = 20,
            max_iterations: int = 50,
            w: float = 0.7,
            c1: float = 1.4,
            c2: float = 1.4,
            temperature: float = 300.0,
            config_entropy: Optional[ConfigEntropy] = None,
            radius_loader: Optional[AtomicRadiusLoader] = None,
            atomic_displacement_limit: float = 0.3,
            logfile: str | None = None,
            trajectory: Any = None,
            append_trajectory: bool = False,
            **kwargs
    ):
        """Initialize the PSO optimizer.

        Args:
            atoms (Atoms): ASE Atoms object to optimize
            n_particles (int): Number of particles in the swarm. Default = 20
            max_iterations (int): Maximum number of iterations. Default = 50
            w (float): Inertia weight. Default = 0.7
            c1 (float): Cognitive parameter. Default = 1.4
            c2 (float): Social parameter. Default = 1.4
            temperature (float): Temperature in K for free energy calculation
            config_entropy (ConfigEntropy): Entropy calculator
            radius_loader (AtomicRadiusLoader): For atomic radius data
            atomic_displacement_limit (float): Max displacement as fraction of radius (default 30%)
            logfile (str | None): File to write log output to, None means no log
            trajectory (Any): ASE trajectory object for recording
            append_trajectory (bool): Whether to append to an existing trajectory
            **kwargs: Additional parameters for the optimizer
        """
        # Call the Dynamics constructor
        super().__init__(
            atoms,
            logfile=logfile,
            trajectory=trajectory,
            append_trajectory=append_trajectory
        )

        # PSO parameters
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive parameter
        self.c2 = c2  # Social parameter

        # Temperature for entropy contribution
        self.temperature = temperature
        self.config_entropy = config_entropy or ConfigEntropy()

        # For atomic radius constraints
        self.radius_loader = radius_loader or AtomicRadiusLoader()
        if not self.radius_loader.loaded:
            self.radius_loader.load_radii()

        # Create particle encoder
        self.encoder = ParticleEncoder(
            self.radius_loader,
            max_atomic_displacement=atomic_displacement_limit
        )

        # Template structure for encoding/decoding
        self.template_structure = AseAtomsAdaptor.get_structure(atoms)

        # PSO state variables
        self.particles = None
        self.velocities = None
        self.pbest_positions = None
        self.pbest_values = None
        self.gbest_position = None
        self.gbest_value = float('inf')
        self.gbest_structure = None
        self.gbest_forces = None

        # For tracking fitness history and metrics
        self.fitness_history = []
        self.iteration = 0
        self.converged = False
        self.convergence_reason = None
        self.constraint_stats = {
            "total_applications": 0,
            "lattice_violations": 0,
            "angle_violations": 0,
            "position_violations": 0,
            "proximity_violations": 0
        }
        self.start_time = time.time()

        # Store fmax for convergence check
        self.fmax = 0.1  # default value

        # Initialize particles
        self.initialize()

    def log(self, message: str, level: str = "info"):
        """Log a message with the specified level.

        Args:
            message (str): Message to log
            level (str): Log level (debug, info, warning, error, critical)
        """
        if level == "debug":
            logger.debug(message)
        elif level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "critical":
            logger.critical(message)

    def initialize(self):
        """Initialize the particles and velocities."""
        # Encode template structure
        base_particle = self.encoder.encode(self.template_structure)
        particle_dim = len(base_particle)

        self.log(f"Initializing {self.n_particles} particles with dimension {particle_dim}")

        # Initialize particles around the base structure
        self.particles = np.zeros((self.n_particles, particle_dim))
        self.velocities = np.zeros((self.n_particles, particle_dim))

        # First particle is the template structure
        self.particles[0] = base_particle

        # Other particles are random perturbations
        for i in range(1, self.n_particles):
            # Add random perturbation to lattice parameters (±5%)
            perturbed = base_particle.copy()

            # Perturb lattice parameters (a, b, c)
            for j in range(3):
                perturbed[j] *= (1 + np.random.uniform(-0.05, 0.05))

            # Perturb lattice angles (alpha, beta, gamma) by ±3 degrees
            for j in range(3, 6):
                perturbed[j] += np.random.uniform(-3, 3)

            # Perturb atomic positions (±0.2 Å)
            n_atoms = len(self.template_structure.sites)
            for j in range(6, 6 + 3 * n_atoms):
                perturbed[j] += np.random.uniform(-0.2, 0.2)

            # Randomly set some swap flags
            for j in range(6 + 3 * n_atoms, len(perturbed)):
                perturbed[j] = 1.0 if np.random.random() < 0.1 else 0.0

            # Apply constraints to ensure physical structure
            perturbed = self.encoder.apply_constraints(perturbed, self.template_structure)
            self.particles[i] = perturbed

            # Initialize velocity as small random values
            self.velocities[i] = np.random.uniform(-0.01, 0.01, particle_dim)

        # Initialize personal best to current positions
        self.pbest_positions = self.particles.copy()

        # Initialize personal best values to infinity
        self.pbest_values = np.full(self.n_particles, float('inf'))

        # Evaluate all particles
        self.log("Evaluating initial particles")
        for i in range(self.n_particles):
            fitness, forces = self.evaluate_particle(self.particles[i])

            # Update personal best if needed
            if fitness < self.pbest_values[i]:
                self.pbest_values[i] = fitness
                self.pbest_positions[i] = self.particles[i].copy()

            # Update global best if needed
            if fitness < self.gbest_value:
                self.gbest_value = fitness
                self.gbest_position = self.particles[i].copy()

                # Store global best structure
                structure = self.encoder.decode(self.gbest_position, self.template_structure)
                self.gbest_structure = structure
                self.gbest_forces = forces

        self.log(f"Initialization complete. Best initial fitness: {self.gbest_value:.6f} eV")

    def evaluate_particle(self, particle: np.ndarray) -> Tuple[float, np.ndarray]:
        """Evaluate the fitness of a particle (free energy).

        Args:
            particle (np.ndarray): Encoded structure particle

        Returns:
            Tuple[float, np.ndarray]: (fitness value, forces)
        """
        try:
            # Decode the particle to a structure
            structure = self.encoder.decode(particle, self.template_structure)

            # Convert to ASE atoms and calculate energy, forces
            atoms = AseAtomsAdaptor().get_atoms(structure)
            atoms.calc = self.atoms.calc  # Use the same calculator

            try:
                # Calculate energy and forces
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()

                # Calculate entropy
                entropy = self.config_entropy.calculate(structure)

                # Calculate free energy (F = E - TS)
                free_energy = energy - self.temperature * entropy

                return free_energy, forces
            except Exception as e:
                # If calculation fails, return a high energy to discourage this configuration
                self.log(f"Error in energy calculation: {e}", level="warning")
                return 1e10, np.zeros((len(atoms), 3))

        except Exception as e:
            # If decoding fails, return a high energy
            self.log(f"Error in particle decoding: {e}", level="error")
            return 1e10, np.zeros((len(self.atoms), 3))

    def update_particles(self):
        """Update all particle positions and velocities using PSO equations."""
        for i in range(self.n_particles):
            # Generate random components
            r1 = np.random.random(len(self.particles[i]))
            r2 = np.random.random(len(self.particles[i]))

            # Update velocity
            cognitive_component = self.c1 * r1 * (self.pbest_positions[i] - self.particles[i])
            social_component = self.c2 * r2 * (self.gbest_position - self.particles[i])

            self.velocities[i] = (self.w * self.velocities[i] +
                                  cognitive_component +
                                  social_component)

            # Update position
            self.particles[i] += self.velocities[i]

            # Apply constraints
            self.particles[i] = self.encoder.apply_constraints(
                self.particles[i], self.template_structure
            )

            # Evaluate new position
            fitness, forces = self.evaluate_particle(self.particles[i])

            # Update personal best
            if fitness < self.pbest_values[i]:
                self.pbest_values[i] = fitness
                self.pbest_positions[i] = self.particles[i].copy()

            # Update global best
            if fitness < self.gbest_value:
                self.gbest_value = fitness
                self.gbest_position = self.particles[i].copy()

                # Store global best structure and forces
                structure = self.encoder.decode(self.gbest_position, self.template_structure)
                self.gbest_structure = structure
                self.gbest_forces = forces

                # Update the atoms object with the best structure
                best_atoms = AseAtomsAdaptor().get_atoms(structure)
                self.atoms.set_positions(best_atoms.get_positions())
                self.atoms.set_cell(best_atoms.get_cell())

    def step(self, f=None):
        """Perform a single PSO iteration.

        Args:
            f (ndarray): Not used in PSO, but required for ASE interface compatibility

        Returns:
            float: Maximum force on the best structure
        """
        self.update_particles()
        self.iteration += 1

        # Store fitness history
        self.fitness_history.append(self.gbest_value)

        # Calculate maximum force for convergence check
        max_force = 0.0
        if self.gbest_forces is not None:
            max_force = np.max(np.sqrt(np.sum(self.gbest_forces ** 2, axis=1)))

        # Log progress
        if self.iteration % 5 == 0 or self.iteration == 1:
            elapsed = time.time() - self.start_time
            self.log(
                f"Iteration {self.iteration}/{self.max_iterations}: "
                f"Best fitness = {self.gbest_value:.6f} eV, "
                f"Max force = {max_force:.6f} eV/Å, "
                f"Time elapsed: {elapsed:.2f}s"
            )

        # Update constraint statistics
        self.constraint_stats["total_applications"] = self.encoder.constraints_applied
        self.constraint_stats["lattice_violations"] = self.encoder.constraint_violations["lattice_params"]
        self.constraint_stats["angle_violations"] = self.encoder.constraint_violations["angles"]
        self.constraint_stats["position_violations"] = self.encoder.constraint_violations["atomic_positions"]
        self.constraint_stats["proximity_violations"] = self.encoder.constraint_violations["proximity"]

        # Check convergence based on fmax or max iterations
        if max_force < self.fmax:
            self.converged = True
            self.convergence_reason = f"Maximum force below threshold: {max_force:.6f} < {self.fmax:.6f}"
            self.log(f"Converged: {self.convergence_reason}", level="info")
        elif self.iteration >= self.max_iterations:
            self.converged = True
            self.convergence_reason = f"Maximum iterations reached: {self.iteration}"
            self.log(f"Stopped: {self.convergence_reason}", level="info")

        return max_force

    def run(self, fmax=0.1, steps=None, **kwargs):
        """Run the PSO optimization until convergence or maximum steps.

        Args:
            fmax (float): Maximum force convergence criterion
            steps (int): Maximum number of steps (iterations)
            **kwargs: Additional arguments for ASE compatibility

        Returns:
            int: Number of steps performed
        """
        self.fmax = fmax
        if steps is not None:
            self.max_iterations = steps

        # Call parent class run method
        return super().run(steps=self.max_iterations)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the optimization run.

        Returns:
            Dict[str, Any]: Dictionary with statistics
        """
        elapsed = time.time() - self.start_time

        stats = {
            "iterations": self.iteration,
            "max_iterations": self.max_iterations,
            "converged": self.converged,
            "convergence_reason": self.convergence_reason,
            "final_fitness": self.gbest_value,
            "fitness_history": self.fitness_history,
            "elapsed_time": elapsed,
            "constraint_stats": self.constraint_stats,
            "fmax": self.fmax,
            "temperature": self.temperature,
            "pso_params": {
                "n_particles": self.n_particles,
                "w": self.w,
                "c1": self.c1,
                "c2": self.c2
            },
            "missing_elements": list(self.radius_loader.get_missing_elements())
        }
        return stats


class HighEntropyOptimizer(StructOptimizer):
    """Optimizer for high-entropy crystal systems considering configurational entropy.

    This optimizer extends the standard CHGNet StructOptimizer to include
    configurational entropy contributions in the optimization target and provides
    the option to use Particle Swarm Optimization (PSO) for global structure search.
    """

    def __init__(
            self,
            model: CHGNet | CHGNetCalculator | None = None,
            temperature: float = 300.0,
            use_pso: bool = True,
            pso_particles: int = 20,
            pso_iterations: int = 50,
            pso_w: float = 0.7,
            pso_c1: float = 1.4,
            pso_c2: float = 1.4,
            optimizer_class: Optimizer | str | None = "FIRE",
            use_device: str | None = None,
            stress_weight: float = 1 / 160.21,
            on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
            atomic_radius_constraint: bool = True,
            atomic_displacement_limit: float = 0.3,
            log_level: str = "INFO",
            **kwargs
    ):
        """Initialize the high-entropy structure optimizer.

        Args:
            model (CHGNet or CHGNetCalculator): model or calculator instance
            temperature (float): temperature in K for entropy contribution
            use_pso (bool): whether to use PSO optimization
            pso_particles (int): number of particles for PSO
            pso_iterations (int): maximum iterations for PSO
            pso_w (float): inertia weight for PSO
            pso_c1 (float): cognitive parameter for PSO
            pso_c2 (float): social parameter for PSO
            optimizer_class (Optimizer or str): fallback optimizer if not using PSO
            use_device (str): device for calculations
            stress_weight (float): stress conversion factor
            on_isolated_atoms (str): how to handle isolated atoms
            atomic_radius_constraint (bool): whether to apply atomic radius constraints
            atomic_displacement_limit (float): maximum displacement as fraction of radius (default 0.3)
            log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            **kwargs: Additional parameters
        """
        # Set up logging
        level = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(level)

        # Create a ConfigEntropy instance
        self.config_entropy = ConfigEntropy()

        # Store high entropy specific parameters
        self.temperature = temperature
        self.use_pso = use_pso
        self.pso_particles = pso_particles
        self.pso_iterations = pso_iterations
        self.pso_w = pso_w
        self.pso_c1 = pso_c1
        self.pso_c2 = pso_c2
        self.atomic_radius_constraint = atomic_radius_constraint
        self.atomic_displacement_limit = atomic_displacement_limit

        # Load atomic radius data
        self.radius_loader = AtomicRadiusLoader()
        self.radius_loader.load_radii()

        # For PSO optimizer
        self.pso_optimizer = None
        self.stats = {}

        # Create the appropriate calculator
        if isinstance(model, CHGNetCalculator):
            # Wrap the existing calculator
            calc = model
            self.calculator_is_high_entropy = False
        else:
            # Create a high entropy calculator
            calc = HighEntropyCHGNetCalculator(
                model=model,
                temperature=temperature,
                config_entropy=self.config_entropy,
                use_device=use_device,
                stress_weight=stress_weight,
                on_isolated_atoms=on_isolated_atoms,
            )
            self.calculator_is_high_entropy = True

        # Initialize parent class
        super().__init__(
            model=calc,
            optimizer_class=optimizer_class,
            use_device=use_device,
            stress_weight=stress_weight,
            on_isolated_atoms=on_isolated_atoms,
        )

        # Update OPTIMIZERS dictionary to include PSO
        global OPTIMIZERS
        OPTIMIZERS["PSO"] = PSOOptimizer

        logger.info("High Entropy Optimizer initialized")

    def calculate_free_energy(self, structure: Structure) -> Tuple[float, float, float]:
        """Calculate free energy including configurational entropy.

        Args:
            structure (Structure): pymatgen Structure object

        Returns:
            Tuple[float, float, float]: (energy, entropy, free_energy)
        """
        atoms = AseAtomsAdaptor().get_atoms(structure)
        atoms.calc = self.calculator

        # Calculate energy
        energy = atoms.get_potential_energy()

        # Calculate entropy
        entropy = self.config_entropy.calculate(structure)

        # Calculate free energy: F = E - TS
        free_energy = energy - self.temperature * entropy

        return energy, entropy, free_energy

    def check_atomic_radii_constraints(self, structure: Structure) -> Tuple[bool, List[Tuple[int, float]]]:
        """Check if atomic positions in the structure satisfy radius constraints.

        Args:
            structure (Structure): pymatgen Structure object

        Returns:
            Tuple[bool, List[Tuple[int, float]]]: 
                (is_valid, list of (atom_index, violation_ratio))
        """
        if not self.atomic_radius_constraint:
            return True, []

        violations = []

        # Get nearest neighbor distances for each atom
        neighbors = structure.get_all_neighbors(3.0)  # 3 Angstrom cutoff

        for i, site in enumerate(structure.sites):
            element = site.species_string
            # Get the radius for this element (in pm)
            radius_pm = self.radius_loader.get_radius(element)
            # Convert pm to Angstrom
            radius_ang = radius_pm / 100.0

            # Check distance to nearest neighbor
            if not neighbors[i]:
                continue  # Skip if no neighbors

            min_dist = min(nn.distance for nn in neighbors[i])
            min_allowed_dist = radius_ang * 1.4  # Allow some deviation (0.7 * 2)

            if min_dist < min_allowed_dist:
                violation_ratio = min_dist / min_allowed_dist
                violations.append((i, violation_ratio))

        is_valid = len(violations) == 0
        return is_valid, violations

    def apply_atomic_radii_constraints(self, structure: Structure) -> Structure:
        """Apply atomic radii constraints to the structure.

        Args:
            structure (Structure): input structure

        Returns:
            Structure: adjusted structure
        """
        if not self.atomic_radius_constraint:
            return structure

        # Create a copy of the structure
        adjusted = structure.copy()

        # Check for violations
        is_valid, violations = self.check_atomic_radii_constraints(adjusted)

        if is_valid:
            return adjusted

        # Apply adjustments to fix violations
        for atom_idx, violation_ratio in violations:
            if violation_ratio < 0.7:  # Severe violation - adjust position
                site = adjusted.sites[atom_idx]
                neighbors = adjusted.get_neighbors(site, 2.0)

                if not neighbors:
                    continue

                # Calculate displacement direction (away from closest neighbor)
                closest = neighbors[0]
                direction = site.coords - closest.coords
                norm = np.linalg.norm(direction)
                if norm > 1e-8:  # Avoid division by zero
                    direction = direction / norm
                else:
                    direction = np.array([1.0, 0.0, 0.0])  # Default direction if atoms are on top of each other

                # Move atom slightly away from its closest neighbor
                element = site.species_string
                radius_pm = self.radius_loader.get_radius(element)
                radius_ang = radius_pm / 100.0

                # Calculate adjustment distance (30% of radius)
                adjustment = direction * (radius_ang * self.atomic_displacement_limit)
                new_coords = site.coords + adjustment

                # Update the site position
                adjusted.replace(atom_idx, site.species, new_coords, properties=site.properties)

        return adjusted

    def relax(
            self,
            atoms: Structure | Atoms,
            temperature: float | None = None,
            fmax: float | None = 0.1,
            steps: int | None = 500,
            relax_cell: bool | None = True,
            ase_filter: str | None = "FrechetCellFilter",
            save_path: str | None = None,
            loginterval: int | None = 1,
            crystal_feas_save_path: str | None = None,
            verbose: bool = True,
            assign_magmoms: bool = True,
            **kwargs,
    ) -> Dict[str, Any]:
        """Relax structure considering configurational entropy.

        Args:
            atoms (Structure or Atoms): Structure to optimize
            temperature (float): Temperature in K, overrides the default
            fmax (float): Force convergence criterion
            steps (int): Maximum optimization steps
            relax_cell (bool): Whether to relax cell parameters
            ase_filter (str): Cell filter type
            save_path (str): Path to save trajectory
            loginterval (int): Interval for logging
            crystal_feas_save_path (str): Path to save crystal features
            verbose (bool): Print verbose output
            assign_magmoms (bool): Assign magnetic moments to final structure
            **kwargs: Additional arguments for optimizer

        Returns:
            dict: Results including final structure, trajectory, energy, entropy, etc.
        """
        # Use provided temperature if given, otherwise use default
        if temperature is not None:
            self.temperature = temperature
            if hasattr(self.calculator, 'temperature'):
                self.calculator.temperature = temperature

        # Convert to pymatgen Structure if needed
        if isinstance(atoms, Atoms):
            structure = AseAtomsAdaptor.get_structure(atoms)
        else:
            structure = atoms

        # Apply atomic radii constraints if enabled
        if self.atomic_radius_constraint:
            structure = self.apply_atomic_radii_constraints(structure)

        if self.use_pso:
            # Convert to ASE Atoms
            atoms = AseAtomsAdaptor().get_atoms(structure)
            atoms.calc = self.calculator

            # Create PSO optimizer
            pso = PSOOptimizer(
                atoms=atoms,
                n_particles=self.pso_particles,
                max_iterations=self.pso_iterations if steps is None else steps,
                w=self.pso_w,
                c1=self.pso_c1,
                c2=self.pso_c2,
                temperature=self.temperature,
                config_entropy=self.config_entropy,
                radius_loader=self.radius_loader,
                atomic_displacement_limit=self.atomic_displacement_limit,
                logfile=None if not verbose else "-",  # "-" means stdout
                trajectory=None,
            )
            self.pso_optimizer = pso

            # Set up trajectory observer
            obs = EntropyTrajectoryObserver(atoms, self.config_entropy, self.temperature)
            pso.attach(obs, interval=loginterval)

            # Set up crystal feature observer if needed
            if crystal_feas_save_path:
                cry_obs = CrystalFeasObserver(atoms)
                pso.attach(cry_obs, interval=loginterval)

            # Run PSO optimization
            pso.run(fmax=fmax, steps=steps)

            # Record final state
            obs()

            # Save trajectory if requested
            if save_path is not None:
                obs.save(save_path)

            # Save crystal features if requested
            if crystal_feas_save_path:
                cry_obs.save(crystal_feas_save_path)

            # Convert final structure back to pymatgen Structure
            final_struct = AseAtomsAdaptor.get_structure(atoms)

            # Assign magnetic moments if requested
            if assign_magmoms:
                for key in final_struct.site_properties:
                    final_struct.remove_site_property(property_name=key)
                final_struct.add_site_property(
                    "magmom", [float(magmom) for magmom in atoms.get_magnetic_moments()]
                )

            # Calculate final values
            final_energy = obs.energies[-1] if obs.energies else 0.0
            final_entropy = obs.entropies[-1] if obs.entropies else 0.0
            final_free_energy = obs.free_energies[-1] if obs.free_energies else 0.0

            # Store optimization statistics
            self.stats = pso.get_stats()

            # Return results
            return {
                "final_structure": final_struct,
                "trajectory": obs,
                "energy": final_energy,
                "entropy": final_entropy,
                "free_energy": final_free_energy,
                "optimizer": pso,
                "stats": self.stats,
                "converged": pso.converged,
                "steps": pso.iteration,
            }
        else:
            # Use standard optimization with entropy consideration
            atoms = AseAtomsAdaptor().get_atoms(structure)

            # Use custom trajectory observer if needed
            if not self.calculator_is_high_entropy:
                from ase import filters
                from ase.filters import Filter
                import sys
                import io
                import contextlib
                import inspect

                valid_filter_names = [
                    name
                    for name, cls in inspect.getmembers(filters, inspect.isclass)
                    if issubclass(cls, Filter)
                ]

                if isinstance(ase_filter, str):
                    if ase_filter in valid_filter_names:
                        ase_filter = getattr(filters, ase_filter)
                    else:
                        raise ValueError(
                            f"Invalid {ase_filter=}, must be one of {valid_filter_names}. "
                        )

                atoms.calc = self.calculator
                stream = sys.stdout if verbose else io.StringIO()
                with contextlib.redirect_stdout(stream):
                    # Use EntropyTrajectoryObserver to track entropy
                    obs = EntropyTrajectoryObserver(atoms, self.config_entropy, self.temperature)

                    if crystal_feas_save_path:
                        cry_obs = CrystalFeasObserver(atoms)

                    if relax_cell:
                        atoms = ase_filter(atoms)
                    optimizer: Optimizer = self.optimizer_class(atoms, **kwargs)
                    optimizer.attach(obs, interval=loginterval)

                    if crystal_feas_save_path:
                        optimizer.attach(cry_obs, interval=loginterval)

                    optimizer.run(fmax=fmax, steps=steps)
                    obs()

                if save_path is not None:
                    obs.save(save_path)

                if crystal_feas_save_path:
                    cry_obs.save(crystal_feas_save_path)

                if isinstance(atoms, Filter):
                    atoms = atoms.atoms
                struct = AseAtomsAdaptor.get_structure(atoms)

                if assign_magmoms:
                    for key in struct.site_properties:
                        struct.remove_site_property(property_name=key)
                    struct.add_site_property(
                        "magmom", [float(magmom) for magmom in atoms.get_magnetic_moments()]
                    )

                # Calculate final values
                final_energy = obs.energies[-1] if obs.energies else 0.0
                final_entropy = obs.entropies[-1] if obs.entropies else 0.0
                final_free_energy = obs.free_energies[-1] if obs.free_energies else 0.0

                return {
                    "final_structure": struct,
                    "trajectory": obs,
                    "energy": final_energy,
                    "entropy": final_entropy,
                    "free_energy": final_free_energy,
                    "optimizer": optimizer,
                    "converged": optimizer.converged if hasattr(optimizer, "converged") else None,
                    "steps": len(obs.energies)
                }
            else:
                # Use standard optimization when using HighEntropyCHGNetCalculator
                result = super().relax(
                    atoms=atoms,
                    fmax=fmax,
                    steps=steps,
                    relax_cell=relax_cell,
                    ase_filter=ase_filter,
                    save_path=save_path,
                    loginterval=loginterval,
                    crystal_feas_save_path=crystal_feas_save_path,
                    verbose=verbose,
                    assign_magmoms=assign_magmoms,
                    **kwargs,
                )

                # Extract the final values
                final_structure = result["final_structure"]

                # Compute energy, entropy and free energy
                energy, entropy, free_energy = self.calculate_free_energy(final_structure)

                # Add entropy and free energy to results
                result["energy"] = energy
                result["entropy"] = entropy
                result["free_energy"] = free_energy

                return result

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from the last optimization run.

        Returns:
            Dict[str, Any]: Dictionary of statistics
        """
        return self.stats

    def get_detailed_entropy(self, structure: Structure) -> Dict[str, Any]:
        """Get detailed entropy information for a structure.

        Args:
            structure (Structure): Structure to analyze

        Returns:
            Dict[str, Any]: Dictionary with detailed entropy information
        """
        return self.config_entropy.calculate_detailed(structure)