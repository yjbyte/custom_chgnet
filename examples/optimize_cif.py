#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
High-entropy crystal optimization example using CHGNet with PSO.
Demonstrates entropy-guided structure optimization for multi-component alloys.
"""

import os
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.composition import Composition
from chgnet.model import StructOptimizer
import matplotlib.pyplot as plt


def create_high_entropy_structure(
        lattice_param: float = 3.6,
        elements: list = ["Fe", "Co", "Ni", "Cr", "Mn"],
        structure_type: str = "fcc",
        supercell: tuple = (2, 2, 2),
) -> Structure:
    """Create a high-entropy alloy structure with random element distribution.

    Args:
        lattice_param: Lattice parameter in Angstroms
        elements: List of element symbols
        structure_type: Crystal structure type ("fcc", "bcc")
        supercell: Supercell dimensions

    Returns:
        Structure object with randomly distributed elements
    """
    if structure_type == "fcc":
        # FCC lattice
        lattice = Lattice.cubic(lattice_param)
        coords = [[0.0, 0.0, 0.0]]
    elif structure_type == "bcc":
        # BCC lattice
        lattice = Lattice.cubic(lattice_param)
        coords = [[0.0, 0.0, 0.0]]
    else:
        raise ValueError("Unsupported structure type")

    # Create primitive structure
    structure = Structure(lattice, elements[:1], coords)

    # Make supercell
    structure.make_supercell(supercell)

    # Randomly distribute elements
    n_sites = len(structure)
    n_elements = len(elements)

    # Create roughly equal distribution
    sites_per_element = n_sites // n_elements
    element_list = []

    for i, element in enumerate(elements):
        if i == len(elements) - 1:
            # Last element gets remaining sites
            element_list.extend([element] * (n_sites - len(element_list)))
        else:
            element_list.extend([element] * sites_per_element)

    # Shuffle for random distribution
    np.random.shuffle(element_list)

    # Replace species
    for i, element in enumerate(element_list):
        structure[i] = element

    return structure


def optimize_high_entropy_crystal(
        structure: Structure,
        temperature: float = 300.0,
        entropy_weight: float = 1.0,
        steps: int = 100,
        swarm_size: int = 25,
) -> dict:
    """Optimize high-entropy crystal structure.

    Args:
        structure: Initial structure
        temperature: Temperature for entropy calculation (K)
        entropy_weight: Weight for entropy term
        steps: Maximum optimization steps
        swarm_size: PSO swarm size

    Returns:
        Optimization results with entropy information
    """
    print(f"Optimizing high-entropy structure: {structure.composition.formula}")
    print(f"Temperature: {temperature} K")
    print(f"Entropy weight: {entropy_weight}")

    # PSO parameters optimized for high-entropy systems
    pso_params = {
        "swarm_size": swarm_size,
        "w": 0.6,  # Lower inertia for better exploration
        "c1": 1.5,  # Higher cognitive parameter
        "c2": 1.5,  # Higher social parameter
        "max_velocity": 0.2,  # Smaller steps for stability
    }

    # High-entropy specific parameters
    high_entropy_params = {
        "enable_entropy_optimization": True,
        "entropy_weight": entropy_weight,
        "enable_atomic_ordering": True,
        "ordering_probability": 0.15,  # Higher swap probability
        "temperature": temperature,
    }

    # Initialize optimizer
    relaxer = StructOptimizer(
        optimizer_class="PSO",
        pso_params=pso_params,
        high_entropy_params=high_entropy_params,
    )

    # Run optimization
    result = relaxer.relax_high_entropy(
        structure,
        fmax=0.1,
        steps=steps,
        temperature=temperature,
        entropy_weight=entropy_weight,
        enable_atomic_ordering=True,
        save_entropy_trajectory=True,
        relax_cell=False,  # Keep cell fixed for alloy optimization
    )

    return result


def analyze_optimization_results(result: dict) -> None:
    """Analyze and plot optimization results.

    Args:
        result: Optimization result dictionary
    """
    trajectory = result["trajectory"]
    entropy_info = result["entropy_info"]

    print("\n=== Optimization Results ===")
    print(f"Final energy: {trajectory.energies[-1]:.6f} eV")
    print(f"Initial energy: {trajectory.energies[0]:.6f} eV")
    print(f"Energy change: {trajectory.energies[-1] - trajectory.energies[0]:.6f} eV")
    print(f"Optimization steps: {len(trajectory.energies)}")

    if entropy_info.get("entropy_optimization", False):
        print(f"\n=== Entropy Information ===")
        print(f"Configurational entropy: {entropy_info['configurational_entropy_per_atom']:.6f} eV/K per atom")
        print(f"Number of elements: {entropy_info['n_elements']}")
        print(f"Composition: {entropy_info['composition']}")
        print(f"Entropy contribution to fitness: {entropy_info['entropy_contribution_to_fitness']:.6f} eV")

    # Plot energy trajectory
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(trajectory.energies)
    plt.xlabel('Optimization Step')
    plt.ylabel('Energy (eV)')
    plt.title('Energy Evolution')
    plt.grid(True)

    # Plot force convergence
    plt.subplot(1, 2, 2)
    force_magnitudes = [np.linalg.norm(f, axis=1).max() for f in trajectory.forces]
    plt.plot(force_magnitudes)
    plt.xlabel('Optimization Step')
    plt.ylabel('Max Force (eV/Å)')
    plt.title('Force Convergence')
    plt.grid(True)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('high_entropy_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function demonstrating high-entropy crystal optimization."""
    print("=== High-Entropy Crystal Optimization Demo ===")

    # Create test high-entropy alloy structures
    elements_sets = [
        ["Fe", "Co", "Ni", "Cr", "Mn"],  # Cantor alloy
        ["Al", "Co", "Cr", "Fe", "Ni"],  # AlCoCrFeNi
        ["Ti", "Zr", "Hf", "Nb", "Ta"],  # Refractory HEA
    ]

    temperatures = [300.0, 600.0, 1000.0]
    entropy_weights = [0.5, 1.0, 2.0]

    results = []

    for i, elements in enumerate(elements_sets):
        print(f"\n--- Testing {'-'.join(elements)} system ---")

        # Create structure
        structure = create_high_entropy_structure(
            lattice_param=3.6,
            elements=elements,
            structure_type="fcc",
            supercell=(2, 2, 2)
        )

        # Test different temperatures and entropy weights
        for temp in temperatures[:1]:  # Use only first temperature for demo
            for ent_weight in entropy_weights[:1]:  # Use only first weight for demo
                print(f"\nOptimizing at T={temp}K, entropy_weight={ent_weight}")

                try:
                    result = optimize_high_entropy_crystal(
                        structure,
                        temperature=temp,
                        entropy_weight=ent_weight,
                        steps=30,  # Reduced for demo
                        swarm_size=15,  # Reduced for demo
                    )

                    results.append({
                        "elements": elements,
                        "temperature": temp,
                        "entropy_weight": ent_weight,
                        "result": result
                    })

                    # Analyze first result in detail
                    if len(results) == 1:
                        analyze_optimization_results(result)

                except Exception as e:
                    print(f"Optimization failed: {e}")

                break  # Only do one entropy weight for demo
            break  # Only do one temperature for demo

    # Summary comparison
    print("\n=== Summary Comparison ===")
    for i, res_data in enumerate(results):
        result = res_data["result"]
        trajectory = result["trajectory"]
        entropy_info = result["entropy_info"]

        print(f"System {i + 1}: {'-'.join(res_data['elements'])}")
        print(f"  Final energy: {trajectory.energies[-1]:.4f} eV")
        if entropy_info.get("entropy_optimization", False):
            print(f"  Config entropy: {entropy_info['configurational_entropy_per_atom']:.6f} eV/K")
            print(f"  Elements: {entropy_info['n_elements']}")
        print()


if __name__ == "__main__":
    main()