#!/usr/bin/env python
"""
CHGNet High Entropy Crystal Structure Optimization Demo Script

This script demonstrates how to use the high-entropy crystal structure optimizer
to optimize crystal structures with configurational entropy considerations.

Usage:
    python demo.py [--temperature 300] [--use_pso true] [--max_steps 50]

The script reads CIF files from the test_data directory, optimizes them,
and saves the optimized structures to test_data/output with "optimized_" prefix.
"""

import os
import argparse
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser, CifWriter

from chgnet.model.model import CHGNet
from chgnet.model.high_entropy_optimizer import HighEntropyOptimizer, ConfigEntropy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("high_entropy_demo")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CHGNet High Entropy Crystal Structure Optimization Demo"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Temperature in K for entropy contribution"
    )
    parser.add_argument(
        "--use_pso",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to use PSO optimization (true/false)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="Maximum optimization steps"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="test_data",
        help="Directory containing input CIF files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_data/output",
        help="Directory to save optimized structures"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cpu, cuda, or mps)"
    )

    return parser.parse_args()


def find_cif_files(directory: str) -> List[str]:
    """Find all CIF files in the given directory.

    Args:
        directory (str): Directory to search

    Returns:
        List[str]: List of paths to CIF files
    """
    cif_files = []
    for file in os.listdir(directory):
        if file.lower().endswith('.cif'):
            cif_files.append(os.path.join(directory, file))
    return cif_files


def load_structure(cif_path: str) -> Optional[Structure]:
    """Load structure from CIF file.

    Args:
        cif_path (str): Path to CIF file

    Returns:
        Optional[Structure]: Loaded structure or None if failed
    """
    try:
        parser = CifParser(cif_path)
        structures = parser.get_structures()
        if structures:
            logger.info(f"Successfully loaded structure from {cif_path}")
            return structures[0]
        else:
            logger.error(f"No structures found in {cif_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading structure from {cif_path}: {e}")
        return None


def optimize_structure(
        structure: Structure,
        temperature: float,
        use_pso: bool,
        max_steps: int,
        device: Optional[str] = None
) -> Dict[str, Any]:
    """Optimize structure using HighEntropyOptimizer.

    Args:
        structure (Structure): Input structure
        temperature (float): Temperature in K
        use_pso (bool): Whether to use PSO
        max_steps (int): Maximum optimization steps
        device (str, optional): Device to use

    Returns:
        Dict[str, Any]: Optimization results
    """
    # Load CHGNet model
    logger.info("Loading CHGNet model...")
    model = CHGNet.load(use_device=device)

    # Create optimizer
    logger.info(f"Creating optimizer with temperature={temperature}K, use_pso={use_pso}")
    optimizer = HighEntropyOptimizer(
        model=model,
        temperature=temperature,
        use_pso=use_pso,
        pso_particles=20 if use_pso else 0,
        pso_iterations=max_steps if use_pso else 0,
        use_device=device
    )

    # Calculate initial entropy
    config_entropy = ConfigEntropy()
    initial_entropy_info = config_entropy.calculate_detailed(structure)

    # Perform optimization
    start_time = time.time()
    logger.info(f"Starting optimization with max_steps={max_steps}")
    result = optimizer.relax(
        structure,
        temperature=temperature,
        steps=max_steps,
        fmax=0.1,
        relax_cell=True,
        verbose=True
    )

    # Calculate final entropy
    final_structure = result["final_structure"]
    final_entropy_info = config_entropy.calculate_detailed(final_structure)

    elapsed_time = time.time() - start_time
    logger.info(f"Optimization completed in {elapsed_time:.2f} seconds")

    # Return results with initial and final entropy information
    return {
        "result": result,
        "initial_entropy_info": initial_entropy_info,
        "final_entropy_info": final_entropy_info,
        "elapsed_time": elapsed_time
    }


def save_structure(structure: Structure, output_path: str) -> None:
    """Save structure to CIF file.

    Args:
        structure (Structure): Structure to save
        output_path (str): Output file path
    """
    try:
        writer = CifWriter(structure, symprec=0.1)
        writer.write_file(output_path)
        logger.info(f"Saved structure to {output_path}")
    except Exception as e:
        logger.error(f"Error saving structure to {output_path}: {e}")


def print_optimization_summary(
        filename: str,
        initial_structure: Structure,
        optimization_results: Dict[str, Any]
) -> None:
    """Print a summary of the optimization.

    Args:
        filename (str): Original filename
        initial_structure (Structure): Initial structure
        optimization_results (Dict[str, Any]): Optimization results
    """
    result = optimization_results["result"]
    initial_entropy_info = optimization_results["initial_entropy_info"]
    final_entropy_info = optimization_results["final_entropy_info"]
    elapsed_time = optimization_results["elapsed_time"]
    final_structure = result["final_structure"]

    print("\n" + "=" * 70)
    print(f"Optimization Summary for: {filename}")
    print("=" * 70)

    # Structure information
    print(f"Formula: {initial_structure.composition.reduced_formula}")
    print(f"Initial structure: {len(initial_structure)} atoms")
    print(f"Final structure:   {len(final_structure)} atoms")

    # Volume change
    initial_volume = initial_structure.volume
    final_volume = final_structure.volume
    volume_change = (final_volume - initial_volume) / initial_volume * 100
    print(f"Volume change: {volume_change:.2f}% (from {initial_volume:.2f} Å³ to {final_volume:.2f} Å³)")

    # Entropy information
    initial_entropy = initial_entropy_info["entropy"]
    final_entropy = final_entropy_info["entropy"]
    entropy_change = final_entropy - initial_entropy
    print(f"Initial entropy: {initial_entropy:.6f} eV/atom")
    print(f"Final entropy:   {final_entropy:.6f} eV/atom")
    print(f"Entropy change:  {entropy_change:.6f} eV/atom ({entropy_change / initial_entropy * 100:.2f}%)")

    # Energy information
    if "energy" in result:
        print(f"Final energy:       {result['energy']:.6f} eV")
        print(f"Final free energy:  {result['free_energy']:.6f} eV")

    # Convergence info
    if "converged" in result:
        print(f"Converged: {result['converged']}")

    print(f"Optimization completed in {elapsed_time:.2f} seconds ({result.get('steps', 'N/A')} steps)")
    print("=" * 70 + "\n")


def main() -> None:
    """Main function."""
    args = parse_arguments()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Find CIF files in input directory
    cif_files = find_cif_files(args.input_dir)
    if not cif_files:
        logger.error(f"No CIF files found in {args.input_dir}")
        return

    logger.info(f"Found {len(cif_files)} CIF files to optimize")

    # Process each CIF file
    for cif_path in cif_files:
        filename = os.path.basename(cif_path)
        logger.info(f"Processing {filename}...")

        # Load structure
        structure = load_structure(cif_path)
        if structure is None:
            continue

        # Optimize structure
        try:
            optimization_results = optimize_structure(
                structure,
                args.temperature,
                args.use_pso,
                args.max_steps,
                args.device
            )

            # Save optimized structure
            output_path = os.path.join(args.output_dir, f"optimized_{filename}")
            save_structure(optimization_results["result"]["final_structure"], output_path)

            # Print summary
            print_optimization_summary(filename, structure, optimization_results)

        except Exception as e:
            logger.error(f"Error optimizing {filename}: {e}")
            continue

    logger.info("All structures processed")


if __name__ == "__main__":
    main()