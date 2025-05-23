"""Particle Swarm Optimization for crystal structures."""

from __future__ import annotations

import random
from typing import List, Optional

import numpy as np
from ase import Atoms
from ase.optimize.optimize import Optimizer
from ase.parallel import world



class Particle:
    """Represents a particle in PSO algorithm, encoding crystal structure information."""

    def __init__(
            self,
            atoms: Atoms,
            velocity_scale: float = 0.02,
            position_scale: float = 0.05
    ):
        """
        Initialize a PSO particle

        Args:
            atoms: ASE Atoms object representing crystal structure
            velocity_scale: Initial velocity scale factor
            position_scale: Position update scale factor
        """
        self.atoms = atoms.copy()
        self.best_atoms = atoms.copy()
        self.position_scale = position_scale

        # Initialize particle positions (atom positions and cell vectors)
        self.positions = atoms.get_positions().copy()
        self.cell = atoms.get_cell().copy()

        # Initialize atom order mapping (initially sequential)
        self.atom_order = list(range(len(atoms)))
        self.best_atom_order = self.atom_order.copy()

        # Initialize velocities
        n_atoms = len(atoms)

        # Atom position velocities
        self.position_velocity = np.random.uniform(
            -velocity_scale, velocity_scale, (n_atoms, 3)
        )

        # Cell parameter velocities
        self.cell_velocity = np.random.uniform(
            -velocity_scale, velocity_scale, (3, 3)
        )

        # Energy and forces
        self.energy = float('inf')
        self.best_energy = float('inf')
        self.forces = None
        self.best_forces = None

    def update_velocity(
            self,
            global_best_atoms: Atoms,
            global_best_atom_order: List[int],
            inertia: float = 0.8,
            cognitive: float = 1.5,
            social: float = 1.5
    ):
        """
        Update particle velocity based on PSO algorithm

        Args:
            global_best_atoms: Global best structure
            global_best_atom_order: Global best atom ordering
            inertia: Inertia weight
            cognitive: Cognitive parameter
            social: Social parameter
        """
        r1, r2 = np.random.random(2)

        # Current, personal best, and global best positions
        current_positions = self.atoms.get_positions()
        best_positions = self.best_atoms.get_positions()
        global_best_positions = global_best_atoms.get_positions()

        # Apply atom ordering to positions
        current_positions = self._apply_atom_ordering(current_positions, self.atom_order)
        best_positions = self._apply_atom_ordering(best_positions, self.best_atom_order)
        global_best_positions = self._apply_atom_ordering(global_best_positions, global_best_atom_order)

        # Update position velocity
        self.position_velocity = (
                inertia * self.position_velocity +
                cognitive * r1 * (best_positions - current_positions) +
                social * r2 * (global_best_positions - current_positions)
        )

        # Current, personal best, and global best cell parameters
        current_cell = self.atoms.get_cell()
        best_cell = self.best_atoms.get_cell()
        global_best_cell = global_best_atoms.get_cell()

        # Update cell velocity
        self.cell_velocity = (
                inertia * self.cell_velocity +
                cognitive * r1 * (best_cell - current_cell) +
                social * r2 * (global_best_cell - current_cell)
        )

    def update_position(self, fix_cell: bool = True):
        """
        Update particle position (atom positions and cell parameters)

        Args:
            fix_cell: Whether to fix the cell parameters
        """
        # Get current atom positions
        current_positions = self.atoms.get_positions()

        # Apply inverse atom ordering to get original order
        ordered_positions = self._apply_inverse_ordering(current_positions, self.atom_order)

        # Update positions
        new_positions = ordered_positions + self.position_scale * self.position_velocity

        # Reapply atom ordering
        new_ordered_positions = self._apply_atom_ordering(new_positions, self.atom_order)

        # Update atom positions
        self.atoms.set_positions(new_ordered_positions)

        # Update cell parameters if not fixed
        if not fix_cell:
            current_cell = self.atoms.get_cell()
            new_cell = current_cell + self.position_scale * self.cell_velocity

            # Ensure cell is valid (non-singular)
            while np.linalg.det(new_cell) <= 0:
                # Reduce velocity and try again
                self.cell_velocity *= 0.5
                new_cell = current_cell + self.position_scale * self.cell_velocity

        # 更新晶格后添加
        if not fix_cell:
            # 检查体积变化
            old_vol = np.abs(np.linalg.det(current_cell))
            new_vol = np.abs(np.linalg.det(new_cell))
            # 如果体积变化过大，进行调整
            if new_vol < 0.5 * old_vol or new_vol > 2.0 * old_vol:
                scale = (old_vol / new_vol) ** (1 / 3)
                new_cell *= scale
                self.atoms.set_cell(new_cell, scale_atoms=True)

            # Apply new cell parameters
            self.atoms.set_cell(new_cell, scale_atoms=True)

        # 应用周期边界条件
            self.atoms.set_scaled_positions(self.atoms.get_scaled_positions() % 1.0)
        # Update atom order

    def perturb_atom_order(self, swap_probability: float = 0.3,different_elements_only=False):
        """
        Randomly swap atoms to increase exploration

        Args:
            swap_probability: Probability of atom swapping
        """
        if random.random() < swap_probability:
            n_atoms = len(self.atom_order)
            if n_atoms >= 2:
                # Randomly select two different atoms to swap
                i, j = random.sample(range(n_atoms), 2)

                if (not different_elements_only or self.atoms.get_chemical_symbols()[i] != self.atoms.get_chemical_symbols()[j]):
                    self.atom_order[i], self.atom_order[j] = self.atom_order[j], self.atom_order[i]

                # Only swap different elements (optional)
                if self.atoms.get_chemical_symbols()[i] != self.atoms.get_chemical_symbols()[j]:
                    self.atom_order[i], self.atom_order[j] = self.atom_order[j], self.atom_order[i]

    def _apply_atom_ordering(self, positions: np.ndarray, order: List[int]) -> np.ndarray:
        """Apply atom ordering to positions array"""
        result = np.zeros_like(positions)
        for i, idx in enumerate(order):
            result[i] = positions[idx]
        return result

    def _apply_inverse_ordering(self, positions: np.ndarray, order: List[int]) -> np.ndarray:
        """Apply inverse atom ordering to positions array"""
        result = np.zeros_like(positions)
        for i, idx in enumerate(order):
            result[idx] = positions[i]
        return result

    def update_best(self):
        """Update particle's personal best if current is better"""
        if self.energy < self.best_energy:
            self.best_energy = self.energy
            self.best_atoms = self.atoms.copy()
            self.best_forces = self.forces.copy() if self.forces is not None else None
            self.best_atom_order = self.atom_order.copy()


class PSO(Optimizer):
    """Particle Swarm Optimization for crystal structure optimization."""

    def __init__(
            self,
            atoms: Atoms,
            restart: Optional[str] = None,
            logfile: Optional[str] = '-',
            trajectory: Optional[str] = None,
            master: Optional[bool] = None,
            n_particles: int = 10,
            velocity_scale: float = 0.02,
            position_scale: float = 0.05,
            inertia_start: float = 0.9,
            inertia_end: float = 0.4,
            cognitive: float = 2.0,
            social: float = 2.0,
            swap_probability: float = 0.2,
            different_elements_only: bool = False,
            finalize_with_lbfgs: bool = True,
            **kwargs
    ):
        """
        Initialize PSO optimizer

        Args:
            atoms: ASE Atoms object to optimize
            restart: Filename for restart file
            logfile: File to log optimization progress
            trajectory: Trajectory filename
            master: Control log output
            n_particles: Number of particles in swarm
            velocity_scale: Scale factor for initial velocities
            position_scale: Scale factor for position updates
            inertia_start: Initial inertia weight
            inertia_end: Final inertia weight
            cognitive: Cognitive parameter weight
            social: Social parameter weight
            swap_probability: Probability of atom swapping
            finalize_with_lbfgs: Whether to refine with LBFGS at the end
            **kwargs: Additional arguments passed to parent class
        """
        # 从kwargs中提取PSO特定参数，以免传递给父类
        self.fix_cell = kwargs.pop('fix_cell', True)
        max_iterations = kwargs.pop('max_iterations', 1000)

        # 删除任何额外的不支持的参数以避免传递给父类
        # 这些是用户代码中可能出现的参数
        for param in ['entropy_weight', 'use_device', 'check_cuda_mem',
                      'stress_weight', 'on_isolated_atoms', 'return_site_energies']:
            if param in kwargs:
                kwargs.pop(param)

        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master, **kwargs)

        # PSO parameters
        self.n_particles = n_particles
        self.velocity_scale = velocity_scale
        self.position_scale = position_scale
        self.inertia_start = inertia_start
        self.inertia_end = inertia_end
        self.cognitive = cognitive
        self.social = social
        self.swap_probability = swap_probability
        self.different_elements_only = different_elements_only
        self.finalize_with_lbfgs = finalize_with_lbfgs
        # 设置最大迭代次数
        self.max_iterations = max_iterations
        # self.entropy_weight = entropy_weight,  # 添加熵权重参数



        # Initialize particles
        self.particles = [
            Particle(
                atoms,
                velocity_scale=velocity_scale,
                position_scale=position_scale
            )
            for _ in range(n_particles)
        ]

        # Global best
        self.global_best_atoms = atoms.copy()
        self.global_best_atom_order = list(range(len(atoms)))
        self.global_best_energy = float('inf')
        self.global_best_forces = None

        # Iteration counter
        self.iteration = 0
        # self.max_iterations = 1000  # Can be set by caller

        # Initialize first particle
        self.calculate_energy_forces(self.particles[0])

    def step(self, f=None):
        """Take a step in the optimization process"""
        if self.iteration == 0:
            # First iteration: evaluate all particles
            for i in range(1, self.n_particles):  # Skip first particle (already calculated)
                self.calculate_energy_forces(self.particles[i])
                self.particles[i].update_best()

                # Update global best
                if self.particles[i].energy < self.global_best_energy:
                    self.update_global_best(self.particles[i])
        else:
            # Calculate inertia weight (linearly decreasing)
            progress = min(1.0, self.iteration / self.max_iterations)
            inertia = self.inertia_start - (self.inertia_start - self.inertia_end) * progress

            # Update all particles
            for particle in self.particles:
                # Update velocity and position
                particle.update_velocity(
                    self.global_best_atoms,
                    self.global_best_atom_order,
                    inertia=inertia,
                    cognitive=self.cognitive,
                    social=self.social
                )
                particle.update_position(fix_cell=self.fix_cell)

                # Perturb atom order
                particle.perturb_atom_order(swap_probability=self.swap_probability,different_elements_only=self.different_elements_only)

                # Evaluate energy and forces
                self.calculate_energy_forces(particle)

                # Update personal best
                particle.update_best()

                # Update global best
                if particle.energy < self.global_best_energy:
                    self.update_global_best(particle)

        # Update optimizer atoms with global best
        self.atoms.set_positions(self.global_best_atoms.get_positions())
        if not self.fix_cell:
            self.atoms.set_cell(self.global_best_atoms.get_cell(), scale_atoms=True)

        # Increment iteration counter
        self.iteration += 1

        # Check if we need to finalize with LBFGS
        max_force = np.sqrt((self.global_best_forces ** 2).sum(axis=1).max())
        converged = max_force < self.fmax

        if converged and self.finalize_with_lbfgs and self.iteration > 1:
            self.finalize_with_local_optimizer()
            return True

        return converged

    def calculate_energy_forces(self, particle: Particle):
        """Calculate energy and forces for a particle"""
        # Set calculator
        particle.atoms.calc = self.atoms.calc

        # Calculate energy and forces
        try:
            particle.energy = particle.atoms.get_potential_energy()
            particle.forces = particle.atoms.get_forces()
        except Exception as e:
            # Handle calculation errors
            particle.energy = float('inf')
            particle.forces = np.zeros((len(particle.atoms), 3))
            if world.rank == 0:
                print(f"Warning: Energy calculation failed: {e}")

    def update_global_best(self, particle: Particle):
        """Update global best from particle"""
        self.global_best_energy = particle.energy
        self.global_best_atoms = particle.atoms.copy()
        self.global_best_forces = particle.forces.copy() if particle.forces is not None else None
        self.global_best_atom_order = particle.atom_order.copy()

        # Call all registered observers
        self.call_observers()

    def finalize_with_local_optimizer(self):
        """Finalize optimization with LBFGS for fine-tuning"""
        from ase.optimize import LBFGS

        if world.rank == 0:
            print("\nPSO optimization converged. Refining with LBFGS...\n")

        # Create LBFGS optimizer with global best structure
        self.atoms.set_positions(self.global_best_atoms.get_positions())
        if not self.fix_cell:
            self.atoms.set_cell(self.global_best_atoms.get_cell(), scale_atoms=True)

        # Run LBFGS optimization
        lbfgs = LBFGS(
            self.atoms,
            logfile=self.logfile,
            trajectory=self.trajectory
        )
        lbfgs.run(fmax=self.fmax, steps=min(50, self.max_iterations // 5))