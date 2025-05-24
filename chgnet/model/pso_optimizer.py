"""Enhanced Particle Swarm Optimization for crystal structures."""

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
            velocity_scale: float = 0.1,
            position_scale: float = 0.15,
            initial_perturbation: bool = True
    ):
        self.atoms = atoms.copy()
        self.best_atoms = atoms.copy()
        self.position_scale = position_scale

        # 添加初始扰动以增加粒子多样性
        if initial_perturbation:
            positions = self.atoms.get_positions()
            # 添加更大的随机扰动
            perturbation = np.random.normal(0, 0.2, positions.shape)
            self.atoms.set_positions(positions + perturbation)

            # 随机交换原子位置增加多样性
            if len(atoms) > 1:
                n_swaps = max(1, len(atoms) // 8)
                for _ in range(n_swaps):
                    i, j = random.sample(range(len(atoms)), 2)
                    symbols = self.atoms.get_chemical_symbols()
                    # 优先交换不同元素，增加构型熵
                    if symbols[i] != symbols[j]:
                        pos = self.atoms.get_positions()
                        pos[i], pos[j] = pos[j].copy(), pos[i].copy()
                        self.atoms.set_positions(pos)

        # Initialize velocities
        n_atoms = len(atoms)
        self.position_velocity = np.random.normal(0, velocity_scale, (n_atoms, 3))
        self.cell_velocity = np.random.normal(0, velocity_scale * 0.1, (3, 3))

        # Energy and forces
        self.energy = float('inf')
        self.best_energy = float('inf')
        self.forces = None
        self.best_forces = None

    def update_velocity(
            self,
            global_best_atoms: Atoms,
            inertia: float = 0.8,
            cognitive: float = 1.5,
            social: float = 1.5
    ):
        """Update particle velocity based on PSO algorithm"""
        r1, r2 = np.random.random(2)

        current_positions = self.atoms.get_positions()
        best_positions = self.best_atoms.get_positions()
        global_best_positions = global_best_atoms.get_positions()

        # Update position velocity
        self.position_velocity = (
                inertia * self.position_velocity +
                cognitive * r1 * (best_positions - current_positions) +
                social * r2 * (global_best_positions - current_positions)
        )

        # 限制速度大小避免过大跳跃
        max_velocity = 0.5
        self.position_velocity = np.clip(self.position_velocity, -max_velocity, max_velocity)

    def update_position(self, fix_cell: bool = True):
        """Update particle position"""
        current_positions = self.atoms.get_positions()
        new_positions = current_positions + self.position_scale * self.position_velocity
        self.atoms.set_positions(new_positions)

        # Apply periodic boundary conditions
        scaled_positions = self.atoms.get_scaled_positions()
        scaled_positions = scaled_positions % 1.0
        self.atoms.set_scaled_positions(scaled_positions)

    def perturb_atom_order(self, swap_probability: float = 0.3, different_elements_only: bool = True):
        """Enhanced atom swapping for high-entropy alloys"""
        if random.random() < swap_probability:
            n_atoms = len(self.atoms)
            if n_atoms >= 2:
                symbols = self.atoms.get_chemical_symbols()

                if different_elements_only:
                    # 专门针对高熵合金：优先交换不同元素
                    different_pairs = []
                    for i in range(n_atoms):
                        for j in range(i + 1, n_atoms):
                            if symbols[i] != symbols[j]:
                                different_pairs.append((i, j))

                    if different_pairs:
                        # 多次交换增加构型熵
                        n_swaps = min(3, len(different_pairs))
                        for _ in range(n_swaps):
                            i, j = random.choice(different_pairs)
                            positions = self.atoms.get_positions()
                            positions[i], positions[j] = positions[j].copy(), positions[i].copy()
                            self.atoms.set_positions(positions)
                            # 移除已交换的对，避免重复
                            different_pairs.remove((i, j))
                            if not different_pairs:
                                break
                else:
                    # 随机交换
                    i, j = random.sample(range(n_atoms), 2)
                    positions = self.atoms.get_positions()
                    positions[i], positions[j] = positions[j].copy(), positions[i].copy()
                    self.atoms.set_positions(positions)

    def update_best(self):
        """Update particle's personal best if current is better"""
        if self.energy < self.best_energy:
            self.best_energy = self.energy
            self.best_atoms = self.atoms.copy()
            self.best_forces = self.forces.copy() if self.forces is not None else None


class PSO(Optimizer):
    """Enhanced Particle Swarm Optimization for crystal structure optimization."""

    def __init__(
            self,
            atoms: Atoms,
            restart: Optional[str] = None,
            logfile: Optional[str] = '-',
            trajectory: Optional[str] = None,
            master: Optional[bool] = None,
            n_particles: int = 20,
            velocity_scale: float = 0.1,
            position_scale: float = 0.15,
            inertia_start: float = 0.9,
            inertia_end: float = 0.4,
            cognitive: float = 2.0,
            social: float = 2.0,
            swap_probability: float = 0.3,
            different_elements_only: bool = True,
            finalize_with_lbfgs: bool = True,
            diversity_threshold: float = 1e-6,
            **kwargs
    ):
        # 从kwargs中提取PSO特定参数
        self.fix_cell = kwargs.pop('fix_cell', True)
        max_iterations = kwargs.pop('max_iterations', 1000)

        # 删除不支持的参数
        for param in ['entropy_weight', 'use_device', 'check_cuda_mem',
                      'stress_weight', 'on_isolated_atoms', 'return_site_energies']:
            kwargs.pop(param, None)

        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master, **kwargs)

        # PSO parameters - 现在都是可调节的
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
        self.max_iterations = max_iterations
        self.diversity_threshold = diversity_threshold

        # 输出参数信息，方便调试
        if world.rank == 0:
            print(f"PSO Parameters:")
            print(f"  n_particles: {self.n_particles}")
            print(f"  velocity_scale: {self.velocity_scale}")
            print(f"  position_scale: {self.position_scale}")
            print(f"  swap_probability: {self.swap_probability}")
            print(f"  different_elements_only: {self.different_elements_only}")

        # Initialize particles with diversity
        self.particles = []
        for i in range(n_particles):
            particle = Particle(
                atoms,
                velocity_scale=velocity_scale,
                position_scale=position_scale,
                initial_perturbation=(i > 0)
            )
            self.particles.append(particle)

        # Global best
        self.global_best_atoms = atoms.copy()
        self.global_best_energy = float('inf')
        self.global_best_forces = None

        # Iteration tracking
        self.iteration = 0
        self.stagnation_counter = 0
        self.last_best_energy = float('inf')

        # Initialize first particle
        self.calculate_energy_forces(self.particles[0])

    def step(self, f=None):
        """Take a step in the optimization process"""
        if self.iteration == 0:
            # First iteration: evaluate all particles
            for i in range(1, self.n_particles):
                self.calculate_energy_forces(self.particles[i])
                self.particles[i].update_best()

                if self.particles[i].energy < self.global_best_energy:
                    self.update_global_best(self.particles[i])

            # Update first particle
            self.particles[0].update_best()
            if self.particles[0].energy < self.global_best_energy:
                self.update_global_best(self.particles[0])

        else:
            # Calculate adaptive inertia weight
            progress = min(1.0, self.iteration / self.max_iterations)
            inertia = self.inertia_start - (self.inertia_start - self.inertia_end) * progress

            # Check for stagnation
            energy_improvement = abs(self.last_best_energy - self.global_best_energy)
            if energy_improvement < self.diversity_threshold:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

            # Add diversity if stagnated
            if self.stagnation_counter > 5:
                self.add_diversity()
                self.stagnation_counter = 0

            # Update all particles
            for particle in self.particles:
                particle.update_velocity(
                    self.global_best_atoms,
                    inertia=inertia,
                    cognitive=self.cognitive,
                    social=self.social
                )
                particle.update_position(fix_cell=self.fix_cell)

                # Enhanced atom swapping for high-entropy alloys
                particle.perturb_atom_order(
                    swap_probability=self.swap_probability,
                    different_elements_only=self.different_elements_only
                )

                # Evaluate energy and forces
                self.calculate_energy_forces(particle)
                particle.update_best()

                # Update global best
                if particle.energy < self.global_best_energy:
                    self.update_global_best(particle)

        # Store last best energy
        self.last_best_energy = self.global_best_energy

        # Update optimizer atoms with global best
        self.atoms.set_positions(self.global_best_atoms.get_positions())
        if not self.fix_cell:
            self.atoms.set_cell(self.global_best_atoms.get_cell(), scale_atoms=True)

        self.iteration += 1

        # Check convergence
        if self.global_best_forces is not None:
            max_force = np.sqrt((self.global_best_forces ** 2).sum(axis=1).max())
            converged = max_force < self.fmax

            if converged and self.finalize_with_lbfgs and self.iteration > 1:
                self.finalize_with_local_optimizer()
                return True

            return converged

        return False

    def add_diversity(self):
        """Add diversity when optimization stagnates"""
        if world.rank == 0:
            print(f"Adding diversity at iteration {self.iteration}")

        # 重新初始化一半的粒子
        n_reinit = self.n_particles // 2
        for i in range(n_reinit):
            particle = self.particles[i + n_reinit]

            # 从最佳结构开始添加更大扰动
            particle.atoms = self.global_best_atoms.copy()
            positions = particle.atoms.get_positions()

            # 更大的扰动
            perturbation = np.random.normal(0, 0.4, positions.shape)
            particle.atoms.set_positions(positions + perturbation)

            # 重新初始化速度
            n_atoms = len(particle.atoms)
            particle.position_velocity = np.random.normal(0, self.velocity_scale * 2, (n_atoms, 3))

            # 重新评估
            self.calculate_energy_forces(particle)

    def calculate_energy_forces(self, particle: Particle):
        """Calculate energy and forces for a particle"""
        particle.atoms.calc = self.atoms.calc

        try:
            particle.energy = particle.atoms.get_potential_energy()
            particle.forces = particle.atoms.get_forces()
        except Exception as e:
            particle.energy = float('inf')
            particle.forces = np.zeros((len(particle.atoms), 3))
            if world.rank == 0:
                print(f"Warning: Energy calculation failed: {e}")

    def update_global_best(self, particle: Particle):
        """Update global best from particle"""
        self.global_best_energy = particle.energy
        self.global_best_atoms = particle.atoms.copy()
        self.global_best_forces = particle.forces.copy() if particle.forces is not None else None

        # Call all registered observers
        self.call_observers()

    def finalize_with_local_optimizer(self):
        """Finalize optimization with LBFGS for fine-tuning"""
        from ase.optimize import LBFGS

        if world.rank == 0:
            print("\nPSO optimization converged. Refining with LBFGS...\n")

        self.atoms.set_positions(self.global_best_atoms.get_positions())
        if not self.fix_cell:
            self.atoms.set_cell(self.global_best_atoms.get_cell(), scale_atoms=True)

        lbfgs = LBFGS(
            self.atoms,
            logfile=self.logfile,
            trajectory=self.trajectory
        )
        lbfgs.run(fmax=self.fmax, steps=min(50, self.max_iterations // 5))