"""
高熵结构优化 - PSO优化模块

这个模块包含了粒子群优化算法用于高熵合金结构优化的核心组件
"""

from chgnet.optimization.particle import PSOParticle
from chgnet.optimization.population import initialize_population
from chgnet.optimization.structure_ops import generate_structure_from_particle_stage1
from chgnet.optimization.entropy import calculate_configurational_entropy
from chgnet.optimization.fitness import calculate_fitness_stage1

__all__ = [
    "PSOParticle",
    "initialize_population", 
    "generate_structure_from_particle_stage1",
    "calculate_configurational_entropy",
    "calculate_fitness_stage1"
]