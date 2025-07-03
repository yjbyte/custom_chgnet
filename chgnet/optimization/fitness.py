"""
适应度函数模块

实现阶段一PSO的适应度评估功能，目标函数为 F = E_CHGNet - T * S_config
"""

from __future__ import annotations
from typing import Dict, Optional, Union
import numpy as np
from pymatgen.core import Structure
from chgnet.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
from chgnet.optimization.particle import PSOParticle
from chgnet.optimization.structure_ops import generate_structure_from_particle_stage1
from chgnet.optimization.entropy import calculate_configurational_entropy


def calculate_fitness_stage1(
    structure: Structure,
    T: float,
    site_groups_definition: Dict[str, Dict],
    chgnet_calculator: CHGNetCalculator,
    particle: Optional[PSOParticle] = None
) -> float:
    """
    计算阶段一适应度函数值：F = E_CHGNet - T * S_config
    
    Args:
        structure (Structure): 由PSO粒子生成的当前晶体结构
        T (float): 温度超参数 (K)
        site_groups_definition (Dict[str, Dict]): 位点群定义
        chgnet_calculator (CHGNetCalculator): CHGNet计算器实例
        particle (PSOParticle, optional): 对应的PSO粒子，用于更高效的熵计算
        
    Returns:
        float: 适应度值 F (eV)
        
    Note:
        更低的适应度值表示更好的配置
        能量单位：CHGNet输出eV
        熵单位：使用k_B单位，通过温度参数T平衡
    """
    try:
        # 计算CHGNet能量
        energy_chgnet = _calculate_chgnet_energy(structure, chgnet_calculator)
        
        # 计算配置熵
        s_config = calculate_configurational_entropy(
            particle=particle,
            structure=structure,
            site_groups_definition=site_groups_definition
        )
        
        # 计算适应度：F = E_CHGNet - T * S_config
        # T * S_config 项会将熵的单位从 eV/K 转换为 eV
        fitness = energy_chgnet - T * s_config
        
        return fitness
        
    except Exception as e:
        # 如果计算失败，返回一个很大的适应度值（表示不可行解）
        print(f"适应度计算失败: {e}")
        return float('inf')


def _calculate_chgnet_energy(
    structure: Structure,
    chgnet_calculator: CHGNetCalculator
) -> float:
    """
    使用CHGNet计算结构能量
    
    Args:
        structure (Structure): 晶体结构
        chgnet_calculator (CHGNetCalculator): CHGNet计算器
        
    Returns:
        float: 结构总能量 (eV)
    """
    # 将pymatgen Structure转换为ASE Atoms
    from pymatgen.io.ase import AseAtomsAdaptor
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(structure)
    
    # 设置计算器
    atoms.set_calculator(chgnet_calculator)
    
    # 计算能量
    energy = atoms.get_potential_energy()
    
    return energy


def evaluate_particle_fitness(
    particle: PSOParticle,
    base_structure: Structure,
    site_groups_definition: Dict[str, Dict],
    T: float,
    chgnet_calculator: CHGNetCalculator
) -> float:
    """
    评估单个PSO粒子的适应度
    
    Args:
        particle (PSOParticle): PSO粒子
        base_structure (Structure): 基础结构
        site_groups_definition (Dict[str, Dict]): 位点群定义
        T (float): 温度参数
        chgnet_calculator (CHGNetCalculator): CHGNet计算器
        
    Returns:
        float: 粒子适应度值
    """
    # 生成对应的晶体结构
    structure = generate_structure_from_particle_stage1(
        base_structure, particle, site_groups_definition
    )
    
    # 计算适应度
    fitness = calculate_fitness_stage1(
        structure, T, site_groups_definition, chgnet_calculator, particle
    )
    
    # 更新粒子适应度
    particle.fitness = fitness
    particle.update_best_position()
    
    return fitness


def evaluate_population_fitness(
    population: list[PSOParticle],
    base_structure: Structure,
    site_groups_definition: Dict[str, Dict],
    T: float,
    chgnet_calculator: CHGNetCalculator,
    parallel: bool = False
) -> list[float]:
    """
    评估整个种群的适应度
    
    Args:
        population (List[PSOParticle]): 粒子种群
        base_structure (Structure): 基础结构
        site_groups_definition (Dict[str, Dict]): 位点群定义
        T (float): 温度参数
        chgnet_calculator (CHGNetCalculator): CHGNet计算器
        parallel (bool): 是否使用并行计算
        
    Returns:
        List[float]: 种群适应度值列表
    """
    fitness_values = []
    
    if parallel:
        # TODO: 实现并行计算版本
        # 可以使用multiprocessing或joblib来并行化适应度计算
        pass
    
    # 串行计算版本
    for particle in population:
        fitness = evaluate_particle_fitness(
            particle, base_structure, site_groups_definition, T, chgnet_calculator
        )
        fitness_values.append(fitness)
    
    return fitness_values


def get_fitness_statistics(population: list[PSOParticle]) -> Dict[str, float]:
    """
    获取种群适应度统计信息
    
    Args:
        population (List[PSOParticle]): 粒子种群
        
    Returns:
        Dict[str, float]: 统计信息字典
    """
    fitness_values = [p.fitness for p in population if p.fitness != float('inf')]
    
    if not fitness_values:
        return {
            "min_fitness": float('inf'),
            "max_fitness": float('inf'),
            "mean_fitness": float('inf'),
            "std_fitness": 0.0,
            "valid_particles": 0
        }
    
    return {
        "min_fitness": min(fitness_values),
        "max_fitness": max(fitness_values),
        "mean_fitness": np.mean(fitness_values),
        "std_fitness": np.std(fitness_values),
        "valid_particles": len(fitness_values),
        "total_particles": len(population)
    }


def create_chgnet_calculator(
    model_path: Optional[str] = None,
    device: str = "cpu"
) -> CHGNetCalculator:
    """
    创建CHGNet计算器实例
    
    Args:
        model_path (str, optional): 预训练模型路径，None则使用默认模型
        device (str): 计算设备 ("cpu" 或 "cuda")
        
    Returns:
        CHGNetCalculator: CHGNet计算器实例
    """
    if model_path is None:
        # 使用预训练模型
        model = CHGNet.load()
    else:
        # 加载指定路径的模型
        model = CHGNet.load(path=model_path)
    
    calculator = CHGNetCalculator(model=model, use_device=device)
    return calculator


def validate_fitness_calculation(
    structure: Structure,
    T: float,
    site_groups_definition: Dict[str, Dict],
    chgnet_calculator: CHGNetCalculator
) -> Dict[str, Union[float, bool]]:
    """
    验证适应度计算的合理性
    
    Args:
        structure (Structure): 测试结构
        T (float): 温度参数
        site_groups_definition (Dict[str, Dict]): 位点群定义
        chgnet_calculator (CHGNetCalculator): CHGNet计算器
        
    Returns:
        Dict[str, Union[float, bool]]: 验证结果
    """
    try:
        # 计算各个组件
        energy = _calculate_chgnet_energy(structure, chgnet_calculator)
        entropy = calculate_configurational_entropy(
            structure=structure,
            site_groups_definition=site_groups_definition
        )
        fitness = calculate_fitness_stage1(
            structure, T, site_groups_definition, chgnet_calculator
        )
        
        # 验证计算结果的合理性
        energy_valid = not (np.isnan(energy) or np.isinf(energy))
        entropy_valid = not (np.isnan(entropy) or np.isinf(entropy)) and entropy >= 0
        fitness_valid = not (np.isnan(fitness) or np.isinf(fitness))
        
        return {
            "energy_eV": energy,
            "entropy_eV_per_K": entropy,
            "fitness_eV": fitness,
            "T_times_S": T * entropy,
            "energy_valid": energy_valid,
            "entropy_valid": entropy_valid,
            "fitness_valid": fitness_valid,
            "overall_valid": energy_valid and entropy_valid and fitness_valid
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "overall_valid": False
        }


def compare_fitness_components(
    structure1: Structure,
    structure2: Structure,
    T: float,
    site_groups_definition: Dict[str, Dict],
    chgnet_calculator: CHGNetCalculator
) -> Dict[str, Dict[str, float]]:
    """
    比较两个结构的适应度组件
    
    Args:
        structure1, structure2 (Structure): 要比较的两个结构
        T (float): 温度参数
        site_groups_definition (Dict[str, Dict]): 位点群定义
        chgnet_calculator (CHGNetCalculator): CHGNet计算器
        
    Returns:
        Dict[str, Dict[str, float]]: 比较结果
    """
    results = {}
    
    for i, structure in enumerate([structure1, structure2], 1):
        validation = validate_fitness_calculation(
            structure, T, site_groups_definition, chgnet_calculator
        )
        results[f"structure_{i}"] = validation
    
    # 计算差值
    if results["structure_1"]["overall_valid"] and results["structure_2"]["overall_valid"]:
        results["difference"] = {
            "delta_energy": results["structure_2"]["energy_eV"] - results["structure_1"]["energy_eV"],
            "delta_entropy": results["structure_2"]["entropy_eV_per_K"] - results["structure_1"]["entropy_eV_per_K"],
            "delta_fitness": results["structure_2"]["fitness_eV"] - results["structure_1"]["fitness_eV"]
        }
    
    return results