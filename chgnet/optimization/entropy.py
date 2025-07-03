"""
配置熵计算模块

实现高熵合金结构的配置熵计算功能
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import math
from collections import Counter
from pymatgen.core import Structure
from chgnet.optimization.particle import PSOParticle

# 玻尔兹曼常数 (eV/K)
K_B = 8.617333262145e-5


def calculate_configurational_entropy(
    particle: PSOParticle = None,
    structure: Structure = None,
    site_groups_definition: Dict[str, Dict] = None,
    temperature: float = 300.0
) -> float:
    """
    计算配置熵 S_config
    
    配置熵的计算公式：S_config = -k_B * Σ(n_i * ln(n_i/N))
    其中 n_i 是第i种原子的数量，N是总原子数
    
    Args:
        particle (PSOParticle, optional): PSO粒子
        structure (Structure, optional): 晶体结构  
        site_groups_definition (Dict[str, Dict], optional): 位点群定义
        temperature (float): 温度 (K)，用于热力学计算
        
    Returns:
        float: 配置熵值 (eV/K)
        
    Note:
        可以通过particle或structure计算，优先使用particle
    """
    if particle is not None:
        return _calculate_entropy_from_particle(particle, site_groups_definition)
    elif structure is not None:
        return _calculate_entropy_from_structure(structure, site_groups_definition)
    else:
        raise ValueError("必须提供particle或structure中的一个")


def _calculate_entropy_from_particle(
    particle: PSOParticle,
    site_groups_definition: Dict[str, Dict]
) -> float:
    """
    从PSO粒子计算配置熵
    
    Args:
        particle (PSOParticle): PSO粒子
        site_groups_definition (Dict[str, Dict]): 位点群定义
        
    Returns:
        float: 配置熵值 (eV/K)
    """
    total_entropy = 0.0
    
    # 对每个位点群分别计算配置熵
    for group_name, arrangement in particle.site_groups_arrangement.items():
        group_entropy = _calculate_group_configurational_entropy(arrangement)
        total_entropy += group_entropy
    
    return total_entropy


def _calculate_entropy_from_structure(
    structure: Structure,
    site_groups_definition: Dict[str, Dict]
) -> float:
    """
    从晶体结构计算配置熵
    
    Args:
        structure (Structure): 晶体结构
        site_groups_definition (Dict[str, Dict]): 位点群定义
        
    Returns:
        float: 配置熵值 (eV/K)
    """
    total_entropy = 0.0
    
    for group_name, group_info in site_groups_definition.items():
        site_indices = group_info["site_indices"]
        
        # 提取该位点群的原子种类
        elements_in_group = []
        for site_idx in site_indices:
            elements_in_group.append(structure[site_idx].specie.Z)
        
        arrangement = np.array(elements_in_group)
        group_entropy = _calculate_group_configurational_entropy(arrangement)
        total_entropy += group_entropy
    
    return total_entropy


def _calculate_group_configurational_entropy(arrangement: np.ndarray) -> float:
    """
    计算单个位点群的配置熵
    
    使用混合熵公式：S = -k_B * Σ(x_i * ln(x_i))
    其中 x_i = n_i/N 是第i种元素的摩尔分数
    
    Args:
        arrangement (np.ndarray): 位点群内的原子排列
        
    Returns:
        float: 该位点群的配置熵 (eV/K)
    """
    if len(arrangement) == 0:
        return 0.0
    
    # 计算元素计数
    element_counts = Counter(arrangement)
    total_atoms = len(arrangement)
    
    entropy = 0.0
    for count in element_counts.values():
        if count > 0:
            x_i = count / total_atoms
            entropy -= x_i * math.log(x_i)
    
    return entropy * K_B


def calculate_ideal_mixing_entropy(composition: Dict[str, int]) -> float:
    """
    计算理想混合熵
    
    Args:
        composition (Dict[str, int]): 元素组成 {"元素": 数量}
        
    Returns:
        float: 理想混合熵 (eV/K)
    """
    total_atoms = sum(composition.values())
    if total_atoms == 0:
        return 0.0
    
    entropy = 0.0
    for count in composition.values():
        if count > 0:
            x_i = count / total_atoms
            entropy -= x_i * math.log(x_i)
    
    return entropy * K_B


def calculate_maximum_configurational_entropy(
    site_groups_definition: Dict[str, Dict]
) -> float:
    """
    计算给定位点群定义下的最大配置熵
    
    最大配置熵出现在所有元素完全随机分布时
    
    Args:
        site_groups_definition (Dict[str, Dict]): 位点群定义
        
    Returns:
        float: 最大配置熵 (eV/K)
    """
    total_max_entropy = 0.0
    
    for group_name, group_info in site_groups_definition.items():
        elements = group_info["elements"]
        element_counts = Counter(elements)
        group_max_entropy = calculate_ideal_mixing_entropy(
            {str(elem): count for elem, count in element_counts.items()}
        )
        total_max_entropy += group_max_entropy
    
    return total_max_entropy


def calculate_entropy_efficiency(
    current_entropy: float,
    site_groups_definition: Dict[str, Dict]
) -> float:
    """
    计算配置熵效率（当前熵/最大可能熵）
    
    Args:
        current_entropy (float): 当前配置熵
        site_groups_definition (Dict[str, Dict]): 位点群定义
        
    Returns:
        float: 熵效率 (0-1)
    """
    max_entropy = calculate_maximum_configurational_entropy(site_groups_definition)
    if max_entropy == 0:
        return 0.0
    return current_entropy / max_entropy


def calculate_entropy_temperature_effect(
    base_entropy: float,
    temperature: float,
    reference_temperature: float = 300.0
) -> float:
    """
    计算温度对配置熵的影响
    
    在理想情况下，配置熵不直接依赖温度，但可以考虑热振动的贡献
    
    Args:
        base_entropy (float): 基础配置熵
        temperature (float): 当前温度 (K)
        reference_temperature (float): 参考温度 (K)
        
    Returns:
        float: 温度修正后的熵值
    """
    # 简化的温度修正：主要影响热振动贡献
    # 这里使用简单的线性近似
    temperature_factor = temperature / reference_temperature
    return base_entropy * temperature_factor


def get_entropy_analysis_report(
    particle: PSOParticle,
    site_groups_definition: Dict[str, Dict],
    temperature: float = 300.0
) -> Dict[str, float]:
    """
    生成配置熵分析报告
    
    Args:
        particle (PSOParticle): PSO粒子
        site_groups_definition (Dict[str, Dict]): 位点群定义
        temperature (float): 温度 (K)
        
    Returns:
        Dict[str, float]: 包含各种熵指标的报告
    """
    current_entropy = calculate_configurational_entropy(
        particle=particle,
        site_groups_definition=site_groups_definition,
        temperature=temperature
    )
    
    max_entropy = calculate_maximum_configurational_entropy(site_groups_definition)
    efficiency = calculate_entropy_efficiency(current_entropy, site_groups_definition)
    
    # 计算每个位点群的熵贡献
    group_entropies = {}
    for group_name, arrangement in particle.site_groups_arrangement.items():
        group_entropies[f"entropy_{group_name}"] = _calculate_group_configurational_entropy(arrangement)
    
    report = {
        "total_configurational_entropy": current_entropy,
        "maximum_possible_entropy": max_entropy,
        "entropy_efficiency": efficiency,
        "temperature": temperature,
        **group_entropies
    }
    
    return report


def compare_particle_entropies(
    particles: List[PSOParticle],
    site_groups_definition: Dict[str, Dict]
) -> List[Tuple[int, float]]:
    """
    比较多个粒子的配置熵
    
    Args:
        particles (List[PSOParticle]): 粒子列表
        site_groups_definition (Dict[str, Dict]): 位点群定义
        
    Returns:
        List[Tuple[int, float]]: [(粒子索引, 配置熵值)] 按熵值降序排列
    """
    entropy_list = []
    
    for i, particle in enumerate(particles):
        entropy = calculate_configurational_entropy(
            particle=particle,
            site_groups_definition=site_groups_definition
        )
        entropy_list.append((i, entropy))
    
    # 按熵值降序排列
    entropy_list.sort(key=lambda x: x[1], reverse=True)
    
    return entropy_list