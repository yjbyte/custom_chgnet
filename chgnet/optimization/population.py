"""
种群初始化模块

实现生成阶段一PSO初始粒子种群的功能，确保每个粒子严格遵守位点群约束条件
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np
from chgnet.optimization.particle import PSOParticle


def initialize_population(
    population_size: int,
    site_groups_definition: Dict[str, Dict],
    random_seed: int = None
) -> List[PSOParticle]:
    """
    初始化PSO粒子种群
    
    Args:
        population_size (int): 种群大小
        site_groups_definition (Dict[str, Dict]): 位点群定义
        random_seed (int, optional): 随机种子，用于结果复现
        
    Returns:
        List[PSOParticle]: 初始化的粒子种群列表
        
    Example:
        >>> site_groups_def = {
        ...     "group_A": {
        ...         "site_indices": [0, 1, 2, 3],
        ...         "elements": [6, 6, 14, 14],  # 2个C原子，2个Si原子
        ...         "element_counts": {"C": 2, "Si": 2}
        ...     }
        ... }
        >>> population = initialize_population(10, site_groups_def)
        >>> len(population)
        10
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    population = []
    
    for i in range(population_size):
        particle = PSOParticle(site_groups_definition)
        
        # 为了增加初始种群的多样性，对每个位点群应用不同的随机化策略
        for group_name in particle.site_groups_arrangement.keys():
            if i % 3 == 0:
                # 策略1：完全随机排列
                _randomize_group_arrangement(particle, group_name)
            elif i % 3 == 1:
                # 策略2：局部随机交换
                _local_shuffle_group(particle, group_name)
            else:
                # 策略3：分段随机排列
                _segment_shuffle_group(particle, group_name)
        
        population.append(particle)
    
    return population


def _randomize_group_arrangement(particle: PSOParticle, group_name: str):
    """
    对指定位点群进行完全随机排列
    
    Args:
        particle (PSOParticle): 目标粒子
        group_name (str): 位点群名称
    """
    arrangement = particle.site_groups_arrangement[group_name].copy()
    np.random.shuffle(arrangement)
    particle.site_groups_arrangement[group_name] = arrangement


def _local_shuffle_group(particle: PSOParticle, group_name: str):
    """
    对位点群进行局部随机交换（增加局部有序性）
    
    Args:
        particle (PSOParticle): 目标粒子
        group_name (str): 位点群名称
    """
    arrangement = particle.site_groups_arrangement[group_name].copy()
    n = len(arrangement)
    
    # 进行若干次局部交换
    num_swaps = max(1, n // 3)
    for _ in range(num_swaps):
        i = np.random.randint(0, n)
        j = np.random.randint(max(0, i-2), min(n, i+3))  # 在邻近范围内选择交换位置
        arrangement[i], arrangement[j] = arrangement[j], arrangement[i]
    
    particle.site_groups_arrangement[group_name] = arrangement


def _segment_shuffle_group(particle: PSOParticle, group_name: str):
    """
    对位点群进行分段随机排列
    
    Args:
        particle (PSOParticle): 目标粒子
        group_name (str): 位点群名称
    """
    arrangement = particle.site_groups_arrangement[group_name].copy()
    n = len(arrangement)
    
    if n <= 2:
        np.random.shuffle(arrangement)
    else:
        # 将数组分成2-3段，每段内部随机排列
        num_segments = np.random.randint(2, min(4, n+1))
        segment_boundaries = np.sort(np.random.choice(range(1, n), num_segments-1, replace=False))
        segment_boundaries = np.concatenate([[0], segment_boundaries, [n]])
        
        for i in range(len(segment_boundaries) - 1):
            start, end = segment_boundaries[i], segment_boundaries[i+1]
            segment = arrangement[start:end].copy()
            np.random.shuffle(segment)
            arrangement[start:end] = segment
    
    particle.site_groups_arrangement[group_name] = arrangement


def validate_population_constraints(
    population: List[PSOParticle],
    site_groups_definition: Dict[str, Dict]
) -> bool:
    """
    验证整个种群是否满足约束条件
    
    Args:
        population (List[PSOParticle]): 粒子种群
        site_groups_definition (Dict[str, Dict]): 位点群定义
        
    Returns:
        bool: 是否所有粒子都满足约束
    """
    for particle in population:
        for group_name, group_info in site_groups_definition.items():
            if group_name not in particle.site_groups_arrangement:
                return False
                
            expected_elements = sorted(group_info["elements"])
            actual_elements = sorted(particle.site_groups_arrangement[group_name].tolist())
            
            if expected_elements != actual_elements:
                return False
    
    return True


def get_population_diversity_metrics(population: List[PSOParticle]) -> Dict[str, float]:
    """
    计算种群多样性指标
    
    Args:
        population (List[PSOParticle]): 粒子种群
        
    Returns:
        Dict[str, float]: 包含各种多样性指标的字典
    """
    if not population:
        return {"diversity": 0.0, "unique_arrangements": 0}
    
    # 计算独特排列的数量
    unique_arrangements = set()
    for particle in population:
        # 将所有位点群的排列连接成一个元组作为标识符
        arrangement_signature = tuple()
        for group_name in sorted(particle.site_groups_arrangement.keys()):
            arrangement_signature += tuple(particle.site_groups_arrangement[group_name])
        unique_arrangements.add(arrangement_signature)
    
    diversity = len(unique_arrangements) / len(population)
    
    return {
        "diversity": diversity,
        "unique_arrangements": len(unique_arrangements),
        "total_particles": len(population)
    }


def create_diverse_population(
    population_size: int,
    site_groups_definition: Dict[str, Dict],
    min_diversity: float = 0.8,
    max_attempts: int = 100,
    random_seed: int = None
) -> List[PSOParticle]:
    """
    创建具有指定最小多样性的种群
    
    Args:
        population_size (int): 种群大小
        site_groups_definition (Dict[str, Dict]): 位点群定义
        min_diversity (float): 最小多样性要求 (0-1)
        max_attempts (int): 最大尝试次数
        random_seed (int, optional): 随机种子
        
    Returns:
        List[PSOParticle]: 满足多样性要求的粒子种群
    """
    best_population = None
    best_diversity = 0.0
    
    for attempt in range(max_attempts):
        population = initialize_population(
            population_size, 
            site_groups_definition, 
            random_seed=random_seed + attempt if random_seed else None
        )
        
        diversity_metrics = get_population_diversity_metrics(population)
        current_diversity = diversity_metrics["diversity"]
        
        if current_diversity > best_diversity:
            best_diversity = current_diversity
            best_population = population
        
        if current_diversity >= min_diversity:
            break
    
    return best_population if best_population else initialize_population(
        population_size, site_groups_definition, random_seed
    )