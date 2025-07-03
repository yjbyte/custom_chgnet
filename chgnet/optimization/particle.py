"""
PSO粒子表示模块

实现阶段一PSO的粒子数据结构，用于描述晶体中原子在位点群内的排列方式
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np
import copy


class PSOParticle:
    """
    PSO粒子类，用于表示晶体结构中原子在可交换位点群内的排列
    
    粒子编码原子在位点群内的具体排列，确保每个位点群内的元素种类和数量保持不变
    
    Attributes:
        site_groups_arrangement (Dict[str, np.ndarray]): 位点群排列字典
            键为位点群名称，值为该位点群内的原子排列数组
            数组索引对应位点索引，数组值对应原子种类（用原子序数表示）
        site_groups_definition (Dict[str, Dict]): 位点群定义信息
        fitness (float): 粒子适应度值
        velocity (Dict[str, np.ndarray]): 粒子速度（用于PSO更新）
        best_position (Dict[str, np.ndarray]): 粒子历史最佳位置
        best_fitness (float): 粒子历史最佳适应度
    """
    
    def __init__(self, site_groups_definition: Dict[str, Dict]):
        """
        初始化PSO粒子
        
        Args:
            site_groups_definition (Dict[str, Dict]): 位点群定义
                格式: {
                    "group_name": {
                        "site_indices": [0, 1, 2, ...],  # 该群组包含的位点索引
                        "elements": [6, 6, 14, 14, ...],  # 该群组包含的原子种类（原子序数）
                        "element_counts": {"C": 2, "Si": 2}  # 元素计数
                    }
                }
        """
        self.site_groups_definition = copy.deepcopy(site_groups_definition)
        self.site_groups_arrangement = {}
        self.fitness = float('inf')
        self.velocity = {}
        self.best_position = {}
        self.best_fitness = float('inf')
        
        # 为每个位点群初始化随机排列
        self._initialize_random_arrangement()
        
    def _initialize_random_arrangement(self):
        """为每个位点群初始化随机的原子排列"""
        for group_name, group_info in self.site_groups_definition.items():
            elements = group_info["elements"]
            # 创建随机排列
            arrangement = np.array(elements)
            np.random.shuffle(arrangement)
            self.site_groups_arrangement[group_name] = arrangement
            
    def get_arrangement_for_group(self, group_name: str) -> np.ndarray:
        """
        获取指定位点群的原子排列
        
        Args:
            group_name (str): 位点群名称
            
        Returns:
            np.ndarray: 该位点群的原子排列数组
        """
        return self.site_groups_arrangement.get(group_name, np.array([]))
    
    def set_arrangement_for_group(self, group_name: str, arrangement: np.ndarray):
        """
        设置指定位点群的原子排列
        
        Args:
            group_name (str): 位点群名称
            arrangement (np.ndarray): 新的原子排列
        """
        if not self._validate_arrangement(group_name, arrangement):
            raise ValueError(f"无效的排列：不满足位点群 {group_name} 的约束条件")
        self.site_groups_arrangement[group_name] = arrangement.copy()
    
    def _validate_arrangement(self, group_name: str, arrangement: np.ndarray) -> bool:
        """
        验证给定排列是否满足位点群约束
        
        Args:
            group_name (str): 位点群名称
            arrangement (np.ndarray): 待验证的排列
            
        Returns:
            bool: 是否满足约束
        """
        if group_name not in self.site_groups_definition:
            return False
            
        expected_elements = sorted(self.site_groups_definition[group_name]["elements"])
        actual_elements = sorted(arrangement.tolist())
        
        return expected_elements == actual_elements
    
    def swap_atoms_in_group(self, group_name: str, index1: int, index2: int):
        """
        在指定位点群内交换两个原子的位置
        
        Args:
            group_name (str): 位点群名称
            index1, index2 (int): 要交换的原子在该位点群内的索引
        """
        if group_name not in self.site_groups_arrangement:
            raise ValueError(f"位点群 {group_name} 不存在")
            
        arrangement = self.site_groups_arrangement[group_name]
        if index1 >= len(arrangement) or index2 >= len(arrangement):
            raise ValueError("索引超出范围")
            
        arrangement[index1], arrangement[index2] = arrangement[index2], arrangement[index1]
    
    def copy(self) -> 'PSOParticle':
        """创建粒子的深拷贝"""
        new_particle = PSOParticle(self.site_groups_definition)
        new_particle.site_groups_arrangement = {
            k: v.copy() for k, v in self.site_groups_arrangement.items()
        }
        new_particle.fitness = self.fitness
        new_particle.best_fitness = self.best_fitness
        new_particle.best_position = {
            k: v.copy() for k, v in self.best_position.items()
        }
        return new_particle
    
    def update_best_position(self):
        """更新粒子的历史最佳位置"""
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = {
                k: v.copy() for k, v in self.site_groups_arrangement.items()
            }
    
    def get_total_sites(self) -> int:
        """获取所有位点群的总位点数"""
        return sum(len(arr) for arr in self.site_groups_arrangement.values())
    
    def get_arrangement_summary(self) -> Dict[str, Dict]:
        """
        获取排列的摘要信息
        
        Returns:
            Dict: 包含每个位点群排列信息的字典
        """
        summary = {}
        for group_name, arrangement in self.site_groups_arrangement.items():
            unique, counts = np.unique(arrangement, return_counts=True)
            summary[group_name] = {
                "arrangement": arrangement.tolist(),
                "element_distribution": dict(zip(unique.tolist(), counts.tolist()))
            }
        return summary
    
    def __str__(self) -> str:
        """粒子的字符串表示"""
        return f"PSOParticle(fitness={self.fitness:.4f}, groups={list(self.site_groups_arrangement.keys())})"
    
    def __repr__(self) -> str:
        return self.__str__()