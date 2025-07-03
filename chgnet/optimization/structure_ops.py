"""
结构生成模块

实现从PSO粒子生成晶体结构的功能
"""

from __future__ import annotations
from typing import Dict, Union
import numpy as np
from pymatgen.core import Structure, Element
from ase import Atoms
from chgnet.optimization.particle import PSOParticle


def generate_structure_from_particle_stage1(
    base_structure: Structure,
    particle_arrangement: PSOParticle,
    site_groups_definition: Dict[str, Dict]
) -> Structure:
    """
    根据PSO粒子排列信息生成新的晶体结构
    
    Args:
        base_structure (Structure): 基础结构，定义晶胞和总原子数
        particle_arrangement (PSOParticle): PSO粒子，描述原子在位点群内的排列
        site_groups_definition (Dict[str, Dict]): 位点群信息
        
    Returns:
        Structure: 根据粒子排列修改后的新结构
        
    Example:
        >>> from pymatgen.core import Structure, Lattice
        >>> lattice = Lattice.cubic(4.0)
        >>> base_struct = Structure(lattice, ["C", "C", "Si", "Si"], 
        ...                        [[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]])
        >>> site_groups_def = {
        ...     "group_A": {
        ...         "site_indices": [0, 1, 2, 3],
        ...         "elements": [6, 6, 14, 14]
        ...     }
        ... }
        >>> particle = PSOParticle(site_groups_def)
        >>> new_struct = generate_structure_from_particle_stage1(
        ...     base_struct, particle, site_groups_def)
    """
    # 创建新结构的副本
    new_structure = base_structure.copy()
    
    # 根据粒子排列修改结构中的原子种类
    for group_name, group_info in site_groups_definition.items():
        if group_name not in particle_arrangement.site_groups_arrangement:
            continue
            
        site_indices = group_info["site_indices"]
        arrangement = particle_arrangement.site_groups_arrangement[group_name]
        
        # 验证位点索引有效性
        if max(site_indices) >= len(new_structure):
            raise ValueError(f"位点索引超出结构范围: max={max(site_indices)}, structure_len={len(new_structure)}")
        
        if len(site_indices) != len(arrangement):
            raise ValueError(f"位点数量与排列长度不匹配: {len(site_indices)} vs {len(arrangement)}")
        
        # 更新指定位点的原子种类
        for site_idx, atomic_number in zip(site_indices, arrangement):
            element = Element.from_Z(atomic_number)
            new_structure[site_idx] = element
    
    return new_structure


def structure_to_ase_atoms(structure: Structure) -> Atoms:
    """
    将pymatgen Structure转换为ASE Atoms对象
    
    Args:
        structure (Structure): pymatgen结构
        
    Returns:
        Atoms: ASE原子对象
    """
    atoms = Atoms(
        symbols=[site.specie.symbol for site in structure],
        positions=structure.cart_coords,
        cell=structure.lattice.matrix,
        pbc=True
    )
    return atoms


def ase_atoms_to_structure(atoms: Atoms) -> Structure:
    """
    将ASE Atoms对象转换为pymatgen Structure
    
    Args:
        atoms (Atoms): ASE原子对象
        
    Returns:
        Structure: pymatgen结构
    """
    from pymatgen.io.ase import AseAtomsAdaptor
    adaptor = AseAtomsAdaptor()
    return adaptor.get_structure(atoms)


def validate_structure_consistency(
    original_structure: Structure,
    modified_structure: Structure,
    site_groups_definition: Dict[str, Dict]
) -> bool:
    """
    验证修改后的结构是否与原结构保持一致性
    
    Args:
        original_structure (Structure): 原始结构
        modified_structure (Structure): 修改后的结构
        site_groups_definition (Dict[str, Dict]): 位点群定义
        
    Returns:
        bool: 结构是否一致
    """
    # 检查晶胞参数
    if not np.allclose(original_structure.lattice.matrix, modified_structure.lattice.matrix):
        return False
    
    # 检查原子数量
    if len(original_structure) != len(modified_structure):
        return False
    
    # 检查原子位置（只检查坐标，不检查元素类型）
    if not np.allclose(original_structure.frac_coords, modified_structure.frac_coords):
        return False
    
    # 检查位点群约束
    all_modified_sites = set()
    for group_name, group_info in site_groups_definition.items():
        site_indices = group_info["site_indices"]
        expected_elements = sorted(group_info["elements"])
        
        actual_elements = []
        for site_idx in site_indices:
            actual_elements.append(modified_structure[site_idx].specie.Z)
            all_modified_sites.add(site_idx)
        
        if sorted(actual_elements) != expected_elements:
            return False
    
    # 检查未修改位点的元素是否保持不变
    for i in range(len(original_structure)):
        if i not in all_modified_sites:
            if original_structure[i].specie != modified_structure[i].specie:
                return False
    
    return True


def get_structure_composition_summary(structure: Structure) -> Dict[str, int]:
    """
    获取结构的组成摘要
    
    Args:
        structure (Structure): 晶体结构
        
    Returns:
        Dict[str, int]: 元素组成字典
    """
    composition = {}
    for site in structure:
        element = site.specie.symbol
        composition[element] = composition.get(element, 0) + 1
    return composition


def create_test_structure() -> tuple[Structure, Dict[str, Dict]]:
    """
    创建用于测试的示例结构和位点群定义
    
    Returns:
        tuple[Structure, Dict[str, Dict]]: 测试结构和位点群定义
    """
    from pymatgen.core import Lattice
    
    # 创建一个4原子的立方结构
    lattice = Lattice.cubic(4.0)
    species = ["C", "C", "Si", "Si"]
    coords = [
        [0.0, 0.0, 0.0],      # 位点0
        [0.5, 0.5, 0.0],      # 位点1  
        [0.5, 0.0, 0.5],      # 位点2
        [0.0, 0.5, 0.5]       # 位点3
    ]
    
    structure = Structure(lattice, species, coords)
    
    # 定义位点群：所有位点为一个可交换的群组
    site_groups_definition = {
        "group_A": {
            "site_indices": [0, 1, 2, 3],
            "elements": [6, 6, 14, 14],  # 2个C，2个Si
            "element_counts": {"C": 2, "Si": 2}
        }
    }
    
    return structure, site_groups_definition


def create_complex_test_structure() -> tuple[Structure, Dict[str, Dict]]:
    """
    创建更复杂的测试结构，包含多个位点群
    
    Returns:
        tuple[Structure, Dict[str, Dict]]: 复杂测试结构和位点群定义
    """
    from pymatgen.core import Lattice
    
    # 创建一个8原子的结构
    lattice = Lattice.cubic(6.0)
    species = ["Al", "Al", "Ti", "Ti", "Ni", "Ni", "Cr", "Cr"]
    coords = [
        [0.0, 0.0, 0.0],      # 位点0 - 群组A
        [0.5, 0.0, 0.0],      # 位点1 - 群组A
        [0.0, 0.5, 0.0],      # 位点2 - 群组A
        [0.5, 0.5, 0.0],      # 位点3 - 群组A
        [0.0, 0.0, 0.5],      # 位点4 - 群组B
        [0.5, 0.0, 0.5],      # 位点5 - 群组B
        [0.0, 0.5, 0.5],      # 位点6 - 群组B
        [0.5, 0.5, 0.5]       # 位点7 - 群组B
    ]
    
    structure = Structure(lattice, species, coords)
    
    # 定义两个独立的位点群
    site_groups_definition = {
        "group_A": {
            "site_indices": [0, 1, 2, 3],
            "elements": [13, 13, 22, 22],  # 2个Al，2个Ti
            "element_counts": {"Al": 2, "Ti": 2}
        },
        "group_B": {
            "site_indices": [4, 5, 6, 7],
            "elements": [28, 28, 24, 24],  # 2个Ni，2个Cr
            "element_counts": {"Ni": 2, "Cr": 2}
        }
    }
    
    return structure, site_groups_definition