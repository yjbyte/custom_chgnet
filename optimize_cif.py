#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
optimize_cif.py

直接在脚本内指定 CIF 文件路径，执行 CHGNet 结构优化，
输出优化后的 CIF 文件 optimized_<原文件名>.cif。
"""

import os
from pymatgen.core.structure import Structure
from chgnet.model import StructOptimizer

def optimize_cif(input_cif: str,
                 fmax: float = 0.1,
                 steps: int = 500,
                 relax_cell: bool = True) -> str:
    """
    使用 CHGNet 对 CIF 文件中的结构进行 Relax 优化。

    Args:
        input_cif (str): 待优化的 CIF 文件路径。
        fmax (float): 最大力容忍度，单位 eV/Å。默认 0.1。
        steps (int): 最大优化步数。默认 500。
        relax_cell (bool): 是否同时松弛晶胞。默认 True。

    Returns:
        output_cif (str): 优化后 CIF 文件路径。
    """
    # 1. 读取原始结构
    structure = Structure.from_file(input_cif)

    # 2. 初始化 CHGNet 结构优化器
    relaxer = StructOptimizer(model=None,
                              optimizer_class="FIRE",
                              use_device=None)

    # 3. 执行结构优化
    result = relaxer.relax(structure,
                           fmax=fmax,
                           steps=steps,
                           relax_cell=relax_cell,
                           verbose=True)

    # 4. 获取优化后结构
    opt_struct = result["final_structure"]

    # 5. 构造输出文件名
    base = os.path.basename(input_cif)
    output_cif = os.path.join(os.path.dirname(input_cif),
                              f"optimized_{base}")

    # 6. 写出优化后结构为 CIF 文件
    opt_struct.to(filename=output_cif)

    return output_cif

if __name__ == "__main__":
    # ✅ 在这里填写你的 CIF 文件路径
    input_cif_path = "test_data/Ru27La27F27N27O27_5.cif"  # ←←← 修改为你的文件路径，例如："Zr27Nb27O81.cif"

    # ✅ 执行优化
    result_file = optimize_cif(input_cif_path,
                               fmax=0.1,
                               steps=500,
                               relax_cell=True)

    print(f"优化完成，结果已保存到: {result_file}")
