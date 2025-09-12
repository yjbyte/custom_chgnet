"""
使用 CryStAl 方法（晶体结构优化算法）对晶体结构进行批量优化的脚本。
此脚本可以处理单个CIF文件或文件夹中的所有CIF文件。
支持基于构型熵的优化。
"""

import os
import time
import argparse
import glob
import numpy as np
from pymatgen.core import Structure
from chgnet.model.dynamics import StructOptimizer, calculate_distance_entropy


def compare_structures(original: Structure, optimized: Structure, model, use_entropy: bool = False,
                       entropy_bin_width: float = 0.2, entropy_cutoff: float = 10.0) -> dict:
    """
    比较优化前后的结构，返回相关信息。

    Args:
        original (Structure): 原始结构
        optimized (Structure): 优化后的结构
        model: 用于能量计算的CHGNet模型
        use_entropy (bool): 是否计算并比较构型熵
        entropy_bin_width (float): 构型熵计算中使用的分箱宽度
        entropy_cutoff (float): 构型熵计算中使用的距离截断

    Returns:
        dict: 包含比较信息的字典
    """
    # 计算能量
    orig_pred = model.predict_structure(original, task='e')
    opt_pred = model.predict_structure(optimized, task='e')

    orig_energy = float(orig_pred['e'])
    opt_energy = float(opt_pred['e'])
    energy_diff = opt_energy - orig_energy

    # 计算原子位移
    orig_coords = original.cart_coords
    opt_coords = optimized.cart_coords

    # 计算每个原子的位移距离
    displacements = np.linalg.norm(opt_coords - orig_coords, axis=1)
    max_disp = np.max(displacements)
    avg_disp = np.mean(displacements)

    # 计算晶格参数的变化
    orig_latt = original.lattice.abc
    opt_latt = optimized.lattice.abc

    result = {
        "original_energy": orig_energy,
        "optimized_energy": opt_energy,
        "energy_difference": energy_diff,
        "energy_improvement_percentage": (energy_diff / abs(orig_energy)) * 100 if orig_energy != 0 else 0,
        "max_atomic_displacement": max_disp,
        "average_atomic_displacement": avg_disp,
        "original_lattice": orig_latt,
        "optimized_lattice": opt_latt
    }

    # 如果启用熵计算，则计算并添加熵值信息
    if use_entropy:
        orig_entropy = calculate_distance_entropy(original, bin_width=entropy_bin_width, cutoff=entropy_cutoff)
        opt_entropy = calculate_distance_entropy(optimized, bin_width=entropy_bin_width, cutoff=entropy_cutoff)
        entropy_diff = opt_entropy - orig_entropy

        result.update({
            "original_entropy": orig_entropy,
            "optimized_entropy": opt_entropy,
            "entropy_difference": entropy_diff,
            "entropy_change_percentage": (entropy_diff / abs(orig_entropy)) * 100 if orig_entropy != 0 else 0
        })

    return result


def optimize_structure(input_file, output_file, optimizer, crystal_params, entropy_params):
    """
    优化单个结构文件并保存结果。

    Args:
        input_file (str): 输入CIF文件路径
        output_file (str): 输出CIF文件路径
        optimizer (StructOptimizer): 结构优化器实例
        crystal_params (dict): CryStAl算法参数字典
        entropy_params (dict): 熵计算参数字典

    Returns:
        dict: 比较结果字典
    """
    try:
        # 加载初始结构
        print(f"\n处理文件: {input_file}")
        initial_structure = Structure.from_file(input_file)
        print(f"结构信息: {len(initial_structure)}个原子, 化学式: {initial_structure.composition.formula}")

        # 执行优化
        print("开始进行CryStAl结构优化...")
        if entropy_params['use_entropy']:
            print(f"熵优化已启用: T = {entropy_params['temperature']}, "
                  f"bin_width = {entropy_params['entropy_bin_width']}, "
                  f"cutoff = {entropy_params['entropy_cutoff']}")

        start_time = time.time()

        # 调用 relax_crystal 方法
        final_structure = optimizer.relax_crystal(
            initial_structure,
            n_molecules=crystal_params['n_molecules'],
            max_iter=crystal_params['max_iter'],
            n_corners=crystal_params['n_corners'],
            fission_rate=crystal_params['fission_rate'],
            use_entropy=entropy_params['use_entropy'],
            temperature=entropy_params['temperature'],
            entropy_bin_width=entropy_params['entropy_bin_width'],
            entropy_cutoff=entropy_params['entropy_cutoff']
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"优化完成！用时: {elapsed_time:.2f}秒")

        # 比较优化前后的结构
        comparison = compare_structures(
            initial_structure,
            final_structure,
            optimizer.calculator.model,
            use_entropy=entropy_params['use_entropy'],
            entropy_bin_width=entropy_params['entropy_bin_width'],
            entropy_cutoff=entropy_params['entropy_cutoff']
        )

        print(f"原始能量: {comparison['original_energy']:.6f} eV/atom")
        print(f"优化后能量: {comparison['optimized_energy']:.6f} eV/atom")
        print(
            f"能量变化: {comparison['energy_difference']:.6f} eV/atom ({comparison['energy_improvement_percentage']:.2f}%)")
        print(f"最大原子位移: {comparison['max_atomic_displacement']:.6f} Å")
        print(f"平均原子位移: {comparison['average_atomic_displacement']:.6f} Å")

        if entropy_params['use_entropy']:
            print(f"\n原始构型熵: {comparison['original_entropy']:.6f}")
            print(f"优化后构型熵: {comparison['optimized_entropy']:.6f}")
            print(f"熵变化: {comparison['entropy_difference']:.6f} ({comparison['entropy_change_percentage']:.2f}%)")

        # 保存优化后的结构
        final_structure.to(filename=output_file)
        print(f"\n优化后的结构已保存到: {output_file}")

        return comparison

    except Exception as e:
        print(f"处理文件 {input_file} 时出错: {str(e)}")
        return None


def process_input_path(input_path, output_dir, optimizer, crystal_params, entropy_params):
    """
    处理输入路径（可以是单个文件或目录）。

    Args:
        input_path (str): 输入文件或目录路径
        output_dir (str): 输出目录
        optimizer (StructOptimizer): 结构优化器实例
        crystal_params (dict): CryStAl算法参数字典
        entropy_params (dict): 熵计算参数字典
    """
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(input_path):
        cif_files = glob.glob(os.path.join(input_path, "*.cif"))
        if not cif_files:
            print(f"警告：在目录 {input_path} 中未找到CIF文件")
            return

        print(f"在目录 {input_path} 中找到 {len(cif_files)} 个CIF文件")
        summary = []
        for cif_file in cif_files:
            base_name = os.path.splitext(os.path.basename(cif_file))[0]
            output_file = os.path.join(output_dir, f"{base_name}_optimized.cif")
            result = optimize_structure(cif_file, output_file, optimizer, crystal_params, entropy_params)
            if result:
                result['input_file'] = cif_file
                result['output_file'] = output_file
                summary.append(result)

        if summary:
            print("\n==== 批处理摘要 ====")
            print(f"成功优化: {len(summary)}/{len(cif_files)} 个结构")
            energy_percentages = [result['energy_improvement_percentage'] for result in summary]
            avg_energy_percentage = np.mean(energy_percentages)
            best_improvement = min(summary, key=lambda x: x['energy_difference'])
            print(f"\n能量改善最大的结构: {os.path.basename(best_improvement['input_file'])}")
            print(f"  能量改善: {best_improvement['energy_difference']:.6f} eV/atom ({best_improvement['energy_improvement_percentage']:.2f}%)")
            print(f"\n平均能量变化百分比: {avg_energy_percentage:.2f}%")

            if entropy_params['use_entropy']:
                entropy_percentages = [result['entropy_change_percentage'] for result in summary]
                avg_entropy_percentage = np.mean(entropy_percentages)
                max_entropy_change = max(summary, key=lambda x: x['entropy_difference'])
                print(f"\n熵增加最大的结构: {os.path.basename(max_entropy_change['input_file'])}")
                print(f"  熵增加: {max_entropy_change['entropy_difference']:.6f} ({max_entropy_change['entropy_change_percentage']:.2f}%)")
                print(f"\n平均熵变化百分比: {avg_entropy_percentage:.2f}%")

    elif os.path.isfile(input_path) and input_path.lower().endswith('.cif'):
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_optimized.cif")
        optimize_structure(input_path, output_file, optimizer, crystal_params, entropy_params)
    else:
        print(f"错误: 输入路径 {input_path} 不是有效的CIF文件或目录")


def main():
    """主函数，执行CryStAl结构优化流程。"""
    parser = argparse.ArgumentParser(description='使用CryStAl方法优化晶体结构')
    parser.add_argument('--input', '-i', type=str, required=False, default='../test_data',help='输入CIF文件或包含CIF文件的目录路径')
    parser.add_argument('--output', '-o', type=str, required=False, default='../test_data/crystal_optimizer',help='输出目录，用于保存优化后的结构')

    # CryStAl相关参数
    parser.add_argument('--molecules', type=int, default=20, help='CryStAl算法的分子数量 (默认: 20)')
    parser.add_argument('--iterations', '-n', type=int, default=50, help='CryStAl算法的最大迭代次数 (默认: 50)')
    parser.add_argument('--corners', type=int, default=4, help='构成晶格的角分子数量 (默认: 4)')
    parser.add_argument('--fission_rate', type=float, default=0.1, help='分裂算子产生新分子的比例 (默认: 0.1)')

    # 熵相关参数
    parser.add_argument('--use_entropy', action='store_true', default=True, help='是否启用构型熵优化')
    parser.add_argument('--temperature', type=float, default=0.1, help='熵贡献权重因子 (默认: 0.1)')
    parser.add_argument('--bin_width', type=float, default=0.2, help='熵计算的分箱宽度 (默认: 0.2)')
    parser.add_argument('--cutoff', type=float, default=10.0, help='熵计算的距离截断 (默认: 10.0)')

    args = parser.parse_args()

    # 初始化优化器
    print("初始化CHGNet优化器...")
    optimizer = StructOptimizer()

    # 设置CryStAl参数
    crystal_params = {
        'n_molecules': args.molecules,
        'max_iter': args.iterations,
        'n_corners': args.corners,
        'fission_rate': args.fission_rate
    }

    # 设置熵相关参数
    entropy_params = {
        'use_entropy': args.use_entropy,
        'temperature': args.temperature,
        'entropy_bin_width': args.bin_width,
        'entropy_cutoff': args.cutoff
    }

    print(f"CryStAl参数: 分子数={crystal_params['n_molecules']}, 迭代次数={crystal_params['max_iter']}, "
          f"角分子数={crystal_params['n_corners']}, 分裂率={crystal_params['fission_rate']}")

    if entropy_params['use_entropy']:
        print(f"熵优化已启用: T={entropy_params['temperature']}, bin_width={entropy_params['entropy_bin_width']}, "
              f"cutoff={entropy_params['entropy_cutoff']}")
    else:
        print("熵优化未启用")

    # 处理输入路径
    process_input_path(args.input, args.output, optimizer, crystal_params, entropy_params)

    print("\n所有优化任务完成！")


if __name__ == "__main__":
    main()