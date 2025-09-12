"""
使用 ASO 方法（原子搜索优化算法）对晶体结构进行批量优化的脚本。
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
        original (Structure): 原始结构。
        optimized (Structure): 优化后的结构。
        model: 用于能量计算的CHGNet模型。
        use_entropy (bool): 是否计算并比较构型熵。
        entropy_bin_width (float): 构型熵计算中使用的分箱宽度。
        entropy_cutoff (float): 构型熵计算中使用的距离截断。

    Returns:
        dict: 包含比较信息的字典。
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


def optimize_structure(input_file, output_file, optimizer, aso_params, entropy_params):
    """
    使用ASO算法优化单个结构文件并保存结果。

    Args:
        input_file (str): 输入CIF文件路径。
        output_file (str): 输出CIF文件路径。
        optimizer (StructOptimizer): 结构优化器实例。
        aso_params (dict): ASO算法参数字典。
        entropy_params (dict): 熵计算参数字典。

    Returns:
        dict: 比较结果字典，如果出错则返回None。
    """
    try:
        print(f"\n处理文件: {input_file}")
        initial_structure = Structure.from_file(input_file)
        print(f"结构信息: {len(initial_structure)}个原子, 化学式: {initial_structure.composition.formula}")

        print("开始进行ASO结构优化...")
        if entropy_params['use_entropy']:
            print(f"熵优化已启用: T = {entropy_params['temperature']}, "
                  f"bin_width = {entropy_params['entropy_bin_width']}, "
                  f"cutoff = {entropy_params['entropy_cutoff']}")

        start_time = time.time()
        final_structure = optimizer.relax_aso(
            initial_structure,
            n_atoms=aso_params['n_atoms'],
            max_iter=aso_params['max_iter'],
            alpha=aso_params['alpha'],
            beta=aso_params['beta'],
            use_entropy=entropy_params['use_entropy'],
            temperature=entropy_params['temperature'],
            entropy_bin_width=entropy_params['entropy_bin_width'],
            entropy_cutoff=entropy_params['entropy_cutoff']
        )
        end_time = time.time()
        print(f"优化完成！用时: {end_time - start_time:.2f}秒")

        comparison = compare_structures(
            initial_structure, final_structure, optimizer.calculator.model,
            use_entropy=entropy_params['use_entropy'],
            entropy_bin_width=entropy_params['entropy_bin_width'],
            entropy_cutoff=entropy_params['entropy_cutoff']
        )

        print(f"原始能量: {comparison['original_energy']:.6f} eV/atom")
        print(f"优化后能量: {comparison['optimized_energy']:.6f} eV/atom")
        print(f"能量变化: {comparison['energy_difference']:.6f} eV/atom ({comparison['energy_improvement_percentage']:.2f}%)")
        print(f"最大原子位移: {comparison['max_atomic_displacement']:.6f} Å")
        print(f"平均原子位移: {comparison['average_atomic_displacement']:.6f} Å")

        if entropy_params['use_entropy']:
            print(f"\n原始构型熵: {comparison['original_entropy']:.6f}")
            print(f"优化后构型熵: {comparison['optimized_entropy']:.6f}")
            print(f"熵变化: {comparison['entropy_difference']:.6f} ({comparison['entropy_change_percentage']:.2f}%)")

        final_structure.to(filename=output_file)
        print(f"\n优化后的结构已保存到: {output_file}")
        return comparison

    except Exception as e:
        print(f"处理文件 {input_file} 时出错: {str(e)}")
        return None


def process_input_path(input_path, output_dir, optimizer, aso_params, entropy_params):
    """
    处理输入路径（可以是单个文件或目录）。
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
            result = optimize_structure(cif_file, output_file, optimizer, aso_params, entropy_params)
            if result:
                result['input_file'] = cif_file
                summary.append(result)
        if summary:
            print("\n==== 批处理摘要 ====")
            print(f"成功优化: {len(summary)}/{len(cif_files)} 个结构")
            energy_percentages = [r['energy_improvement_percentage'] for r in summary]
            print(f"\n平均能量改善百分比: {np.mean(energy_percentages):.2f}%")
            best_improvement = min(summary, key=lambda x: x['energy_difference'])
            print(f"能量改善最大的结构: {os.path.basename(best_improvement['input_file'])}")
            print(f"  能量改善: {best_improvement['energy_difference']:.6f} eV/atom ({best_improvement['energy_improvement_percentage']:.2f}%)")
            if entropy_params['use_entropy']:
                entropy_percentages = [r['entropy_change_percentage'] for r in summary]
                print(f"\n平均熵变百分比: {np.mean(entropy_percentages):.2f}%")
                max_entropy_change = max(summary, key=lambda x: x['entropy_difference'])
                print(f"熵增加最大的结构: {os.path.basename(max_entropy_change['input_file'])}")
                print(f"  熵增加: {max_entropy_change['entropy_difference']:.6f} ({max_entropy_change['entropy_change_percentage']:.2f}%)")

    elif os.path.isfile(input_path) and input_path.lower().endswith('.cif'):
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_optimized.cif")
        optimize_structure(input_path, output_file, optimizer, aso_params, entropy_params)
    else:
        print(f"错误: 输入路径 {input_path} 不是有效的CIF文件或目录")


def main():
    """主函数，执行ASO结构优化流程。"""
    parser = argparse.ArgumentParser(description='使用ASO方法优化晶体结构')
    parser.add_argument('--input', '-i', type=str, required=False, default='../test_data',help='输入CIF文件或包含CIF文件的目录路径')
    parser.add_argument('--output', '-o', type=str, required=False, default='../test_data/aso_optimizer',help='输出目录，用于保存优化后的结构')

    # ASO相关参数
    parser.add_argument('--atoms', type=int, default=20, help='ASO算法的原子数量 (默认: 20)')
    parser.add_argument('--iterations', '-n', type=int, default=50, help='ASO算法的最大迭代次数 (默认: 50)')
    parser.add_argument('--alpha', type=float, default=50.0, help='深度权重 (默认: 50.0)')
    parser.add_argument('--beta', type=float, default=0.2, help='乘数权重 (默认: 0.2)')

    # 熵相关参数
    parser.add_argument('--use_entropy', action='store_true', default=True, help='是否启用构型熵优化')
    parser.add_argument('--temperature', type=float, default=0.1, help='熵贡献权重因子 (默认: 0.1)')
    parser.add_argument('--bin_width', type=float, default=0.2, help='熵计算的分箱宽度 (默认: 0.2)')
    parser.add_argument('--cutoff', type=float, default=10.0, help='熵计算的距离截断 (默认: 10.0)')

    args = parser.parse_args()

    print("初始化CHGNet优化器...")
    optimizer = StructOptimizer()

    aso_params = {
        'n_atoms': args.atoms,
        'max_iter': args.iterations,
        'alpha': args.alpha,
        'beta': args.beta
    }
    entropy_params = {
        'use_entropy': args.use_entropy,
        'temperature': args.temperature,
        'entropy_bin_width': args.bin_width,
        'entropy_cutoff': args.cutoff
    }

    print(f"ASO参数: 原子数={aso_params['n_atoms']}, 迭代次数={aso_params['max_iter']}, "
          f"alpha={aso_params['alpha']}, beta={aso_params['beta']}")
    if entropy_params['use_entropy']:
        print(f"熵优化已启用: T={entropy_params['temperature']}, bin_width={entropy_params['entropy_bin_width']}, "
              f"cutoff={entropy_params['entropy_cutoff']}")
    else:
        print("熵优化未启用")

    process_input_path(args.input, args.output, optimizer, aso_params, entropy_params)
    print("\n所有优化任务完成！")


if __name__ == "__main__":
    main()