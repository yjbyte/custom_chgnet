from chgnet.model import CHGNet
from chgnet.model.dynamics import StructOptimizer
from pymatgen.core import Structure
import os
# 加载模型
model = CHGNet.load()

# 加载初始结构
input_cif = "chgnet/data/perov_supercell/train/4.cif"  # 输入文件路径
structure = Structure.from_file(input_cif)

# 创建优化器，熵权重和设备应该在这里指定
optimizer = StructOptimizer(
    model=model,
    optimizer_class="PSO",
    entropy_weight=0.1  # 在这里设置熵权重
)


result = optimizer.relax(
    structure,
    fmax=0.05,                # 力收敛阈值
    steps=100,                # 最大迭代次数
    relax_cell=True,          # 允许晶格优化
    verbose=True,             # 显示详细信息
    n_particles=15,           # 使用15个粒子
    swap_probability=0.2
    # 原子交换概率
)

# 获取优化后的结构
relaxed_structure = result["final_structure"]

base = os.path.basename(input_cif)
output_cif = os.path.join(os.path.dirname(input_cif),
                              f"optimized_{base}")
relaxed_structure.to(filename=output_cif)
print(f"优化完成，结果已保存到: {output_cif}")

