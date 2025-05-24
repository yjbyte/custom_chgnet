from chgnet.model import CHGNet
from chgnet.model.dynamics import StructOptimizer
from pymatgen.core import Structure
import os

# 加载模型
model = CHGNet.load()

# 加载初始结构
# input_cif = "chgnet/data/perov_supercell/train/4.cif"
input_cif = "test_data/Ru27La27F27N27O27_5.cif"
structure = Structure.from_file(input_cif)

# 创建优化器 - 修复了关键问题
optimizer = StructOptimizer(
    model=model,
    optimizer_class="PSO",
    entropy_weight=0.1  # 设置熵权重
)

# 注意：必须在 relax 调用中设置 return_site_energies=True 才能使熵贡献生效
# 但这个参数目前不能直接传递，需要修改代码

result = optimizer.relax(
    structure,
    fmax=0.05,
    steps=100,
    relax_cell=True,
    verbose=True,
    n_particles=20,  # 增加粒子数
    swap_probability=0.3,  # 增加交换概率
    different_elements_only=True,  # 只交换不同元素
    # 新增可调节参数
    velocity_scale=0.1,    # 可调节的初始速度缩放
    position_scale=0.15,   # 可调节的位置更新缩放
    inertia_start=0.9,     # PSO惯性权重开始值
    inertia_end=0.4,       # PSO惯性权重结束值
    cognitive=2.0,         # 认知参数
    social=2.0,            # 社会参数
    finalize_with_lbfgs=True  # 最后用LBFGS精化
)

# 获取优化后的结构
relaxed_structure = result["final_structure"]

base = os.path.basename(input_cif)
output_cif = os.path.join(os.path.dirname(input_cif), f"optimized_{base}")
relaxed_structure.to(filename=output_cif)
print(f"优化完成，结果已保存到: {output_cif}")