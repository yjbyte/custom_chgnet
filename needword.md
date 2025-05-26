# 高熵晶体结构优化功能需求文档

## 1. 项目背景
当前项目是chgnet的开源项目,基于chgnet进行的二次开发,主要的开发方式是修改目标函数,修改优化器类型，添加相关接口，设计相关编码等功能

## 2. 总体目标

在chgnet项目的基础上，修改结构优化功能，特别针对高熵晶体系统，实现考虑构型熵的结构优化，并引入更适合全局优化的PSO算法。

## 3. 功能需求

### 3.1 目标函数修改
- 将原目标函数从纯能量优化扩展为"能量+构型熵"组合
- 引入温度参数T，用于调节熵的贡献
- 目标函数形式：F = E - T·S，其中E为CHGNet预测能量，S为构型熵

### 3.2 优化算法替换
- 将现有的FIRE优化器替换为PSO(粒子群优化)算法
- 保留兼容现有ASE优化器的接口，允许用户选择使用PSO或其他优化器
- PSO实现需考虑高维搜索空间的特点

### 3.3 粒子编码设计
- 每个粒子需包含三部分编码：
  1. **结构编码**：基本晶体结构的表示
  2. **顺序变化**：原子位置交换的可能性
  3. **晶格畸变**：针对晶格参数和原子位置的微调

### 3.4 原子半径约束
- 使用提供的`chgnet/data/Atomic_radius_table.csv`文件中的原子半径数据
- 晶格畸变限制在对应原子半径的30%范围内
- 对于表中不存在的原子，使用默认值100pm，并记录提示信息

## 4. 技术规范

### 4.1 构型熵计算
- 使用标准构型熵计算公式：S_config = -k_B * Σ[p_i * ln(p_i)]
- p_i代表第i种元素的原子分数
- 单位需与能量单位保持一致(eV/atom)

### 4.2 PSO算法实现
- 实现标准的PSO算法框架
- 支持以下参数设置：
  - 粒子数量(默认20~30)
  - 最大迭代次数(默认50~100)
  - 个体和全局最优权重(w, c1, c2)
  - 收敛条件


## 5. 实现细节

### 5.1 代码结构
- 创建新的模块`chgnet.model.high_entropy_optimizer.py`
- 保持与现有`dynamics.py`的兼容性和一致接口风格

### 5.2 类设计
- `HighEntropyOptimizer`类：继承自`StructOptimizer`，添加熵相关功能
- `PSOOptimizer`类：实现PSO算法，兼容ASE优化器接口
- `ConfigEntropy`类：提供构型熵计算功能

### 5.3 关键方法
- `calculate_config_entropy(structure)`：计算给定结构的构型熵
- `pso_step()`：执行PSO算法的单步迭代
- `relax(structure, temperature=300, ...)`：扩展的结构优化函数

## 6. 接口设计

### 6.1 用户接口
```python
optimizer = HighEntropyOptimizer(
    model=model,
    temperature=300,  # 默认温度(K)
    use_pso=True,     # 是否使用PSO算法
    pso_particles=20, # PSO粒子数量
    pso_iterations=50 # PSO最大迭代次数
)

result = optimizer.relax(
    structure,
    temperature=500,  # 可选，覆盖默认温度
    fmax=0.1,         # 力收敛阈值
    steps=500,        # 最大步数
    relax_cell=True   # 是否优化晶胞
)
```

### 6.2 返回值
```python
{
    "final_structure": optimized_structure,
    "trajectory": trajectory_observer,
    "energy": final_energy,
    "entropy": final_entropy,
    "free_energy": final_free_energy
}
```


## 8. 注意事项

1. 保持与原项目的兼容性，不影响原有功能
2. 代码需符合项目的风格规范
3. 提供详细的文档和注释
4. 确保性能不会显著下降
5. 考虑物理合理性，避免生成非物理结构
