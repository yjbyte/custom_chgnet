from __future__ import annotations

import contextlib
# 添加新的导入
import csv
import inspect
import io
import os
import pickle
import sys
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Callable
from typing import Literal
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import (
    BaseCalculator,
    Calculator,
)
from ase.calculators.calculator import all_changes, all_properties
from ase.md.npt import NPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen, NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.optimize.bfgs import BFGS
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.optimize.fire import FIRE
from ase.optimize.lbfgs import LBFGS, LBFGSLineSearch
from ase.optimize.mdmin import MDMin
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from pymatgen.analysis.eos import BirchMurnaghan
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from chgnet.model.model import CHGNet
from chgnet.utils import determine_device

if TYPE_CHECKING:
    from ase.io import Trajectory
    from ase.optimize.optimize import Optimizer
    from typing_extensions import Self

    from chgnet import PredTask

# We would like to thank M3GNet develop team for this module
# source: https://github.com/materialsvirtuallab/m3gnet

OPTIMIZERS = {
    "FIRE": FIRE,
    "BFGS": BFGS,
    "LBFGS": LBFGS,
    "LBFGSLineSearch": LBFGSLineSearch,
    "MDMin": MDMin,
    "SciPyFminCG": SciPyFminCG,
    "SciPyFminBFGS": SciPyFminBFGS,
    "BFGSLineSearch": BFGSLineSearch,
}


def load_atomic_radii(csv_path: Optional[str] = None) -> Dict[str, float]:
    """
    加载原子半径数据，从CSV文件中读取原子的离子半径。

    Args:
        csv_path (str, optional): 原子半径数据CSV文件的路径。如果为None，则使用默认路径。

    Returns:
        Dict[str, float]: 原子符号到离子半径(pm)的映射字典
    """
    if csv_path is None:
        # 假设CSV文件位于chgnet/data/目录下
        module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(module_dir, 'data', 'Atomic_radius_table.csv')

    atomic_radii = {}

    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                atom = row['Atom'].strip()
                # 按顺序检查A、B、O三列，使用第一个非空值
                radius = None
                for col in ['Ion radius at site A', 'Ion radius at site B', 'Ion radius at site O']:
                    if row[col] and row[col].strip():
                        radius = float(row[col].strip())
                        break

                if radius is not None:
                    atomic_radii[atom] = radius
    except Exception as e:
        print(f"加载原子半径数据时出错: {e}")

    return atomic_radii


def simple_pso(
        objective_func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        n_particles: int = 20,
        max_iter: int = 100,
        c1: float = 0.5,
        c2: float = 0.5,
        w: float = 0.9
) -> Tuple[np.ndarray, float]:
    """
    一个粒子群优化(PSO)算法实现。

    Args:
        objective_func: 目标函数，接收粒子位置(numpy数组)并返回一个标量值
        bounds: 每个维度的搜索范围，格式为[(min_1, max_1), (min_2, max_2), ...]
        n_particles: 粒子数量
        max_iter: 最大迭代次数
        c1: 认知参数
        c2: 社会参数
        w: 惯性权重

    Returns:
        Tuple[np.ndarray, float]: 最优位置和对应的目标函数值
    """
    # 获取问题维度
    dim = len(bounds)

    # 设置速度限制（基于搜索空间的大小）
    v_bounds = []
    for i in range(dim):
        v_min = -(bounds[i][1] - bounds[i][0]) * 0.25  # 速度限制为搜索范围的10%
        v_max = (bounds[i][1] - bounds[i][0]) * 0.25
        v_bounds.append((v_min, v_max))

    # 初始化粒子位置和速度
    positions = np.zeros((n_particles, dim))
    velocities = np.zeros((n_particles, dim))

    # 随机初始化粒子位置和速度
    for i in range(dim):
        positions[:, i] = np.random.uniform(
            bounds[i][0], bounds[i][1], size=n_particles
        )
        velocities[:, i] = np.random.uniform(
            v_bounds[i][0], v_bounds[i][1], size=n_particles
        )

    # 初始化每个粒子的最佳位置和值
    pbest_pos = positions.copy()
    pbest_val = np.array([objective_func(pos) for pos in positions])

    # 初始化全局最佳位置和值
    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    # 主循环
    for _ in range(max_iter):
        # 更新每个粒子
        for i in range(n_particles):
            # 生成随机系数
            r1 = np.random.random(dim)
            r2 = np.random.random(dim)

            # 更新速度
            cognitive_velocity = c1 * r1 * (pbest_pos[i] - positions[i])
            social_velocity = c2 * r2 * (gbest_pos - positions[i])
            velocities[i] = w * velocities[i] + cognitive_velocity + social_velocity

            # 应用速度限制
            for j in range(dim):
                velocities[i, j] = np.clip(velocities[i, j], v_bounds[j][0], v_bounds[j][1])

            # 更新位置
            positions[i] += velocities[i]

            # 应用位置边界限制
            for j in range(dim):
                positions[i, j] = np.clip(positions[i, j], bounds[j][0], bounds[j][1])

            # 评估新位置
            val = objective_func(positions[i])

            # 更新粒子最佳位置
            if val < pbest_val[i]:
                pbest_pos[i] = positions[i].copy()
                pbest_val[i] = val

                # 更新全局最佳位置
                if val < gbest_val:
                    gbest_pos = positions[i].copy()
                    gbest_val = val

    # 返回全局最优位置和对应的值
    return gbest_pos, gbest_val


def calculate_distance_entropy(structure: Structure, bin_width: float = 0.2, cutoff: float = 10.0) -> float:
    """
    计算基于同种元素间距分布的构型熵。

    Args:
        structure (Structure): Pymatgen Structure对象。
        bin_width (float): 距离分布直方图的分箱宽度 (Å)。
        cutoff (float): 计算原子间距时考虑的最大距离 (Å)。

    Returns:
        float: 计算得到的构型熵值 (k=1)。
    """
    # a. 按元素分组
    sites_by_element = defaultdict(list)
    for site in structure:
        sites_by_element[site.specie.symbol].append(site)

    all_distances = []
    # b. 计算同种元素间距
    for symbol, sites in sites_by_element.items():
        if len(sites) < 2:
            continue

        coords = np.array([site.coords for site in sites])

        # 计算该组内所有唯一原子对之间的距离
        # 使用itertools.combinations来获取所有唯一的对
        from itertools import combinations
        for i, j in combinations(range(len(coords)), 2):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < cutoff:
                all_distances.append(dist)

    # c. 处理空距离
    if not all_distances:
        return 0.0

    # d. 分箱与概率计算
    bins = np.arange(0, cutoff + bin_width, bin_width)
    hist, _ = np.histogram(all_distances, bins=bins)

    # 只考虑非零计数的箱
    non_zero_hist = hist[hist > 0]

    if non_zero_hist.size == 0:
        return 0.0

    total_count = np.sum(non_zero_hist)
    probabilities = non_zero_hist / total_count

    # e. 计算香农熵 (k=1)
    entropy = -np.sum(probabilities * np.log(probabilities))

    # f. 返回熵值
    return entropy


class CHGNetCalculator(Calculator):
    """CHGNet Calculator for ASE applications."""

    implemented_properties = ("energy", "forces", "stress", "magmoms", "energies")

    def __init__(
            self,
            model: CHGNet | None = None,
            *,
            use_device: str | None = None,
            check_cuda_mem: bool = False,
            stress_weight: float = units.GPa,  # GPa to eV/A^3
            on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
            return_site_energies: bool = False,
            **kwargs,
    ) -> None:
        """Provide a CHGNet instance to calculate various atomic properties using ASE.

        Args:
            model (CHGNet): instance of a chgnet model. If set to None,
                the pretrained CHGNet is loaded.
                Default = None
            use_device (str, optional): The device to be used for predictions,
                either "cpu", "cuda", or "mps". If not specified, the default device is
                automatically selected based on the available options.
                Default = None
            check_cuda_mem (bool): Whether to use cuda with most available memory
                Default = False
            stress_weight (float): the conversion factor to convert GPa to eV/A^3.
                Default = 1/160.21
            on_isolated_atoms ('ignore' | 'warn' | 'error'): how to handle Structures
                with isolated atoms.
                Default = 'warn'
            return_site_energies (bool): whether to return the energy of each atom
            **kwargs: Passed to the Calculator parent class.
        """
        super().__init__(**kwargs)

        # Determine the device to use
        device = determine_device(use_device=use_device, check_cuda_mem=check_cuda_mem)
        self.device = device

        # Move the model to the specified device
        if model is None:
            self.model = CHGNet.load(verbose=False, use_device=self.device)
        else:
            self.model = model.to(self.device)
        self.model.graph_converter.set_isolated_atom_response(on_isolated_atoms)
        self.stress_weight = stress_weight
        self.return_site_energies = return_site_energies
        print(f"CHGNet will run on {self.device}")

    @classmethod
    def from_file(cls, path: str, use_device: str | None = None, **kwargs) -> Self:
        """Load a user's CHGNet model and initialize the Calculator."""
        return cls(
            model=CHGNet.from_file(path),
            use_device=use_device,
            **kwargs,
        )

    @property
    def version(self) -> str | None:
        """The version of CHGNet."""
        return self.model.version

    @property
    def n_params(self) -> int:
        """The number of parameters in CHGNet."""
        return self.model.n_params

    def calculate(
            self,
            atoms: Atoms | None = None,
            properties: list | None = None,
            system_changes: list | None = None,
            task: PredTask = "efsm",
    ) -> None:
        """Calculate various properties of the atoms using CHGNet.

        Args:
            atoms (Atoms | None): The atoms object to calculate properties for.
            properties (list | None): The properties to calculate.
                Default is all properties.
            system_changes (list | None): The changes made to the system.
                Default is all changes.
            task (PredTask): The task to perform. One of "e", "ef", "em", "efs", "efsm".
                Default = "efsm"
        """
        properties = properties or all_properties
        system_changes = system_changes or all_changes
        super().calculate(
            atoms=atoms,
            properties=properties,
            system_changes=system_changes,
        )

        # Run CHGNet
        structure = AseAtomsAdaptor.get_structure(atoms)
        graph = self.model.graph_converter(structure)
        model_prediction = self.model.predict_graph(
            graph.to(self.device),
            task=task,
            return_crystal_feas=True,
            return_site_energies=self.return_site_energies,
        )

        # Convert Result
        extensive_factor = len(structure) if self.model.is_intensive else 1
        key_map = dict(
            e=("energy", extensive_factor),
            f=("forces", 1),
            m=("magmoms", 1),
            s=("stress", self.stress_weight),
        )
        self.results |= {
            long_key: model_prediction[key] * factor
            for key, (long_key, factor) in key_map.items()
            if key in model_prediction
        }
        self.results["free_energy"] = self.results["energy"]
        self.results["crystal_fea"] = model_prediction["crystal_fea"]
        if self.return_site_energies:
            self.results["energies"] = model_prediction["site_energies"]


class StructOptimizer:
    """Wrapper class for structural relaxation."""

    def __init__(
            self,
            model: CHGNet | CHGNetCalculator | None = None,
            optimizer_class: Optimizer | str | None = "FIRE",
            use_device: str | None = None,
            stress_weight: float = units.GPa,
            on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
    ) -> None:
        """Provide a trained CHGNet model and an optimizer to relax crystal structures.

        Args:
            model (CHGNet): instance of a CHGNet model or CHGNetCalculator.
                If set to None, the pretrained CHGNet is loaded.
                Default = None
            optimizer_class (Optimizer,str): choose optimizer from ASE.
                Default = "FIRE"
            use_device (str, optional): The device to be used for predictions,
                either "cpu", "cuda", or "mps". If not specified, the default device is
                automatically selected based on the available options.
                Default = None
            stress_weight (float): the conversion factor to convert GPa to eV/A^3.
                Default = 1/160.21
            on_isolated_atoms ('ignore' | 'warn' | 'error'): how to handle Structures
                with isolated atoms.
                Default = 'warn'
        """
        if isinstance(optimizer_class, str):
            if optimizer_class in OPTIMIZERS:
                optimizer_class = OPTIMIZERS[optimizer_class]
            else:
                raise ValueError(
                    f"Optimizer instance not found. Select from {list(OPTIMIZERS)}"
                )

        self.optimizer_class: Optimizer = optimizer_class

        if isinstance(model, CHGNetCalculator):
            self.calculator = model
        else:
            self.calculator = CHGNetCalculator(
                model=model,
                stress_weight=stress_weight,
                use_device=use_device,
                on_isolated_atoms=on_isolated_atoms,
            )

    @property
    def version(self) -> str:
        """The version of CHGNet."""
        return self.calculator.model.version

    @property
    def n_params(self) -> int:
        """The number of parameters in CHGNet."""
        return self.calculator.model.n_params

    def relax(
            self,
            atoms: Structure | Atoms,
            *,
            fmax: float | None = 0.3,
            steps: int | None = 500,
            relax_cell: bool | None = True,
            ase_filter: str | None = "FrechetCellFilter",
            save_path: str | None = None,
            loginterval: int | None = 1,
            crystal_feas_save_path: str | None = None,
            verbose: bool = True,
            assign_magmoms: bool = True,
            **kwargs,
    ) -> dict[str, Structure | TrajectoryObserver]:
        """Relax the Structure/Atoms until maximum force is smaller than fmax.

        Args:
            atoms (Structure | Atoms): A Structure or Atoms object to relax.
            fmax (float | None): The maximum force tolerance for relaxation.
                Default = 0.3
            steps (int | None): The maximum number of steps for relaxation.
                Default = 500
            relax_cell (bool | None): Whether to relax the cell as well.
                Default = True
            ase_filter (str | ase.filters.Filter): The filter to apply to the atoms
                object for relaxation. Default = FrechetCellFilter
                Default used to be ExpCellFilter which was removed due to bug reported
                in https://gitlab.com/ase/ase/-/issues/1321 and fixed in
                https://gitlab.com/ase/ase/-/merge_requests/3024.
            save_path (str | None): The path to save the trajectory.
                Default = None
            loginterval (int | None): Interval for logging trajectory and crystal
                features. Default = 1
            crystal_feas_save_path (str | None): Path to save crystal feature vectors
                which are logged at a loginterval rage
                Default = None
            verbose (bool): Whether to print the output of the ASE optimizer.
                Default = True
            assign_magmoms (bool): Whether to assign magnetic moments to the final
                structure. Default = True
            **kwargs: Additional parameters for the optimizer.

        Returns:
            dict[str, Structure | TrajectoryObserver]:
                A dictionary with 'final_structure' and 'trajectory'.
        """
        from ase import filters
        from ase.filters import Filter

        valid_filter_names = [
            name
            for name, cls in inspect.getmembers(filters, inspect.isclass)
            if issubclass(cls, Filter)
        ]

        if isinstance(ase_filter, str):
            if ase_filter in valid_filter_names:
                ase_filter = getattr(filters, ase_filter)
            else:
                raise ValueError(
                    f"Invalid {ase_filter=}, must be one of {valid_filter_names}. "
                )

        if isinstance(atoms, Structure):
            atoms = AseAtomsAdaptor().get_atoms(atoms)

        atoms.calc = self.calculator  # assign model used to predict forces

        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)

            if crystal_feas_save_path:
                cry_obs = CrystalFeasObserver(atoms)

            if relax_cell:
                atoms = ase_filter(atoms)
            optimizer: Optimizer = self.optimizer_class(atoms, **kwargs)
            optimizer.attach(obs, interval=loginterval)

            if crystal_feas_save_path:
                optimizer.attach(cry_obs, interval=loginterval)

            optimizer.run(fmax=fmax, steps=steps)
            obs()

        if save_path is not None:
            obs.save(save_path)

        if crystal_feas_save_path:
            cry_obs.save(crystal_feas_save_path)

        if isinstance(atoms, Filter):
            atoms = atoms.atoms
        struct = AseAtomsAdaptor.get_structure(atoms)

        if assign_magmoms:
            for key in struct.site_properties:
                struct.remove_site_property(property_name=key)
            struct.add_site_property(
                "magmom", [float(magmom) for magmom in atoms.get_magnetic_moments()]
            )
        return {"final_structure": struct, "trajectory": obs}

    def relax_pso(self, atoms: Structure, n_particles: int = 10, max_iter: int = 50, c1: float = 0.5, c2: float = 0.5,
                  w: float = 0.9, use_entropy: bool = False, temperature: float = 0.1,
                  entropy_bin_width: float = 0.2, entropy_cutoff: float = 10.0) -> Structure:
        """
        使用粒子群优化(PSO)对晶体结构进行优化。
        通过在原子坐标上施加小的扰动来寻找能量最低的结构配置。

        Args:
            atoms (Structure): 要优化的Pymatgen Structure对象
            n_particles (int): PSO算法中的粒子数量，默认为20
            max_iter (int): PSO算法的最大迭代次数，默认为50
            c1 (float): PSO的认知参数，默认为0.5
            c2 (float): PSO的社会参数，默认为0.5
            w (float): PSO的惯性权重，默认为0.9
            use_entropy (bool): 是否在优化目标中包含构型熵。默认为 False。
            temperature (float): 熵的贡献权重因子 T。默认为 0.1。
            entropy_bin_width (float): 熵计算中距离分箱的宽度。默认为 0.2。
            entropy_cutoff (float): 熵计算中原子间距的最大截断距离。默认为 10.0。

        Returns:
            Structure: 优化后的晶体结构
        """
        # 1. 获取原子符号和离子半径映射
        atomic_radii = load_atomic_radii()
        default_radius = 120.0  # 默认半径值(pm)

        # 2. 获取结构信息
        n_atoms = len(atoms)
        species = [site.specie.symbol for site in atoms]

        # 获取初始分数坐标
        initial_frac_coords = np.array([site.frac_coords for site in atoms])

        # 3. 计算每个原子的最大位移边界
        bounds = []

        for i, atom_symbol in enumerate(species):
            # 查找原子半径，如果不存在则使用默认值
            radius_pm = atomic_radii.get(atom_symbol, default_radius)

            # 计算笛卡尔坐标系下的最大位移(Å)
            # 半径从pm转换为Å(除以100)，然后乘以10%
            max_displacement_A = radius_pm / 100.0 * 0.25

            # 将笛卡尔坐标下的最大位移转换为分数坐标
            lattice_inv = atoms.lattice.inv_matrix

            # 计算三个轴方向上的最大分数坐标位移
            cart_displacements = [
                [max_displacement_A, 0, 0],  # x方向
                [0, max_displacement_A, 0],  # y方向
                [0, 0, max_displacement_A]  # z方向
            ]

            # 计算这些笛卡尔位移在分数坐标下的大小
            frac_displacements = []
            for cart_disp in cart_displacements:
                frac_disp = np.dot(lattice_inv, cart_disp)
                frac_displacements.append(np.linalg.norm(frac_disp))

            # 设置x, y, z三个方向的边界
            for j in range(3):
                # 分数坐标下的边界
                bounds.append((-frac_displacements[j], frac_displacements[j]))

        # 4. 定义目标函数
        def objective_function(particle_position: np.ndarray) -> float:
            """
            PSO优化的目标函数，计算给定原子位移下结构的能量。
            如果use_entropy为True，则返回 E_CHGNet - T * S_dist。

            Args:
                particle_position (np.ndarray): 所有原子的位移向量，格式为[dx1, dy1, dz1, dx2, dy2, dz2, ...]

            Returns:
                float: 结构的能量值（可能包含熵贡献）。
            """
            # 重塑位移向量为(n_atoms, 3)形状
            displacements = particle_position.reshape((-1, 3))

            # 计算新的分数坐标
            new_frac_coords = initial_frac_coords + displacements

            # 创建一个新的结构对象
            new_structure = atoms.copy()

            # 更新所有原子的位置
            for i in range(n_atoms):
                new_structure.replace(i, species[i], new_frac_coords[i], properties=atoms[i].properties)

            # 使用CHGNet计算能量
            prediction = self.calculator.model.predict_structure(new_structure, task='e')
            energy_chgnet = float(prediction['e'])

            if use_entropy:
                s_dist = calculate_distance_entropy(
                    new_structure,
                    bin_width=entropy_bin_width,
                    cutoff=entropy_cutoff
                )
                return energy_chgnet - temperature * s_dist
            else:
                return energy_chgnet

        # 5. 调用PSO优化器
        print(f"开始PSO优化，使用 {n_particles} 个粒子和 {max_iter} 次迭代...")
        if use_entropy:
            print(f"熵优化已启用: T = {temperature}, bin_width = {entropy_bin_width}, cutoff = {entropy_cutoff}")

        gbest_position, gbest_energy = simple_pso(
            objective_func=objective_function,
            bounds=bounds,
            n_particles=n_particles,
            max_iter=max_iter,
            c1=c1,
            c2=c2,
            w=w
        )
        print(f"PSO优化完成，最佳目标函数值: {gbest_energy}")

        # 6. 处理优化结果
        # 重塑最佳位置为(n_atoms, 3)形状
        best_displacements = gbest_position.reshape((-1, 3))

        # 计算最终的原子分数坐标
        final_frac_coords = initial_frac_coords + best_displacements

        # 创建最终结构
        final_structure = atoms.copy()

        # 更新所有原子的位置
        for i in range(n_atoms):
            final_structure.replace(i, species[i], final_frac_coords[i], properties=atoms[i].properties)

        # 7. 返回优化后的结构
        return final_structure



class TrajectoryObserver:
    """Trajectory observer is a hook in the relaxation process that saves the
    intermediate structures.
    """

    def __init__(self, atoms: Atoms) -> None:
        """Create a TrajectoryObserver from an Atoms object.

        Args:
            atoms (Atoms): the structure to observe.
        """
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.magmoms: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self) -> None:
        """The logic for saving the properties of an Atoms during the relaxation."""
        self.energies.append(self.compute_energy())
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.magmoms.append(self.atoms.get_magnetic_moments())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def __len__(self) -> int:
        """The number of steps in the trajectory."""
        return len(self.energies)

    def compute_energy(self) -> float:
        """Calculate the potential energy.

        Returns:
            energy (float): the potential energy.
        """
        return self.atoms.get_potential_energy()

    def save(self, filename: str) -> None:
        """Save the trajectory to file.

        Args:
            filename (str): filename to save the trajectory
        """
        out_pkl = {
            "energy": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "magmoms": self.magmoms,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }
        with open(filename, "wb") as file:
            pickle.dump(out_pkl, file)


class CrystalFeasObserver:
    """CrystalFeasObserver is a hook in the relaxation and MD process that saves the
    intermediate crystal feature structures.
    """

    def __init__(self, atoms: Atoms) -> None:
        """Create a CrystalFeasObserver from an Atoms object."""
        self.atoms = atoms
        self.crystal_feature_vectors: list[np.ndarray] = []

    def __call__(self) -> None:
        """Record Atoms crystal feature vectors after an MD/relaxation step."""
        self.crystal_feature_vectors.append(self.atoms._calc.results["crystal_fea"])  # noqa: SLF001

    def __len__(self) -> int:
        """Number of recorded steps."""
        return len(self.crystal_feature_vectors)

    def save(self, filename: str) -> None:
        """Save the crystal feature vectors to filename in pickle format."""
        out_pkl = {"crystal_feas": self.crystal_feature_vectors}
        with open(filename, "wb") as file:
            pickle.dump(out_pkl, file)


class MolecularDynamics:
    """Molecular dynamics class."""

    def __init__(
            self,
            atoms: Atoms | Structure,
            *,
            model: CHGNet | CHGNetCalculator | None = None,
            ensemble: str = "nvt",
            thermostat: str = "Berendsen_inhomogeneous",
            temperature: int = 300,
            starting_temperature: int | None = None,
            timestep: float = 2.0,
            pressure: float = 1.01325e-4,
            taut: float | None = None,
            taup: float | None = None,
            bulk_modulus: float | None = None,
            trajectory: str | Trajectory | None = None,
            logfile: str | None = None,
            loginterval: int = 1,
            crystal_feas_logfile: str | None = None,
            append_trajectory: bool = False,
            on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
            return_site_energies: bool = False,
            use_device: str | None = None,
    ) -> None:
        """Initialize the MD class.

        Args:
            atoms (Atoms): atoms to run the MD
            model (CHGNet): instance of a CHGNet model or CHGNetCalculator.
                If set to None, the pretrained CHGNet is loaded.
                Default = None
            ensemble (str): choose from 'nve', 'nvt', 'npt'
                Default = "nvt"
            thermostat (str): Thermostat to use
                choose from "Nose-Hoover", "Berendsen", "Berendsen_inhomogeneous"
                Default = "Berendsen_inhomogeneous"
            temperature (float): temperature for MD simulation, in K
                Default = 300
            starting_temperature (float): starting temperature of MD simulation, in K
                if set as None, the MD starts with the momentum carried by ase.Atoms
                if input is a pymatgen.core.Structure, the MD starts at 0K
                Default = None
            timestep (float): time step in fs
                Default = 2
            pressure (float): pressure in GPa
                Can be 3x3 or 6 np.array if thermostat is "Nose-Hoover"
                Default = 1.01325e-4 GPa = 1 atm
            taut (float): time constant for temperature coupling in fs.
                The temperature will be raised to target temperature in approximate
                10 * taut time.
                Default = 100 * timestep
            taup (float): time constant for pressure coupling in fs
                Default = 1000 * timestep
            bulk_modulus (float): bulk modulus of the material in GPa.
                Used in NPT ensemble for the barostat pressure coupling.
                The DFT bulk modulus can be found for most materials at
                https://next-gen.materialsproject.org/

                In NPT ensemble, the effective damping time for pressure is multiplied
                by compressibility. In LAMMPS, Bulk modulus is defaulted to 10
                see: https://docs.lammps.org/fix_press_berendsen.html
                and: https://gitlab.com/ase/ase/-/blob/master/ase/md/nptberendsen.py

                If bulk modulus is not provided here, it will be calculated by CHGNet
                through Birch Murnaghan equation of state (EOS).
                Note the EOS fitting can fail because of non-parabolic potential
                energy surface, which is common with soft system like liquid and gas.
                In such case, user should provide an input bulk modulus for better
                barostat coupling, otherwise a guessed bulk modulus = 2 GPa will be used
                (water's bulk modulus)

                Default = None
            trajectory (str or Trajectory): Attach trajectory object
                Default = None
            logfile (str): open this file for recording MD outputs
                Default = None
            loginterval (int): write to log file every interval steps
                Default = 1
            crystal_feas_logfile (str): open this file for recording crystal features
                during MD. Default = None
            append_trajectory (bool): Whether to append to prev trajectory.
                If false, previous trajectory gets overwritten
                Default = False
            on_isolated_atoms ('ignore' | 'warn' | 'error'): how to handle Structures
                with isolated atoms.
                Default = 'warn'
            return_site_energies (bool): whether to return the energy of each atom
            use_device (str): the device for the MD run
                Default = None
        """
        self.ensemble = ensemble
        self.thermostat = thermostat
        if isinstance(atoms, Structure | Molecule):
            atoms = AseAtomsAdaptor().get_atoms(atoms)
            # atoms = atoms.to_ase_atoms()

        if starting_temperature is not None:
            MaxwellBoltzmannDistribution(
                atoms, temperature_K=starting_temperature, force_temp=True
            )
            Stationary(atoms)

        self.atoms = atoms
        if isinstance(model, Calculator | BaseCalculator):
            self.atoms.calc = model
        else:
            self.atoms.calc = CHGNetCalculator(
                model=model,
                use_device=use_device,
                on_isolated_atoms=on_isolated_atoms,
                return_site_energies=return_site_energies,
            )

        if taut is None:
            taut = 100 * timestep
        if taup is None:
            taup = 1000 * timestep

        if ensemble.lower() == "nve":
            """
            VelocityVerlet (constant N, V, E) molecular dynamics.

            Note: it's recommended to use smaller timestep for NVE compared to other
            ensembles, since the VelocityVerlet algorithm assumes a strict conservative
            force field.
            """
            self.dyn = VelocityVerlet(
                atoms=self.atoms,
                timestep=timestep * units.fs,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )
            print("NVE-MD created")

        elif ensemble.lower() == "nvt":
            """
            Constant volume/temperature molecular dynamics.
            """
            if thermostat.lower() == "nose-hoover":
                """
                Nose-hoover (constant N, V, T) molecular dynamics.
                ASE implementation currently only supports upper triangular lattice
                """
                self.upper_triangular_cell()
                self.dyn = NPT(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    externalstress=pressure
                                   * units.GPa,  # ase NPT does not like externalstress=None
                    ttime=taut * units.fs,
                    pfactor=None,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )
                print("NVT-Nose-Hoover MD created")
            elif thermostat.lower().startswith("berendsen"):
                """
                Berendsen (constant N, V, T) molecular dynamics.
                """
                self.dyn = NVTBerendsen(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    taut=taut * units.fs,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )
                print("NVT-Berendsen-MD created")
            else:
                raise ValueError(
                    "Thermostat not supported, choose in 'Nose-Hoover', 'Berendsen', "
                    "'Berendsen_inhomogeneous'"
                )

        elif ensemble.lower() == "npt":
            """
            Constant pressure/temperature molecular dynamics.
            """
            # Bulk modulus is needed for pressure damping time
            if bulk_modulus is not None:
                bulk_modulus_au = bulk_modulus / 160.2176  # GPa to eV/A^3
                compressibility_au = 1 / bulk_modulus_au
            else:
                try:
                    # Fit bulk modulus by equation of state
                    eos = EquationOfState(model=self.atoms.calc)
                    eos.fit(atoms=atoms, steps=500, fmax=0.3, verbose=False)
                    bulk_modulus = eos.get_bulk_modulus(unit="GPa")
                    bulk_modulus_au = eos.get_bulk_modulus(unit="eV/A^3")
                    compressibility_au = eos.get_compressibility(unit="A^3/eV")
                    print(
                        f"Completed bulk modulus calculation: "
                        f"k = {bulk_modulus:.3}GPa, {bulk_modulus_au:.3}eV/A^3"
                    )
                except Exception:
                    bulk_modulus_au = 2 / 160.2176
                    compressibility_au = 1 / bulk_modulus_au
                    warnings.warn(
                        "Warning!!! Equation of State fitting failed, setting bulk "
                        "modulus to 2 GPa. NPT simulation can proceed with incorrect "
                        "pressure relaxation time."
                        "User input for bulk modulus is recommended.",
                        stacklevel=2,
                    )
            self.bulk_modulus = bulk_modulus

            if thermostat.lower() == "nose-hoover":
                """
                Combined Nose-Hoover and Parrinello-Rahman dynamics, creating an
                NPT (or N,stress,T) ensemble.
                see: https://gitlab.com/ase/ase/-/blob/master/ase/md/npt.py
                ASE implementation currently only supports upper triangular lattice
                """
                self.upper_triangular_cell()
                ptime = taup * units.fs
                self.dyn = NPT(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    externalstress=pressure * units.GPa,
                    ttime=taut * units.fs,
                    pfactor=bulk_modulus * units.GPa * ptime * ptime,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )
                print("NPT-Nose-Hoover MD created")

            elif thermostat.lower() == "berendsen_inhomogeneous":
                """
                Inhomogeneous_NPTBerendsen thermo/barostat
                This is a more flexible scheme that fixes three angles of the unit
                cell but allows three lattice parameter to change independently.
                see: https://gitlab.com/ase/ase/-/blob/master/ase/md/nptberendsen.py
                """

                self.dyn = Inhomogeneous_NPTBerendsen(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    pressure_au=pressure * units.GPa,
                    taut=taut * units.fs,
                    taup=taup * units.fs,
                    compressibility_au=compressibility_au,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                )
                print("NPT-Berendsen-inhomogeneous-MD created")

            elif thermostat.lower() == "npt_berendsen":
                """
                This is a similar scheme to the Inhomogeneous_NPTBerendsen.
                This is a less flexible scheme that fixes the shape of the
                cell - three angles are fixed and the ratios between the three
                lattice constants.
                see: https://gitlab.com/ase/ase/-/blob/master/ase/md/nptberendsen.py
                """

                self.dyn = NPTBerendsen(
                    atoms=self.atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    pressure_au=pressure * units.GPa,
                    taut=taut * units.fs,
                    taup=taup * units.fs,
                    compressibility_au=compressibility_au,
                    trajectory=trajectory,
                    logfile=logfile,
                    loginterval=loginterval,
                    append_trajectory=append_trajectory,
                )
                print("NPT-Berendsen-MD created")
            else:
                raise ValueError(
                    "Thermostat not supported, choose in 'Nose-Hoover', 'Berendsen', "
                    "'Berendsen_inhomogeneous'"
                )

        self.trajectory = trajectory
        self.logfile = logfile
        self.loginterval = loginterval
        self.timestep = timestep
        self.crystal_feas_logfile = crystal_feas_logfile

    def run(self, steps: int) -> None:
        """Thin wrapper of ase MD run.

        Args:
            steps (int): number of MD steps
        """
        if self.crystal_feas_logfile:
            obs = CrystalFeasObserver(self.atoms)
            self.dyn.attach(obs, interval=self.loginterval)

        self.dyn.run(steps)

        if self.crystal_feas_logfile:
            obs.save(self.crystal_feas_logfile)

    def set_atoms(self, atoms: Atoms) -> None:
        """Set new atoms to run MD.

        Args:
            atoms (Atoms): new atoms for running MD
        """
        calculator = self.atoms.calc
        self.atoms = atoms
        self.dyn.atoms = atoms
        self.dyn.atoms.calc = calculator

    def upper_triangular_cell(self, *, verbose: bool | None = False) -> None:
        """Transform to upper-triangular cell.
        ASE Nose-Hoover implementation only supports upper-triangular cell
        while ASE's canonical description is lower-triangular cell.

        Args:
            verbose (bool): Whether to notify user about upper-triangular cell
                transformation. Default = False
        """
        if not NPT._isuppertriangular(self.atoms.get_cell()):  # noqa: SLF001
            a, b, c, alpha, beta, gamma = self.atoms.cell.cellpar()
            angles = np.radians((alpha, beta, gamma))
            sin_a, sin_b, _sin_g = np.sin(angles)
            cos_a, cos_b, cos_g = np.cos(angles)
            cos_p = (cos_g - cos_a * cos_b) / (sin_a * sin_b)
            cos_p = np.clip(cos_p, -1, 1)
            sin_p = (1 - cos_p ** 2) ** 0.5

            new_basis = [
                (a * sin_b * sin_p, a * sin_b * cos_p, a * cos_b),
                (0, b * sin_a, b * cos_a),
                (0, 0, c),
            ]

            self.atoms.set_cell(new_basis, scale_atoms=True)
            if verbose:
                print("Transformed to upper triangular unit cell.", flush=True)


class EquationOfState:
    """Class to calculate equation of state."""

    def __init__(
            self,
            model: CHGNet | CHGNetCalculator | None = None,
            optimizer_class: Optimizer | str | None = "FIRE",
            use_device: str | None = None,
            stress_weight: float = units.GPa,
            on_isolated_atoms: Literal["ignore", "warn", "error"] = "error",
    ) -> None:
        """Initialize a structure optimizer object for calculation of bulk modulus.

        Args:
            model (CHGNet): instance of a CHGNet model or CHGNetCalculator.
                If set to None, the pretrained CHGNet is loaded.
                Default = None
            optimizer_class (Optimizer,str): choose optimizer from ASE.
                Default = "FIRE"
            use_device (str, optional): The device to be used for predictions,
                either "cpu", "cuda", or "mps". If not specified, the default device is
                automatically selected based on the available options.
                Default = None
            stress_weight (float): the conversion factor to convert GPa to eV/A^3.
                Default = 1/160.21
            on_isolated_atoms ('ignore' | 'warn' | 'error'): how to handle Structures
                with isolated atoms.
                Default = 'error'
        """
        self.relaxer = StructOptimizer(
            model=model,
            optimizer_class=optimizer_class,
            use_device=use_device,
            stress_weight=stress_weight,
            on_isolated_atoms=on_isolated_atoms,
        )
        self.fitted = False

    def fit(
            self,
            atoms: Structure | Atoms,
            *,
            n_points: int = 11,
            fmax: float | None = 0.3,
            steps: int | None = 500,
            verbose: bool | None = False,
            **kwargs,
    ) -> None:
        """Relax the Structure/Atoms and fit the Birch-Murnaghan equation of state.

        Args:
            atoms (Structure | Atoms): A Structure or Atoms object to relax.
            n_points (int): Number of structures used in fitting the equation of states
            fmax (float | None): The maximum force tolerance for relaxation.
                Default = 0.3
            steps (int | None): The maximum number of steps for relaxation.
                Default = 500
            verbose (bool): Whether to print the output of the ASE optimizer.
                Default = False
            **kwargs: Additional parameters for the optimizer.
        """
        if isinstance(atoms, Atoms):
            atoms = AseAtomsAdaptor.get_structure(atoms)
        primitive_cell = atoms.get_primitive_structure()
        local_minima = self.relaxer.relax(
            primitive_cell,
            relax_cell=True,
            fmax=fmax,
            steps=steps,
            verbose=verbose,
            **kwargs,
        )

        volumes, energies = [], []
        for idx in np.linspace(-0.25, 0.25, n_points):
            structure_strained = local_minima["final_structure"].copy()
            structure_strained.apply_strain([idx, idx, idx])
            result = self.relaxer.relax(
                structure_strained,
                relax_cell=False,
                fmax=fmax,
                steps=steps,
                verbose=verbose,
                **kwargs,
            )
            volumes.append(result["final_structure"].volume)
            energies.append(result["trajectory"].energies[-1])
        self.bm = BirchMurnaghan(volumes=volumes, energies=energies)
        self.bm.fit()
        self.fitted = True

    def get_bulk_modulus(self, unit: Literal["eV/A^3", "GPa"] = "eV/A^3") -> float:
        """Get the bulk modulus of from the fitted Birch-Murnaghan equation of state.

        Args:
            unit (str): The unit of bulk modulus. Can be "eV/A^3" or "GPa"
                Default = "eV/A^3"

        Returns:
            float: Bulk Modulus

        Raises:
            ValueError: If the equation of state is not fitted.
        """
        if self.fitted is False:
            raise ValueError(
                "Equation of state needs to be fitted first through self.fit()"
            )
        if unit == "eV/A^3":
            return self.bm.b0
        if unit == "GPa":
            return self.bm.b0_GPa
        raise ValueError("unit has to be eV/A^3 or GPa")

    def get_compressibility(self, unit: str = "A^3/eV") -> float:
        """Get the bulk modulus of from the fitted Birch-Murnaghan equation of state.

        Args:
            unit (str): The unit of bulk modulus. Can be "A^3/eV",
            "GPa^-1" "Pa^-1" or "m^2/N"
                Default = "A^3/eV"

        Returns:
            Bulk Modulus (float)
        """
        if self.fitted is False:
            raise ValueError(
                "Equation of state needs to be fitted first through self.fit()"
            )
        if unit == "A^3/eV":
            return 1 / self.bm.b0
        if unit == "GPa^-1":
            return 1 / self.bm.b0_GPa
        if unit in {"Pa^-1", "m^2/N"}:
            return 1 / (self.bm.b0_GPa * 1e9)
        raise NotImplementedError("unit has to be one of A^3/eV, GPa^-1 Pa^-1 or m^2/N")