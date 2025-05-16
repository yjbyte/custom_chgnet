import os
import random
from ase import io
from ase.build import bulk
from ase.io import write
from ase import Atoms
from ase.optimize import BFGS
from ase.calculators.emt import EMT
import numpy as np
# python 版本是3.8


def length(structure, weidu):
    # 定义一个函数，用于获取晶体结构的晶胞c轴长度。
    # filename = structure  # 变量filename存储传入的晶体结构文件名。
    # atoms = io.read(filename)  # 使用io.read读取晶体结构文件。

    cell = structure.get_cell()  # 获取晶胞矩阵。
    # 假设晶胞是正交的，c轴长度是晶胞矩阵第三行的向量长度。
    c_length = cell[weidu][weidu]  # 计算c轴长度。
    return c_length  # 返回c轴长度。


def z_axis(structure1, structure2):
    c1 = length(structure1, 2)  # 调用length函数获取structure1c轴长度。
    c2 = length(structure2, 2)  # 调用length函数获取structure2c轴长度。

    # 获取a b 轴的长度
    # a1 = length(structure1, 0)
    # a2 = length(structure2, 0)
    # b1 = length(structure1, 1)
    # b2 = length(structure2, 1)

    # 计算structure1 和 structure2 对应 a b 轴的差
    # a = a1 if a1 > a2 else a2
    # b = b1 if b1 > b2 else b2

    # 计算系数
    scale = (c1 + c2) / c1

    print('structure1:')
    # 获取所有原子的元素类型
    chemical_symbols1 = structure1.get_chemical_symbols()
    for index, atom in enumerate(structure1):
        element = chemical_symbols1[index]
        position = atom.position
        # 恢复原子坐标
        atom.position[2] /= scale
        print(f"Atom {index} - Element: {element}, Coordinates: {position}")

    print('structure2:')
    # 获取所有原子的元素类型
    chemical_symbols2 = structure2.get_chemical_symbols()
    for index, atom in enumerate(structure2):
        element = chemical_symbols2[index]
        position = atom.position
        print(f"Atom {index} - Element: {element}, Coordinates: {position}")

    translation_vector = [0, 0, c1]  # 定义沿z轴的平移向量。

    structure2.translate(translation_vector)  # 将第二个晶体结构沿z轴平移。

    # 创建一个更大的晶胞来容纳两个晶体结构。
    supercell_matrix = list(structure1.get_cell())  # 获取原始晶胞矩阵。

    supercell_matrix[2][2] += c2  # 扩大晶胞在z方向的尺寸。
    # supercell_matrix[1][1] = b  # 扩大晶胞在y方向的尺寸。
    # supercell_matrix[0][0] = a  # 扩大晶胞在x方向的尺寸。

    new_supercell = structure1 * 1  # 创建超晶胞。

    new_supercell.set_cell(supercell_matrix, scale_atoms=True)  # 设置新的晶胞矩阵。
    new_supercell.extend(structure2)  # 将平移后的晶体结构添加到超晶胞中。

    new_supercell.pbc = (True, True, True)  # 设置周期性边界条件。

    print('new_supercell')
    # 获取所有原子的元素类型
    chemical_symbols = new_supercell.get_chemical_symbols()
    for index, atom in enumerate(new_supercell):
        element = chemical_symbols[index]
        position = atom.position
        print(f"Atom {index} - Element: {element}, Coordinates: {position}")

    return new_supercell


def y_axis(structure1, structure2):
    b1 = length(structure1, 1)  # 调用length函数获取structure1b轴长度。
    b2 = length(structure2, 1)  # 调用length函数获取structure2b轴长度。

    # # 获取a b 轴的长度
    # a1 = length(structure1, 0)
    # a2 = length(structure2, 0)
    # c1 = length(structure1, 2)
    # c2 = length(structure2, 2)

    # 计算structure1 和 structure2 对应 a b 轴的差
    # a = a1 if a1 > a2 else a2
    # c = c1 if c1 > c2 else c2

    # 计算系数
    scale = (b1 + b2) / b1

    print('structure1:')
    # 获取所有原子的元素类型
    chemical_symbols1 = structure1.get_chemical_symbols()
    for index, atom in enumerate(structure1):
        element = chemical_symbols1[index]
        position = atom.position
        # 恢复原子坐标
        atom.position[1] /= scale
        print(f"Atom {index} - Element: {element}, Coordinates: {position}")

    print('structure2:')
    # 获取所有原子的元素类型
    chemical_symbols2 = structure2.get_chemical_symbols()
    for index, atom in enumerate(structure2):
        element = chemical_symbols2[index]
        position = atom.position
        print(f"Atom {index} - Element: {element}, Coordinates: {position}")

    translation_vector = [0, b1, 0]  # 定义沿b轴的平移向量。

    structure2.translate(translation_vector)  # 将第二个晶体结构沿z轴平移。

    # 创建一个更大的晶胞来容纳两个晶体结构。
    supercell_matrix = list(structure1.get_cell())  # 获取原始晶胞矩阵。

    # supercell_matrix[2][2] = c  # 扩大晶胞在z方向的尺寸。
    supercell_matrix[1][1] += b2  # 扩大晶胞在y方向的尺寸。
    # supercell_matrix[0][0] = a  # 扩大晶胞在x方向的尺寸。

    new_supercell = structure1 * 1  # 创建超晶胞。

    new_supercell.set_cell(supercell_matrix, scale_atoms=True)  # 设置新的晶胞矩阵。
    new_supercell.extend(structure2)  # 将平移后的晶体结构添加到超晶胞中。

    new_supercell.pbc = (True, True, True)  # 设置周期性边界条件。

    print('new_supercell')
    # 获取所有原子的元素类型
    chemical_symbols = new_supercell.get_chemical_symbols()
    for index, atom in enumerate(new_supercell):
        element = chemical_symbols[index]
        position = atom.position
        print(f"Atom {index} - Element: {element}, Coordinates: {position}")

    return new_supercell


def x_axis(structure1, structure2):
    a1 = length(structure1, 0)  # 调用length函数获取structure1b轴长度。
    a2 = length(structure2, 0)  # 调用length函数获取structure2b轴长度。

    # 获取 b c 轴的长度
    # b1 = length(structure1, 1)
    # b2 = length(structure2, 1)
    # c1 = length(structure1, 2)
    # c2 = length(structure2, 2)

    # # 计算structure1 和 structure2 对应 a b 轴的差
    # b = b1 if b1 > b2 else b2
    # c = c1 if c1 > c2 else c2

    # 计算系数
    scale = (a1 + a2) / a1

    print('structure1:')
    # 获取所有原子的元素类型
    chemical_symbols1 = structure1.get_chemical_symbols()
    for index, atom in enumerate(structure1):
        element = chemical_symbols1[index]
        position = atom.position
        # 恢复原子坐标
        atom.position[0] /= scale
        print(f"Atom {index} - Element: {element}, Coordinates: {position}")

    print('structure2:')
    # 获取所有原子的元素类型
    chemical_symbols2 = structure2.get_chemical_symbols()
    for index, atom in enumerate(structure2):
        element = chemical_symbols2[index]
        position = atom.position
        print(f"Atom {index} - Element: {element}, Coordinates: {position}")

    translation_vector = [a1, 0, 0]  # 定义沿b轴的平移向量。

    structure2.translate(translation_vector)  # 将第二个晶体结构沿z轴平移。

    # 创建一个更大的晶胞来容纳两个晶体结构。
    supercell_matrix = list(structure1.get_cell())  # 获取原始晶胞矩阵。

    # supercell_matrix[2][2] = c  # 扩大晶胞在z方向的尺寸。
    # supercell_matrix[1][1] = b  # 扩大晶胞在y方向的尺寸。
    supercell_matrix[0][0] += a2  # 扩大晶胞在x方向的尺寸。

    new_supercell = structure1 * 1  # 创建超晶胞。

    new_supercell.set_cell(supercell_matrix, scale_atoms=True)  # 设置新的晶胞矩阵。
    new_supercell.extend(structure2)  # 将平移后的晶体结构添加到超晶胞中。

    new_supercell.pbc = (True, True, True)  # 设置周期性边界条件。

    print('new_supercell')
    # 获取所有原子的元素类型
    chemical_symbols = new_supercell.get_chemical_symbols()
    for index, atom in enumerate(new_supercell):
        element = chemical_symbols[index]
        position = atom.position
        print(f"Atom {index} - Element: {element}, Coordinates: {position}")

    return new_supercell


cif_file = os.listdir(r'stacking_data')
# os.chdir(r'G:\Ansys\DiffCSP-main\output\Na3MnCoNiO6')
i = 0
n = 3  # 指定是n*n*n

while i < len(cif_file):
    num = 1
    file_name = cif_file[i][0:-4]
    print(cif_file[i])
    # supercell = io.read(cif_file[i])
    supercell = io.read(os.path.join(r'stacking_data', cif_file[i]))
    while num < n:
        supercell = z_axis(supercell, io.read(os.path.join(r'stacking_data', cif_file[i])))
        num += 1
    num = 1
    # os.chdir(r'G:\Ansys\DiffCSP-main\output\Na3MnCoNiO6\z')
    write(r'stacking_data\z/%s.cif' % file_name, supercell)
    # os.chdir(r'..')

    supercell_y = io.read(r'stacking_data\z/%s.cif' % file_name)

    while num < n:
        supercell_y = y_axis(supercell_y, io.read(r'stacking_data\z/%s.cif' % file_name))
        num += 1
    num = 1

    # os.chdir(r'G:\Ansys\DiffCSP-main\output\Na3MnCoNiO6\y')
    write(r'stacking_data\y/%s.cif' % file_name, supercell_y)
    # os.chdir(r'..')

    supercell_x = io.read(r'stacking_data\y/%s.cif' % file_name)

    while num < n:
        supercell_x = x_axis(supercell_x, io.read(r'stacking_data\y/%s.cif' % file_name))
        num += 1
    # 输出新的CIF文件
    # os.chdir(r'x')  # 该目录下是最终堆叠得到的cif
    write(r'stacked_data\x/%s.cif' % file_name, supercell_x)
    print('-----------------------------------------------------------------------')
    print('-----------------------------------------------------------------------')
    print('-----------------------------------------------------------------------')
    print(f'--------------------------第{i+1}个cif生成完成-------------------------------')
    print('-----------------------------------------------------------------------')
    print('-----------------------------------------------------------------------')
    print('-----------------------------------------------------------------------')
    print('-----------------------------------------------------------------------')
    i += 1
    # os.chdir(r'../cif')



