import numpy as np

def truss3d_element_stiffness(x1, x2, E, A):
    """
    功能：计算三维杆单元的长度、方向余弦、全局坐标系6×6刚度矩阵
    输入：
        x1: 节点1坐标 [x, y, z]
        x2: 节点2坐标 [x, y, z]
        E: 弹性模量 (Pa)
        A: 截面积 (m²)
    输出：
        L: 单元长度
        cx, cy, cz: 方向余弦
        Ke: 6×6单元刚度矩阵
    """
    # 转为numpy数组
    x1 = np.array(x1, dtype=float)
    x2 = np.array(x2, dtype=float)

    # 坐标差
    dx = x2[0] - x1[0]
    dy = x2[1] - x1[1]
    dz = x2[2] - x1[2]

    # 单元长度
    L = np.sqrt(dx**2 + dy**2 + dz**2)

    # 退化单元判断
    if L < 1e-15:
        raise Exception("错误：两个节点重合，单元退化，无法计算！")

    # 方向余弦
    cx = dx / L
    cy = dy / L
    cz = dz / L

    # 构造6×6刚度矩阵
    k = E * A / L
    Ke = np.zeros((6, 6))

    Ke[0,0] = k * cx**2
    Ke[0,1] = k * cx*cy
    Ke[0,2] = k * cx*cz
    Ke[0,3] = -k * cx**2
    Ke[0,4] = -k * cx*cy
    Ke[0,5] = -k * cx*cz

    Ke[1,0] = Ke[0,1]
    Ke[1,1] = k * cy**2
    Ke[1,2] = k * cy*cz
    Ke[1,3] = -k * cx*cy
    Ke[1,4] = -k * cy**2
    Ke[1,5] = -k * cy*cz

    Ke[2,0] = Ke[0,2]
    Ke[2,1] = Ke[1,2]
    Ke[2,2] = k * cz**2
    Ke[2,3] = -k * cx*cz
    Ke[2,4] = -k * cy*cz
    Ke[2,5] = -k * cz**2

    Ke[3,0] = Ke[0,3]
    Ke[3,1] = Ke[1,3]
    Ke[3,2] = Ke[2,3]
    Ke[3,3] = k * cx**2
    Ke[3,4] = k * cx*cy
    Ke[3,5] = k * cx*cz

    Ke[4,0] = Ke[0,4]
    Ke[4,1] = Ke[1,4]
    Ke[4,2] = Ke[2,4]
    Ke[4,3] = Ke[3,4]
    Ke[4,4] = k * cy**2
    Ke[4,5] = k * cy*cz

    Ke[5,0] = Ke[0,5]
    Ke[5,1] = Ke[1,5]
    Ke[5,2] = Ke[2,5]
    Ke[5,3] = Ke[3,5]
    Ke[5,4] = Ke[4,5]
    Ke[5,5] = k * cz**2

    return L, (cx, cy, cz), Ke


def truss3d_element_stress(x1, x2, E, A, de):
    """
    功能：由节点位移计算单元应变、应力、轴力
    输入：
        de: 节点位移 [u1,v1,w1,u2,v2,w2]
    输出：
        epsilon: 应变
        sigma: 应力 (Pa)
        N: 轴力 (N)
    """
    de = np.array(de, dtype=float)
    L, (cx, cy, cz), _ = truss3d_element_stiffness(x1, x2, E, A)

    # 轴向伸长量
    delta = cx*(de[3]-de[0]) + cy*(de[4]-de[1]) + cz*(de[5]-de[2])

    # 应变
    epsilon = delta / L

    # 应力与轴力
    sigma = E * epsilon
    N = sigma * A

    return epsilon, sigma, N


# ===================== 算例1：沿x轴杆单元 =====================
print("="*50)
print("               算例1：沿x轴一维杆单元")
print("="*50)

x1 = [0,0,0]
x2 = [2,0,0]
E = 200e9
A = 1.0e-4
de = [0,0,0, 1e-3,0,0]

L1, dir1, Ke1 = truss3d_element_stiffness(x1,x2,E,A)
eps1, sig1, N1 = truss3d_element_stress(x1,x2,E,A,de)

print(f"单元长度 L = {L1:.2f} m")
print(f"方向余弦 (cx,cy,cz) = ({dir1[0]:.0f}, {dir1[1]:.0f}, {dir1[2]:.0f})")
print(f"轴向应变 ε = {eps1:.2e}")
print(f"轴向应力 σ = {sig1/1e6:.2f} MPa")
print(f"轴力 N = {N1:.1e} N")

# ===================== 算例2：空间任意方向杆单元 =====================
print("\n" + "="*50)
print("             算例2：空间任意方向杆单元")
print("="*50)

x1 = [0,0,0]
x2 = [1,2,2]
E = 210e9
A = 2.0e-4
de = [0,0,0, 1e-3, 2e-3, 2e-3]

L2, dir2, Ke2 = truss3d_element_stiffness(x1,x2,E,A)
eps2, sig2, N2 = truss3d_element_stress(x1,x2,E,A,de)

print(f"单元长度 L = {L2:.1f} m")
print(f"方向余弦 (cx,cy,cz) = ({dir2[0]:.3f}, {dir2[1]:.3f}, {dir2[2]:.3f})")
print(f"轴向应变 ε = {eps2:.2e}")
print(f"轴向应力 σ = {sig2/1e6:.2f} MPa")
print(f"轴力 N = {N2:.1e} N")

# 验证刚度矩阵对称性
sym_check = np.allclose(Ke2, Ke2.T)
print(f"\n刚度矩阵是否对称: {sym_check}")