import numpy as np
import json

# -------------------------- 1. 前处理模块 --------------------------
def read_model(json_path):
    """读取JSON模型文件，统一转为0基索引"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nsd = data["nsd"]         # 空间维数
    ndof_node = data["ndof"]  # 单节点自由度
    nnp = data["nnp"]         # 总节点数
    nel = data["nel"]         # 总单元数
    nen = data["nen"]         # 单单元节点数
    E = np.array(data["E"], dtype=float)
    A = np.array(data["CArea"], dtype=float)

    # 节点坐标
    x = np.array(data["x"], dtype=float)
    y = np.array(data["y"], dtype=float)

    # 单元连接 IEN: 1基 → 0基
    IEN = np.array(data["IEN"], dtype=int) - 1

    # 约束自由度 & 约束值: 1基 → 0基
    fixed_dof = np.array(data["fixed_dof"], dtype=int) - 1
    fixed_val = np.array(data["fixed_value"], dtype=float)

    # 节点载荷: 1基 → 0基
    force_dof = np.array(data["force_dof"], dtype=int) - 1
    force_val = np.array(data["force_value"], dtype=float)

    # 初始化总体载荷向量
    total_dof = nnp * ndof_node
    F = np.zeros(total_dof)
    for dof, val in zip(force_dof, force_val):
        F[dof] = val

    return nsd, ndof_node, nnp, nel, nen, E, A, x, y, IEN, fixed_dof, fixed_val, F

# -------------------------- 2. 生成对号矩阵 LM --------------------------
def build_LM(IEN, ndof_node, nel, nen):
    """
    构造对号矩阵 LM: (单元内自由度数量, 单元数)
    LM[i,e] = 第e个单元第i个局部自由度 对应的 全局自由度编号
    """
    local_dof_num = nen * ndof_node
    LM = np.zeros((local_dof_num, nel), dtype=int)
    for e in range(nel):
        idx = 0
        for node_id in IEN[e]:
            for d in range(ndof_node):
                LM[idx, e] = node_id * ndof_node + d
                idx += 1
    return LM

# -------------------------- 3. 单元刚度矩阵计算 --------------------------
def get_ke_1d(e, n1, n2, E, A, x):
    """一维杆单元刚度矩阵"""
    L = abs(x[n2] - x[n1])
    ke = (E[e] * A[e] / L) * np.array([[1, -1], [-1, 1]])
    return ke, L

def get_ke_2d(e, n1, n2, E, A, x, y):
    """二维桁架单元刚度矩阵 + 方向余弦"""
    dx = x[n2] - x[n1]
    dy = y[n2] - y[n1]
    L = np.hypot(dx, dy)
    c = dx / L
    s = dy / L
    k0 = E[e] * A[e] / L

    ke = k0 * np.array([
        [c**2,  c*s, -c**2, -c*s],
        [c*s,  s**2, -c*s, -s**2],
        [-c**2, -c*s, c**2,  c*s],
        [-c*s, -s**2, c*s,  s**2]
    ])
    return ke, L, c, s

# -------------------------- 4. 总体刚度矩阵组装 --------------------------
def assemble_global_K(K, ke, LM, e):
    """根据LM矩阵，将单元刚度ke累加到总刚K"""
    local_dofs = LM[:, e]
    for i in range(len(local_dofs)):
        for j in range(len(local_dofs)):
            gi = local_dofs[i]
            gj = local_dofs[j]
            K[gi, gj] += ke[i, j]

# -------------------------- 5. 缩减法求解方程 --------------------------
def solve_equation(K, F, fixed_dof, fixed_val):
    """缩减法处理位移边界，求解位移+约束反力"""
    total_dof = K.shape[0]
    all_dof = np.arange(total_dof)
    free_dof = np.setdiff1d(all_dof, fixed_dof)

    # 分块矩阵
    Kff = K[np.ix_(free_dof, free_dof)]
    Kef = K[np.ix_(fixed_dof, free_dof)]
    Ff = F[free_dof]
    d_e = fixed_val

    # 求解自由位移: Kff * df = Ff - Kef^T * de
    df = np.linalg.solve(Kff, Ff - Kef.T @ d_e)

    # 重构完整位移向量
    d = np.zeros(total_dof)
    d[free_dof] = df
    d[fixed_dof] = d_e

    # 计算约束反力
    reaction = K @ d - F
    return d, reaction, free_dof

# -------------------------- 6. 后处理：单元应力、轴力 --------------------------
def post_1d(e, n1, n2, E, A, x, d, LM):
    ke, L = get_ke_1d(e, n1, n2, E, A, x)
    local_dofs = LM[:, e]
    de = d[local_dofs]
    sigma = (E[e] / L) * np.array([-1, 1]) @ de
    force = sigma * A[e]
    return L, sigma, force

def post_2d(e, n1, n2, E, A, x, y, d, LM):
    ke, L, c, s = get_ke_2d(e, n1, n2, E, A, x, y)
    local_dofs = LM[:, e]
    de = d[local_dofs]
    sigma = (E[e] / L) * np.array([-c, -s, c, s]) @ de
    force = sigma * A[e]
    return L, c, s, sigma, force

# -------------------------- 主执行函数 --------------------------
def run_fem(json_path, is_2d):
    # 1. 前处理
    nsd, ndof_node, nnp, nel, nen, E, A, x, y, IEN, fixed_dof, fixed_val, F = read_model(json_path)
    total_dof = nnp * ndof_node
    K = np.zeros((total_dof, total_dof))

    # 2. 生成对号矩阵LM
    LM = build_LM(IEN, ndof_node, nel, nen)
    print("=" * 50)
    print("【对号矩阵 LM】")
    print(LM)

    # 3. 逐单元计算刚度 + 组装总刚
    for e in range(nel):
        n1, n2 = IEN[e]
        if is_2d:
            ke, _, _, _ = get_ke_2d(e, n1, n2, E, A, x, y)
        else:
            ke, _ = get_ke_1d(e, n1, n2, E, A, x)
        assemble_global_K(K, ke, LM, e)

    # 输出总刚矩阵、对称性、奇异性
    print("\n【总体刚度矩阵 K】")
    print(np.round(K, 4))
    is_sym = np.allclose(K, K.T)
    print(f"矩阵对称性: {is_sym}")
    try:
        np.linalg.inv(K)
        print("施加边界前: 矩阵非奇异")
    except np.linalg.LinAlgError:
        print("施加边界前: 矩阵奇异（桁架/杆结构刚体位移，正常）")

    # 4. 求解位移、约束反力
    d, reaction, free_dof = solve_equation(K, F, fixed_dof, fixed_val)
    print("\n【全局节点位移】")
    print(np.round(d, 6))
    print("\n【约束反力】")
    print(np.round(reaction, 6))

    # 5. 后处理：单元结果
    print("\n【单元计算结果】")
    for e in range(nel):
        n1, n2 = IEN[e]
        if is_2d:
            L, c, s, sigma, force = post_2d(e, n1, n2, E, A, x, y, d, LM)
            print(f"单元{e+1}: 长度={L:.4f}, 方向余弦c={c:.4f}, s={s:.4f}, 应力={sigma:.6f}, 轴力={force:.6f}")
        else:
            L, sigma, force = post_1d(e, n1, n2, E, A, x, d, LM)
            print(f"单元{e+1}: 长度={L:.4f}, 应力={sigma:.6f}, 轴力={force:.6f}")
    print("=" * 50 + "\n")

# -------------------------- 程序入口 --------------------------
if __name__ == "__main__":
    # ========== 算例1：一维两单元杆结构（自动生成JSON） ==========
    case1_data = {
        "Title": "1D Two-Bar Structure",
        "nsd": 1,
        "ndof": 1,
        "nnp": 3,
        "nel": 2,
        "nen": 2,
        "E": [100.0, 200.0],
        "CArea": [1.0, 1.0],
        "x": [0.0, 1.0, 2.0],
        "y": [0.0, 0.0, 0.0],
        "IEN": [[1, 2], [2, 3]],
        "fixed_dof": [1],
        "fixed_value": [0.0],
        "force_dof": [3],
        "force_value": [10.0]
    }
    with open("case1_1d.json", "w", encoding="utf-8") as f:
        json.dump(case1_data, f, indent=2)
    print(">>>>>>>>>> 开始计算 算例1：一维杆结构 <<<<<<<<<<")
    run_fem("case1_1d.json", is_2d=False)

    # ========== 算例2：二维两杆桁架结构（自动生成JSON） ==========
    case2_data = {
        "Title": "2D Truss Structure",
        "nsd": 2,
        "ndof": 2,
        "nnp": 3,
        "nel": 2,
        "nen": 2,
        "E": [1.0, 1.0],
        "CArea": [1.0, 1.0],
        "x": [1.0, 0.0, 1.0],
        "y": [0.0, 0.0, 1.0],
        "IEN": [[1, 3], [2, 3]],
        "fixed_dof": [1, 2, 3, 4],
        "fixed_value": [0.0, 0.0, 0.0, 0.0],
        "force_dof": [5, 6],
        "force_value": [10.0, 0.0]
    }
    with open("case2_2d.json", "w", encoding="utf-8") as f:
        json.dump(case2_data, f, indent=2)
    print(">>>>>>>>>> 开始计算 算例2：二维桁架结构 <<<<<<<<<<")
    run_fem("case2_2d.json", is_2d=True)