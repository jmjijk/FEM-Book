import numpy as np
import json
import time
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from math import pi

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# ==============================
# 一、复用 2.3 作业全部基础模块
# 模型读取、单元刚度、LM矩阵、总刚组装、自由度分块
# ==============================
def read_model(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    nsd = data["nsd"]
    ndof_node = data["ndof"]
    nnp = data["nnp"]
    nel = data["nel"]
    nen = data["nen"]
    E = np.array(data["E"], dtype=float)
    A = np.array(data["CArea"], dtype=float)
    x = np.array(data["x"], dtype=float)
    y = np.array(data["y"], dtype=float)
    IEN = np.array(data["IEN"], dtype=int) - 1
    fixed_dof = np.array(data["fixed_dof"], dtype=int) - 1
    fixed_val = np.array(data["fixed_value"], dtype=float)
    force_dof = np.array(data["force_dof"], dtype=int) - 1
    force_val = np.array(data["force_value"], dtype=float)

    total_dof = nnp * ndof_node
    F = np.zeros(total_dof)
    for dof, val in zip(force_dof, force_val):
        F[dof] = val
    return nsd, ndof_node, nnp, nel, nen, E, A, x, y, IEN, fixed_dof, fixed_val, F

def build_LM(IEN, ndof_node, nel, nen):
    local_dof_num = nen * ndof_node
    LM = np.zeros((local_dof_num, nel), dtype=int)
    for e in range(nel):
        idx = 0
        for node_id in IEN[e]:
            for d in range(ndof_node):
                LM[idx, e] = node_id * ndof_node + d
                idx += 1
    return LM

def get_ke_1d(e, n1, n2, E, A, x):
    L = abs(x[n2] - x[n1])
    ke = (E[e] * A[e] / L) * np.array([[1, -1], [-1, 1]])
    return ke, L

def get_ke_2d(e, n1, n2, E, A, x, y):
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

def assemble_global_K(K, ke, LM, e):
    local_dofs = LM[:, e]
    for i in range(len(local_dofs)):
        for j in range(len(local_dofs)):
            gi = local_dofs[i]
            gj = local_dofs[j]
            K[gi, gj] += ke[i, j]

def partition_dof(K, F, fixed_dof, fixed_val):
    total_dof = K.shape[0]
    all_dof = np.arange(total_dof)
    free_dof = np.setdiff1d(all_dof, fixed_dof)
    Kff = K[np.ix_(free_dof, free_dof)]
    Kef = K[np.ix_(fixed_dof, free_dof)]
    Ff = F[free_dof]
    rhs = Ff - Kef.T @ fixed_val
    return free_dof, Kff, rhs

# 后处理：重构位移、约束反力、单元应力/轴力
def post_process_truss(d, K, F, fixed_dof, LM, IEN, E, A, x, y, is_2d):
    reaction = K @ d - F
    nel = IEN.shape[0]
    unit_res = []
    for e in range(nel):
        n1, n2 = IEN[e]
        local_dofs = LM[:, e]
        de = d[local_dofs]
        if is_2d:
            _, L, c, s = get_ke_2d(e, n1, n2, E, A, x, y)
            sigma = (E[e] / L) * np.array([-c, -s, c, s]) @ de
        else:
            _, L = get_ke_1d(e, n1, n2, E, A, x)
            sigma = (E[e] / L) * np.array([-1, 1]) @ de
        force = sigma * A[e]
        unit_res.append((L, sigma, force))
    return reaction, unit_res

# ==============================
# 二、自研 LDL^T 求解器（核心任务1）
# K = L @ D @ L.T，L单位下三角，D对角阵
# ==============================
def ldlt_factor(K):
    """
    LDL^T 分解，输入对称矩阵K，返回 L, D
    检测非正主元并抛出异常
    """
    n = K.shape[0]
    A = K.copy().astype(float)
    L = np.eye(n)
    D = np.zeros(n)

    for j in range(n):
        # 计算对角元 D[j]
        sum_d = 0.0
        for k in range(j):
            sum_d += L[j, k] ** 2 * D[k]
        D[j] = A[j, j] - sum_d

        # 检测非正主元
        if D[j] <= 1e-12:
            raise ValueError(f"分解失败：第{j}个主元非正，矩阵非正定/奇异")

        # 计算L第j列下方元素
        for i in range(j + 1, n):
            sum_l = 0.0
            for k in range(j):
                sum_l += L[i, k] * L[j, k] * D[k]
            L[i, j] = (A[i, j] - sum_l) / D[j]
    return L, D

def ldlt_solve(L, D, R):
    """
    求解 L D L^T x = R
    三步：前代 -> 对角求解 -> 回代
    """
    n = len(R)
    # 1. 前代: L y = R
    y = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i, j] * y[j]
        y[i] = R[i] - s

    # 2. 对角求解: D z = y
    z = y / D

    # 3. 回代: L^T x = z
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        s = 0.0
        for j in range(i+1, n):
            s += L[j, i] * x[j]
        x[i] = z[i] - s
    return x

def residual_norm(K, x, R):
    """计算残差、残差2-范数"""
    r = R - K @ x
    r_norm = np.linalg.norm(r, 2)
    return r, r_norm

# ==============================
# 三、统一求解接口
# ==============================
def solve_equilibrium(K_FF, rhs, method="ldlt", **options):
    """
    统一求解接口
    :param K_FF: 缩减刚度矩阵
    :param rhs: 右端向量
    :param method: ldlt / sparse
    :return: 解向量x, 残差范数, 耗时
    """
    t0 = time.perf_counter()
    if method == "ldlt":
        try:
            L, D = ldlt_factor(K_FF)
            x = ldlt_solve(L, D, rhs)
        except Exception as e:
            raise e
    elif method == "sparse":
        K_sp = csr_matrix(K_FF)
        x = spsolve(K_sp, rhs)
    else:
        raise NotImplementedError("仅支持 ldlt / sparse")
    t1 = time.perf_counter()
    r, r_norm = residual_norm(K_FF, x, rhs)
    return x, r_norm, t1 - t0

# ==============================
# 四、任务2：病态矩阵、误差、条件数分析
# ==============================
def ill_condition_test():
    print("\n" + "="*60)
    print("【任务2：病态矩阵测试】")
    K = np.array([[1.0, 1.0], [1.0, 1.0001]])
    a_exact = np.array([1.0, 1.0])
    R = K @ a_exact
    cond = np.linalg.cond(K)
    print(f"矩阵K:\n{K}")
    print(f"精确解 a_exact = {a_exact}")
    print(f"右端 R = {R}")
    print(f"矩阵条件数 cond(K) = {cond:.2e}")

    # 1. 双精度求解
    x_dp, r_dp, _ = solve_equilibrium(K, R, method="ldlt")
    err_dp = np.linalg.norm(x_dp - a_exact) / np.linalg.norm(a_exact)
    rel_r_dp = r_dp / np.linalg.norm(R)
    print(f"\n双精度结果:")
    print(f"数值解: {x_dp}, 相对残差: {rel_r_dp:.2e}, 相对误差: {err_dp:.2e}")

    # 2. 四舍五入4位有效数字
    K_4dig = np.round(K, 4)
    R_4dig = np.round(R, 4)
    x_4, r_4, _ = solve_equilibrium(K_4dig, R_4dig, method="ldlt")
    err_4 = np.linalg.norm(x_4 - a_exact) / np.linalg.norm(a_exact)
    rel_r_4 = r_4 / np.linalg.norm(R_4dig)
    print(f"4位有效数字结果:")
    print(f"数值解: {x_4}, 相对残差: {rel_r_4:.2e}, 相对误差: {err_4:.2e}")
    print("结论：病态矩阵残差很小，但解误差可能很大")

# ==============================
# 五、算例1：三对角对称正定矩阵
# ==============================
def tridiagonal_test():
    print("\n" + "="*60)
    print("【算例1：多阶三对角矩阵测试】")
    for n in [10, 100, 500, 1000]:
        K = np.zeros((n, n))
        for i in range(n):
            K[i,i] = 2.0
            if i > 0:
                K[i,i-1] = -1.0
                K[i-1,i] = -1.0
        a_exact = np.ones(n)
        R = K @ a_exact
        x, r_norm, t = solve_equilibrium(K, R, method="ldlt")
        err = np.linalg.norm(x - a_exact) / np.linalg.norm(a_exact)
        print(f"阶数 n={n:4d} | 耗时={t:.4f}s | 残差范数={r_norm:.2e} | 相对误差={err:.2e}")

# ==============================
# 六、算例2：非正定矩阵检测
# ==============================
def non_positive_test():
    print("\n" + "="*60)
    print("【算例2：非正定矩阵检测】")
    K = np.array([[1.0, 2.0], [2.0, 1.0]])
    R = np.array([1.0, 1.0])
    print(f"测试矩阵 K:\n{K}")
    try:
        L, D = ldlt_factor(K)
        x = ldlt_solve(L, D, R)
        print("分解成功，解：", x)
    except ValueError as e:
        print("检测到非正定，提示：", e)

# ==============================
# 七、算例0：复用2.3桁架模型（一维杆 + 二维桁架）
# ==============================
def run_truss_case():
    # 算例0-1 一维杆
    print("\n" + "="*60)
    print("【算例0-1：2.3 一维两单元杆】")
    case1 = {
        "Title": "1D bar", "nsd":1, "ndof":1, "nnp":3, "nel":2, "nen":2,
        "E":[100,200], "CArea":[1,1], "x":[0,1,2], "y":[0,0,0],
        "IEN":[[1,2],[2,3]], "fixed_dof":[1], "fixed_value":[0.0],
        "force_dof":[3], "force_value":[10.0]
    }
    with open("case1_1d.json", "w") as f:
        json.dump(case1, f, indent=2)
    nsd, ndof_node, nnp, nel, nen, E, A, x, y, IEN, fixed_dof, fixed_val, F = read_model("case1_1d.json")
    total_dof = nnp * ndof_node
    K = np.zeros((total_dof, total_dof))
    LM = build_LM(IEN, ndof_node, nel, nen)
    for e in range(nel):
        n1, n2 = IEN[e]
        ke, _ = get_ke_1d(e, n1, n2, E, A, x)
        assemble_global_K(K, ke, LM, e)
    free_dof, Kff, rhs = partition_dof(K, F, fixed_dof, fixed_val)
    x_free, r_norm, t = solve_equilibrium(Kff, rhs, method="ldlt")
    # 重构全局位移
    d = np.zeros(total_dof)
    d[free_dof] = x_free
    d[fixed_dof] = fixed_val
    reaction, unit_res = post_process_truss(d, K, F, fixed_dof, LM, IEN, E, A, x, y, is_2d=False)
    print(f"总刚矩阵 K:\n{np.round(K,4)}")
    print(f"节点位移 d = {np.round(d,4)}")
    print(f"约束反力 = {np.round(reaction,4)}")
    for idx, (L, sig, force) in enumerate(unit_res):
        print(f"单元{idx+1}: 长度={L:.2f}, 应力={sig:.4f}, 轴力={force:.4f}")

    # 算例0-2 二维桁架
    print("\n" + "="*60)
    print("【算例0-2：2.3 二维两杆桁架】")
    case2 = {
        "Title":"2D truss","nsd":2,"ndof":2,"nnp":3,"nel":2,"nen":2,
        "E":[1.0,1.0],"CArea":[1.0,1.0],"x":[1.0,0.0,1.0],"y":[0.0,0.0,1.0],
        "IEN":[[1,3],[2,3]],"fixed_dof":[1,2,3,4],"fixed_value":[0,0,0,0],
        "force_dof":[5,6],"force_value":[10,0]
    }
    with open("case2_2d.json", "w") as f:
        json.dump(case2, f, indent=2)
    nsd, ndof_node, nnp, nel, nen, E, A, x, y, IEN, fixed_dof, fixed_val, F = read_model("case2_2d.json")
    total_dof = nnp * ndof_node
    K = np.zeros((total_dof, total_dof))
    LM = build_LM(IEN, ndof_node, nel, nen)
    for e in range(nel):
        n1, n2 = IEN[e]
        ke, _, _, _ = get_ke_2d(e, n1, n2, E, A, x, y)
        assemble_global_K(K, ke, LM, e)
    free_dof, Kff, rhs = partition_dof(K, F, fixed_dof, fixed_val)
    x_free, r_norm, t = solve_equilibrium(Kff, rhs, method="ldlt")
    d = np.zeros(total_dof)
    d[free_dof] = x_free
    d[fixed_dof] = fixed_val
    reaction, unit_res = post_process_truss(d, K, F, fixed_dof, LM, IEN, E, A, x, y, is_2d=True)
    print(f"节点位移 u3={d[4]:.6f}, v3={d[5]:.6f}")
    for idx, (L, sig, force) in enumerate(unit_res):
        print(f"单元{idx+1}: 应力={sig:.6f}, 轴力={force:.6f}")

# ==============================
# 八、算例4：二维Poisson方程 Q4单元 有限元 + 稀疏求解
# ==============================
def poisson_q4_fem(nx, ny):
    """
    单位正方形 Poisson 方程 -Δu = 2π²sin(πx)sin(πy)，四边u=0
    Q4双线性四边形单元
    """
    print(f"\n{'='*60}\n【算例4：Poisson方程 Q4单元 nx={nx}, ny={ny}】")
    # 1. 网格生成
    x_coord = np.linspace(0, 1, nx+1)
    y_coord = np.linspace(0, 1, ny+1)
    X, Y = np.meshgrid(x_coord, y_coord)
    nnp = (nx+1)*(ny+1)
    nel = nx * ny
    node_xy = np.zeros((nnp, 2))
    idx = 0
    for j in range(ny+1):
        for i in range(nx+1):
            node_xy[idx] = [X[j,i], Y[j,i]]
            idx += 1
    # 单元IEN (Q4: 4节点)
    IEN = np.zeros((nel,4), dtype=int)
    e_idx = 0
    for j in range(ny):
        for i in range(nx):
            n0 = j*(nx+1) + i
            n1 = n0 + 1
            n2 = n0 + (nx+1) + 1
            n3 = n0 + (nx+1)
            IEN[e_idx] = [n0, n1, n2, n3]
            e_idx += 1

    # 2. 单元刚度 & 单元载荷 (Q4 积分 2×2高斯点)
    def q4_element(nd_xy):
        xi_gauss = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        w_gauss = np.array([1.0, 1.0])
        ke = np.zeros((4,4))
        fe = np.zeros(4)
        for xi in xi_gauss:
            for eta in xi_gauss:
                N = 0.25 * np.array([
                    (1-xi)*(1-eta), (1+xi)*(1-eta),
                    (1+xi)*(1+eta), (1-xi)*(1+eta)
                ])
                dN_dxi = 0.25 * np.array([
                    -(1-eta), 1-eta, 1+eta, -(1+eta)
                ])
                dN_deta = 0.25 * np.array([
                    -(1-xi), -(1+xi), 1+xi, 1-xi
                ])
                # 雅可比
                J = np.zeros((2,2))
                for i in range(4):
                    J[0,0] += dN_dxi[i] * nd_xy[i,0]
                    J[0,1] += dN_dxi[i] * nd_xy[i,1]
                    J[1,0] += dN_deta[i] * nd_xy[i,0]
                    J[1,1] += dN_deta[i] * nd_xy[i,1]
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)
                dN = np.vstack((invJ @ np.vstack((dN_dxi, dN_deta))))
                # 右端项 f = 2π² sin(πx)sin(πy)
                xg = N @ nd_xy[:,0]
                yg = N @ nd_xy[:,1]
                f_val = 2 * pi**2 * np.sin(pi*xg) * np.sin(pi*yg)
                # 组装 ke, fe
                ke += (dN.T @ dN) * detJ
                fe += N * f_val * detJ
        return ke, fe

    # 3. 总体组装
    t_assemble = time.perf_counter()
    total_dof = nnp
    K = np.zeros((total_dof, total_dof))
    F = np.zeros(total_dof)
    for e in range(nel):
        nds = IEN[e]
        nd_xy = node_xy[nds]
        ke, fe = q4_element(nd_xy)
        for i in range(4):
            for j in range(4):
                K[nds[i], nds[j]] += ke[i,j]
            F[nds[i]] += fe[i]
    t_assemble = time.perf_counter() - t_assemble

    # 4. 边界条件：四边u=0
    fixed_dof = []
    for n in range(nnp):
        xn, yn = node_xy[n]
        if abs(xn)<1e-6 or abs(xn-1)<1e-6 or abs(yn)<1e-6 or abs(yn-1)<1e-6:
            fixed_dof.append(n)
    fixed_dof = np.array(fixed_dof)
    fixed_val = np.zeros_like(fixed_dof)
    free_dof, Kff, rhs = partition_dof(K, F, fixed_dof, fixed_val)

    # 5. 稀疏求解
    t_solve = time.perf_counter()
    x_free, r_norm, _ = solve_equilibrium(Kff, rhs, method="sparse")
    t_solve = time.perf_counter() - t_solve
    u = np.zeros(total_dof)
    u[free_dof] = x_free

    # 6. 误差计算
    u_exact = np.sin(pi*node_xy[:,0]) * np.sin(pi*node_xy[:,1])
    err_node = np.abs(u - u_exact)
    max_err = np.max(err_node)
    l2_err = np.linalg.norm(u - u_exact) / np.linalg.norm(u_exact)
    nnz = np.count_nonzero(K)

    print(f"节点数:{nnp}, 单元数:{nel}, 非零元:{nnz}")
    print(f"装配时间:{t_assemble:.4f}s, 求解时间:{t_solve:.4f}s")
    print(f"最大节点误差:{max_err:.2e}, L2相对误差:{l2_err:.2e}, 相对残差:{r_norm/np.linalg.norm(F):.2e}")

    # 7. 绘图
    U_mat = u.reshape(ny+1, nx+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.contourf(X, Y, U_mat, 20, cmap="jet")
    plt.colorbar()
    plt.title(f"Poisson 数值解 nx={nx}")
    Err_mat = err_node.reshape(ny+1, nx+1)
    plt.subplot(1,2,2)
    plt.contourf(X, Y, Err_mat, 20, cmap="jet")
    plt.colorbar()
    plt.title("误差分布")
    plt.tight_layout()
    plt.show()

# ==============================
# 主程序入口：依次运行所有算例
# ==============================
if __name__ == "__main__":
    # 1. 2.3桁架衔接算例
    run_truss_case()
    # 2. 病态矩阵误差分析
    ill_condition_test()
    # 3. 三对角矩阵性能测试
    tridiagonal_test()
    # 4. 非正定矩阵检测
    non_positive_test()
    # 5. Poisson方程稀疏有限元（小规模网格）
    poisson_q4_fem(nx=20, ny=20)