import numpy as np
import matplotlib.pyplot as plt

# 解决matplotlib中文乱码
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 手动实现双曲余切 coth(x) = (e^x + e^-x)/(e^x - e^-x)
def coth(x):
    if abs(x) < 1e-12:
        return 0.0
    exp_x = np.exp(x)
    exp_negx = np.exp(-x)
    return (exp_x + exp_negx) / (exp_x - exp_negx)

def alpha_supg(Pe):
    """SUPG最优稳定参数 alpha_opt = coth(Pe) - 1/Pe"""
    if abs(Pe) < 1e-10:
        return 0.0
    return coth(Pe) - 1.0 / Pe

def element_matrix(kappa, v, le, alpha):
    """
    生成两节点线性单元对流扩散单元刚度矩阵 Ke(2×2)
    稳定化扩散系数 kappa_bar = kappa + alpha * v * le / 2
    """
    k_bar = kappa + alpha * v * le / 2.0
    # 扩散项单元矩阵
    K_diff = (k_bar / le) * np.array([[1, -1], [-1, 1]])
    # 对流项单元矩阵（非对称）
    K_adv = (v / 2.0) * np.array([[-1, 1], [-1, 1]])
    Ke = K_diff + K_adv
    return Ke

def exact_sol(x, v, kappa, L=1.0):
    """精确解 theta(x) = (exp(vx/kappa)-1)/(exp(vL/kappa)-1)"""
    Pe_global = v * L / kappa
    denom = np.exp(Pe_global) - 1
    return (np.exp(v * x / kappa) - 1) / denom

def solve_advection_diffusion(nel, L, v, target_Pe, alpha):
    """
    组装总刚、施加边界条件求解
    返回：节点坐标x, 数值解theta_num, 精确解theta_ex, 总刚矩阵K_global
    """
    le = L / nel
    # 由单元Pe反求扩散系数kappa：Pe = v*le/(2*kappa)
    kappa = v * le / (2 * target_Pe)

    # 1. 生成等距节点
    nnp = nel + 1
    x = np.linspace(0, L, nnp)
    ndof = nnp
    K_global = np.zeros((ndof, ndof))
    F_global = np.zeros((ndof, 1))

    # 2. 循环单元组装总刚
    for e in range(nel):
        Ke = element_matrix(kappa, v, le, alpha)
        # 单元局部自由度映射至全局
        i0, i1 = e, e + 1
        K_global[i0:i1+1, i0:i1+1] += Ke

    # 3. 施加本质边界条件 theta(0)=0, theta(L)=1
    # 左边界 x=0
    bc0 = 0.0
    row0 = 0
    K_global[row0, :] = 0.0
    K_global[row0, row0] = 1.0
    F_global[row0] = bc0

    # 右边界 x=L
    bcL = 1.0
    rowL = ndof - 1
    K_global[rowL, :] = 0.0
    K_global[rowL, rowL] = 1.0
    F_global[rowL] = bcL

    # 4. 求解线性方程组 K*theta=F
    theta_num = np.linalg.solve(K_global, F_global).flatten()
    theta_ex = exact_sol(x, v, kappa, L)

    return x, theta_num, theta_ex, K_global

def matrix_analysis(K):
    """分析矩阵：对称性、正定性"""
    # 对称误差：矩阵与其转置的二范数
    sym_err = np.linalg.norm(K - K.T)
    is_sym = sym_err < 1e-10

    # 特征值判断正定：所有实特征值大于0
    eig_vals = np.linalg.eigvals(K)
    min_eig = np.min(np.real(eig_vals))
    is_pos_def = min_eig > -1e-10

    print(f"===== 矩阵分析结果 =====")
    print(f"矩阵对称误差 ||K-K^T|| = {sym_err:.2e}")
    print(f"矩阵是否对称：{is_sym}")
    print(f"矩阵最小实特征值 = {min_eig:.2e}")
    print(f"矩阵是否正定：{is_pos_def}\n")
    return sym_err, min_eig, is_sym, is_pos_def

def plot_compare(Pe, res_dict):
    """绘制同一Pe下三种格式对比图"""
    plt.figure(figsize=(10,6))
    # 加密采样点绘制光滑解析解曲线
    x_ex = np.linspace(0,1,200)
    le = 1 / 20
    kappa = 1 * le / (2 * Pe)
    theta_ex_line = exact_sol(x_ex, v=1, kappa=kappa, L=1)
    plt.plot(x_ex, theta_ex_line, 'k--', lw=2, label='精确解析解')

    # 三种数值解绘图
    marker_list = ['o', 's', '^']
    label_list = ['标准Galerkin格式 α=0', '迎风格式 α=1', 'SUPG稳定格式 α_opt']
    key_list = ["gal", "upwind", "supg"]
    for idx, key in enumerate(key_list):
        x, num, ex, _ = res_dict[key]
        plt.plot(x, num, f'-{marker_list[idx]}', lw=1.5, label=label_list[idx])

    plt.xlabel("空间坐标 x")
    plt.ylabel(r"浓度 $\theta(x)$")
    plt.title(f"一维对流扩散数值解对比，单元Peclet数 Pe = {Pe}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_convergence_curve(nel_list, err_gal_list, err_up_list, err_supg_list):
    """附加题：绘制网格加密误差收敛曲线（对数坐标）"""
    plt.figure(figsize=(10,6))
    plt.loglog(nel_list, err_gal_list, 'o-', linewidth=2, label="标准Galerkin α=0")
    plt.loglog(nel_list, err_up_list, 's-', linewidth=2, label="迎风格式 α=1")
    plt.loglog(nel_list, err_supg_list, '^-', linewidth=2, label="SUPG稳定格式 α_opt")
    plt.xlabel("单元数量 nel")
    plt.ylabel("最大节点绝对误差 max|θ_num-θ_ex|")
    plt.title("Pe=3.0 不同网格密度下误差收敛曲线")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.show()

# ====================== 主程序入口 ======================
if __name__ == "__main__":
    # ========== 基础作业固定参数 ==========
    L = 1.0
    nel_base = 20
    v = 1.0
    Pe_list = [0.1, 3.0]

    # 存储所有工况结果
    all_results = {}
    error_table = []

    # 基础作业主计算流程
    for Pe in Pe_list:
        print("="*60)
        print(f"============ 当前单元Peclet数 Pe = {Pe} ============")
        print("="*60)
        res_tmp = {}
        # 1. 标准Galerkin α=0
        x_gal, num_gal, ex_gal, K_gal = solve_advection_diffusion(nel_base, L, v, Pe, alpha=0)
        err_gal = np.max(np.abs(num_gal - ex_gal))
        res_tmp["gal"] = (x_gal, num_gal, ex_gal, K_gal)
        error_table.append([Pe, "标准Galerkin(α=0)", err_gal])

        # 2. 迎风格式 α=1
        x_up, num_up, ex_up, K_up = solve_advection_diffusion(nel_base, L, v, Pe, alpha=1)
        err_up = np.max(np.abs(num_up - ex_up))
        res_tmp["upwind"] = (x_up, num_up, ex_up, K_up)
        error_table.append([Pe, "迎风格式(α=1)", err_up])

        # 3. SUPG最优alpha
        a_opt = alpha_supg(Pe)
        print(f"Pe={Pe} 最优稳定参数 α_opt = {a_opt:.6f}")
        x_supg, num_supg, ex_supg, K_supg = solve_advection_diffusion(nel_base, L, v, Pe, alpha=a_opt)
        err_supg = np.max(np.abs(num_supg - ex_supg))
        res_tmp["supg"] = (x_supg, num_supg, ex_supg, K_supg)
        error_table.append([Pe, f"SUPG稳定格式 α_opt={a_opt:.4f}", err_supg])

        all_results[Pe] = res_tmp

        # Pe=3时执行矩阵性质分析（作业任务4）
        if abs(Pe - 3.0) < 1e-6:
            print("\n-------- Pe=3.0 标准Galerkin总刚矩阵分析 --------")
            matrix_analysis(K_gal)

        # 绘制当前Pe对比曲线
        plot_compare(Pe, res_tmp)

    # 输出基础作业全局误差汇总表格
    print("\n" + "="*80)
    print("【基础作业：各离散格式最大节点绝对误差汇总表】")
    print(f"{'Peclet数':<8}{'计算格式':<26}{'最大节点误差':<20}")
    print("-"*80)
    for row in error_table:
        pe_val, method, err = row
        print(f"{pe_val:<8}{method:<26}{err:.6e}")
    print("="*80)

    # ====================== 附加题：网格加密收敛分析 ======================
    print("\n\n==================== 附加题：网格加密收敛分析 Pe=3.0 ====================")
    Pe_add = 3.0
    nel_list = [10, 20, 40, 80]  # 不同网格单元数量
    err_gal_list = []
    err_up_list = []
    err_supg_list = []
    add_error_table = []

    for nel in nel_list:
        print(f"\n当前网格单元数 nel = {nel}")
        # Galerkin α=0
        _, num_gal, ex_gal, _ = solve_advection_diffusion(nel, L, v, Pe_add, alpha=0)
        err_gal = np.max(np.abs(num_gal - ex_gal))
        err_gal_list.append(err_gal)

        # 迎风格式 α=1
        _, num_up, ex_up, _ = solve_advection_diffusion(nel, L, v, Pe_add, alpha=1)
        err_up = np.max(np.abs(num_up - ex_up))
        err_up_list.append(err_up)

        # SUPG
        a_opt = alpha_supg(Pe_add)
        _, num_supg, ex_supg, _ = solve_advection_diffusion(nel, L, v, Pe_add, alpha=a_opt)
        err_supg = np.max(np.abs(num_supg - ex_supg))
        err_supg_list.append(err_supg)

        add_error_table.append([nel, err_gal, err_up, err_supg])
        print(f"标准Galerkin误差：{err_gal:.6e} | 迎风格式误差：{err_up:.6e} | SUPG误差：{err_supg:.6e}")

    # 打印附加题误差表格
    print("\n" + "="*80)
    print("【附加题 Pe=3.0 不同网格密度误差表】")
    print(f"{'单元数nel':<10}{'Galerkin误差':<18}{'迎风格式误差':<18}{'SUPG误差':<18}")
    print("-"*80)
    for row in add_error_table:
        n, eg, eu, es = row
        print(f"{n:<10}{eg:.6e}        {eu:.6e}        {es:.6e}")
    print("="*80)

    # 绘制收敛曲线
    plot_convergence_curve(nel_list, err_gal_list, err_up_list, err_supg_list)