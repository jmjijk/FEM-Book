import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 设置绘图参数
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 基础参数定义
pi_exact = np.pi
n_values = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
h_values = 1 / n_values

# 2. 计算正多边形逼近的π值和误差
pi_n = n_values * np.sin(np.pi / n_values)
e_n = np.abs(pi_exact - pi_n)

# 3. 模拟外插后的误差
extrapolated_h = 1 / np.array([4, 16, 64, 256])
extrapolated_pi = np.array([3.414213562373096, 3.141418327933211,
                            3.141592658918053, 3.141592653589786])
extrapolated_error = np.abs(pi_exact - extrapolated_pi)

# 拟合蓝色曲线（原始误差）的斜率
log_h_blue = np.log10(h_values)
log_e_blue = np.log10(e_n)
slope_blue, intercept_blue = np.polyfit(log_h_blue, log_e_blue, 1)

# 拟合红色曲线（外插误差）的斜率
log_h_red = np.log10(extrapolated_h)
log_e_red = np.log10(extrapolated_error)
slope_red, intercept_red = np.polyfit(log_h_red, log_e_red, 1)

# 4. 绘图
plt.figure(figsize=(8, 5))

# 绘制原始误差曲线（蓝色三角）
plt.loglog(h_values, e_n, 'b-^', label=r'实际误差 $e_n=|\pi-\pi_n|$', markersize=4)
# 绘制理论误差参考线（虚线）
theoretical_error = (np.pi**3 / 6) * h_values**2
plt.loglog(h_values, theoretical_error, 'k:', label=r'理论误差 $O(h^2)$')
# 绘制外插后的误差曲线（红色三角）
plt.loglog(extrapolated_h, extrapolated_error, 'r-^', label='外插后误差', markersize=4)

# 自动标注实际拟合的斜率（保留两位小数）
plt.text(0.02, 1e-2, f'slope:{slope_blue:.2f}', fontsize=9, color='blue')
plt.text(0.02, 1e-10, f'slope:{slope_red:.2f}', fontsize=9, color='red')

# 坐标轴设置
plt.xlabel(r'$h=1/n$')
plt.ylabel(r'$e_n=|\pi-\pi_n|$')
plt.title('正多边形逼近圆周率的误差收敛曲线')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=8)

plt.show()

# 打印拟合结果，方便你核对
print(f"蓝色曲线（原始误差）拟合斜率: {slope_blue:.4f}")
print(f"红色曲线（外插误差）拟合斜率: {slope_red:.4f}")