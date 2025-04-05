import numpy as np
import matplotlib.pyplot as plt

# 中文和负号显示支持
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

#定义类别颜色映射
colors = {
    r'$\omega_1$': '#a0d2eb',   # 蓝色
    r'$\omega_2$': '#f9b5ac',   # 红色
}

samples_1 = np.array([
    [0.68, 1.34], [0.93, 0.89], [0.9, 1.66], [1.08, 0.65], [1.26, 0.57],
    [0.91, 1.51], [1.5, 1.26], [1.26, 1.31], [0.92, 1.26], [1.04, 0.99],
    [-0.99, -1.54], [-1.16, -1.23], [-0.77, -1.01], [-1.3, -1.28], [-1.27, -0.96],
    [-0.99, -1.13], [-0.94, -0.93], [-1.21, -0.99], [-1.71, -0.82], [-0.99, -0.92]
])

samples_2 = np.array([
    [0.99, -0.6], [1.03, -1.08], [1.56, -0.98], [1.26, -1.16], [1.14, -0.8],
    [0.9, -1.25], [1.18, -0.7], [1.3, -0.9], [1.42, -0.82], [0.54, -1.2],
    [-0.78, 0.68], [-1.02, 1.29], [-0.84, 1.12], [-0.73, 1.06], [-0.54, 1.0],
    [-1.14, 0.58], [-1.09, 0.51], [-1.12, 1.33], [-0.14, 0.82], [-1.45, 1.31]
])
plt.figure(figsize=(8, 8))
#绘制类别1样本点
plt.scatter(samples_1[:, 0], samples_1[:, 1], label=r'$\omega_1$', color=colors[r'$\omega_1$'])

#绘制类别2样本点
plt.scatter(samples_2[:, 0], samples_2[:, 1], label=r'$\omega_2$', color=colors[r'$\omega_2$'])

#绘制坐标轴与辅助线
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.title("两类样本点示意图", fontsize=14)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
