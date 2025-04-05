import numpy as np
import matplotlib.pyplot as plt

# 中文显示支持
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 三个类别的中心点
centers = {
    r'$\omega_1$': np.array([1, 1]),
    r'$\omega_2$': np.array([-1, 1]),
    r'$\omega_3$': np.array([0, -1])
}
colors = {
    r'$\omega_1$': '#a0d2eb',   # 蓝色
    r'$\omega_2$': '#f9b5ac',   # 红色
    r'$\omega_3$': '#b7e4c7'    # 绿色
}
class_ids = {r'$\omega_1$': 0, r'$\omega_2$': 1, r'$\omega_3$': 2}

# 网格
x = np.linspace(-3, 3, 500)
y = np.linspace(-3, 3, 500)
X, Y = np.meshgrid(x, y)
grid = np.stack([X, Y], axis=-1)

# 计算每个点到三个中心点的平方距离
distances = np.zeros((X.shape[0], X.shape[1], 3))
for i, label in enumerate(centers):
    center = centers[label]
    distances[:, :, i] = np.sum((grid - center) ** 2, axis=2)

# 得到每个点所属类别（距离最近）
predicted_class = np.argmin(distances, axis=2)

# 生成样本点
np.random.seed(42)  # 设置随机种子以保证可重复性
n_samples = 100  # 每个类别生成的样本数量

samples = {
    r'$\omega_1$': np.random.normal([1, 1], 0.6, (n_samples, 2)),
    r'$\omega_2$': np.random.normal([-1, 1], 0.6, (n_samples, 2)),
    r'$\omega_3$': np.random.normal([0, -1], 0.6, (n_samples, 2))
}

# 绘图
plt.figure(figsize=(8, 8))

# 分类区域填色
plt.contourf(X, Y, predicted_class, levels=[-0.5, 0.5, 1.5, 2.5], 
             colors=[colors[r'$\omega_1$'], colors[r'$\omega_2$'], colors[r'$\omega_3$']], alpha=0.6)

# 决策边界（类与类之间）
contour = plt.contour(X, Y, predicted_class, levels=[0.5, 1.5], colors='k', linewidths=2)
plt.clabel(contour, inline=True, fmt='决策边界', fontsize=10)

# 类别中心点
for label, center in centers.items():
    plt.plot(center[0], center[1], 'ko', markersize=8)
    plt.text(center[0]+0.1, center[1]+0.1, label, fontsize=14, weight='bold')

# 绘制样本点
for label, points in samples.items():
    plt.scatter(points[:, 0], points[:, 1], label=label, alpha=0.6, s=50)

# 坐标轴与标题
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.title("三类基于距离的分段线性判别函数与样本点")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
