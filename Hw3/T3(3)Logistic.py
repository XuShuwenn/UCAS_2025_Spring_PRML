import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文显示和负号
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

# 样本数据
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

X = np.vstack([samples_1, samples_2])
y = np.array([0] * 20 + [1] * 20).reshape(-1, 1)

# 特征扩展（x1 * x2）并添加偏置项
x1, x2 = X[:, 0].reshape(-1, 1), X[:, 1].reshape(-1, 1)
X_expanded = np.hstack([x1, x2, x1 * x2, np.ones((X.shape[0], 1))])

# Sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 二元交叉熵损失函数
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-8  # 防止log(0)
    return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

# 训练逻辑回归模型
def train_logistic(X, y, lr=0.1, epochs=200):
    weights = np.random.randn(X.shape[1], 1) * 0.01  # 初始化权重
    loss_list, acc_list = [], []

    for _ in range(epochs):
        logits = np.dot(X, weights)
        y_pred = sigmoid(logits)

        loss = cross_entropy_loss(y, y_pred)
        loss_list.append(loss)

        grad = np.dot(X.T, (y_pred - y)) / X.shape[0]
        weights -= lr * grad

        acc = np.mean((y_pred >= 0.5).astype(int) == y)
        acc_list.append(acc)

    return weights, loss_list, acc_list

# 训练模型
weights, loss_list, acc_list = train_logistic(X_expanded, y, lr=0.1, epochs=200)

# 可视化损失和精度曲线
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(loss_list, label='交叉熵损失')
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.title('训练过程中的损失')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(acc_list, label='分类精度', color='green')
plt.xlabel('迭代次数')
plt.ylabel('精度')
plt.title('训练过程中的精度')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 绘制决策面
x1_range = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 400)
x2_range = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 400)
xx, yy = np.meshgrid(x1_range, x2_range)

# 手动构造特征
grid_x1, grid_x2 = xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)
grid_features = np.hstack([grid_x1, grid_x2, grid_x1 * grid_x2, np.ones_like(grid_x1)])

# 计算 sigmoid 输出
probs = sigmoid(np.dot(grid_features, weights)).reshape(xx.shape)

# 可视化决策边界
plt.figure(figsize=(8, 6))
w = weights.ravel()  # 展平为一维向量
plt.text(0.05, 0.95, f"${w[0]:.2f}x_1 + {w[1]:.2f}x_2 + {w[2]:.2f}x_1x_2 + {w[3]:.2f} = 0$", 
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
plt.contour(xx, yy, probs, levels=[0.5], colors='black', linestyles='--', linewidths=2)
plt.contourf(xx, yy, probs, levels=100, cmap='RdBu', alpha=0.6)

# 绘制样本点
plt.scatter(samples_1[:, 0], samples_1[:, 1], c='blue', label='类别 0')
plt.scatter(samples_2[:, 0], samples_2[:, 1], c='red', label='类别 1')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('决策边界及概率分布')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
