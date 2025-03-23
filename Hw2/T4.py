import numpy as np
import matplotlib.pyplot as plt
#常量定义
mean1=[2,0]
cov=[[3,-1],[-1,3]]
num=100
mean2=[1,2]
#生成样本
sample1=np.random.multivariate_normal(mean1,cov,num)
sample2=np.random.multivariate_normal(mean2,cov,num)
#添加偏置项
sample1_new=np.hstack([sample1,np.ones((num,1))])
sample2_new=np.hstack([sample2,np.ones((num,1))])
#合并样本方便使用
X = np.vstack([sample1_new, sample2_new])
y=np.array([1 if i<num else -1 for i in range(2*num)])#添加真实标签

#1.Fisher线性判别法
m1=np.mean(sample1, axis=0)
m2=np.mean(sample2, axis=0)
S_W = np.zeros((2, 2))
for i in range(len(sample1)):
    S_W += np.outer(sample1[i] - m1, sample1[i] - m1)
for i in range(len(sample2)):
    S_W += np.outer(sample2[i] - m2, sample2[i] - m2)
# 投影向量 w_1
w_0=np.linalg.inv(S_W).dot(m1-m2)
origin=-w_0.dot((m1+m2)/2)
w_1=np.append(w_0,origin)#添加偏置项

#2.伪逆方法
X_pinv = np.linalg.pinv(X)
w_2= X_pinv.dot(y)

#3.梯度下降法
w_3 = np.zeros(3)
eta = 0.0005
decay_rate=0.9
decay_steps=40
T=200
for t in range(T):
    y_pred=X.dot(w_3)
    #计算梯度
    grad=-X.T.dot(y-y_pred)#X.T是转置矩阵
    #更新学习率eta
    if t % decay_steps == 0:
        eta = eta * decay_rate
    w_3=w_3-eta*grad
    # 每隔10次迭代输出一次损失函数值
    if t % 10 == 0:
        loss = (1/2*num)*np.mean((y_pred - y) ** 2)
        print(f"Epoch {t}, Loss: {loss:.4f}")

#绘制样本点
plt.figure(figsize=(8, 8))
plt.scatter(sample1[:,0],sample1[:,1],c='r',marker='o',label='Class 1')
plt.scatter(sample2[:,0],sample2[:,1],c='b',marker='o',label='Class2')
plt.title('Multivariate Normal Distribution Samples')

for w in (w_1,w_2,w_3):
    # 归一化向量w
    norm = np.linalg.norm(w)#计算模长
    w_normalized = w / norm
    np.set_printoptions(precision=20)#控制打印精度
    print(w_normalized)
    random_color = np.random.rand(3,)  # 生成一个随机颜色
    #从原点出发绘制投影向量
    plt.quiver(0, 0, w[0], w[1], angles='uv',width=0.003, headwidth=8, headlength=8, headaxislength=8, color=random_color, label='Projection Vector')
    # 计算决策面（线性判别函数）
    x_vals = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    y_vals = (-w[0] / w[1])* x_vals - (w[2] / w[1])
    plt.plot(x_vals, y_vals, color=random_color, label='Decision Boundary')

plt.legend()#设置图例
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()