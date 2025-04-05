import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
#设置matplotlib支持中文
rcParams['font.sans-serif']=['Microsoft YaHei']
rcParams['axes.unicode_minus']=False

#使用ReLU激活函数
def relu(a):
    return np.maximum(0, a)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)  #提升数值稳定性，防止溢出
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 二元交叉熵损失函数
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

#转换为one-hot编码
def to_one_hot(y, num_classes):
    return np.eye(num_classes)[y]

#固定步长学习率衰减法调整学习率
def step_decay(epoch, initial_lr=0.1, decay_factor=0.9, step_size=10):
    return initial_lr * (decay_factor ** (epoch // step_size))

#定义二层感知机网络类
class TwoLayerNet:
    #初始化参数包含隐层size,学习率这两个超参数
    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        self.lr = lr
        self.params = {
            'W1': 0.01 * np.random.randn(input_size, hidden_size),
            'b1': np.zeros(hidden_size),
            'W2': 0.01 * np.random.randn(hidden_size, output_size),
            'b2': np.zeros(output_size)
        }
    #前向传播
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        self.a1 = np.dot(x, W1) + b1
        self.z1 = relu(self.a1)
        self.a2 = np.dot(self.z1, W2) + b2
        self.y = softmax(self.a2)
        return self.y
    #选择交叉熵作为损失函数
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_loss(t, y)
    #计算精度的函数
    def accuracy(self, x, t):
        y = self.predict(x)
        pred = np.argmax(y, axis=1)
        true = np.argmax(t, axis=1)
        return np.mean(pred == true)
    #训练函数
    def train(self, x, t, epochs=200,patience=20):
        loss_list,accuracy_list=[],[]#记录每个epoch的分类精度
        best_acc,wait=0,0      #wait是训练精度没有提升的epoch数
        best_params=None

        for epoch in range(epochs):
            self.predict(x)  # 前向传播
            dy = (self.y - t) / x.shape[0]  # 计算输出层的误差
            grads = {
                'W2': np.dot(self.z1.T, dy),
                'b2': np.sum(dy, axis=0),
            }
            dz1 = np.dot(dy, self.params['W2'].T) * (self.a1>0) #ReLU的梯度
            grads['W1'] = np.dot(x.T, dz1)
            grads['b1'] = np.sum(dz1, axis=0)

            for key in self.params:
                self.params[key] -= self.lr * grads[key]
            
            # 动态调整学习率（步长衰减）
            self.lr = step_decay(epoch)  

            loss = self.loss(x, t)
            accuracy=self.accuracy(x,t)
            loss_list.append(loss)
            accuracy_list.append(accuracy)
            #早停法防止过拟合
            warm_up=50
            if epoch>warm_up:
                if accuracy>best_acc:
                    best_acc=accuracy
                    best_params={k:v.copy()for k,v in self.params.items()}
                    wait=0
                else:
                    wait+=1
                    if wait >=patience:
                        print(f'早停触发，在第{epoch+1}轮提前停止训练')
                        break
            
        if best_params is not None:
            self.params=best_params
    
        return loss_list,accuracy_list

#两组样本点
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

X = np.vstack((samples_1, samples_2))
y = np.array([0]*20 + [1]*20)
y_onehot = to_one_hot(y, 2)

#创建并训练模型
net = TwoLayerNet(input_size=2, hidden_size=10, output_size=2, lr=0.1)
loss_curve,accuracy_curve = net.train(X, y_onehot, epochs=200)

#绘制损失曲线和精度曲线
plt.figure(figsize=(10, 6))
#损失函数曲线
plt.subplot(1, 2, 1)
plt.plot(loss_curve, label='交叉熵损失', color='blue')
plt.xlabel('迭代步数')
plt.ylabel('损失')
plt.title('损失曲线')
plt.legend()
plt.grid(True)
#训练精度曲线
plt.subplot(1, 2, 2)
plt.plot(accuracy_curve, label='训练精度', color='green')
plt.xlabel('迭代步数')
plt.ylabel('精度')
plt.title('精度曲线')
plt.legend()
plt.grid(True)

plt.tight_layout()  # 自动调整布局
plt.show()

#计算决策边界
x1_range=np.linspace(X[:,0].min()-0.5,X[:,0].max()+0.5,400)
x2_range=np.linspace(X[:,1].min()-0.5,X[:,1].max()+0.5,400)
xx,yy=np.meshgrid(x1_range,x2_range)
grid=np.c_[xx.ravel(),yy.ravel()]
Z=np.argmax(net.predict(grid),axis=1).reshape(xx.shape)
#绘制决策边界
plt.contourf(xx,yy,Z,alpha=0.6,cmap='coolwarm')
plt.scatter(samples_1[:,0],samples_1[:,1],c='blue',label='类别0')
plt.scatter(samples_2[:,0],samples_2[:,1],c='red',label='类别1')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('二层感知机决策边界')
plt.legend()
plt.grid(True)
plt.show()
