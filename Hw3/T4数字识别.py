import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 导入训练和测试数据
with h5py.File('usps.h5', 'r') as hf:
    train = hf.get('train')#获取训练数据
    X_train = train.get('data')[:]
    y_train = train.get('target')[:]

    test = hf.get('test')#获取测试数据
    X_test = test.get('data')[:]
    y_test = test.get('target')[:]

#数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#转换为one-hot编码
def to_one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

y_train_onehot = to_one_hot(y_train)
y_test_onehot = to_one_hot(y_test)

#用ReLU作为激活函数和梯度
def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(np.float32)

#softmax函数输出分类概率值
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

#交叉熵损失函数
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# 每200轮学习率减半
def step_decay(epoch, decay_factor=0.9,initial_lr=0.001):
    return initial_lr * (decay_factor** (epoch // 200))

#Xavier初始化权重参数
def xavier_init(input_size, output_size):
    return np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))
#定义二层感知机
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        self.lr = lr
        self.params = {
            'W1': xavier_init(input_size, hidden_size),
            'b1': np.zeros(hidden_size),
            'W2': xavier_init(hidden_size, output_size),
            'b2': np.zeros(output_size)
        }
    #前向传播
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        self.a1 = np.dot(x, W1) + b1
        self.z1 = relu(self.a1)  # 使用ReLU激活函数
        self.a2 = np.dot(self.z1, W2) + b2
        self.y = softmax(self.a2)
        return self.y
    # 计算损失
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_loss(t, y)
    # 计算精度
    def accuracy(self, x, t):
        y = self.predict(x)
        pred = np.argmax(y, axis=1)
        true = np.argmax(t, axis=1)
        return np.mean(pred == true)

    def train(self, x, t, epochs=200):
        loss_list, acc_list, test_err_list = [], [], []
        for epoch in range(epochs):
            self.predict(x)
            dy = (self.y - t) / x.shape[0]
            grads = {
                'W2': np.dot(self.z1.T, dy),
                'b2': np.sum(dy, axis=0),
            }
            dz1 = np.dot(dy, self.params['W2'].T) * relu_grad(self.a1)#使用ReLU
            grads['W1'] = np.dot(x.T, dz1)
            grads['b1'] = np.sum(dz1, axis=0)
            # 参数更新
            for key in self.params:
                self.params[key] -= self.lr * grads[key]
            #self.lr=step_decay(epoch)
            # 记录训练损失和精度
            loss = self.loss(x, t)
            acc = self.accuracy(x, t)
            test_err = 1 - self.accuracy(X_test, y_test_onehot)

            loss_list.append(loss)
            acc_list.append(acc)
            test_err_list.append(test_err)

        return loss_list, acc_list, test_err_list

# 创建并训练模型
input_size = X_train.shape[1]
hidden_size = 512  # 增加隐藏层节点数量
output_size = 10
learning_rate = 0.002
epochs = 1000  # 增加训练轮数

net = TwoLayerNet(input_size, hidden_size, output_size, lr=learning_rate)

# 训练模型
loss_curve, acc_curve, test_err_curve = net.train(X_train, y_train_onehot, epochs=epochs)

# 绘制损失函数和测试错误率曲线
plt.figure(figsize=(12, 6))

# 绘制训练损失曲线
plt.subplot(1, 2, 1)
plt.plot(loss_curve, label='训练损失')
plt.grid(True)
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.title('目标函数曲线')
plt.legend()

#绘制精度曲线
plt.subplot(1, 2, 2)
plt.plot(acc_curve, label='训练精度')
plt.yticks(np.arange(0, 1, 0.05))  
plt.grid(True)

# 绘制测试错误率曲线
plt.plot(test_err_curve, label='测试错误率')
plt.yticks(np.arange(0, 1, 0.05))  
plt.grid(True)  
plt.xlabel('迭代次数')
plt.ylabel('精度')
plt.title('测试错误率曲线')
plt.legend()

plt.tight_layout()
plt.show()
