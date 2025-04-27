import h5py
import numpy as np
from sklearn.metrics import accuracy_score

#导入训练和测试数据
with h5py.File('usps.h5', 'r') as hf:
    train = hf.get('train')
    X_train = train.get('data')[:]
    y_train = train.get('target')[:]

    test = hf.get('test')
    X_test = test.get('data')[:]
    y_test = test.get('target')[:]

#选择7,9是因为这俩数字比较像
selected_classes = [7,9]

#筛选训练集和测试集
train_mask = np.isin(y_train, selected_classes)
X_train_filtered = X_train[train_mask]
y_train_filtered = y_train[train_mask]

test_mask = np.isin(y_test, selected_classes)#返回bool数组
X_test_filtered = X_test[test_mask]
y_test_filtered = y_test[test_mask]

#标签映射为7和9
label_map = {selected_classes[0]: 7, selected_classes[1]: 9}
y_train_filtered = np.vectorize(label_map.get)(y_train_filtered)
y_test_filtered = np.vectorize(label_map.get)(y_test_filtered)

mean_1=np.mean(X_train_filtered[y_train_filtered==7],axis=0)
mean_2=np.mean(X_train_filtered[y_train_filtered==9],axis=0)

cov_1=np.cov(X_train_filtered[y_train_filtered==7],rowvar=False)
cov_2=np.cov(X_train_filtered[y_train_filtered==9],rowvar=False)
#正则化：确保协方差矩阵的行列式不为零
epsilon = 0.05
cov_1 += epsilon * np.eye(cov_1.shape[0])
cov_2 += epsilon * np.eye(cov_2.shape[0])
#求决策面参数
W1=-np.linalg.pinv(cov_1)
W2=-np.linalg.pinv(cov_2)
w1=2*np.linalg.pinv(cov_1).dot(mean_1)
w2=2*np.linalg.pinv(cov_2).dot(mean_2)
#计算频率(直接用两个类别的数量代替，最后是一样的)
n1 = np.sum(y_train_filtered == 7)
n2 = np.sum(y_train_filtered == 9)
#算行列式
det_cov_1=np.linalg.det(cov_1)
det_cov_2=np.linalg.det(cov_2)

w10 = -mean_1.T.dot(np.linalg.pinv(cov_1)).dot(mean_1) - np.log(det_cov_1) + 2 * np.log(n1)
w20 = -mean_2.T.dot(np.linalg.pinv(cov_2)).dot(mean_2) - np.log(det_cov_2) + 2 * np.log(n2)

def predict(X):
    result = np.einsum('ij,jk,ik->i', X, (W1 - W2), X) + X.dot(w1 - w2) + (w10 - w20)
    return np.where(result > 0, 7, 9)

# 预测
train_preds = predict(X_train_filtered)
test_preds = predict(X_test_filtered)

# 精度评估
train_acc = accuracy_score(y_train_filtered, train_preds)
test_acc = accuracy_score(y_test_filtered, test_preds)

print("二次贝叶斯分类器 (7 vs 9)")
print(f"训练准确率: {train_acc * 100:.3f}%")
print(f"测试准确率: {test_acc * 100:.3f}%")

eigvals = np.linalg.eigvals(cov_1)
print("最大特征值:", np.max(eigvals))
print("最小特征值:", np.min(eigvals))
print("条件数（max/min）:", np.max(eigvals) / np.min(eigvals))
