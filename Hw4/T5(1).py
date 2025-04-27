import h5py
import numpy as np
from sklearn.metrics import accuracy_score


# 导入训练和测试数据
with h5py.File('usps.h5', 'r') as hf:
    train = hf.get('train')
    X_train = train.get('data')[:]
    y_train = train.get('target')[:]

    test = hf.get('test')
    X_test = test.get('data')[:]
    y_test = test.get('target')[:]

#选择7,9是因为这俩数字比较像
selected_classes = [7,9]

# 筛选训练集
train_mask = np.isin(y_train, selected_classes)
X_train_filtered = X_train[train_mask]
y_train_filtered = y_train[train_mask]

# 筛选测试集
test_mask = np.isin(y_test, selected_classes)
X_test_filtered = X_test[test_mask]
y_test_filtered = y_test[test_mask]

# 标签映射为7和9
label_map = {selected_classes[0]: 7, selected_classes[1]: 9}
y_train_filtered = np.vectorize(label_map.get)(y_train_filtered)
y_test_filtered = np.vectorize(label_map.get)(y_test_filtered)

mean_1=np.mean(X_train_filtered[y_train_filtered==7],axis=0)
mean_2=np.mean(X_train_filtered[y_train_filtered==9],axis=0)

cov_1=np.cov(X_train_filtered[y_train_filtered==7],rowvar=False)
cov_2=np.cov(X_train_filtered[y_train_filtered==9],rowvar=False)

n1 = np.sum(y_train_filtered == 7)
n2 = np.sum(y_train_filtered == 9)
print(n1,n2)
shared_cov=(n1*cov_1+n2*cov_2)/(n1+n2)#用加权平均估计协方差矩阵

w_LDA=np.linalg.pinv(shared_cov).dot(mean_1-mean_2)#超平面法向量
b_LDA=-0.5*(mean_1+mean_2).dot(np.linalg.pinv(shared_cov).dot(mean_1-mean_2))+np.log(n1/n2)#截距b

#LDA预测函数
def LDA_predict(X):
    return np.where(X.dot(w_LDA) + b_LDA > 0, 7, 9)

Sw=(cov_1*(n1-1))+(cov_2*(n2-1))
w_fisher=np.linalg.pinv(Sw).dot(mean_1-mean_2)

#投影后的中心
m=(mean_1.dot(w_fisher)+mean_2.dot(w_fisher))/2

#Fisher线性判别函数
def fisher_predict(X):
    proj=X.dot(w_fisher)
    return np.where(proj>m, 7, 9)

LDA_train_preds=LDA_predict(X_train_filtered)
LDA_test_preds=LDA_predict(X_test_filtered)
#精度
LDA_train_acc=accuracy_score(y_train_filtered,LDA_train_preds)
LDA_test_acc=accuracy_score(y_test_filtered,LDA_test_preds)

fisher_train_preds=fisher_predict(X_train_filtered)
fisher_test_preds=fisher_predict(X_test_filtered)
fisher_train_acc=accuracy_score(y_train_filtered,fisher_train_preds)
fisher_test_acc=accuracy_score(y_test_filtered,fisher_test_preds)

print("方法对比（对 USPS 数据中的7vs9）")
print(f"LDA    训练准确率: {LDA_train_acc * 100:.3f}% | 测试准确率: {LDA_test_acc * 100:.3f}%")
print(f"Fisher 训练准确率: {fisher_train_acc * 100:.3f}% | 测试准确率: {fisher_test_acc * 100:.3f}%")
