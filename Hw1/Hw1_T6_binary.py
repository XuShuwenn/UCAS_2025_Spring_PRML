import matplotlib.pyplot as plt
import numpy as np
from math import pi
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support

# 假设观测数据的标签（真实类别）
x= np.arange(21)
# 假设这里的观测量取值范围是 [0, 20]，如果观测量 <= 10，则类别为正类（1），否则为负类（-1）
true_labels = np.array([1 if i <= 10 else -1 for i in range(21)])  # 生成一个标签数组

#theta1s=[0.115,0.12,0.125,0.13,0.135]
#theta2s=[0.20,0.25,0.30]

predicted_scores=np.sin(0.125*pi*x-0.250*pi)
thresholds = np.linspace(-1,1,16) #15个阈值
fpr_list =[]
tpr_list =[]

for threshold in thresholds:
    #根据当前阈值将预测概率转换为预测标签  
    predicted_labels = np.where(predicted_scores >= threshold, 1, -1)
    #计算精确率、召回率和F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary')
    #输出计算结果
    print(f'threshold: {threshold:.2f}, recall: {recall:.2f}, precision: {precision:.2f}, f1: {f1:.2f}')
    # 计算 TP, FP, TN, FN
    TP = float(np.sum((predicted_labels == 1) & (true_labels == 1)))  # 真正例
    FP = float(np.sum((predicted_labels== 1) & (true_labels == -1)))  # 假正例
    TN = float(np.sum((predicted_labels== -1) & (true_labels == -1)))  # 真负例
    FN = float(np.sum((predicted_labels== -1) & (true_labels== 1)))  # 假负例
    # 计算 TPR 和 FPR
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # 避免除0错误
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0   
    # 存储结果
    tpr_list.append(TPR)
    fpr_list.append(FPR)

roc_auc=auc(fpr_list,tpr_list)
print(roc_auc)
# 绘制ROC曲线
plt.figure(figsize=(10,8))
plt.plot(fpr_list,tpr_list, lw=2, linestyle='-',label="ROC Curve" )
plt.plot([0, 1], [0, 1], color='gray', linestyle='--',label="Random Guess")  # 随机猜测线
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

plt.show()
