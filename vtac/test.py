import numpy as np

# 五次测试的数据
tpr_values = [27.2, 28.0, 23.2, 28.0, 27.2]
tnr_values = [96.119, 95.821, 95.224, 94.328, 95.821]
score_values = [43.204, 43.415, 41.232, 42.805, 43.083]
acc_values = [77.391, 77.391, 75.652, 76.304, 77.174]
ppv_values = [0.723, 0.714, 0.644, 0.648, 0.708]
auc_values = [0.779, 0.802, 0.773, 0.78, 0.812]
f1_values = [0.395, 0.402, 0.341, 0.391, 0.393]

# 将所有数据存储在一个字典中，方便遍历
data = {
    "TPR": tpr_values,
    "TNR": tnr_values,
    "Score": score_values,
    "Acc": acc_values,
    "PPV": ppv_values,
    "AUC": auc_values,
    "F1": f1_values
}

# 计算每个变量的平均值和标准差
results = {}
for name, values in data.items():
    mean = np.mean(values)
    std_dev = np.std(values, ddof=1)  # 使用 ddof=1 计算样本标准差
    results[name] = {"mean": mean, "std_dev": std_dev}

for name, values in results.items():
    print(f"{name}:")
    print(f"  平均值 (Mean): {values['mean']:.4f}")
    print(f"  标准差 (Standard Deviation): {values['std_dev']:.4f}")
    print("-" * 20)