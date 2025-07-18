import numpy as np
import matplotlib.pyplot as plt

# --- 1. 数据定义 ---

# 定义模型名称和指标标签 (以一个通用且清晰的顺序排列)
models = ['Resnet101', 'Transformer', 'QualityFM-Base']
labels = np.array(['ACC', 'TPR', 'TNR', 'PPV', 'F1', 'AUC'])
n_vars = len(labels)

data = {
    'Resnet101':      [0.70435, 0.808, 0.66567, 0.474, 0.598, 0.679],
    'Transformer':    [0.7587, 0.816, 0.73731, 0.537, 0.648, 0.729],
    'QualityFM-Base': [0.7767, 0.7881, 0.7722, 0.5740, 0.6642, 0.8565]
}


# 计算每个指标在雷达图上的角度
angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
angles += angles[:1] # 闭合雷达图

# 创建画布和子图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 循环绘制每个模型的数据
colors = ['green', 'red', 'blue']
for i, model in enumerate(models):
    # 将列表中的第一个值附加到末尾以闭合图形
    plot_data = data[model] + data[model][:1]
    # 使用 label=model 以便后续生成图例
    ax.plot(angles, plot_data, color=colors[i], linewidth=2, linestyle='solid', label=model)
    ax.fill(angles, plot_data, color=colors[i], alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

ax.set_ylim(0, 1.0)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])

# 添加图例，并将其放置在图表区域之外
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))

# 显示图表
plt.show()