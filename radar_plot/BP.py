import numpy as np
import matplotlib.pyplot as plt

# --- 1. 数据定义 ---

# 定义模型名称和指标标签
models = ['QualityFM-base', 'Transformer', 'ResNet101']
labels = np.array(['SBP MAE', 'SBP |ME|', 'SBP MASE',
                   'DBP MAE', 'DBP |ME|', 'DBP MASE'])
n_vars = len(labels)

# 原始数据 (ME使用其绝对值, MASE百分比转换为小数)
raw_data = {
    'QualityFM-base': [21.86, abs(0.08), 1.1630, 9.12, abs(0.24), 1.1522],
    'Transformer':    [21.39, abs(-2.34), 1.1381, 9.64, abs(-1.13), 1.2179],
    'ResNet101':      [21.33, abs(-0.91), 1.1346, 9.85, abs(-1.82), 1.2452]
}

# --- 2. 数据归一化 (Normalization) ---
# 将数据按指标分组，以便对每个指标独立进行归一化
data_by_metric = np.array([raw_data[model] for model in models]).T

# 定义归一化函数：数值越低，得分越高（范围0-1）
def normalize_lower_is_better(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val: # 防止所有值都相同时除以零
        return np.ones_like(data)
    return (max_val - data) / (max_val - min_val)

# 对每个指标应用归一化函数
normalized_data_by_metric = np.apply_along_axis(normalize_lower_is_better, 1, data_by_metric)

# 将数据转置回按模型分组，方便绘图
normalized_data = normalized_data_by_metric.T

# --- 3. 图表设置与绘制 ---

# 计算每个指标在雷达图上的角度
angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
angles += angles[:1] # 闭合雷达图

# 创建画布和子图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 循环绘制每个模型的数据
colors = ['blue', 'red', 'green']
for i, model in enumerate(models):
    plot_data = np.concatenate((normalized_data[i], [normalized_data[i][0]]))
    # 下一行中的`label=model`是生成图例内容的关键
    ax.plot(angles, plot_data, color=colors[i], linewidth=2, linestyle='solid', label=model)
    ax.fill(angles, plot_data, color=colors[i], alpha=0.25)

# 设置图表的轴标签（各项指标的名称）
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# 设置图表的径向轴（Y轴）刻度
ax.set_ylim(0, 1.05)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.50", "0.75", "Best (1.0)"], color="grey", size=8)

# 添加图表标题
ax.set_title('Normalized Model Performance Comparison (Lower Error is Better)', size=16, color='black', y=1.1)

# ==========================================================
# --- 4. 生成图例 ---
# 下面这行代码会收集所有在绘制时定义的`label`，并创建一个图例
# `loc`和`bbox_to_anchor`用于将图例放置在图表区域之外，防止遮挡数据
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
# ==========================================================

# 显示或保存图表
# plt.savefig('model_error_comparison_radar.png', dpi=300, bbox_inches='tight')
plt.show()