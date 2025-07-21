import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# 设置全局字体为 Times New Roman，并全局加粗
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'


# 第一组数据
data1 = {
    'Resnet101':      [0.70435, 0.808, 0.66567, 0.474, 0.598, 0.679],
    'Transformer':    [0.7587, 0.816, 0.73731, 0.537, 0.648, 0.729],
    'QualityFM-Base': [0.7767, 0.7881, 0.7722, 0.5740, 0.6642, 0.8565]
}

# 第二组数据
data2 = {
    'Resnet101':      [0.7936, 0.7625, 0.8109, 0.7300, 0.7066, 0.8666],
    'Transformer':     [0.8493, 0.7911, 0.8805, 0.7824, 0.7856, 0.9110],
    'QualityFM-Base':  [0.8657, 0.8075, 0.8970, 0.8100, 0.8071, 0.9359]
}

labels = np.array(['Acc', "TPR", "TNR", "PPV", "F1", "AUC"])
kinds = list(data1.keys())

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
angles_closed = np.concatenate((angles, [angles[0]]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 12), subplot_kw={'projection': 'polar'})

# --- 图表 1 (ax1) ---
# 将坐标轴标签（'Acc', 'TPR'等）设置为粗体
ax1.set_thetagrids(angles * 180 / np.pi, labels, fontsize=40)
ax1.set_rlim(0.4, 0.9)
# 设置刻度标签的大小
ax1.tick_params(labelsize=40)
# 【新增】将径向刻度标签（Y轴）设置为粗体

# 美学优化：设置网格线样式
ax1.grid(color='grey', linestyle='--', linewidth=1)
# 美学优化：设置图表外圈线条宽度
ax1.spines['polar'].set_linewidth(1.5)


for kind in kinds:
    data_values = data1[kind][:]
    data_values.append(data_values[0])
    # 美学优化：增加线条宽度和标记点大小
    ax1.plot(angles_closed, data_values, 'o-', label=kind, linewidth=3, markersize=10)
    ax1.fill(angles_closed, data_values, alpha=0.25)


# --- 图表 2 (ax2) ---
# 将坐标轴标签设置为粗体
ax2.set_thetagrids(angles * 180 / np.pi, labels, fontsize=40)
ax2.set_rlim(0.6, 1)
ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
# 设置刻度标签的大小
ax2.tick_params(labelsize=40)
# 【新增】将径向刻度标签（Y轴）设置为粗体


# 美学优化：设置网格线样式
ax2.grid(color='grey', linestyle='--', linewidth=1)
# 美学优化：设置图表外圈线条宽度
ax2.spines['polar'].set_linewidth(1.5)

for kind in kinds:
    data_values = data2[kind][:]
    data_values.append(data_values[0])
    # 美学优化：增加线条宽度和标记点大小
    ax2.plot(angles_closed, data_values, 'o-', label=kind, linewidth=3, markersize=10)
    ax2.fill(angles_closed, data_values, alpha=0.25)


# --- 图例和标题 ---
handles, legend_labels = ax1.get_legend_handles_labels()
# 【修改】使用 prop 参数将图例文字设置为粗体
fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
           prop={'size': 40}, ncol=len(kinds)) # 可选 ncol 让图例水平排列

# 【修改】将子图标题设置为粗体
fig.text(0.3, 0.1, '(a) VTaC', ha='center', va='center', fontsize=44)
fig.text(0.73, 0.1, '(b) MIMIC PERform AF', ha='center', va='center', fontsize=44
         )

plt.subplots_adjust(top=0.85, bottom=0.15) # 调整布局为图例和标题留出空间
plt.savefig("radar_charts_vertical_legend_text.png", dpi=300, bbox_inches='tight')
