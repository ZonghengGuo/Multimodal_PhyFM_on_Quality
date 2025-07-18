import matplotlib.pyplot as plt
import numpy as np

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
# 闭合图形
angles_closed = np.concatenate((angles, [angles[0]]))

# 创建一个1行2列的图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 14), subplot_kw={'projection': 'polar'})

# --- 绘制第一个雷达图 ---
ax1.set_thetagrids(angles * 180 / np.pi, labels)
ax1.set_title("VTaC", position=(0.5, 1.1), size=14)
ax1.set_rlim(0.4, 0.9)

# 注意：为了让代码可以重复运行，这里对data_values做了副本
for kind in kinds:
    data_values = data1[kind][:] # 使用[:]创建副本
    data_values.append(data_values[0])
    ax1.plot(angles_closed, data_values, 'o-', label=kind)
    ax1.fill(angles_closed, data_values, alpha=0.25)

# --- 绘制第二个雷达图 ---
ax2.set_thetagrids(angles * 180 / np.pi, labels)
ax2.set_title("MIMIC PERform AF", position=(0.5, 1.1), size=14)
ax2.set_rlim(0.65, 0.95)

# 注意：为了让代码可以重复运行，这里对data_values做了副本
for kind in kinds:
    data_values = data2[kind][:] # 使用[:]创建副本
    data_values.append(data_values[0])
    ax2.plot(angles_closed, data_values, 'o-', label=kind)
    ax2.fill(angles_closed, data_values, alpha=0.25)

handles, legend_labels = ax1.get_legend_handles_labels()

fig.legend(handles, legend_labels, loc='upper center',  fontsize=16)

# 调整布局并保存图像
# # 注意：调整了tight_layout的rect来为图例腾出更多空间
# plt.tight_layout(rect=[0, 0.05, 1, 0.9])
plt.savefig("radar_charts_vertical_legend.png")