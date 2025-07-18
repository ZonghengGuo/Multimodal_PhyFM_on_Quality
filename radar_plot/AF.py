import numpy as np
import matplotlib.pyplot as plt

# --- Data Definition ---

# Define the metric labels
labels = np.array(['ACC', 'TPR', 'TNR', 'PPV', 'F1', 'AUC'])
n_vars = len(labels)

# Performance data for each model
transformer_quality_stats = [0.8493, 0.7911, 0.8805, 0.7824, 0.7856, 0.9110]
resnet101_stats = [0.7936, 0.7625, 0.8109, 0.7300, 0.7066, 0.8666]
pwsa_base_stats = [0.8657, 0.8075, 0.8970, 0.8100, 0.8071, 0.9359]


# --- Chart Setup and Plotting ---

# Calculate the angle for each metric
angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
# To close the radar_plot chart, the first angle needs to be appended to the end
angles += angles[:1]

# Append the first value of each model's data to the end to close the plot
transformer_quality_stats += transformer_quality_stats[:1]
resnet101_stats += resnet101_stats[:1]
pwsa_base_stats += pwsa_base_stats[:1]

# Create the figure and subplot with a polar projection
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Plot the data for the Transformer-quality model
ax.plot(angles, transformer_quality_stats, color='red', linewidth=2, linestyle='solid', label='Transformer-quality')
ax.fill(angles, transformer_quality_stats, 'red', alpha=0.25)

# Plot the data for the ResNet101 model
ax.plot(angles, resnet101_stats, color='green', linewidth=2, linestyle='solid', label='ResNet101')
ax.fill(angles, resnet101_stats, 'green', alpha=0.25)

# Plot the data for the PWSA-Base model
ax.plot(angles, pwsa_base_stats, color='blue', linewidth=2, linestyle='solid', label='PWSA-Base')
ax.fill(angles, pwsa_base_stats, 'blue', alpha=0.25)

# Set the tick labels for the chart (the metric names)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# --- MODIFIED SECTION: Set axis range from 0 to 1 ---
# Set the circular gridlines and their labels for the 0-1 range
ax.set_rlabel_position(30)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
ax.set_ylim(0, 1.0) # Set the radial axis limits from 0 to 1
# --- END OF MODIFIED SECTION ---

# Add a legend and position it outside the plot area
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))



# --- Display or Save the Chart ---

# To save the chart as an image file, uncomment the following line
# plt.savefig('model_performance_radar_chart_0-1_range.png', dpi=300, bbox_inches='tight')

# Display the chart
plt.show()