import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Preparation ---
# Set up the labels for the x-axis, grouping by SBP and DBP
labels = ['SBP\nMAE', 'SBP\nME', 'SBP\nMASE (%)',
          'DBP\nMAE', 'DBP\nME', 'DBP\nMASE (%)']

# Data for 'with-pretraining' (Sky Blue)
with_means = [19.46, 4.85, 103.51, 7.86, -0.11, 99.29]
# Standard deviation only applies to ME, so other values are 0
with_std = [0, 23.84, 0, 0, 10.42, 0]

# Data for 'without-Pretraining' (Rosy Red)
without_means = [19.70, 1.70, 104.82, 8.81, -3.09, 111.30]
# Standard deviation only applies to ME
without_std = [0, 23.84, 0, 0, 10.36, 0]

# --- 2. Plotting Setup ---
x = np.arange(len(labels))  # Label locations
width = 0.4  # Bar width

# Create figure and axes
fig, ax = plt.subplots(figsize=(14, 8))

# Define the colors from the previous configuration
with_color = '#56B4E9'  # Sky Blue
without_color = '#EE6677' # Rosy Red

# Plot the bars
rects1 = ax.bar(x - width/2, with_means, width,
                label='w/ pre-train', color=with_color,
                yerr=with_std, capsize=5)

rects2 = ax.bar(x + width/2, without_means, width,
                label='w/o pre-train', color=without_color,
                yerr=without_std, capsize=5)

# --- 3. Add Chart Elements & Styling ---
# Add a horizontal line at y=0 for reference
ax.axhline(0, color='grey', linewidth=0.8)

# Add a vertical line to visually separate SBP and DBP groups
ax.axvline(2.5, color='grey', linestyle='--', linewidth=1)

# Set y-axis label
ax.set_ylabel('Error Value', fontsize=12)
# Set chart title

# Set x-axis ticks and labels
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
# Display legend
ax.legend(fontsize=11)

# Add value labels on top of the bars
ax.bar_label(rects1, padding=5, fmt='%.2f', fontsize=9)
ax.bar_label(rects2, padding=5, fmt='%.2f', fontsize=9)

# Adjust y-axis limits to ensure all labels and error bars are visible
ax.set_ylim(-30, 140)

# Optimize layout
fig.tight_layout()

# Display the plot
plt.show()