import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Preparation ---
# Define x-axis labels
labels = ['ACC', 'TPR', 'TNR', 'PPV', 'F1', 'AUC']

# Data for "with-pretraining"
with_pretraining_means = [0.8221, 0.7983, 0.8307, 0.6333, 0.7063, 0.9099]
with_pretraining_std = [0.072, 0.0649, 0.0645, 0.0331, 0.0243, 0.0156]

# Data for "without-Pretraining"
without_pretraining_means = [0.7678, 0.2672, 0.9546, 0.6874, 0.3844, 0.7892]
without_pretraining_std = [0.0774, 0.0812, 0.0712, 0.0382, 0.0246, 0.0168]


# --- 2. Plotting Setup ---
x = np.arange(len(labels))  # Label locations
width = 0.35  # Bar width

# Create figure and axes
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the bars with the final color scheme
# w/ pre-train -> Sky Blue
rects1 = ax.bar(x - width/2,
                with_pretraining_means,
                width,
                label='w/ pre-train',
                color='#56B4E9',  # Keep this blue
                yerr=with_pretraining_std,
                capsize=5)

# w/o pre-train -> Rosy Red
rects2 = ax.bar(x + width/2,
                without_pretraining_means,
                width,
                label='w/o pre-train',
                color='#EE6677',  # <--- Use Rosy Red
                yerr=without_pretraining_std,
                capsize=5)

# --- 3. Add Chart Elements ---
# Set y-axis label
ax.set_ylabel('Mean Value', fontsize=12)
# Set chart title
# Set x-axis ticks and labels
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
# Display legend
ax.legend()

# # Add value labels on top of the bars
# ax.bar_label(rects1, padding=3, fmt='%.4f')
# ax.bar_label(rects2, padding=3, fmt='%.4f')

# Optimize layout
fig.tight_layout()

# Display the plot
plt.show()