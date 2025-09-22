import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the data
df = pd.read_csv("data.csv", header=None)
col1, col2, col3, col4 = df[0], df[1], df[2], df[3]

# Custom x-tick labels: col2 rounded + col1 in brackets
xticks = col2
xtick_labels = [f"{x:.2f} [{n}]" for x, n in zip(col2, col1)]

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# Left plot: col3 vs col2
axes[0].plot(col2, col3, "o", linewidth=2, color="tab:blue", label="col3")
axes[0].set_xlabel("col2 [col1]")
axes[0].set_ylabel("col3")
axes[0].set_xticks(xticks)
axes[0].set_xticklabels(xtick_labels)  # straight, not oblique
axes[0].grid(True, linestyle="--", alpha=0.6)
axes[0].legend()
axes[0].invert_xaxis()

# Right plot: col4 vs col2
axes[1].plot(col2, col4, "o-", linewidth=2, color="tab:orange", label="col4")
axes[1].set_xlabel("col2 [col1]")
axes[1].set_ylabel("col4")
axes[1].set_xticks(xticks)
axes[1].set_xticklabels(xtick_labels)
axes[1].grid(True, linestyle="--", alpha=0.6)
axes[1].legend()
axes[1].invert_xaxis()

plt.suptitle("Two-panel plot (x-axis flipped, labels rounded)", fontsize=14)
plt.show()
