import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(0, 10, 500)
y = np.sin(x) + 0.1 * np.random.randn(500)

# Create the figure and two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot on the first axis (full plot)
ax1.plot(x, y, label="Full Plot")
ax1.set_title("Full Plot")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# Zoom region you want to focus on
x_zoom_min, x_zoom_max = 2, 4
y_zoom_min, y_zoom_max = -1.5, 1.5

# Plot on the second axis (zoomed-in plot)
ax2.plot(x, y, label="Zoomed In", color='orange')
ax2.set_xlim(x_zoom_min, x_zoom_max)
ax2.set_ylim(y_zoom_min, y_zoom_max)
ax2.set_title("Zoomed In")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

# Optional: Highlight zoom region in the first plot
rect = plt.Rectangle(
    (x_zoom_min, y_zoom_min),
    x_zoom_max - x_zoom_min,
    y_zoom_max - y_zoom_min,
    linewidth=1,
    edgecolor='red',
    facecolor='none',
    linestyle='--'
)
ax1.add_patch(rect)

plt.tight_layout()
fig.savefig("comparison_plot.png", bbox_inches="tight", dpi=300)
