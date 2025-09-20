import matplotlib.pyplot as plt
import numpy as np

# Open file

PATH = "graph.txt"
store = []

with open(PATH, "r") as f:
  lines = f.readlines()
  for line in lines:
    store.append((line.split("&"))[1:])
    store[-1][-1] = store[-1][-1][:5]
    store[-1] = [float(v) for v in store[-1]]

# Plot bar graph in clusters of 5
labels = ["Initial MDE Estimate", 
          "Proportional Rescaling", 
          "$\lambda=5\\times|\\bar{R}|_F$", 
          "$\lambda=0.5\\times|\\bar{R}|_F$", 
          "$\lambda=0.05\\times|\\bar{R}|_F$"]

model = [ "DepthAnythingV2-Small",
          "DepthAnythingV2-Base",
          "DepthAnythingV2-Large",
          "MetricV2-Small",
          "MetricV2-Large"
]

x = range(len(labels))
width = 0.15
fig, ax = plt.subplots(3,1,figsize=(17, 10))

for k in range(3):
  for i in range(0, 5):
      values = [store[u][k] for u in range(i, 25+i, 5)]
      print(values)

      # Plot values in a cluster with labels as legend
      ax[k].bar([p + width*(i) for p in x], values, width=width, label=labels[i])
      ax[k].set_xticks([p + 2*width for p in x])
      ax[k].set_xticklabels(model, fontsize=14)

# Use latex for legend put outside the plot
ax[1].legend(loc='center right', bbox_to_anchor=(1.2, 0.5), ncol=1, fontsize=12)

ax[0].set_ylabel("RMSE/m", fontsize=14)
ax[0].set_ylim(0,0.9)
ax[0].tick_params(axis='y', labelsize=14)
ax[1].set_ylabel("MAE/m", fontsize=14)
ax[1].set_ylim(0,0.65)
ax[1].tick_params(axis='y', labelsize=14)
ax[2].set_ylabel("DELTA1", fontsize=14)
ax[2].set_ylim(0.85,1.02)
ax[2].tick_params(axis='y', labelsize=14)

# Add horizontal grid lines
for k in range(3):
    ax[k].yaxis.grid(True)
    ax[k].set_axisbelow(True)

plt.tight_layout()
plt.savefig("bar_graph.png")