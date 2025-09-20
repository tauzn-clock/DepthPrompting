from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

img_index = [0,12,45,9]

N = 4
M = 9

fig, ax = plt.subplots(N, M, figsize=(20, 7), sharex=True, sharey=True)

# Turn off all axes initially
for i in range(N):
    for j in range(M):
        ax[i,j].axis('off')

# Reduce space between subplots
plt.subplots_adjust(wspace=0.01, hspace=0.01)

# Load and display images


for i in range(len(img_index)):
    gt_path = f"/scratchdata/compressed_sensing/gt/gt_{img_index[i]}.png"
    gt = np.array(Image.open(gt_path))
    ax[i,0].imshow(gt, cmap='gray')




# Disable all ticks and axis lines
for i in range(N):
    ax[i,0].axis('on')
    ax[i,0].spines['top'].set_visible(False)
    ax[i,0].spines['right'].set_visible(False)
    ax[i,0].spines['left'].set_visible(False)
    ax[i,0].spines['bottom'].set_visible(False)
    ax[i,0].set_xticks([])
    ax[i,0].set_yticks([])
    
ax[N-1,0].set_xlabel('Ground Truth')
ax[N-1,1].set_xlabel('Depth Prompting')
ax[N-1,2].set_xlabel('Depth Prompting Error')
ax[N-1,3].set_xlabel('DepthFormer + CS')
ax[N-1,4].set_xlabel('DepthFormer + CS Error')
ax[N-1,5].set_xlabel('Metric-3D-small + CS')
ax[N-1,6].set_xlabel('Metric-3D-small + CS Error')

fig.savefig("test.png", dpi=300, bbox_inches='tight')