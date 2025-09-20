from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

N = 5
M = 6

fig, ax = plt.subplots(N, M, figsize=(13, 8), sharex=True, sharey=True)

# Turn off all axes initially
for i in range(N):
    for j in range(M):
        #ax[i,j].axis('off')
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
        ax[i,j].spines['top'].set_visible(False)
        ax[i,j].spines['right'].set_visible(False)
        ax[i,j].spines['left'].set_visible(False)
        ax[i,j].spines['bottom'].set_visible(False)


# Reduce space between subplots
plt.subplots_adjust(wspace=0.03, hspace=0.03)

index = 11
sampled = [62381, 55450, 34656, 13862, 6931, 3466]

# Find global min and max
vmin = 0
vmax = 10

error_vmin = 0
error_vmax = 3

gt_img = np.array(Image.open(f"/scratchdata/compressed_sensing/gt_half/gt_half_{index}.png")) / 1000.0
print("GT max:", gt_img.max(), "min:", gt_img.min(), gt_img.dtype)

for s in range(len(sampled)):
    sampled_path = f"/scratchdata/compressed_sensing/depthprompting/sample_{sampled[s]}_sample_{index}.png"
    sampled_img = np.array(Image.open(sampled_path)) / 1000.0
    ax[0,s].imshow(sampled_img, cmap='gray', vmin=vmin, vmax=vmax)

    depthprompting_path = f"/scratchdata/compressed_sensing/depthprompting/pred_{sampled[s]}_{index}.png"
    depthprompting_img = np.array(Image.open(depthprompting_path)) / 1000.0
    ax[1,s].imshow(depthprompting_img, cmap='gray', vmin=vmin, vmax=vmax)

    depthprompting_error = abs(depthprompting_img.astype(float) - gt_img.astype(float))
    print(depthprompting_error.max(), depthprompting_error.min(), depthprompting_error.dtype)
    ax[2,s].imshow(depthprompting_error, cmap='inferno', vmin=error_vmin, vmax=error_vmax)

    cs_path = f"/scratchdata/compressed_sensing/depthformer/pred_{sampled[s]}_{index}.png"
    cs_img = np.array(Image.open(cs_path)) / 1000.0
    ax[3,s].imshow(cs_img, cmap='gray', vmin=vmin, vmax=vmax)

    cs_error = abs(cs_img.astype(float) - gt_img.astype(float))
    print(cs_error.max(), cs_error.min(), cs_error.dtype)
    ax[4,s].imshow(cs_error, cmap='inferno', vmin=error_vmin, vmax=error_vmax)


plt.rcParams.update({'font.size': 8})

# Set global colorbars
cbar_ax = fig.add_axes([0.92, 0.55, 0.010, 0.35])  # [left, bottom, width, height]
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='gray', norm=plt.Normalize(vmin=vmin, vmax=vmax)), cax=cbar_ax)
cbar.set_label('Depth (m)', rotation=270, labelpad=12)

cbar_ax2 = fig.add_axes([0.92, 0.1, 0.010, 0.35])  # [left, bottom, width, height]
cbar2 = fig.colorbar(plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(vmin=error_vmin, vmax=error_vmax)), cax=cbar_ax2)
cbar2.set_label('Absolute Error (m)', rotation=270, labelpad=12)

TITLE_FONT_SIZE = 12

ax[0,0].title.set_fontsize(TITLE_FONT_SIZE)
ax[0,1].title.set_fontsize(TITLE_FONT_SIZE)
ax[0,2].title.set_fontsize(TITLE_FONT_SIZE)
ax[0,3].title.set_fontsize(TITLE_FONT_SIZE)
ax[0,4].title.set_fontsize(TITLE_FONT_SIZE)
ax[0,5].title.set_fontsize(TITLE_FONT_SIZE)
ax[0,0].set_title('Sampled Ratio = 0.90')
ax[0,1].set_title('0.80')
ax[0,2].set_title('0.50')
ax[0,3].set_title('0.20')
ax[0,4].set_title('0.10')
ax[0,5].set_title('0.05')

# Set axis size

AXIS_FONT_SIZE = 8

ax[0,0].set_ylabel('Measurement Samples', fontsize=AXIS_FONT_SIZE)
ax[1,0].set_ylabel('Depth Prompting', fontsize=AXIS_FONT_SIZE)
ax[2,0].set_ylabel('Rel. Error (DP)', fontsize=AXIS_FONT_SIZE)
ax[3,0].set_ylabel('DepthFormer + CS', fontsize=AXIS_FONT_SIZE)
ax[4,0].set_ylabel('Rel. Error (CS)', fontsize=AXIS_FONT_SIZE)


fig.savefig("image_details.png", dpi=300, bbox_inches='tight')