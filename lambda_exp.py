from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

N = 6
M = 6

fig, ax = plt.subplots(N, M, figsize=(13, 10), sharex=True, sharey=True)

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

index = [11, 40, 105]
coord = [(60,250), (170,170), (250,260)]
X = 80 * 2
Y = 60 * 2

# Find global min and max
vmin = 0
vmax = 10


for s in range(len(index)):
    img = Image.open(f"/scratchdata/compressed_sensing/gt/gt_{index[s]}.png")
    img = np.array(img) / 1000.0
    ax[0,2*s].imshow(img, cmap='inferno', vmin=vmin, vmax=vmax)

    # Zoom in on a specific region (optional)
    x_start, y_start = coord[s]
    x_end, y_end = x_start + X, y_start + Y
    crop = img[y_start:y_end, x_start:x_end]
    crop = np.array(Image.fromarray((crop * 255).astype(np.uint8)).resize((img.shape[1], img.shape[0]), Image.LANCZOS))
    ax[0,2*s+1].imshow(crop, cmap='inferno', vmin=vmin, vmax=vmax)
    # Zoom in on a specific region (optional)
    x_start, y_start = coord[s]
    x_end, y_end = x_start + X, y_start + Y
    # Draw rectangle on original image
    ax[0,2*s].add_patch(plt.Rectangle((x_start, y_start), X, Y, edgecolor='green', facecolor='none', lw=1))
    crop = img[y_start:y_end, x_start:x_end]
    crop = np.array(Image.fromarray(crop).resize((img.shape[1], img.shape[0]), Image.LANCZOS))
    ax[0,2*s+1].imshow(crop, cmap='inferno', vmin=vmin, vmax=vmax)

    img = Image.open(f"/scratchdata/compressed_sensing/metric3dsmall_est/{index[s]}.png")
    img = np.array(img) / 1000.0
    ax[1,2*s].imshow(img, cmap='inferno', vmin=vmin, vmax=vmax)
    # Zoom in on a specific region (optional)
    x_start, y_start = coord[s]
    x_end, y_end = x_start + X, y_start + Y
    # Draw rectangle on original image
    ax[1,2*s].add_patch(plt.Rectangle((x_start, y_start), X, Y, edgecolor='green', facecolor='none', lw=1))
    crop = img[y_start:y_end, x_start:x_end]
    crop = np.array(Image.fromarray(crop).resize((img.shape[1], img.shape[0]), Image.LANCZOS))
    print(crop.min(), crop.max())
    ax[1,2*s+1].imshow(crop, cmap='inferno', vmin=vmin, vmax=vmax)

    img = Image.open(f"/scratchdata/compressed_sensing/metric3dsmall_prop/{index[s]}.png")
    img = np.array(img) / 1000.0
    ax[2,2*s].imshow(img, cmap='inferno', vmin=vmin, vmax=vmax)
    # Zoom in on a specific region (optional)
    x_start, y_start = coord[s]
    x_end, y_end = x_start + X, y_start + Y
    ax[2,2*s].add_patch(plt.Rectangle((x_start, y_start), X, Y, edgecolor='green', facecolor='none', lw=1))
    crop = img[y_start:y_end, x_start:x_end]
    crop = np.array(Image.fromarray(crop).resize((img.shape[1], img.shape[0]), Image.LANCZOS))
    print(crop.min(), crop.max())
    ax[2,2*s+1].imshow(crop, cmap='inferno', vmin=vmin, vmax=vmax)

    img = Image.open(f"/scratchdata/compressed_sensing/metric3dsmall_5/{index[s]}.png")
    img = np.array(img) / 1000.0
    ax[3,2*s].imshow(img, cmap='inferno', vmin=vmin, vmax=vmax)
    # Zoom in on a specific region (optional)
    x_start, y_start = coord[s]
    x_end, y_end = x_start + X, y_start + Y
    ax[3,2*s].add_patch(plt.Rectangle((x_start, y_start), X, Y, edgecolor='green', facecolor='none', lw=1))
    crop = img[y_start:y_end, x_start:x_end]
    crop = np.array(Image.fromarray(crop).resize((img.shape[1], img.shape[0]), Image.LANCZOS))
    print(crop.min(), crop.max())
    ax[3,2*s+1].imshow(crop, cmap='inferno', vmin=vmin, vmax=vmax)
    
    img = Image.open(f"/scratchdata/compressed_sensing/metric3dsmall_05/{index[s]}.png")
    img = np.array(img) / 1000.0
    ax[4,2*s].imshow(img, cmap='inferno', vmin=vmin, vmax=vmax)
    # Zoom in on a specific region (optional)
    x_start, y_start = coord[s]
    x_end, y_end = x_start + X, y_start + Y
    ax[4,2*s].add_patch(plt.Rectangle((x_start, y_start), X, Y, edgecolor='green', facecolor='none', lw=1))
    crop = img[y_start:y_end, x_start:x_end]
    crop = np.array(Image.fromarray(crop).resize((img.shape[1], img.shape[0]), Image.LANCZOS))
    print(crop.min(), crop.max())
    ax[4,2*s+1].imshow(crop, cmap='inferno', vmin=vmin, vmax=vmax)

    img = Image.open(f"/scratchdata/compressed_sensing/metric3dsmall_005/{index[s]}.png")
    img = np.array(img) / 1000.0
    ax[5, 2*s].imshow(img, cmap='inferno', vmin=vmin, vmax=vmax)
    # Zoom in on a specific region (optional)
    x_start, y_start = coord[s]
    x_end, y_end = x_start + X, y_start + Y
    ax[5,2*s].add_patch(plt.Rectangle((x_start, y_start), X, Y, edgecolor='green', facecolor='none', lw=1))
    crop = img[y_start:y_end, x_start:x_end]
    crop = np.array(Image.fromarray(crop).resize((img.shape[1], img.shape[0]), Image.LANCZOS))
    print(crop.min(), crop.max())
    ax[5,2*s+1].imshow(crop, cmap='inferno', vmin=vmin, vmax=vmax)


plt.rcParams.update({'font.size': 8})

# Set global colorbars
cbar_ax = fig.add_axes([0.92, 0.3, 0.010, 0.40])  # [left, bottom, width, height]
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(vmin=vmin, vmax=vmax)), cax=cbar_ax)
cbar.set_label('Depth (m)', rotation=270, labelpad=12)

# Set axis size

AXIS_FONT_SIZE = 8

ax[0,0].set_ylabel('GT', fontsize=AXIS_FONT_SIZE)
ax[1,0].set_ylabel('Initial MDE Estimate', fontsize=AXIS_FONT_SIZE)
ax[2,0].set_ylabel('Proportional Rescaling', fontsize=AXIS_FONT_SIZE)
ax[3,0].set_ylabel(r"$\lambda=5\times|\bar{R}|_F$", fontsize=AXIS_FONT_SIZE)
ax[4,0].set_ylabel(r"$\lambda=0.5\times|\bar{R}|_F$", fontsize=AXIS_FONT_SIZE)
ax[5,0].set_ylabel(r"$\lambda=0.05\times|\bar{R}|_F$", fontsize=AXIS_FONT_SIZE)


fig.savefig("lambda.png", dpi=300, bbox_inches='tight')