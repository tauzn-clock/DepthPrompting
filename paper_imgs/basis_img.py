import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import scipy.fft as spfft

import sys
sys.path.append("/DepthPrompting")

from depthanything_interface import get_model
import tqdm
import pickle

vits_basis_path = "basis_vits.pkl"
vitb_basis_path = "basis_vitb.pkl"
vitl_basis_path = "basis_vitl.pkl"
metric3d_vits_path = "/scratchdata/basis_metric3d_vit_small.pkl"
metric3d_vitb_path = "/scratchdata/basis_metric3d_vit_large.pkl"

with open(vits_basis_path, "rb") as f:
    vits_basis = pickle.load(f)
with open(vitb_basis_path, "rb") as f:
    vitb_basis = pickle.load(f)
with open(vitl_basis_path, "rb") as f:
    vitl_basis = pickle.load(f)
with open(metric3d_vits_path, "rb") as f:
    metric3d_vits = pickle.load(f)
with open(metric3d_vitb_path, "rb") as f:
    metric3d_vitb = pickle.load(f)

print(vits_basis.max(), vitb_basis.max(), vitl_basis.max())
print(vits_basis.sum(), vitb_basis.sum(), vitl_basis.sum())

size = vits_basis.shape[0] * vits_basis.shape[1]
indices = np.linspace(0, size - 1, size, dtype=int)
print(len(indices))

vits_flat = vits_basis.flatten()
vits_flat = np.sort(vits_flat)[::-1]
vitb_flat = vitb_basis.flatten()
vitb_flat = np.sort(vitb_flat)[::-1]
vitl_flat = vitl_basis.flatten()
vitl_flat = np.sort(vitl_flat)[::-1]
metric3d_vitb_flat = metric3d_vitb.flatten()
metric3d_vitb_flat = np.sort(metric3d_vitb_flat)[::-1]
metric3d_vits_flat = metric3d_vits.flatten()
metric3d_vits_flat = np.sort(metric3d_vits_flat)[::-1]

# Global set font size

start = 0
end = 5000

fig, ax = plt.subplots(1, 2, figsize=(20, 8))
# Set line style and width
ax[0].plot(indices[start:end], metric3d_vits_flat[start:end], label="Metric3DV2-Small", color="red", linewidth=4, linestyle='--')
ax[0].plot(indices[start:end], metric3d_vitb_flat[start:end], label="Metric3DV2-Large", color="blue", linewidth=4, linestyle='--')
ax[0].plot(indices[start:end], vits_flat[start:end], label="DepthAnythingV2-Small", color="red", linewidth=4, linestyle=':')
ax[0].plot(indices[start:end], vitb_flat[start:end], label="DepthAnythingV2-Base", color="green", linewidth=4, linestyle=':')
ax[0].plot(indices[start:end], vitl_flat[start:end], label="DepthAnythingV2-Large", color="blue", linewidth=4, linestyle=':')

ax[0].set_xlim(start, end)
ax[0].set_ylim(0.1, 10)

# Set y axis to logarithmic scale
ax[0].set_yscale("log")

# Add grid lines
ax[0].grid(True)
ax[0].minorticks_on()
ax[0].grid(True, which='major', linestyle='-', linewidth=1.6, color='gray')
ax[0].grid(True, which='minor', linestyle='--', linewidth=0.8, color='lightgray')

ax[0].set_xlabel("Basis Index", fontsize=24)
ax[0].set_ylabel("Magnitude (log scale)", fontsize=24)
ax[0].tick_params(axis='both', which='major', labelsize=20)

start = 0
end = 5000
ax[1].plot(indices[start:end], metric3d_vits_flat[start:end], label="Metric3DV2-Small", color="red", linewidth=4, linestyle='--')
ax[1].plot(indices[start:end], metric3d_vitb_flat[start:end], label="Metric3DV2-Large", color="blue", linewidth=4, linestyle='--')
ax[1].plot(indices[start:end], vits_flat[start:end], label="DepthAnythingV2-Small", color="red", linewidth=4, linestyle=':')
ax[1].plot(indices[start:end], vitb_flat[start:end], label="DepthAnythingV2-Base", color="green", linewidth=4, linestyle=':')
ax[1].plot(indices[start:end], vitl_flat[start:end], label="DepthAnythingV2-Large", color="blue", linewidth=4, linestyle=':')
ax[1].set_yscale("log")
ax[1].grid(True)
ax[1].minorticks_on()
ax[1].grid(True, which='major', linestyle='-', linewidth=1.6, color='gray')
ax[1].grid(True, which='minor', linestyle='--', linewidth=0.8, color='lightgray')

ax[1].set_xlabel("Basis Index", fontsize=24)
#ax[1].set_ylabel("Magnitude (log scale)", fontsize=24)

xmin = 500
xmax = 5000
ymin = 0.1
ymax = 0.5
ax[1].set_xlim(xmin, xmax)
ax[1].set_ylim(ymin, ymax)
ax[1].tick_params(axis='both', which='major', labelsize=20)

# Legend
ax[1].legend(fontsize=20)

# Highlight zoom region in the first plot

"""
rect = plt.Rectangle(
    (xmin, ymin),
    xmax - xmin-10,
    ymax - ymin,
    linewidth=3,
    edgecolor='black',
    facecolor='none',
    linestyle='--'
)
ax[0].add_patch(rect)
"""

fig.savefig("basis.png", bbox_inches="tight", dpi=300)