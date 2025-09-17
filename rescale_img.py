import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import scipy.fft as spfft
import torch

import sys
sys.path.append("/DepthPrompting")

from metric3d_interface import get_model
import tqdm

import sys
sys.path.append("/DepthPrompting/pylbfgs")
from compressed_sensing import rescale_ratio, rescale_ratio_proportional

model_type = "metric3d_vit_small" # "vitb" or "vitl"

model = get_model(model_type)
model.to("cuda:0")
#model.eval()

INDEX = 120

RGB_PATH = f"/scratchdata/compressed_sensing/rgb/rgb_{INDEX}.png"
DEPTH_PATH = f"/scratchdata/compressed_sensing/gt/gt_{INDEX}.png"
SAMPLE_PATH = f"/scratchdata/compressed_sensing/metric3dsmall_sample/{INDEX}.png"
METRIC_5 = f"/scratchdata/compressed_sensing/metric3dsmall_5/{INDEX}.png"
METRIC_05 = f"/scratchdata/compressed_sensing/metric3dsmall_05/{INDEX}.png"
METRIC_005 = f"/scratchdata/compressed_sensing/metric3dsmall_005/{INDEX}.png"

rgb = Image.open(RGB_PATH)
depth = Image.open(DEPTH_PATH)
sample = Image.open(SAMPLE_PATH)
metric_5 = Image.open(METRIC_5)
metric_05 = Image.open(METRIC_05)
metric_005 = Image.open(METRIC_005)

rgb = np.array(rgb)
plt.imsave(f"rgb_{INDEX}.png", rgb)

rgb = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
rgb = rgb.cuda() if torch.cuda.is_available() else rgb
est, _, _ = model.inference({'input': rgb})
est = torch.nn.functional.interpolate(
        est, size=(rgb.shape[2], rgb.shape[3]), mode='bilinear', align_corners=False)
est = est[0, 0].cpu().numpy()

print(est.min(), est.max())


depth = np.array(depth) / 1000.0
sample = np.array(sample) / 1000.0
metric_5 = np.array(metric_5) / 1000.0
metric_05 = np.array(metric_05) / 1000.0
metric_005 = np.array(metric_005) / 1000.0

vmax = max(depth.max(), sample.max(), metric_5.max(), metric_05.max(), metric_005.max(), est.max())
vmin = min(depth.min(), sample.min(), metric_5.min(), metric_05.min(), metric_005.min(), est.min())
plt.imsave(f"depth_{INDEX}.png", depth, cmap='inferno', vmin=vmin, vmax=vmax)
plt.imsave(f"sample_{INDEX}.png", sample, cmap='inferno', vmin=vmin, vmax=vmax)
plt.imsave(f"est_{INDEX}.png", est, cmap='inferno', vmin=vmin, vmax=vmax)
plt.imsave(f"metric_5_{INDEX}.png", metric_5, cmap='inferno', vmin=vmin, vmax=vmax)
plt.imsave(f"metric_05_{INDEX}.png", metric_05, cmap='inferno', vmin=vmin, vmax=vmax)
plt.imsave(f"metric_005_{INDEX}.png", metric_005, cmap='inferno', vmin=vmin, vmax=vmax)

exit()

model_type = "metric3d_vit_small" # "vitb" or "vitl"

model = get_model(model_type)
model.to("cuda:0")
#model.eval()


INDEX = 20

RGB_PATH = f"/scratchdata/compressed_sensing/rgb/{INDEX}.png"
DEPTH_PATH = f"/scratchdata/compressed_sensing/depth/{INDEX}.png"

rgb = Image.open(RGB_PATH)
depth = Image.open(DEPTH_PATH)
rgb = np.array(rgb)
depth = np.array(depth) / 1000.0
print(depth.min(), depth.max())
plt.imsave(f"rgb_{INDEX}.png", rgb)

rgb = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
rgb = rgb.cuda() if torch.cuda.is_available() else rgb
est, _, _ = model.inference({'input': rgb})
est = torch.nn.functional.interpolate(
        est, size=(rgb.shape[2], rgb.shape[3]), mode='bilinear', align_corners=False)
est = est[0, 0].cpu().numpy()

print(est.min(), est.max())

ratio = rescale_ratio(depth, est,relative_C=0.05)
mask = depth > 0.0
final = est * ratio
final = final * (1-mask) + depth * mask

print(ratio.min(), ratio.max())
vmax = max(final.max(), depth.max(), est.max())
vmin = min(final.min(), depth.min(), est.min())
plt.imsave(f"depth_{INDEX}.png", depth, cmap='inferno', vmin=vmin, vmax=vmax)
plt.imsave(f"est_{INDEX}.png", est, cmap='inferno', vmin=vmin, vmax=vmax)
plt.imsave(f"final_{INDEX}.png", final, cmap='inferno', vmin=vmin, vmax=vmax)