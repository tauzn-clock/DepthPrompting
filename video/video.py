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

DIR_PATH = "/scratchdata/processed/alcove3"

for INDEX in range(20,400):
  RGB_PATH = f"{DIR_PATH}/rgb/{INDEX}.png"
  DEPTH_PATH = f"{DIR_PATH}/depth/{INDEX}.png"

  rgb = Image.open(RGB_PATH)
  depth = Image.open(DEPTH_PATH)
  rgb = Image.open(RGB_PATH)
  depth = Image.open(DEPTH_PATH)
  rgb = np.array(rgb)
  rgb_ori = rgb.copy()
  depth = np.array(depth) / 1000.0

  rgb = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
  rgb = rgb.cuda() if torch.cuda.is_available() else rgb
  est, _, _ = model.inference({'input': rgb})
  est = torch.nn.functional.interpolate(
          est, size=(rgb.shape[2], rgb.shape[3]), mode='bilinear', align_corners=False)
  est = est[0, 0].cpu().numpy()

  ratio = rescale_ratio(depth, est,relative_C=0.05)
  mask = depth > 0.0
  final = est * ratio
  final = final * (1-mask) + depth * mask

  print(rgb_ori.min(), rgb_ori.max())
  print(depth.min(), depth.max())
  print(est.min(), est.max())
  print(final.min(), final.max())

  fig, axs = plt.subplots(2, 2, figsize=(8,6))
  fig.tight_layout()

  # Reduce space between subplots
  plt.subplots_adjust(wspace=0.005, hspace=0.005)

  axs[0,0].imshow(rgb_ori.astype(np.uint8))
  axs[0,0].set_title("RGB")
  axs[0,0].axis('off')

  vmax = 20
  vmin = 0

  axs[0,1].imshow(depth, cmap='inferno', vmin=vmin, vmax=vmax)
  axs[0,1].set_title("Measured Depth")
  axs[0,1].axis('off')

  axs[1,0].imshow(est, cmap='inferno', vmin=vmin, vmax=vmax)
  axs[1,0].set_title("MDE Predicted Depth")
  axs[1,0].axis('off')

  axs[1,1].imshow(final, cmap='inferno', vmin=vmin, vmax=vmax)
  axs[1,1].set_title("Our Rescaled Depth")
  axs[1,1].axis('off')

  # Set global colorbars 
  cbar_ax = fig.add_axes([0.98, 0.15, 0.010, 0.80])  # [left, bottom, width, height]
  cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(vmin=vmin, vmax=vmax)), cax=cbar_ax)
  cbar.set_label('Depth (m)', rotation=270, labelpad=12)

  plt.savefig(f"{INDEX}.png", bbox_inches='tight', dpi=300)
