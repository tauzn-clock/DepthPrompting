import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import scipy.fft as spfft
import torch
import tqdm

def metric(gt, pred):
  error = np.abs(gt - pred)
  rmse = np.sqrt(np.mean(error**2))
  mae = np.mean(error)
  delta1 = np.mean((np.maximum(gt / pred, pred / gt) < 1.25).astype(np.float32))
  return [rmse, mae, delta1]

#Numpy set seed
np.random.seed(0)

import sys
sys.path.append("/DepthPrompting")
sys.path.append("/DepthPrompting/pylbfgs")
from compressed_sensing import rescale_ratio, rescale_ratio_proportional
from metric3d_interface import get_model

model_type = "metric3d_vit_small" # "vitb" or "vitl"

model = get_model(model_type)
model.to("cuda:0")

R = 0.5

store = []

for i in tqdm.tqdm(range(654)):
  RGB_PATH = f"/scratchdata/compressed_sensing/rgb/rgb_{i}.png"
  DEPTH_PATH = f"/scratchdata/compressed_sensing/gt/gt_{i}.png"

  rgb = Image.open(RGB_PATH)
  rgb = np.array(rgb)
  rgb = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
  rgb = rgb.cuda() if torch.cuda.is_available() else rgb

  est, _, _ = model.inference({'input': rgb})
  est = torch.nn.functional.interpolate(
  est, size=(rgb.shape[2], rgb.shape[3]), mode='bilinear', align_corners=False)
  est = est[0, 0].cpu().numpy()
  #Image.fromarray((est*1000).astype(np.uint16)).save(f"/scratchdata/compressed_sensing/metric3dsmall_est/{i}.png")

  depth = Image.open(DEPTH_PATH, mode='r')
  depth = np.array(depth) / 1000.0

  indices = np.random.choice(est.size, size=int(est.size * R), replace=False)
  mask = np.zeros(est.shape, dtype=bool)
  mask[np.unravel_index(indices, est.shape)] = True
  sample = depth * mask
  #Image.fromarray((sample*1000).astype(np.uint16)).save(f"/scratchdata/compressed_sensing/metric3dsmall_sample/{i}.png")


  #ratio = rescale_ratio(sample, est,relative_C=5)
  ratio = rescale_ratio_proportional(sample, est)
  final = est * ratio
  final = final * (1-mask) + sample * mask
  Image.fromarray((final*1000).astype(np.uint16)).save(f"/scratchdata/compressed_sensing/metric3dsmall_prop/{i}.png")

  store.append(metric(depth, final))

store = np.array(store)
print(store.mean(axis=0))