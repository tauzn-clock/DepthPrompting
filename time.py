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
from depthanything_interface import get_model

model_type = "vits" # "vitb" or "vitl"

model = get_model(model_type)
model.to("cuda:0")
model.eval()

R = 0.5

store = []

for i in tqdm.tqdm(range(654)):
  RGB_PATH = f"/scratchdata/compressed_sensing/rgb/rgb_{i}.png"
  DEPTH_PATH = f"/scratchdata/compressed_sensing/gt/gt_{i}.png"

  rgb = cv2.imread(RGB_PATH)
  #rgb = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
  #rgb = rgb.cuda() if torch.cuda.is_available() else rgb

  est = model.infer_image(rgb)
  #est = est.unsqueeze(0).unsqueeze(0).cpu().numpy()
  #print(est.max(), est.min(), est.shape)

  depth = Image.open(DEPTH_PATH, mode='r')
  depth = np.array(depth) / 1000.0

  indices = np.random.choice(est.size, size=int(est.size * R), replace=False)
  mask = np.zeros(est.shape, dtype=bool)
  mask[np.unravel_index(indices, est.shape)] = True
  sample = depth * mask


  #ratio = rescale_ratio(sample, est,relative_C=5)
  ratio = rescale_ratio_proportional(sample, est)
  final = est * ratio
  final = final * (1-mask) + sample * mask

  store.append(metric(depth, final))

store = np.array(store)
print(store.mean(axis=0))