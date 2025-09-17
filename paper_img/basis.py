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

model_type = "metric3d_vit_large" # "vitb" or "vitl"

model = get_model(model_type)
model.to("cuda:0")
#model.eval()

total_basis = []

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
    #print(est.min(), est.max())
    depth = Image.open(DEPTH_PATH, mode='r')
    depth = np.array(depth) / 1000.0
    #print(est.shape, depth.shape)
    #print(depth.min(), depth.max())

    ratio = np.log(depth / est)
    print(ratio.min(), ratio.max())

    basis = spfft.dctn(ratio, norm='ortho')
    basis = np.abs(basis)
    total_basis.append(basis)
    print(basis.min(), basis.max())

total_basis = np.array(total_basis)
avg = np.mean(total_basis, axis=0)

# Save basis with pickle
import pickle
with open(f"/scratchdata/basis_{model_type}.pkl", "wb") as f:
    pickle.dump(avg, f)