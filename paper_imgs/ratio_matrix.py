import sys
sys.path.append("/DepthPrompting")

from depthanything_interface import get_model

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

INDEX = 11

RGB_PATH = f"/scratchdata/compressed_sensing/rgb/rgb_{INDEX}.png"
rgb = cv2.imread(RGB_PATH)
rgb = rgb[12:-12, 16:-16, :]

GT_PATH = f"/scratchdata/compressed_sensing/gt/gt_{INDEX}.png"
gt = Image.open(GT_PATH, mode='r')
gt = np.array(gt) / 1000.0
print(gt.min(), gt.max())
print(gt.shape)

model = get_model("vitl")
model.eval()
model.to("cuda:0")
pred_l = model.infer_image(rgb) # HxW depth map in meters in numpy
print(pred_l.min(), pred_l.max())
print(pred_l.shape)

ratio_l = gt / pred_l
print(ratio_l.min(), ratio_l.max())

model = get_model("vitb")
model.eval()
model.to("cuda:0")
pred_b = model.infer_image(rgb) # HxW depth map in meters in numpy
print(pred_b.min(), pred_b.max())
print(pred_b.shape)

ratio_b = gt / pred_b
print(ratio_b.min(), ratio_b.max())

model = get_model("vits")
model.eval()
model.to("cuda:0")
pred_s = model.infer_image(rgb) # HxW depth map in meters in numpy
print(pred_s.min(), pred_s.max())
print(pred_s.shape)

ratio_s = gt / pred_s
print(ratio_s.min(), ratio_s.max())

fig, ax = plt.subplots(1, 3, figsize=(12, 8), sharex=True, sharey=True)
# Turn off all axes initially
for i in range(3):
    ax[i].axis('off')

ax[2].imshow(np.log(ratio_l), cmap='inferno')
ax[2].set_title("ViT-L Ratio")
ax[1].imshow(np.log(ratio_b), cmap='inferno')
ax[1].set_title("ViT-B Ratio")
ax[0].imshow(np.log(ratio_s), cmap='inferno')
ax[0].set_title("ViT-S Ratio")

fig.savefig(f"ratio_matrix_{INDEX}.png", bbox_inches='tight', dpi=200)