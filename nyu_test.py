from utils.metric_func import *
from utils.util_func import *

from config import args

from data.nyu import NYU as NYU_Dataset

import matplotlib.pyplot as plt

args.patch_height, args.patch_width = 240, 320
args.max_depth = 10.0
args.split_json = './data/data_split/nyu.json'
target_vals = convert_str_to_num(args.nyu_val_samples, 'int')
val_dataset = NYU_Dataset(args, 'test', num_sample_test=1)
print('Dataset is NYU')
num_sparse_dep = args.num_sample

test_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    num_workers=4, pin_memory=False, drop_last=False)


for i, sample in enumerate(test_loader):
    for k in sample.keys():
        print(k, sample[k].shape)

    break

img = sample['rgb'][0].cpu().numpy().squeeze().transpose(1, 2, 0)
img = (img - img.min()) / (img.max() - img.min())
depth = sample['dep'][0].squeeze()
gt = sample['gt'][0].squeeze()

print(img.max(), img.min())

plt.imsave('img.png', img)
plt.imsave('depth.png', depth)
plt.imsave('gt.png', gt)