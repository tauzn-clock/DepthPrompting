from data.our import OUR
from utils.metric_func import *
from utils.util_func import *
from config import args
from model_list import import_model
import matplotlib.pyplot as plt

from collections import OrderedDict 
import torch
import json

# Config to NYU
args.patch_height, args.patch_width = 480, 640
args.max_depth = 10.0
args.split_json = './data/data_split/nyu.json'
args.fx = 5.1885790117450188e+02
args.fy = 5.1946961112127485e+02
args.cx = 3.2558244941119034e+02 - 8.0 * 2
args.cy = 2.5373616633400465e+02 - 6.0 * 2
args.scale = 1000.0
args.data_length = 1449

val_dataset = OUR(args, 'test', num_sample_test=100)
print(val_dataset.mode)
print('Dataset is NYU')
print("Pretrain Paramter Path:", args.pretrain)

test_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, shuffle=False,
                num_workers=4, pin_memory=False, drop_last=False)
    
model = import_model(args)

checkpoint = torch.load(args.pretrain)
try:
    loaded_state_dict = checkpoint['state_dict']
except:
    loaded_state_dict = checkpoint
new_state_dict = OrderedDict()
for n, v in loaded_state_dict.items():
    name = n.replace("module.", "")
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model = model.cuda()
print('Load pretrained weight')

model.eval()

for i, sample in enumerate(test_loader):
    sample = {key: val.to('cuda') for key, val in sample.items() if val is not None}
    print(sample['dep'].shape)
    output = model(sample)
    
    # pred_init is the initial estimate
    # pred is constrained by inputs
    
    print(output.keys())
    print(output['pred_init'].max(), output['pred_init'].min(), output['pred_init'].shape)
    plt.imsave("pred_init.png", output['pred_init'][0, 0].detach().cpu().numpy(), cmap='gray')
    print(output['pred'].max(), output['pred'].min(), output['pred'].shape)
    plt.imsave("pred.png", output['pred'][0, 0].detach().cpu().numpy(), cmap='gray')
        
    break