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
args.patch_height, args.patch_width = 240, 320
args.max_depth = 10.0
args.split_json = './data/data_split/nyu.json'
args.fx = 5.1885790117450188e+02 / 2.0
args.fy = 5.1946961112127485e+02 / 2.0
args.cx = 3.2558244941119034e+02 / 2.0 - 8.0
args.cy = 2.5373616633400465e+02 / 2.0 - 6.0
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
    output = model(sample)
    
    print(output)
    
    break