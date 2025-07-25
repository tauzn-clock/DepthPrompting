import torch
from collections import OrderedDict
from config import args as args_config
from model_list import import_model

DEVICE = "cuda:0"

args = args_config

args.data_name = "NYU" 
args.pretrain = "/scratchdata/depthformer_nyu.pth" 
args.model_name = "depth_prompt_main" 
args.patch_height = 240 
args.patch_width = 320 
args.prop_kernel = 9 
args.prop_time = 18 
args.loss = "L1L2_SILogloss_init2" 
args.dir_data = "/scratchdata/nyudepthv2"

model = import_model(args)
model.cuda()

checkpoint = torch.load(args.pretrain, map_location='cpu')
loaded_state_dict = checkpoint['state_dict']
new_state_dict = OrderedDict()
for n, v in loaded_state_dict.items():
    name = "module."+n
    new_state_dict[name] = v
model.load_state_dict(new_state_dict, strict=False)
model.cuda()
model.to(DEVICE)

print('Load pretrained weight')
print(checkpoint.keys())

args.max_depth = 10.0
args.split_json = './data/data_split/nyu.json'
target_vals = [1]

from data.nyu import NYU as NYU_Dataset
val_datasets = [NYU_Dataset(args, 'test', num_sample_test=v) for v in target_vals]
test_loaders = [torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    num_workers=4, pin_memory=False, drop_last=False) for val_dataset in val_datasets]

import matplotlib.pyplot as plt
model.eval()
for i, sample in enumerate(test_loaders[0]):
    sample = {key: val.to(DEVICE) for key, val in sample.items() if val is not None}
    
    print(sample.keys())
    print(sample['rgb'].shape)
    rgb = sample['rgb'][0,0].detach().cpu().numpy()
    print(rgb.max(), rgb.min())
    plt.imsave("rgb.png", rgb)
    
    output = model(sample)
    
    print(output.keys())
    
    pred = output["pred"][0,0].detach().cpu().numpy()
    print(pred.max(), pred.min())
    plt.imsave("pred.png", pred)
    
    pred_init = output["pred_init"][0,0].detach().cpu().numpy()
    print(pred_init.max(), pred_init.min())
    plt.imsave("pred_init.png", pred_init)
    
    #print(len(output["pred_inter"]))
    #pred_inter = output["pred_inter"][0,0].detach().cpu().numpy()
    #print(pred_inter.max(), pred_inter.min())
    #plt.imsave("pred_inter.png", pred_inter)
    
    #print(output["guidance"].shape)
    #print(output["confidence"].shape)
    
    break