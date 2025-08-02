import os
import random
import time
import warnings
from collections import OrderedDict
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

from utils.metric_func import *
from utils.util_func import *

from tqdm import tqdm
from config import args as args_config
from model_list import import_model

from depthanything_interface import *

import matplotlib.pyplot as plt
import sys
sys.path.append("/DepthPrompting/pylbfgs")
from compressed_sensing import rescale_ratio

from PIL import Image

args = args_config
best_rmse = 10.0

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])
    
    if args.data_name == 'NYU':
        from data.nyu import NYU as NYU_Dataset
        args.patch_height, args.patch_width = 240, 320
        args.max_depth = 10.0
        args.split_json = './data/data_split/nyu.json'
        target_vals = convert_str_to_num(args.nyu_val_samples, 'int')
        val_datasets = [NYU_Dataset(args, 'test', num_sample_test=v) for v in target_vals]
        print('Dataset is NYU')
        num_sparse_dep = args.num_sample        
    elif args.data_name == 'KITTIDC':
        from data.kittidc import KITTIDC as KITTI_dataset
        args.patch_height, args.patch_width = 240, 1216
        args.max_depth = 80.0
        args.split_json = './data/data_split/kitti_dc.json'
        target_vals = convert_str_to_num(args.kitti_val_lidars, 'int')
        val_datasets = [KITTI_dataset(args, 'test', num_lidars_test=v) for v in target_vals]
        print('Dataset is KITTI')
        num_sparse_dep = args.lidar_lines
    elif args.data_name == 'VOID':
        from data.void import VOID
        args.nyu_val_samples = str(args.void_sparsity)
        target_vals = convert_str_to_num(args.nyu_val_samples, 'int')
        dataset = VOID(args, 'test')
        val_datasets = [dataset]
        num_sparse_dep = args.num_sample
    elif args.data_name == 'SUNRGBD':
        from data.sun_rgbd import SUN_RGBD
        args.split_json = './data/data_split/allsplit.mat'
        args.max_depth = 10.0
        target_vals = convert_str_to_num(args.nyu_val_samples, 'int')
        val_datasets = [SUN_RGBD(args, 'test', num_sample_test=v) for v in target_vals]
        num_sparse_dep = args.num_sample
    elif args.data_name == 'IPAD':
        from data.ipad import iPad as IPAD_dataset
        args.max_depth = 10.0
        target_vals = convert_str_to_num(args.nyu_val_samples, 'int')
        val_datasets = [IPAD_dataset(args, 'test', num_sample_test=v) for v in target_vals]
        print('[IPAD Dataset] Split: {} | MaxDepth: {} | H,W: {},{} | Num Sample: {}'.format(args.split_json, args.max_depth, args.patch_height, args.patch_width, args.num_sample))
        num_sparse_dep = args.num_sample
    elif args.data_name == 'NUSCENE':
        from data.nuscene import NUSCENE
        args.max_depth = 80.0
        dataset = NUSCENE(args, 'test')
        target_vals = convert_str_to_num(args.kitti_val_lidars, 'int')
        val_datasets = [dataset]
        num_sparse_dep = args.num_sample
    elif args.data_name == 'SCENENET':
        from data.scenenet import SCENENET
        args.patch_height, args.patch_width = 240, 320
        args.data_path = './data/data_split/scenenet.csv'
        target_vals = convert_str_to_num(args.nyu_val_samples, 'int')
        val_datasets = [SCENENET(args, 'test', num_sample_test=v) for v in target_vals]
        print("Using SCENENET")
        num_sparse_dep = args.num_sample 
        print(num_sparse_dep)
    elif args.data_name == 'OUR':
        from data.our import OUR
        import json
        #assert args.patch_height == 240 
        #assert args.patch_width == 320
        with open(os.path.join(args.dir_data, "camera_info.json"), 'r') as f:
            camera_info = json.load(f)
            print("Camera Info:", camera_info)
        args.fx = camera_info['P'][0]
        args.fy = camera_info['P'][5]
        args.cx = camera_info['P'][2]
        args.cy = camera_info['P'][6]
        
        args.data_length = 393
        args.max_depth = 10.0   
        args.scale = 1000.0    
        target_vals = convert_str_to_num(args.nyu_val_samples, 'int')
        val_datasets = [OUR(args, 'test', num_sample_test=v) for v in target_vals]
        print('Dataset is NYU')
        num_sparse_dep = args.num_sample        
    else:
        print("Please Choice Dataset !!")
        raise NotImplementedError
    print("Using depth_anything model for evaluation...")
    model = get_model()
    model.to("cuda:0")
    model.eval()
    args.num_sparse_dep = num_sparse_dep

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')        

    print('MaxDepth: {} | H,W: {},{}'.format(args.max_depth, args.patch_height, args.patch_width))

    if args.visualization:
        print("Save directory: ", args.save_dir)
        os.makedirs(args.save_dir, exist_ok=True)
        os.system("chmod -R 777 {}".format(args.save_dir))
        from utils import visualize
        visual = visualize.visualize(args)
    else:
        visual = None

    test_loaders = [torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=False, drop_last=False) for val_dataset in val_datasets]

    avg_rmse = AverageMeter('avg_rmse', ':6.4f')
    avg_mae = AverageMeter('avg_mae', ':6.4f')
    avg_delta1 = AverageMeter('avg_de;ta1', ':6.4f')
    
    print('\n\n=== Arguments ===')
    cnt = 0
    for key in sorted(vars(args)):
        print(key, ':',  getattr(args, key), end='  |  ')
        cnt += 1
        if (cnt + 1) % 5 == 0:
            print('')
    print('\n')
        
    for target_val, val_loader in zip(target_vals, test_loaders):
        val_rmse, val_mae, val_delta1 = test(val_loader, model, args, visual, target_val)
        avg_rmse.update(val_rmse)
        avg_mae.update(val_mae)
        avg_delta1.update(val_delta1)
    print("Test for various Sampels/Lidars:",target_vals)
    
    store_metrics = []
    
    for target_val_,rmse_,mae_,delta1_ in zip(target_vals,avg_rmse.list,avg_mae.list,avg_delta1.list):
        print('{:.4f}/{:.4f}/{:.4f}'.format(rmse_,mae_,delta1_),end=" ")
        store_metrics.append([target_val_,float(rmse_),float(mae_),float(delta1_)])
        
    import csv
    import time
    TIMESTAMP = str(time.time())
    with open(f"./metrics/tmp{TIMESTAMP}.csv","w",newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Sample","RMSE","MAE","DELTA1"])
        writer.writerows(store_metrics)
    print("\n [Average RMSE/MAE/DELTA1] ==> {:2.4f}/{:2.4f}/{:2.4f}\n".format(avg_rmse.avg,avg_mae.avg,avg_delta1.avg))
    
def test(test_loader, model, args, visual, target_sample):
    rmse = AverageMeter('RMSE', ':.4f')
    mae = AverageMeter('MAE', ':.4f')
    delta1 = AverageMeter('DELTA1',':.4f')
    model.eval()
    pbar = tqdm(total=len(test_loader))

    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            sample = {key: val.to('cuda') for key, val in sample.items() if val is not None}
            raw_img = sample["rgb_h5"][0].detach().cpu().numpy()
            raw_img = raw_img[...,::-1]
            
            target_shape = sample["dep"][0,0].shape
            raw_img = cv2.resize(raw_img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

            depth_pred = model.infer_image(raw_img)
            depth_pred = torch.tensor(depth_pred, device='cuda').unsqueeze(0).unsqueeze(0)
            #depth_pred = rescale_pred(sample["dep"][0,0].detach().cpu().numpy(), depth_pred)
            #depth_pred = torch.from_numpy(depth_pred).unsqueeze(0).unsqueeze(0).to("cuda")

            output = {'pred_init': depth_pred, 'pred': depth_pred}
                        
            if True:
                sampled_pts = sample["dep"][0,0].detach().cpu().numpy()
                pred_init = output["pred_init"][0,0].detach().cpu().numpy()
                _,_,H,W = output["pred"].shape
                R = int(sample["num_sample"]) / (H*W)
                ratio = rescale_ratio(sampled_pts, pred_init,relative_C=1/R)
                mask = sampled_pts > 0.0
                depth_pred = pred_init * ratio
                depth_pred = depth_pred * (1-mask) + sampled_pts * mask
                output["pred"] = torch.tensor(depth_pred, device='cuda').unsqueeze(0).unsqueeze(0)

            if target_sample==0: 
                rmse_result, mae_result, abs_rel_result = eval_metric2(sample, output['pred_init'], args)
            else: rmse_result, mae_result, abs_rel_result = eval_metric2(sample, output['pred'], args)
            #print(rmse_result, mae_result, abs_rel_result)
            #exit()

            #from debug import debug
            #debug(sample, output)
            #exit()

            rmse.update(rmse_result, sample['gt'].size(0))
            mae.update(mae_result, sample['gt'].size(0))
            delta1.update(abs_rel_result,sample['gt'].size(0))

            if args.visualization:
                visual.data_put(sample, output)
                path_ = os.path.join(args.save_dir,'sample_{:04d}'.format(target_sample))
                os.makedirs(path_, exist_ok=True)
                if args.data_name ==  'IPAD':
                    visual.save_all_nyu_gt_sparse_rgb_errormap(idx=i, path_to_save=path_)    
                elif args.data_name == 'NUSCENE':
                    visual.save_all_kitti_gt_sparse_rgb_errormap(idx=i, path_to_save=path_)
                    visual.depth(type='pred', idx=i, path_to_save=path_)
                    visual.depth(type='sparse', idx=i, path_to_save=path_)
                    visual.RGB(idx=i, path_to_save=path_)

            if args.use_raw_depth_as_input:
                error_str = '{} | #:{} | '.format('Test', 'raw')
            else:
                error_str = '{} | #:{:3d} | '.format('Test', int(target_sample))

            pbar.set_description(error_str)
            pbar.update(test_loader.batch_size)

        if args.use_raw_depth_as_input:
            error_str_new = '[{}] #:{} | RMSE/MAE/DELTA1: {:.4f}/{:.4f}/{:.4f}'.format('Test', 'raw', rmse.avg, mae.avg, delta1.avg)
        else:
            error_str_new = '[{}] #:{:3d} | RMSE/MAE: {:.4f}/{:.4f}/{:.4f}'.format('Test', int(target_sample), rmse.avg, mae.avg, delta1.avg)
            

        pbar.set_description(error_str_new)
        pbar.close()

    return rmse.avg, mae.avg, delta1.avg

if __name__ == '__main__':
    main()
