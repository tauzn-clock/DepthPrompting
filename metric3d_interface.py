import torch

def get_model():
    model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
    return model