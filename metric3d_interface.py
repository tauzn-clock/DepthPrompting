import torch

def get_model():
    model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
    return model

if __name__ == "__main__":
    model = get_model()
    model = model.cuda() if torch.cuda.is_available() else model
    #model.eval()

    from PIL import Image
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt

    rgb = Image.open('/scratchdata/InformationOptimisation/rgb/3.png').convert('RGB')
    rgb = np.array(rgb)

    plt.imsave("rgb.png", rgb)

    print(rgb.shape)
    rgb = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    rgb = rgb.cuda() if torch.cuda.is_available() else rgb
    print(rgb.max(), rgb.min(), rgb.mean())
    print(rgb.shape)
    print(rgb.dtype)

    model = model.cuda() if torch.cuda.is_available() else model
    pred_depth, confidence, output_dict = model.inference({'input': rgb})
    print(pred_depth.shape)
    pred_normal = output_dict['prediction_normal'][:, :3, :, :] # only available for Metric3Dv2 i.e., ViT models
    normal_confidence = output_dict['prediction_normal'][:, 3, :, :] # see https://arxiv.org/abs/2109.09881 for details

    print(pred_depth.max(), pred_depth.min(), pred_depth.mean())
    plt.imsave("pred_depth.png", pred_depth[0, 0].cpu().numpy())
    plt.imsave("pred_normal.png", (pred_normal[0].cpu().numpy().transpose(1, 2, 0)+1) / 2)

    print(output_dict.keys())
    print(output_dict['prediction'].shape, output_dict['prediction'].max(), output_dict['prediction'].min(), output_dict['prediction'].mean())