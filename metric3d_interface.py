import torch

def get_model():
    model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
    return model

if __name__ == "__main__":
    model = get_model()
    model.to("cuda:0")
    model.eval()

    from PIL import Image
    import numpy as np
    from matplotlib import pyplot as plt

    rgb = Image.open('/scratchdata/nyu_plane/rgb/0.png').convert('RGB')
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

