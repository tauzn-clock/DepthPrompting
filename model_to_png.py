import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def output_to_png(depth_map, output_path):
    print(output_path)
    print(depth_map.shape)

    img = depth_map[0, 0].detach().cpu().numpy() * 1000
    img = img.astype('uint16')
    depth_image = Image.fromarray(img)
    depth_image.save(output_path)

if __name__ == "__main__":
    depth = Image.open("./depth_maps/pred_0.png", mode='r')
    depth = np.array(depth)

    print(np.sum(depth==0)/depth.size)

    print(depth.max(), depth.min())

    plt.imshow(depth, cmap='gray')
    plt.savefig("test.png", bbox_inches='tight')