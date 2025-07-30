import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image
import torch.nn.functional as F

def img_over_pcd(points, img, filepath=None):

    # Visualize the point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    color = img.reshape(-1, 3) / 255.0  # Normalize color values to [0, 1]

    point_cloud.colors = o3d.utility.Vector3dVector(color)

    tf = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    point_cloud.transform(tf)

    if filepath is not None:
        # Visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add the point cloud to the visualizer
        vis.add_geometry(point_cloud)

        opt = vis.get_render_option()
        opt.point_size = 2

        view_control = vis.get_view_control()
        view_control.set_zoom(0.6) 
        view_control.rotate(0, -100)

        img = np.array(vis.capture_screen_float_buffer(True))
        left = img.shape[1]
        right = 0
        top = img.shape[0]
        bottom = 0

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if np.sum(img[i,j]) < 3:
                    left = min(left, j)
                    right = max(right, j)
                    top = min(top, i)
                    bottom = max(bottom, i)

        output = img[top:bottom, left:right]
        
        plt.imsave(filepath, output)
    else:
        return point_cloud

def get_3d(depth, INTRINSICS):
    H, W = depth.shape
    depth = depth.flatten()
    # Generate a grid of (x, y) coordinates
    x, y = np.meshgrid(np.arange(W), np.arange(H))

    # Flatten the arrays
    x = x.flatten()
    y = y.flatten()

    # Calculate 3D coordinates
    fx, fy, cx, cy = INTRINSICS[0], INTRINSICS[1], INTRINSICS[2], INTRINSICS[3]
    z = depth

    x_3d = (x - cx) * z / fx
    y_3d = (y - cy) * z / fy

    # Create a point cloud
    points = np.vstack((x_3d, y_3d, z)).T
    
    return points

def debug(sample, output):

    rgb = sample['rgb'][0].detach().cpu().numpy()
    rgb = np.sum(rgb, axis=0)  # Sum over channels
    gt = sample['gt'][0,0].detach().cpu().numpy()
    sampled_depth = sample['dep'][0,0].detach().cpu().numpy()
    #print(rgb.shape,(gt.shape[0], gt.shape[1]))
    #rgb = F.interpolate(rgb, size=(3, gt.shape[0], gt.shape[1]), mode='bilinear', align_corners=False)#[0]#detach().cpu().numpy()
    
    plt.imsave("rgb.png", rgb)
    plt.imsave("gt.png", gt)
    plt.imsave("sampled_depth.png", sampled_depth)

    INTRINSICS = [5.1885790117450188e+02 / 2.0, 
                  5.1946961112127485e+02 / 2.0, 
                  3.2558244941119034e+02 / 2.0 - 8.0,
                  2.5373616633400465e+02 / 2.0 - 6.0]

    raw_img = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb)) * 255
    print(raw_img.shape)
    # Convert to uint8 and reshape to (H, W, 3)
    raw_img = raw_img.astype(np.uint8)
    raw_img = np.stack([raw_img] * 3, axis=-1)  # Repeat the grayscale channel to create a 3-channel image

    final_depth = output['pred'][0,0].detach().cpu().numpy()

    pts_3d = get_3d(final_depth, INTRINSICS)
    pcd = img_over_pcd(pts_3d, raw_img)
    o3d.io.write_point_cloud("estimated.ply", pcd)

    sampled_depth_pts3d = get_3d(sampled_depth, INTRINSICS)
    sampled_pcd = img_over_pcd(sampled_depth_pts3d, raw_img)
    o3d.io.write_point_cloud("sampled.ply", sampled_pcd)

    gt_pt3d = get_3d(gt, INTRINSICS)
    gt_pcd = img_over_pcd(gt_pt3d, raw_img)
    o3d.io.write_point_cloud("gt.ply", gt_pcd)


