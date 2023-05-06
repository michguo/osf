"""Data loader for OSF data."""

import os
import torch
import numpy as np
import imageio 
import json
import cv2

import cam_utils


trans_t = lambda t: torch.tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]
], dtype=torch.float)

rot_phi = lambda phi: torch.tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]
], dtype=torch.float)

rot_theta = lambda th: torch.tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]
], dtype=torch.float)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def convert_cameras_to_nerf_format(anno):
    """
    Args:
        anno: List of annotations for each example. Each annotation is represented by a
            dictionary that must contain the key `RT` which is the world-to-camera
            extrinsics matrix with shape [3, 4], in [right, down, forward] coordinates.

    Returns:
        c2w: [N, 4, 4] np.float32. Array of camera-to-world extrinsics matrices in
            [right, up, backwards] coordinates.
    """
    c2w_list = []
    for a in anno:
        # Convert from w2c to c2w.
        w2c = np.array(a['RT'] + [[0.0, 0.0, 0.0, 1.0]])
        c2w = cam_utils.w2c_to_c2w(w2c)

        # Convert from [right, down, forwards] to [right, up, backwards]
        c2w[:3, 1] *= -1  # down -> up
        c2w[:3, 2] *= -1  # forwards -> back
        c2w_list.append(c2w)
    c2w = np.array(c2w_list)
    return c2w


def load_osf_data(basedir, spiral_radius, half_res=False, testskip=1,
        n_render_spiral=40, render_spiral_angles=None, same_splits=False, render_anno=False,
        test_indices=None, use_train_anno_as_test=False, is_nerf=False):
    """
    Returns:
        imgs: [N, H, W, 4] np.float32. Array of images in RGBA format, and normalized
            between [0, 1].
        poses: [N, 4, 4] np.float32. Camera poses corresponding to each image.
        metadata: [N, M] np.float32. Metadata corresponding to each image.
            Currently, we only support light positions as the metadata (M=3).
    """
    with open(os.path.join(basedir, 'anno_train.json'), 'r') as fp:
        anno_train = json.load(fp)

    with open(os.path.join(basedir, 'anno_test.json'), 'r') as fp:
        anno_test = json.load(fp)
    
    if test_indices is not None:
        if use_train_anno_as_test:
            anno_test = [anno_train[i] for i in test_indices]
        else:
            anno_test = [anno_test[i] for i in test_indices]

    # Convert camera matrices into NeRF format.
    c2w_train = convert_cameras_to_nerf_format(anno_train)
    for i in range(len(anno_train)):
        anno_train[i]['c2w'] = c2w_train[i]

    c2w_test = convert_cameras_to_nerf_format(anno_test)
    for i in range(len(anno_test)):
        anno_test[i]['c2w'] = c2w_test[i]

    # Prepare splits.
    splits = ['train', 'val', 'test']
    metas = {}
    metas['train'] = anno_train
    metas['val'] = anno_test
    metas['test'] = anno_test

    all_imgs = []
    all_poses = []
    all_metadata = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        metadata = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        if len(meta) > 0:
            for frame in meta[::skip]:
                fname = os.path.join(basedir, 'rgba', frame['filename'])
                imgs.append(imageio.imread(fname))  # [H, W, 4]
                poses.append(frame['c2w'])  # [4, 4]
                if is_nerf:
                    metadata.append(np.ones_like(frame['light_pos']))
                else:
                    metadata.append(frame['light_pos'])  # [3,]

            # Normalize images, and keep all 4 channels (RGBA) so that we can use
            # the alpha channel to convert transparent pixels to white if
            # `white_bkgd=True`.
            imgs = (np.array(imgs) / 255.).astype(np.float32)  # [N, H, W, 4]

            poses = np.array(poses).astype(np.float32)  # [N, 4, 4]
            metadata = np.array(metadata).astype(np.float32)  # [N, 3]
            counts.append(counts[-1] + imgs.shape[0])
            all_imgs.append(imgs)
            all_poses.append(poses)
            all_metadata.append(metadata)
        else:
            counts.append(0)

    # Create a list where each element contains example indices for each split.
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    i_split[2] = i_split[1]  # Make test equal to val
    imgs = np.concatenate(all_imgs, 0)  # [N, H, W, 4]
    poses = np.concatenate(all_poses, 0)  # [N, 4, 4]
    metadata = np.concatenate(all_metadata, 0)  # [N, 3]

    # Extract the height and width from the shape of the first image example.
    H, W = imgs[0].shape[:2]
    # Compute the focal length.
    focal = metas["train"][0]['K'][0][0]

    # Render poses for generating spiral videos.
    if spiral_radius is None:
        translations = poses[:, :3, 3]  # [N, 3]
        center = [0, 0, 0]  # For now hard code center to be origin.
        dists = np.linalg.norm(translations - center, axis=1)  # [N,]
        spiral_radius = np.max(dists)
    if render_spiral_angles is None:
        # If no angles are provided, simply discretize 360 degrees inot n_render_spiral.
        render_spiral_angles = np.linspace(-180,180,n_render_spiral+1)[:-1]
    render_poses = torch.stack([pose_spherical(
            theta=angle, phi=-30.0, radius=spiral_radius) for angle in render_spiral_angles],0)  # [N, 4, 4]

    # Set the render lighting to be the viewdir of the first render pose, plus a few
    # degrees.
    light_angle = render_spiral_angles[0] + 15  # 20  # Shift the first angle by few degrees
    light_render_pose = pose_spherical(  # [4, 4]
            theta=light_angle, phi=-30.0, radius=1.0)
    light_pos = light_render_pose[:3, 3][None, :]  # [1, 3]
    render_metadata = light_pos

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    # Temporarily render with annotations.
    if render_anno:
        render_poses = poses
        render_metadata = metadata
    return imgs, poses, render_poses, [H, W, focal], i_split, metadata, render_metadata
