"""Utility functions for shadows."""
import torch

import ray_utils
import run_osf_helpers


def create_ray_batch(ray_batch, pts, metadata, use_viewdirs, lightdirs_method):
    num_primary_rays = pts.size()[0]  # R
    num_primary_samples = pts.size()[1]  # S

    rays_o = pts.view(-1, 3)  # [R?S, 3]
    rays_i = ray_batch[:, 11:12]  # [R, 1]
    rays_dst = ray_utils.get_lightdirs(  # [R?, S, 3]
        lightdirs_method=lightdirs_method, num_rays=num_primary_rays,
        num_samples=num_primary_samples, rays_i=rays_i, metadata=metadata,
        ray_batch=ray_batch, use_viewdirs=use_viewdirs, normalize=True)
    rays_dst = rays_dst.view(rays_o.size())  # [R?S, 3]

    rays_i = torch.tile(rays_i.unsqueeze(1), (1, num_primary_samples, 1))  # [R?, S, 1]
    rays_i = rays_i.view(-1, 1)  # [R?S, 1]

    shadow_ray_batch = ray_utils.create_ray_batch(rays_o, rays_dst, rays_i, use_viewdirs, rays_d_method="world_origin")
    return shadow_ray_batch


def compute_transmittance(alpha):
    trans = run_osf_helpers.compute_transmittance(alpha=alpha[..., 0])  # [R?S, S]

    last_trans = trans[:, -1]  # [R?S,]
    return last_trans
