import os
import datetime
import secrets
import sys

import torch
import numpy as np
import imageio
import json
import random
import time
import uuid

import indirect_utils
from load_osf import load_osf_data
from intersect import compute_object_intersect_tensors
from ray_utils import transform_rays
from run_osf_helpers import *
from scatter import scatter_coarse_and_fine
import shadow_utils

from tqdm import tqdm, trange

from skimage.metrics import structural_similarity as calculate_ssim
from utils import LPIPS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(fn, chunk):
    """Construct a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], dim=0)
    return ret


def run_network(inputs, viewdirs, lightdirs, fn, embed_fn, embedviews_fn, embedlights_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.

    Args:
        inputs: [R, S, 3] float tensor. Sampled points per day.
        viewdirs: [R, S, 3] float tensor. Viewing directions.
        lightdirs: [R, S, 3] float tensor. Light directions.
        fn: Network function.
        embed_fn: Function to embed input points.
        embedviews_fn: Function to embed viewdirs.
        embedlights_fn: Function to embed viewdirs.
        netchunk: Batch size to feed into the network.
    """
    def prepare_and_embed(embeddirs_fn, dirs):
        """
        Args:
            embeddirs_fn: Function to embed directions.
            dirs: [R, S, 3] float tensor. Per-sample directions.

        Returns:
            edirs: [RS, L] float tensor. Embedded directions.
        """
        dirs_flat = torch.reshape(dirs, [-1, dirs.shape[-1]])  # [RS, 3]
        edirs = embeddirs_fn(dirs_flat)  # [RS, L]
        return edirs

    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        embedded_dirs = prepare_and_embed(embedviews_fn, viewdirs)
        embedded = torch.cat((embedded, embedded_dirs), dim=-1)
    if lightdirs is not None:
        embedded_dirs = prepare_and_embed(embedlights_fn, lightdirs)
        embedded = torch.cat((embedded, embedded_dirs), dim=-1)
    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def compute_shadows(ray_batch, pts, object_params, **kwargs):
    """Compute shadows cast by other objects onto the current object.

    Args:
        ray_batch: [R?, M] float tensor. Batch of primary rays.
        pts: [R?, S, 3] float tensor. Sampled points along rays.
        object_params: List of object parameters that must contain the following items:
                network_fn: Coarse network.
                network_fine: Fine network.
                intersect_bbox: bool. If true, intersect rays with object bbox. Only
                    rays that intersect are evaluated, and sampling bounds are
                    determined by the intersection bounds.
            Additional parameters that are only required if intersect_bbox = True:
                box_center: List of length 3 containing the (x, y, z) center of the bbox.
                box_dims: List of length 3 containing the x, y, z dimensions of the bbox.
            Optional params:
                translation: List of length 3 containing the (x, y, z) object-to-world translation.
        **kwargs: Dict. Additional arguments.

    Returns:
        trans: [R?, S, 1] float tensor. Transmittance along shadow rays.
    """
    if len(object_params) == 0:
        raise ValueError(f'object_params must contain at least one element.')

    # Get batch of shadow rays.
    if kwargs['shadow_lightdirs_method'] == 'random':
        lightdirs_method = 'random_upper'  # Only use upper hemisphere for shadows.
        #lightdirs_method = 'random'  # Only use upper hemisphere for shadows.
    else:
        lightdirs_method = kwargs['shadow_lightdirs_method']
    shadow_ray_batch = shadow_utils.create_ray_batch(  # [R?S, M]
            pts=pts,  # [R?, S, M]
            metadata=kwargs['metadata'],  # [N, 3]
            use_viewdirs=kwargs['use_viewdirs'],
            lightdirs_method=lightdirs_method,
            ray_batch=ray_batch)  # [R?, M]

    # Render rays in batches, but don't render indirect/shadows a second time.
    kwargs['render_shadows'] = False
    kwargs['render_indirect'] = False
    kwargs['chunk'] = kwargs['secondary_chunk']
    all_ret = batchify_rays(
            shadow_ray_batch, object_params=object_params, **kwargs)

    # Compute transmittance.
    trans = shadow_utils.compute_transmittance(  # [R?S,]
            alpha=all_ret['alpha'])  # [R?S, S*O, 1]
    trans = torch.reshape(trans, [pts.shape[0], -1, 1])  # [R?, S, 1]
    return trans  # [R?, S, 1]


def compute_indirect_illumination(rays_i, pts, object_params, **kwargs):
    """Compute shadows cast by other objects onto the current object.
    Args:
        pts: [R?, S, 3] float tensor. Sampled points along rays.
        object_params: List of object parameters that must contain the following items:
                network_fn: Coarse network.
                network_fine: Fine network.
                intersect_bbox: bool. If true, intersect rays with object bbox. Only
                    rays that intersect are evaluated, and sampling bounds are
                    determined by the intersection bounds.
            Additional parameters that are only required if intersect_bbox = True:
                box_center: List of length 3 containing the (x, y, z) center of the bbox.
                box_dims: List of length 3 containing the x, y, z dimensions of the bbox.
            Optional params:
                translation: List of length 3 containing the (x, y, z) object-to-world
                    translation.
        **kwargs: Dict. Additional arguments.
    Returns:
        indirect_radiance: [R?, S, 3] float tensor. Indirect radiance from secondary rays.
    """
    if len(object_params) == 0:
        raise ValueError(f'object_params must contain at least one element.')

    # Get batch of secondary rays.
    ray_batch = indirect_utils.create_ray_batch(  # [R?S, M]
            pts=pts,  # [R?, S, M]
            near=kwargs['near'],
            far=kwargs['far'],
            rays_i=rays_i,
            use_viewdirs=kwargs['use_viewdirs'])

    # Render rays in batches, but don't render indirect/shadows a second time.
    kwargs['render_shadows'] = False
    kwargs['render_indirect'] = False
    kwargs['chunk'] = kwargs['secondary_chunk']
    all_ret = batchify_rays(
            ray_batch, object_params=object_params, **kwargs)

    # Extract indirect radiance, reshape from [RS, 3] to [R, S, 3] to map to samples
    indirect_radiance = torch.reshape(all_ret['rgb_map'], [pts.shape[0], pts.shape[1], -1])  # [R, S, 3]
    # [R?, S, 3]
    return indirect_radiance, ray_batch


def evaluate_single_object(
        ray_batch, params, other_params=None, ret_ray_batch=False, **kwargs):
    """Evaluate rays for a single object.
    Args:
        ray_batch: [R, M] float tensor. All information necessary for sampling along a
            ray, including: ray origin, ray direction, min dist, max dist, and
            unit-magnitude viewing direction, and iamge ID, all in world coordinate
            frame.
        params: Dictionary of object parameters that must contain the following items:
                network_fn: Coarse network.
                network_fine: Fine network.
                intersect_bbox: bool. If true, intersect rays with object bbox. Only
                    rays that intersect are evaluated, and sampling bounds are
                    determined by the intersection bounds.
            Additional parameters that are only required if intersect_bbox = True:
                box_center: List of length 3 containing the (x, y, z) center of the bbox.
                box_dims: List of length 3 containing the x, y, z dimensions of the bbox.
            Optional params:
                translation: List of length 3 containing the (x, y, z) object-to-world
                    translation.
        other_params: List of params in the same format as `params` representing objects
            that we want to cast shadows from onto the current object. Only required if
            `render_shadows` is true.
        **kwargs: Dict. Additional arguments.
    Returns:
        ret0: Coarse network outputs.
        ret: Fine network outputs.
    """
    # Store the original number of rays before intersection.
    num_rays = ray_batch.shape[0]
    num_rays_to_eval = num_rays

    # Transform rays from world to object coordinate frame.
    ray_batch_obj = ray_utils.transform_rays(
        ray_batch, params, kwargs['use_viewdirs'], inverse=True)
    metadata_obj = ray_utils.transform_dirs(kwargs['metadata'], params, inverse=True)
    # ray_batch_obj = ray_batch
    # metadata_obj = kwargs['metadata']

    # Potentially only select a subset of intersecting rays to evaluate.
    ray_batch_obj_to_eval = ray_batch_obj

    if params['intersect_bbox']:
        # Compute ray bbox intersections. Update the batch of rays to only be the set of
        # rays that intersect with the object, and near/far bounds correspond to the
        # intersection bounds.
        # breakpoint()
        ray_batch_obj_to_eval, indices, intersect_mask = compute_object_intersect_tensors(
            ray_batch_obj, box_center=params['box_center'], box_dims=params['box_dims'])  # [R?, M], [R?,]
        num_rays_to_eval = len(indices)
    # print(f"params['exp_dir']: {params['exp_dir']}")
    # print(f"params['intersect_bbox']: {params['intersect_bbox']}")
    # print(f"ray_batch_obj_to_eval.shape: {ray_batch_obj_to_eval.shape}")
    # print(f"ray_batch_obj.shape: {ray_batch_obj.shape}")

    # Sample points along rays and evaluate the network on the points. We skip this if
    # there are no intersections  because the network/concat operations throw errors on
    # zero batch size.
    ret0, ret = None, None
    # print(f"num_rays_to_eval: {num_rays_to_eval}")
    if num_rays_to_eval > 0:
        metadata = kwargs.pop('metadata')
        if "apply_scaling" in params:
            kwargs["apply_scaling"] = params["apply_scaling"]
        if "density_scale" in params:
            kwargs["density_scale"] = params["density_scale"]
        ret0, ret = run_single_object(
                ray_batch_obj_to_eval, metadata=metadata_obj, network_fn=params['network_fn'],
                network_fine=params['network_fine'], lightdirs_method=params['lightdirs_method'], **kwargs)
                #network_fine=params['network_fine'], lightdirs_method='metadata', **kwargs)
        kwargs['metadata'] = metadata
        if ret_ray_batch:
            ret['ray_batch'] = ray_batch
        num_samples = ret['pts'].shape[1]  # S (contains both coarse and fine points)

        # Convert from object back into world coordinate space.

        # Michelle's temporary visualization code
        # if "save_results" in kwargs and kwargs["save_results"]:
        #     save_results(ret["pts"], "pts_body")
        # print(ret["pts"].reshape([-1, 3]).min(dim=0))
        # print(ret["pts"].reshape([-1, 3]).max(dim=0))
        # breakpoint()

        world_pts = ray_utils.transform_points_into_world_coordinate_frame(ret['pts'], params)
        # if "save_results" in kwargs and kwargs["save_results"]:
        #     save_results(ret["pts"], "pts")
        #     save_results(ret["rgb"], "rgb")

        # print(world_pts.reshape([-1, 3]).min(dim=0))
        # print(world_pts.reshape([-1, 3]).max(dim=0))
        # breakpoint()

        ray_batch_to_eval = ray_utils.transform_rays(
                ray_batch_obj_to_eval, params, kwargs['use_viewdirs'])
        # ray_batch_to_eval = ray_batch_obj_to_eval
        # Optionally compute shadows or indirect illumination from other objects onto
        # the current object.
        # if kwargs['render_shadows']:
            # Only keep objects that have bboxes.
        # other_params = [p for p in other_params  if p['intersect_bbox']]
        other_params = [p for p in other_params]
        # if params["intersect_bbox"]:
        rays_i = ray_batch_obj_to_eval[:, -1:]  # [R?, 1]
        if kwargs['render_shadows'] and 'render_shadows' in params and params['render_shadows']:
            shadow_trans = compute_shadows(  # [R?, S, 1]
                    ray_batch_to_eval,  # [R?, M]
                    world_pts,  # [R?, S, 3]
                    other_params,
                    **kwargs)
            ret['rgb'] *= shadow_trans  # [R?, S, 3]
        if kwargs['render_indirect'] and 'render_indirect' in params and params['render_indirect']:
            # [R?, S, 3], [R?S, M]
            indirect_radiance, indirect_ray_batch = compute_indirect_illumination(
                    rays_i, world_pts, other_params, **kwargs)
            indirect_radiance *= 5.3
            # if indirect_radiance.sum() > 0:
            #     breakpoint()
            # ret['rgb'] is the rho
            # we assume 1.0 direct lighting
            # indirect radiance is the radiance along the secondary rays                
            # Obtain rho(x, omega_view, omega_indirect).
            # Run OSF on the same primary sampled points, same view directions, but with the indirect secondary ray directions.
            # TODO: return the view directions from the run_single_object function.
            # [R?S, M]
            indirect_ray_batch_obj = ray_utils.transform_rays(indirect_ray_batch, params, kwargs['use_viewdirs'], inverse=True)
            indirect_dirs = indirect_ray_batch_obj[:, 8:11]
            indirect_dirs = torch.reshape(indirect_dirs, [ret['pts'].shape[0], ret['pts'].shape[1], 3])

            # override light dirs for indirect_rho
            # indirect_dirs[..., 0] = 0
            # indirect_dirs[..., 1] = 0
            # indirect_dirs[..., 2] = -1
            """
            what is indirect_dirs:
                - the viewdirs (incoming) of the indirect rays, which is basically the ray direction (normalized) in the object coordinate frame
            lightdirs input argument:
                - the normalized position of the light source, in the object coordinate frame
            """
            raw = kwargs['network_query_fn'](ret['pts'], ret['viewdirs'], indirect_dirs, params['network_fine'])  # [R, S, 4]
            indirect_rho = normalize_rgb(raw_rgb=raw[..., :3], scaled_sigmoid=kwargs["scaled_sigmoid"])
            # indirect = indirect_rho * indirect_radiance
            # ret['rgb'] += theindirect  # everything should be [R?, S, 3]

            indirect =  indirect_rho * indirect_radiance
            ret['rgb'] += indirect

            if "indirect_only" in params and params["indirect_only"]:
                ret['rgb'] = indirect
            elif "indirect_rho_only" in params and params["indirect_rho_only"]:
                ret['rgb'] = indirect_rho
            elif "indirect_radiance_only" in params and params["indirect_radiance_only"]:
                ret['rgb'] = indirect_radiance
            
            # pts_flattened = ret['pts'].reshape([-1, 3])
            # idxs = torch.where(
            #     torch.logical_and(
            #         torch.logical_and(-0.001 < pts_flattened[:, 0], pts_flattened[:, 0] < 0.001),
            #         torch.logical_and(-0.001 < pts_flattened[:, 1], pts_flattened[:, 1] < 0.001),
            #     )
            # )
            # print(indirect_radiance.reshape([-1, 3]).mean(dim=0))
            # breakpoint()

        if 'light_rgb' in params:
            light_rgb = torch.tensor(params['light_rgb'], dtype=tf.float32)[None, None, :]
            ret0['rgb'] *= light_rgb
            ret['rgb'] *= light_rgb
        # ret['alpha'] = tf.ones_like(ret['alpha']) * 0.01
        if kwargs['light_falloff']:
            rays_i = ray_batch_obj_to_eval[:, -1:]
            metadata_obj_i = metadata_obj[rays_i.long()]
            ret_r = metadata_obj_i - ret['pts']
            ret0_r = metadata_obj_i - ret0['pts']
            breakpoint()
            ret['rgb'] = ret['rgb'] / (ret_r ** 2).sum(dim=-1).unsqueeze(-1)
            ret0['rgb'] = ret0['rgb'] / (ret0_r ** 2).sum(dim=-1).unsqueeze(-1)
    # Scatter the results from intersecting rays back to the original set of rays.
    if params['intersect_bbox']:
        ret0, ret = scatter_coarse_and_fine(ret0, ret, indices, num_rays, **kwargs)
        if ret_ray_batch:
            ret['ray_batch'] = torch.cat((ray_batch, intersect_mask[:, None].float()), dim=1)
    # Only save shadow rays for objects that don't require bbox intersections (o/w we
    # need to scatter shadow rays first)
    # if not params['intersect_bbox'] and shadow_ray_batch is not None:
    #     if ret_ray_batch:
    #         ret['shadow_ray_batch'] = tf.reshape(
    #                 shadow_ray_batch, [num_rays, num_samples, -1])  # [R, S, M*O]
    return ret0, ret


def save_results(tensor: torch.Tensor, name: str):
    x = tensor.cpu().numpy()
    # exp_name = "comp_scene_soap"
    exp_name = "comp_scene_soap_bbox"
    dir_ = os.path.join("/viscam/u/mguo95/workspace/third_party/yuyuchang/osf-pytorch/logs", exp_name, name)
    os.makedirs(dir_, exist_ok=True)
    h = uuid.uuid4().hex
    fname = f"{h}.npy"
    path = os.path.join(dir_, fname)
    assert not os.path.exists(path)
    np.save(path, x)
    num_files = len(os.listdir(dir_))
    # print(f"num_files for {name}: {num_files}")


def evaluate_multiple_objects(ray_batch, object_params, **kwargs):
    """Evaluates multiple objects and sorts them into a single set of results.
    Args:
        ray_batch: [R, M] float  tensor. All information necessary for sampling along a
            ray, including: ray origin, ray direction, min dist, max dist, and
            unit-magnitude viewing direction, and iamge ID, all in world coordinate
            frame.
        object_params: List of object parameters that must contain the following items:
                network_fn: Coarse network.
                network_fine: Fine network.
                intersect_bbox: bool. If true, intersect rays with object bbox. Only
                    rays that intersect are evaluated, and sampling bounds are
                    determined by the intersection bounds.
            Additional parameters that are only required if intersect_bbox = True:
                box_center: List of length 3 containing the (x, y, z) center of the bbox.
                box_dims: List of length 3 containing the x, y, z dimensions of the bbox.
            Optional params:
                translation: List of length 3 containing the (x, y, z) object-to-world
                    translation.
        **kwargs: Dict. Additional arguments.
    Returns:
        ret0: Coarse network results.
        ret: Fine network results.
    """
    if len(object_params) == 0:
        return {}, {}

    # Evaluate set of objects.
    id2ret0, id2ret = {}, {}
    for i, params in enumerate(object_params):
        other_params = [p for j, p in enumerate(object_params) if j != i]
        id2ret0[i], id2ret[i] = evaluate_single_object(
                ray_batch, params, other_params=other_params, **kwargs)

    # Combine results across objects by sorting. Num samples is multiplied by the
    # number of objects.
    ret0 = combine_multi_object_results(id2ret0, object_params)
    ret = combine_multi_object_results(id2ret, object_params)
    return ret0, ret


def render_rays(ray_batch, **kwargs):
    """Volumetric rendering.
    Args:
        ray_batch: [R, M] float tensor. All information necessary for sampling along a
            ray, including: ray origin, ray direction, min dist, max dist, and
            unit-magnitude viewing direction, and iamge ID, all in world coordinate
            frame.
        white_bkgd: bool. If True, assume a white background.
        **kwargs: Dict. Additional arguments.
    Returns:
        ret: [Dict]. Contains the following key-value pairs:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray. From fine model.
            disp_map: [num_rays]. Disparity map. 1 / depth.
            acc_map: [num_rays]. Accumulated opacity along each ray. From fine model.
            raw: [num_rays, num_samples, 4]. Raw predictions from model.
            rgb0: See rgb_map. Output for coarse model.
            disp0: See disp_map. Output for coarse model.
            acc0: See acc_map. Output for coarse model.
            z_std: [num_rays]. Standard deviation of sample distances along each ray.
    """
    ret0, ret = evaluate_multiple_objects(ray_batch, **kwargs)

    # Compose points along rays into the final rendered result in pixel space.
    outputs0 = compose_outputs(z_vals=ret0['z_vals'], rgb=ret0['rgb'],
            alpha=ret0['alpha'], white_bkgd=kwargs['white_bkgd'])
    outputs = compose_outputs(z_vals=ret['z_vals'], rgb=ret['rgb'], alpha=ret['alpha'],
            white_bkgd=kwargs['white_bkgd'])

    # Add composed outputs into the dictionaries we are returning.
    ret0.update(outputs0)
    ret.update(outputs)

    # breakpoint()

    # Merge coarse results into the main dictionary we are returning.
    if kwargs['N_importance'] > 0:
        ret['rgb0'] = ret0['rgb_map']
        ret['disp0'] = ret0['disp_map']
        ret['acc0'] = ret0['acc_map']
        # ret['z_std'] = tf.math.reduce_std(ret['z_samples'], -1)  # [N_rays]

    #if kwargs['check_numerics']:
    #    for k in ret:
    #        tf.debugging.check_numerics(ret[k], 'output {}'.format(k))
    return ret


def batchify_rays(rays_flat, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    chunk = kwargs['chunk']
    for i in range(0, rays_flat.shape[0], chunk):
        # print(f"[batchify_rays] i: {i:05}")
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, rays=None, c2w=None, ndc=True,
           near=0., far=1., c2w_staticcam=None, img_id=0,
           **kwargs):
    """Render rays
    Args:
        H: int. Height of image in pixels.
        W: int. Width of image in pixels.
        focal: float. Focal length of pinhole camera.
        chunk: int. Maximum number of rays to process simultaneously. Used to
            control maximum memory usage. Does not affect final results.
        rays: array of shape [3, batch_size, 3]. Ray origin, direction, and image ID for
            each example in batch.
        c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
        ndc: bool. If True, represent ray origin, direction in NDC coordinates.
        near: float or array of shape [batch_size]. Nearest distance for a ray.
        far: float or array of shape [batch_size]. Farthest distance for a ray.
        use_viewdirs: bool. If True, use viewing direction of a point in space in model.
        translation: List of length 3 containing the (x, y, z) object-to-world
            translation to apply to the object.
        c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
        camera while using other c2w argument for viewing directions.
        Returns:
        rgb_map: [batch_size, 3]. Predicted RGB values for rays.
        disp_map: [batch_size]. Disparity map. Inverse of depth.
        acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
        extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d, rays_i = get_rays(H, W, focal, c2w, img_id=img_id)
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    else:
        # use provided ray batch
        rays_o, rays_d, rays_i = rays

    if kwargs['use_viewdirs']:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d, rays_i = get_rays(H, W, focal, c2w_staticcam, img_id=img_id)

        # Make all directions unit magnitude.
        # shape: [batch_size, 3]
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(
            H, W, focal, torch.tensor([1.], dtype=torch.float), rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()
    rays_i = torch.reshape(rays_i[..., :1], [-1, 1]).float()  # [N, 1]
    rays_near, rays_far = near * \
        torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = torch.cat((rays_o, rays_d, rays_near, rays_far), dim=-1)
    if kwargs['use_viewdirs']:
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction)
        rays = torch.cat((rays, viewdirs), dim=-1)
    rays = torch.cat((rays, rays_i), dim=-1)

    # Render and reshape
    metadata_original = kwargs.pop('metadata')
    if c2w is not None:  # Special metadata for cases when we're rendering full image.
        metadata = kwargs['render_metadata']
    else:
        metadata = metadata_original
    ### try
    # metadata = metadata_original
    if c2w is not None and kwargs['N_importance'] > 128:
        print('Using too many samples for rendering a full image at one pass. Batchifying rays to avoid OOM.')
        n_batch = kwargs['N_importance'] // 128
        all_ret_list = []
        for i in range(n_batch):
            batch_size = rays.shape[0]//n_batch
            ray_batch = rays[i*batch_size:(i+1)*batch_size]
            ret_this = batchify_rays(ray_batch, metadata=metadata, near=near, far=far, **kwargs)
            all_ret_list.append(ret_this)
        all_ret = {}
        for k in all_ret_list[0].keys():
            all_ret[k] = torch.cat([all_ret_list[i][k] for i in range(n_batch)], 0)
    else:
        all_ret = batchify_rays(rays, metadata=metadata, near=near, far=far, **kwargs)
    kwargs['metadata'] = metadata_original

    for k in all_ret:
        #tf.compat.v1.debugging.assert_equal(
        #    all_ret[k].shape[0], tf.math.reduce_prod(sh[:-1]), message=f'k: {k}, {all_ret[k].shape[0]}, {tf.math.reduce_prod(sh[:-1])}')
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)
    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, save_extras=False,
        render_staticview=False, render_staticcam=False, render_moving=False, c2w_staticcam=None, render_start=None,
        render_end=None, save_results=False, skip_metrics=False):

    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    mse_list, psnr_list, ssim_list, lpips_list = [], [], [], []
    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        # if i != 7:
        #     continue
        if render_start is not None and render_end is not None and not(render_start <= i < render_end):
            continue
        print(i, time.time() - t)
        t = time.time()

        c2w = c2w[:3, :4]
        if render_staticview:
            render_kwargs['c2w_staticcam'] = c2w  # Set the projected view to be the render pose of the current frame.
            c2w = c2w_staticcam  # Use first camera as the static viewdir.
        elif render_staticcam:
            # Use render pose as the viewdir, while the camera stays put.
            render_kwargs['c2w_staticcam'] = c2w_staticcam
            # Lightdir is set to be the render pose position, set via metadata.
            render_kwargs['render_metadata'] = c2w[:3, 3][None, :]
        elif render_moving:
            if render_start is not None and render_end is not None:
                raise NotImplementedError
            # c2w = c2w_staticcam
            for obj_idx, params in enumerate(render_kwargs['object_params']):
                moved_params = params.copy()
                if 'translation_delta' in params:
                    moved_params['translation'] = list(np.array(params['translation']) + np.array(params['translation_delta']))
                if 'rotation_delta' in params:
                    moved_params['rotation'] = list(np.array(params['rotation']) + np.array(params['rotation_delta']))
                render_kwargs['object_params'][obj_idx] = moved_params
        render_kwargs["save_results"] = False
        # TODO: fix img_id
        rgb, disp, acc, extras = render(H, W, focal, chunk=chunk, c2w=c2w, img_id=-render_poses.shape[0] + i, **render_kwargs)
        # rgb, disp, acc, extras = render(H, W, focal, chunk=chunk, c2w=c2w, **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        if type(gt_imgs[i]) is np.ndarray:
            # gt_img_pytorch = torch.zeros_like(rgb)
            # gt_img_numpy = gt_img_pytorch.cpu().numpy()
            gt_img_pytorch = torch.tensor(gt_imgs[i], device=device)
            gt_img_numpy = gt_imgs[i]
        else:
            gt_img_pytorch = gt_imgs[i].to(device)
            gt_img_numpy = gt_imgs[i].cpu().numpy()

        if i == 0:
            print(rgb.shape, disp.shape)

        if not skip_metrics:
            mse = img2mse(rgb, gt_img_pytorch)
            psnr = mse2psnr(mse)
            ssim = calculate_ssim(rgb.cpu().numpy(), gt_img_numpy, data_range=gt_img_numpy.max() - gt_img_numpy.min(),
                                multichannel=True)
            lpips = LPIPS.calculate(rgb, gt_img_pytorch)
            mse_list.append(mse.item())
            ssim_list.append(ssim)
            psnr_list.append(psnr.item())
            lpips_list.append(lpips.item())

            if gt_imgs is not None and render_factor == 0:
                # p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
                p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_img_numpy)))
                print(p)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

            if save_extras:
                save_extras_as_npy(savedir, extras, i, 'test')
        # breakpoint()
    
    if not skip_metrics:
        average_mse = sum(mse_list) / len(mse_list)
        average_psnr = sum(psnr_list) / len(psnr_list)
        average_ssim = sum(ssim_list) / len(ssim_list)
        average_lpips = sum(lpips_list) / len(lpips_list)

        numerical_results = {'mse': average_mse, 'psnr': average_psnr, 'ssim': average_ssim, 'lpips': average_lpips}
        numerical_results_filename = os.path.join(savedir, 'numerical_results.json')
        with open(numerical_results_filename, 'w') as numerical_results_file:
            json.dump(numerical_results, numerical_results_file)
   
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args, metadata, render_metadata):
    """Instantiate NeRF's MLP model.

    Args:
        args: Arguments from the arg parser.
        metadata: [N, 3] float tensor. Metadata about each image. Currently only light
            position is provided.
        metadata: [N, 3] float tensor. Metadata about render poses. Currently only light
            position is provided.

    Returns:
        render_kwargs_train: Training render arguments.
        render_kwargs_test: Testing render arguments.
        start: int. Iteration to start training at.
        grad_vars: Variables to apply gradients to.
        models: Dict of models.
    """
    # If object params are not provided, we initialize the params for a single object,
    # and by default we use the parameters that NeRF uses (no bbox intersection),
    # and no input light directions.
    if args.object_params is None:
        args.object_params = [{'intersect_bbox': False, 'lightdirs_method': None}]

    # Currently we only support multi-object rendering during inference.
    n_objects = len(args.object_params)
    if n_objects > 1:
        assert args.render_only

    # The iteration to start at, which can change if an earlier model is loaded later.
    start = 0
    for i, params in enumerate(args.object_params):
        embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

        input_ch_views, input_ch_lights = 0, 0
        embedviews_fn = None
        embedlights_fn = None
        if args.use_viewdirs:
            embedviews_fn, input_ch_views = get_embedder(
                args.multires_views, args.i_embed)
        if args.use_lightdirs:
            embedlights_fn, input_ch_lights = get_embedder(
                args.multires_lights, args.i_embed)
        output_ch = 4
        skips = [4]

        model = NeRF(
            D=args.netdepth, W=args.netwidth,
            input_ch=input_ch, output_ch=output_ch, skips=skips,
            input_ch_views=input_ch_views, input_ch_lights=input_ch_lights,
            use_viewdirs=args.use_viewdirs, use_lightdirs=args.use_lightdirs)
        grad_vars = list(model.parameters())

        model_fine = None
        if args.N_importance > 0:
            model_fine = NeRF(
                D=args.netdepth_fine, W=args.netwidth_fine,
                input_ch=input_ch, output_ch=output_ch, skips=skips,
                input_ch_views=input_ch_views, input_ch_lights=input_ch_lights,
                use_viewdirs=args.use_viewdirs, use_lightdirs=args.use_lightdirs)
            grad_vars += list(model_fine.parameters())

        def network_query_fn(inputs, viewdirs, lightdirs, network_fn): return run_network(
            inputs, viewdirs, lightdirs, network_fn,
            embed_fn=embed_fn,
            embedviews_fn=embedviews_fn,
            embedlights_fn=embedlights_fn,
            netchunk=args.netchunk)

        # Create optimizer
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

        start = 0
        basedir = args.basedir
        expname = args.expname

        # Load model weights, if available.
        def get_ckpts(exp_dir):
            # Find all the coarse model checkpoints.
            return [os.path.join(exp_dir, f) for f in sorted(os.listdir(exp_dir)) if 'tar' in f]
                    #('model_' in f and 'fine' not in f and 'optimizer' not in f)]
        if 'exp_dir' in args.object_params[i]:
            ckpts = get_ckpts(params['exp_dir'])
        elif args.ft_path is not None and args.ft_path != 'None':
            ckpts = [args.ft_path]
        else:
            ckpts = get_ckpts(os.path.join(args.basedir, args.expname))
        print('Found ckpts', ckpts)
        if len(ckpts) > 0 and not args.no_reload:
            ckpt_path = ckpts[-1]
            print('Reloading from', ckpt_path)
            ckpt = torch.load(ckpt_path)
            start = ckpt['global_step']
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            # Load model
            model.load_state_dict(ckpt['network_fn_state_dict'])

            if model_fine is not None:
                model_fine.load_state_dict(ckpt['network_fine_state_dict'])

        args.object_params[i]['network_fn'] = model
        args.object_params[i]['network_fine'] = model_fine

    # This models dictionary will contain all the model components (coarse, fine,
    # optimizer) that will be saved during training. Multi-object saving is currently
    # not supported.
    models = {}
    if n_objects == 1:
        models['model'] = model
        if model_fine is not None:
            models['model_fine'] = model_fine

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'N_samples': args.N_samples,
        'use_viewdirs': args.use_viewdirs,
        'use_lightdirs': args.use_lightdirs,
        'shadow_lightdirs_method': args.shadow_lightdirs_method,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
        'scaled_sigmoid': args.scaled_sigmoid,
        'object_params': args.object_params,
        'metadata': metadata,
        # 'render_metadata': render_metadata,
        'render_metadata': metadata, # use in training and render_only
        'render_shadows': args.render_shadows,
        'render_indirect': args.render_indirect,
        'secondary_chunk': args.secondary_chunk,
        'check_numerics': args.check_numerics,
        'light_falloff': args.light_falloff,
        'directional_light': args.directional_light,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    return render_kwargs_train, render_kwargs_test, start, grad_vars, models, optimizer


def save_extras_as_npy(savedir, extras, i, tag):
    for k in extras.keys():
        extras_dir = os.path.join(savedir, 'extras', k, tag)
        os.makedirs(extras_dir, exist_ok=True)
        extras_path = os.path.join(extras_dir, f'{i:06d}.npy')
        np.save(open(extras_path, 'wb'), extras[k].detach().cpu().numpy())

def config_parser():
    import configargparse
    import yaml
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    #parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str, default='', help='experiment name')
    parser.add_argument("--expname_ts", action="store_true", 
                        help='append timestamp to expname')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str,
                        default='./data/llff/fern', help='input data directory')

    # training options
    parser.add_argument("--is_nerf", action="store_true",
                        help='is nerf?')
    parser.add_argument("--num_iters", type=int, default=200000,
                        help='number of total iterations')
    parser.add_argument("--outmask_keep_ratio", type=float,
                        default=1., help='for out-of-mask black pixels, keep this ratio of pixels')
    parser.add_argument("--outmask_keep_ratio_end", type=float,
                        default=None, help='Increase outmask_keep_ratio to this value after a quarter of training course.')
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int,
                        default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float,
                        default=5e-4, help='learning rate')
    parser.add_argument("--max_lr", type=float,
                        default=1e-3, help='max learning rate for cycle scheduler')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000s)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--secondary_chunk", type=int,
                        help='number of rays processed in parallel for secondary rays, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time (instead of randomly sampling from all train images)')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--random_seed", type=int,
                        help='fix random seed for repeatability')
    parser.add_argument("--light_falloff", action='store_true',
                        help='light falloff with 1/r^2')
    parser.add_argument("--directional_light", action='store_true',
                        help='use directional light')
    
    # pre-crop options
    parser.add_argument("--precrop_iters", type=int, default=10000,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')    

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='include viewdirs as model input')
    parser.add_argument("--use_lightdirs", action='store_true',
                        help='include lightdirs as model input')
    parser.add_argument("--shadow_lightdirs_method", type=str, 
                        help='method for computing lightdirs for shadows.')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D view direction)')
    parser.add_argument("--multires_lights", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D light direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--scaled_sigmoid", action='store_true',
                        help='use scaled sigmoid to normalize rgb predictions')
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_staticview", action='store_true',
                        help='render with static viewpoint')
    parser.add_argument("--render_staticcam", action='store_true',
                        help='render with static camera')
    parser.add_argument("--render_moving", action='store_true',
                        help='render with moving objects')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_anno", action='store_true',
                        help='render using all annotations')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--n_render_spiral", type=int, default=40,
                        help='number of spiral views.')
    parser.add_argument("--render_spiral_angles", type=float, action='append',
                        help='angles to render for spiral.')
    parser.add_argument("--spiral_radius", type=float,
                        help='radius of the spiral views.')
    parser.add_argument('--render_reverse', action='store_true',
                        help='whether to render in reverse order of frames.')
    parser.add_argument('--render_mirror', action='store_true',
                        help='whether to render the original video, followed by its reverse.')
    parser.add_argument('--test_indices', type=str,
                        help='comma separated list of test example indices (indexed into test_anno.json).')
    parser.add_argument("--use_train_anno_as_test", action='store_true',
                        help='whether to use the train annotations file (instead of the test annotations file).')
    parser.add_argument('--render_start', type=int,
                        help='frame index to start rendering from (inclusive).')
    parser.add_argument('--render_end', type=int,
                        help='frame index to end rendering at (exclusive).')
    parser.add_argument('--fps', type=int, default=15,
                        help='frames per second.')
    parser.add_argument("--check_numerics", action='store_true',
                        help='whether to check numerics on the rendered results.')

    # osf options
    parser.add_argument('--object_params', type=yaml.safe_load, action='append', 
                        help='Object parameters. See render_rays() for description.')
    parser.add_argument('--render_shadows', action='store_true',
                        help='whether to render shadows.')
    parser.add_argument('--render_indirect', action='store_true',
                        help='whether to render indirect illumination.')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels / osf / nrf')
    parser.add_argument("--testskip", type=int, default=1,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--same_splits", action='store_true',
                        help='Use the same data for all splits')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # nrf flags
    parser.add_argument("--n_examples", type=int,
                        help='If set, uses the first `n_examples` instead of the full dataset.')
    parser.add_argument("--no_shuffle", action='store_true',
                        help='Turn off shuffling when computing splits.')
    parser.add_argument("--no_split", action='store_true',
                        help='Turn off split so that train/val/test are the same.')
    parser.add_argument("--train_as_val", action='store_true',
                        help='Use train set as val set.')
    parser.add_argument("--start", type=int,
                        help='Start example index.')
    parser.add_argument("--end", type=int,
                        help='End example index.')
    parser.add_argument("--near", type=float, default=0.01,
                        help='Near plane distance.')
    parser.add_argument("--far", type=float, default=4,
                        help='Far plane distnace.')

    # llff flags
    parser.add_argument("--factor", type=int,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')
    parser.add_argument("--i_extras",   type=int, default=10000,
                        help='frequency of saving extras as numpy arrays')
    parser.add_argument("--save_extras", action='store_true',
                        help='whether to save extras')
    parser.add_argument("--save_results", action='store_true',
                        help='whether to save results')
    parser.add_argument("--skip_metrics", action='store_true',
                        help='whether to skip computing metrics')

    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()

    # Force white_bkgd to be off if we are rendering indirect.
    # if args.render_indirect:
    #     args.white_bkgd = False
    if args.secondary_chunk is None:
        args.secondary_chunk = args.chunk

    if args.expname_ts:
        now = datetime.datetime.now()
        ts = now.strftime("%Y%m%d_%H%M%S")
        args.expname = f'{ts}_{args.expname}_{secrets.token_hex(4)}'

    print(f'expname: {args.expname}')

    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        #tf.compat.v1.set_random_seed(args.random_seed)
    
    test_indices = None
    if args.test_indices is not None:
        test_indices = np.array(list(map(int, (args.test_indices).split(","))))

    metadata, render_metadata = None, None
    # Load data
    assert args.dataset_type == 'osf'
    assert args.near is not None and args.far is not None
    near = args.near
    far = args.far

    images_rgba, poses, render_poses, hwf, i_split, metadata, render_metadata = load_osf_data(
        args.datadir, args.spiral_radius, args.half_res, args.testskip,
        args.n_render_spiral, args.render_spiral_angles, same_splits=args.same_splits, render_anno=args.render_anno,
        test_indices=test_indices, use_train_anno_as_test=args.use_train_anno_as_test, is_nerf=args.is_nerf)
    print('Loaded osf', images_rgba.shape, render_poses.shape, hwf, args.datadir)
    i_train, i_val, i_test = i_split

    assert images_rgba.shape[-1] == 4
    images = images_rgba[..., :3]*images_rgba[..., -1:]
    masks = images_rgba[..., -1:]

    c2w_staticcam = render_poses[0][:3, :4]
    if args.render_reverse:
        render_poses = render_poses[::-1]
    
    # Create dummy metadata if not loaded from dataset.
    if metadata is None:
        metadata = torch.tensor([[0, 0, 1]] * len(images), dtype=torch.float)  # [N, 3]
    if render_metadata is None:
        render_metadata = metadata

    # Use for composition test
    # metadata = torch.tensor([[0, 0, 1]] * len(images), dtype=torch.float)  # [N, 3]
    # metadata = torch.tensor([[-0.1, 0.1, 0.99]] * len(images), dtype=torch.float)  # [N, 3]
    # metadata = torch.tensor([[0.1, -0.1, 0.99]] * len(images), dtype=torch.float)  # [N, 3]
    # metadata = torch.tensor([[1, -1, 10]] * len(images), dtype=torch.float)  # [N, 3]
    # metadata = torch.tensor([[-1, -1, 10]] * len(images), dtype=torch.float)  # [N, 3]
    # render_metadata = metadata
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])
    # TODO
    # Render compose
    # render_poses = np.array(poses[i_train])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, models, optimizer = create_nerf(
        args, metadata, render_metadata)
    global_step = start

    bds_dict = {
        'near': torch.tensor(near).float(),
        'far': torch.tensor(far).float(),
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                # images = None
                images = images

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)
            print('Begin rendering', testsavedir)
            rgbs, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test,
                                  gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor,
                                  save_extras=args.save_extras, render_staticview=args.render_staticview,
                                  render_staticcam=args.render_staticcam, render_moving=args.render_moving,
                                  c2w_staticcam=c2w_staticcam, render_start=args.render_start,
                                  render_end=args.render_end, save_results=args.save_results, skip_metrics=args.skip_metrics)
            print('Done rendering', testsavedir)
            fps = args.fps
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'),
                             to8b(rgbs), fps=fps, quality=8)
            if args.render_mirror:
                imageio.mimwrite(os.path.join(testsavedir, 'video_mirrored.mp4'),
                                np.concatenate([to8b(rgbs), to8b(rgbs)[::-1]], axis=0), fps=fps, quality=8)
            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand

    # For random ray batching.
    #
    # Constructs an array 'rays_rgb' of shape [N*H*W, 3, 3] where axis=1 is
    # interpreted as,
    #   axis=0: ray origin in world space
    #   axis=1: ray direction in world space
    #   axis=2: observed RGB color of pixel
    print('get rays')
    # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3], and
    # rays_id=[H, W, 3] for each pixel in the image. This stack() adds a new
    # dimension.
    rays = [get_rays_np(H, W, focal, p, img_id=i) for i, p in
            enumerate(poses[:, :3, :4])]  # List of [ro+rd+ri, H, W, 3]
    rays = np.stack(rays, axis=0)  # [N, ro+rd+ri, H, W, 3]
    print('done, concats')
    # [N, ro+rd+ri+rgb, H, W, 3]
    rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)
    # [N, H, W, ro+rd+ri+rgb, 3]
    rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
    rays_mask = np.transpose(masks[:, None, ...], [0, 2, 3, 1, 4]).squeeze(-1).squeeze(-1)  # [N, H, W]

    rays_rgb = np.stack([rays_rgb[i]
                         for i in i_train], axis=0)  # train images only
    rays_mask = np.stack([rays_mask[i]
                         for i in i_train], axis=0)  # train images only

    rays_rgb_all = np.stack([rays_rgb[i]
                         for i in i_train], axis=0)  # train images only
    rays_mask = np.stack([rays_mask[i]
                         for i in i_train], axis=0)  # train images only

    rays_rgb = select_rays_by_mask(rays_rgb_all, rays_mask, outmask_keep_ratio=args.outmask_keep_ratio)

    # N_iters = 1000000 + 1
    N_iters = args.num_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    start += 1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, total_steps=N_iters)

    # generate an index array for sampling random rays
    n_rays = rays_rgb.shape[0]
    idx_array = np.random.permutation(n_rays)
    n_batches = int(np.ceil(n_rays / N_rand))

    for i in range(start, N_iters):
        time0 = time.time()

        if args.outmask_keep_ratio_end is not None and i == N_iters // 4 + 1:
            print('25% training done; switch outmask_keep_ratio from {} to {}'.format(args.outmask_keep_ratio, args.outmask_keep_ratio_end))
            rays_rgb = select_rays_by_mask(rays_rgb_all, rays_mask, outmask_keep_ratio=args.outmask_keep_ratio_end)
            n_rays = rays_rgb.shape[0]
            idx_array = np.random.permutation(n_rays)
            n_batches = int(np.ceil(n_rays / N_rand))
        if i % n_batches == 0:
            idx_array = np.random.permutation(n_rays)
        idx_this_batch = idx_array[(i % n_batches) * N_rand: (i % n_batches + 1) * N_rand]
        batch = rays_rgb[idx_this_batch]  # [B, ro+rd+ri+rgb, 3]
        batch = torch.tensor(batch)  # [B, ro+rd+ri+rgb, 3]
        batch = torch.transpose(batch, 0, 1).to(device)  # [ro+rd+ri+rgb, B, 3]

        # batch_rays[i, n, xyz] = ray origin/direction/id                                                                                                                                                                                                                                                                                                                                                                                                                                                                    , example_id, 3D position
        # target_s[n, rgb] = example_id, observed color.
        batch_rays, target_s = batch[:-1], batch[-1]  # [ro+rd+ri, N, 3], [N, 3]

        #####  Core optimization loop  #####
        # Make predictions for color, disparity, accumulated opacity.
        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                        verbose=i < 10, retraw=True,
                                        **render_kwargs_train)

        optimizer.zero_grad()
        # Compute MSE loss between predicted and true RGB.
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][..., -1]
        loss = img_loss
        psnr = mse2psnr(img_loss)
        # if args.reg_trans_beta is not None:
        #     loss += args.reg_trans_beta * (tf.log(trans) + tf.log(1-trans))

        # Add MSE loss for coarse-grained model
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss += img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        scheduler.step()

        dt = time.time() - time0

        #####           end            #####

        # Rest is logging

        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                # 'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fn_state_dict': render_kwargs_train['object_params'][0]['network_fn'].state_dict(),
                # 'network_fine_state_dict': render_kwrags_train['network_fine'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['object_params'][0]['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_extras == 0 or i in [0, 2500, 5000] and args.save_extras:
            save_extras_as_npy(os.path.join(basedir, expname), extras, i, 'train')

        if (i % args.i_testset == 0 or i == 1000):
            testsavedir = os.path.join(
                basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, args.chunk, render_kwargs_test,
                            gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        if i % args.i_print == 0:
            # tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()} PSNR: {psnr.item()}")
            print(f"[TRAIN] Iter: {i} Loss: {loss.item()} PSNR: {psnr.item()}")
            print('Current LR: {}'.format(scheduler.get_last_lr()))
            sys.stdout.flush()

        global_step += 1

if __name__ =='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()

