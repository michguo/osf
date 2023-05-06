"""Utility functions for scattering rays."""
import torch


def create_scatter_indices_for_dim(dim, shape, indices=None):
    """Create scatter indices for a given dimension."""
    dim_size = shape[dim]
    N_dims = len(shape)
    reshape = [1] * N_dims
    reshape[dim] = -1

    if indices is None:
        indices = torch.arange(dim_size, dtype=torch.int)  # [dim_size,]

    indices = indices.view(reshape)

    indices = torch.broadcast_to(
        indices, shape)  # [Ro, S, 1] or [Ro, S, C, 1]  [0,1,1,1] vs. [512,64,1,1]

    indices = indices.int()
    return indices


def create_scatter_indices(updates, dim2known_indices):
    """Create scatter indices."""
    updates_expanded = updates.unsqueeze(-1)
    target_shape = updates_expanded.size()
    n_dims = len(updates.size())  # 2 or 3

    dim_indices_list = []
    for dim in range(n_dims):
        indices = None
        if dim in dim2known_indices:
            indices = dim2known_indices[dim]
        dim_indices = create_scatter_indices_for_dim(  # [Ro, S, C, 1]
            dim=dim,
            shape=target_shape,  # [Ro, S, 1] or [Ro, S, C, 1]
            indices=indices)  # [Ro,]
        dim_indices_list.append(dim_indices)
    scatter_indices = torch.cat((dim_indices_list), dim=-1)  # [Ro, S, C, 3]
    return scatter_indices
    

def scatter_nd(tensor, updates, dim2known_indices):
    scatter_indices = create_scatter_indices(  # [Ro, S, C, 3]
        updates=updates,  # [Ro, S]
        dim2known_indices=dim2known_indices)  # [Ro,]
    if len(scatter_indices.shape) == 3:
        scatter_indices = scatter_indices.long().view(-1, 2)
        tensor.data[scatter_indices[:, 0],
               scatter_indices[:, 1]] = updates.view(-1)
    elif len(scatter_indices.shape) == 4:
        scatter_indices = scatter_indices.long().view(-1, 3)
        tensor.data[scatter_indices[:, 0],
               scatter_indices[:, 1],
               scatter_indices[:, 2]] = updates.view(-1)
    return tensor


def scatter_results(intersect, indices, N_rays, keys, N_samples, N_importance=None):
    # We use 'None' to indicate that the intersecting set of rays is equivalent to
    # the full set if rays, so we are done.
    if indices is None:
        return {k: intersect[k] for k in keys}

    scattered_results = {}
    dim2known_indices = {0: indices}  # [R', 1]
    for k in keys:
        if k == 'z_vals':
            tensor = torch.arange(N_samples)  # [S,]
            tensor = tensor.float()
            tensor = torch.stack([tensor] * N_rays)  # [R, S]
        elif k == 'z_samples':
            tensor = torch.arange(N_importance)  # [I,]
            tensor = tensor.float()
            tensor = torch.stack([tensor] * N_rays)  # [R, I]
        elif k == 'raw':
            tensor = torch.full((N_rays, N_samples, 4), 1000.0, dtype=torch.float, requires_grad=True)  # [R, S, 4]
        elif k == 'pts':
            tensor = torch.full((N_rays, N_samples, 3), 1000.0, dtype=torch.float, requires_grad=True)  # [R, S, 3]
        elif 'rgb' in k:
            tensor = torch.zeros((N_rays, N_samples, 3), dtype=torch.float, requires_grad=True)  # [R, S, 3]
        elif 'alpha' in k:
            tensor = torch.zeros((N_rays, N_samples, 1), dtype=torch.float, requires_grad=True)  # [R, S, 1]
        else:
            raise ValueError(f'Invalid key: {k}')
        # No intersections to scatter.
        if len(indices) == 0:
            scattered_results[k] = tensor
        else:
            scattered_v = scatter_nd(  # [R, S, K]
                tensor=tensor,
                updates=intersect[k],  # [Ro, S]
                dim2known_indices=dim2known_indices)
            if k == 'z_samples':
                scattered_v = scattered_v.view(N_rays, N_importance)  # [R, I]
            else:
                if k == 'z_vals':
                    scattered_v = scattered_v.view(N_rays, N_samples)  # [R, S]
                else:
                    scattered_v = scattered_v.view(N_rays, N_samples, tensor.size()[2])  # [R, S, K]
            scattered_results[k] = scattered_v
    return scattered_results


def scatter_coarse_and_fine(
    ret0, ret, indices, num_rays, N_samples, N_importance, **kwargs):
    ret0 = scatter_results(
        intersect=ret0,
        indices=indices,
        N_rays=num_rays,
        keys=['z_vals', 'rgb', 'alpha', 'raw', 'pts'],
        N_samples=N_samples)
    ret = scatter_results(
        intersect=ret,
        indices=indices,
        N_rays=num_rays,
        keys=['z_vals', 'rgb', 'alpha', 'raw', 'z_samples', 'pts'],
        N_samples=N_samples + N_importance,
        N_importance=N_importance)
    return ret0, ret
