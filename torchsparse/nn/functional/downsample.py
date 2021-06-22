from typing import Tuple, Union

import torch

from torchsparse.nn.functional.utils import get_kernel_offsets
from torchsparse.utils import make_ntuple

__all__ = ['spdownsample']


def spdownsample(
        coords: torch.Tensor,
        stride: Union[int, Tuple[int, ...]] = 2,
        kernel_size: Union[int, Tuple[int, ...]] = 2,
        tensor_stride: Union[int, Tuple[int, ...]] = 1) -> torch.Tensor:
    stride = make_ntuple(stride, ndim=3)
    kernel_size = make_ntuple(kernel_size, ndim=3)
    tensor_stride = make_ntuple(tensor_stride, ndim=3)

    sample_stride = [stride[k] * tensor_stride[k] for k in range(3)]

    if all(stride[k] in [1, kernel_size[k]] for k in range(3)):
        sample_stride = torch.tensor(sample_stride,
                                     dtype=torch.int,
                                     device=coords.device).unsqueeze(0)
        coords = coords.clone()
        coords[:, :3] = coords[:, :3] // sample_stride * sample_stride
    else:
        offsets = get_kernel_offsets(kernel_size,
                                     tensor_stride,
                                     device=coords.device)

        xyz = coords[:, :3].unsqueeze(1).repeat(1, offsets.size(0), 1) + offsets
        b = coords[:, 3:].repeat(1, offsets.size(0))
        coords = torch.cat([xyz.view(-1, 3), b.view(-1, 1)], dim=1)

        mask = ((coords[:, 0] % sample_stride[0] == 0)
                & (coords[:, 1] % sample_stride[1] == 0)
                & (coords[:, 2] % sample_stride[2] == 0))
        coords = coords[mask]

    coords = torch.unique(coords, dim=0)
    return coords
