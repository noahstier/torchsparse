from typing import Optional

import torch
from torch.autograd import Function

import torchsparse.backend

__all__ = ['sphash']


class HashGPU(Function):

    @staticmethod
    def forward(ctx, coords):
        if coords.device.type == 'cuda':
            return torchsparse.backend.hash_forward(coords.contiguous())
        elif coords.device.type == 'cpu':
            return torchsparse.backend.cpu_hash_forward(
                coords.int().contiguous())
        else:
            device = coords.device
            return torchsparse.backend.cpu_hash_forward(
                coords.int().contiguous().cpu()).to(device)


class KernelHashGPU(Function):

    @staticmethod
    def forward(ctx, coords: torch.Tensor, offsets: torch.Tensor):
        if coords.device.type == 'cuda':
            return torchsparse.backend.kernel_hash_forward(
                coords.contiguous(), offsets.contiguous())
        elif coords.device.type == 'cpu':
            return torchsparse.backend.cpu_kernel_hash_forward(
                coords.int().contiguous(),
                offsets.int().contiguous())
        else:
            device = coords.device
            return torchsparse.backend.cpu_kernel_hash_forward(
                coords.int().contiguous().cpu(),
                offsets.int().contiguous().cpu()).to(device)


def sphash(coords: torch.Tensor, offsets: Optional[torch.Tensor] = None):
    if offsets is None:
        return HashGPU.apply(coords)
    else:
        return KernelHashGPU.apply(coords, offsets)
