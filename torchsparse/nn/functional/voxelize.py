import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

import torchsparse.backend

__all__ = ['spvoxelize']


class VoxelizeGPU(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, feat, idx, cnt):
        out = torchsparse.backend.insertion_forward(feat.contiguous(),
                                                    idx.int().contiguous(), cnt)
        ctx.for_backwards = (idx.int().contiguous(), cnt, feat.shape[0])
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, top_grad):
        idx, cnt, N = ctx.for_backwards
        bottom_grad = torchsparse.backend.insertion_backward(
            top_grad.contiguous(), idx, cnt, N)
        return bottom_grad, None, None


def spvoxelize(feat, idx, cnt):
    return VoxelizeGPU.apply(feat, idx, cnt)
