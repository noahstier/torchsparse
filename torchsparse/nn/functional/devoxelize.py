import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

import torchsparse.backend

__all__ = ['spdevoxelize', 'calc_ti_weights']


def calc_ti_weights(pc, idx_query, scale: float = 1):
    # TODO(Haotian): normalize the weights to a probability distribution.
    # Note that some indices are "-1".
    with torch.no_grad():
        # don't want points to lie exactly on grid
        p = pc
        # don't use np.floor then convert to torch. numerical errors.
        if scale != 1:
            pf = torch.floor(pc / scale) * scale
        else:
            pf = torch.floor(pc)
        pc = pf + scale

        x = p[:, 0].view(-1, 1)
        y = p[:, 1].view(-1, 1)
        z = p[:, 2].view(-1, 1)

        xf = pf[:, 0].view(-1, 1).float()
        yf = pf[:, 1].view(-1, 1).float()
        zf = pf[:, 2].view(-1, 1).float()

        xc = pc[:, 0].view(-1, 1).float()
        yc = pc[:, 1].view(-1, 1).float()
        zc = pc[:, 2].view(-1, 1).float()

        w0 = (xc - x) * (yc - y) * (zc - z)
        w1 = (xc - x) * (yc - y) * (z - zf)
        w2 = (xc - x) * (y - yf) * (zc - z)
        w3 = (xc - x) * (y - yf) * (z - zf)
        w4 = (x - xf) * (yc - y) * (zc - z)
        w5 = (x - xf) * (yc - y) * (z - zf)
        w6 = (x - xf) * (y - yf) * (zc - z)
        w7 = (x - xf) * (y - yf) * (z - zf)

        w = torch.cat([w0, w1, w2, w3, w4, w5, w6, w7], dim=1)
        w = w.transpose(1, 0).contiguous()
        if scale != 1:
            w /= scale ** 3
        w[idx_query == -1] = 0
        w /= w.sum(0) + 1e-8
    return w


class DevoxelizeFunction(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx, feat, indices, weights):
        if 'cuda' in str(feat.device):
            out = torchsparse.backend.devoxelize_forward_cuda(
                feat.contiguous(),
                indices.contiguous().int(), weights.contiguous())
        else:
            out = torchsparse.backend.devoxelize_forward_cpu(
                feat.contiguous(),
                indices.contiguous().int(), weights.contiguous())

        ctx.for_backwards = (indices.contiguous().int(), weights, feat.shape[0])

        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        indices, weights, n = ctx.for_backwards

        if 'cuda' in str(grad_out.device):
            grad_features = torchsparse.backend.devoxelize_backward_cuda(
                grad_out.contiguous(), indices, weights, n)
        else:
            grad_features = torchsparse.backend.devoxelize_backward_cpu(
                grad_out.contiguous(), indices, weights, n)

        return grad_features, None, None


devoxelize = DevoxelizeFunction.apply


def spdevoxelize(feat, indices, weights):
    return devoxelize(feat, indices, weights)
