from typing import Optional, Tuple, Union

import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

import torchsparse.backend
from torchsparse import SparseTensor
from torchsparse.nn import functional as F
from torchsparse.nn.functional.utils import get_kernel_offsets
from torchsparse.utils import make_ntuple

__all__ = ['conv3d']


class ConvolutionFunction(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.half)
    def forward(ctx,
                features,
                kernel,
                neighbor_map,
                neighbor_offset,
                sizes,
                transposed: bool = False):
        features = features.contiguous()
        kernel = kernel.contiguous()
        if not transposed:
            outputs = torch.zeros(sizes[1],
                                  kernel.size(-1),
                                  dtype=features.dtype,
                                  device=features.device)
        else:
            # TODO(Haotian): ensure the original, upsampled size to be the same.
            outputs = torch.zeros(sizes[0],
                                  kernel.size(-1),
                                  dtype=features.dtype,
                                  device=features.device)

        if features.device.type == 'cuda':
            torchsparse.backend.convolution_forward_cuda(
                features, outputs, kernel, neighbor_map, neighbor_offset,
                transposed)
        else:
            # use the native pytorch XLA APIs for the TPU.
            cur_st = 0
            for kernel_idx in range(kernel.shape[0]):
                cur_ed = cur_st + neighbor_offset[kernel_idx]
                in_map = neighbor_map[cur_st:cur_ed, 0].long()
                out_map = neighbor_map[cur_st:cur_ed, 1].long()
                cur_st += neighbor_offset[kernel_idx]

                if transposed:
                    in_map, out_map = out_map, in_map
                # gather
                cur_feat = features[in_map]
                # gemm
                cur_feat = torch.mm(cur_feat, kernel[kernel_idx])
                # scatter
                outputs[out_map] += cur_feat

        ctx.for_backwards = (features, kernel, neighbor_map, neighbor_offset,
                             transposed)
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        features, kernel, neighbor_map, neighbor_offset, transpose = ctx.for_backwards

        input_size = features.size(0)
        kernel_volume, in_channels, out_channels = kernel.size()

        grad_features = torch.zeros(input_size,
                                    in_channels,
                                    device=features.device,
                                    dtype=features.dtype)
        grad_kernel = torch.zeros(kernel_volume,
                                  in_channels,
                                  out_channels,
                                  device=kernel.device,
                                  dtype=features.dtype)

        if features.device.type == 'cuda':
            torchsparse.backend.convolution_backward_cuda(
                features, grad_features, grad_outputs.contiguous(), kernel,
                grad_kernel, neighbor_map, neighbor_offset, transpose)
        else:
            raise NotImplementedError
        return grad_features, grad_kernel, None, None, None, None


def conv3d(inputs: SparseTensor,
           weight: torch.Tensor,
           kernel_size: Union[int, Tuple[int, ...]],
           bias: Optional[torch.Tensor] = None,
           stride: Union[int, Tuple[int, ...]] = 1,
           dilation: Union[int, Tuple[int, ...]] = 1,
           transposed: bool = False) -> SparseTensor:
    feats = inputs.F
    coords = inputs.C
    cur_stride = inputs.s

    kernel_size = make_ntuple(kernel_size, ndim=3)
    stride = make_ntuple(stride, ndim=3)
    dilation = make_ntuple(dilation, ndim=3)

    if (kernel_size == (1, 1, 1) and stride == (1, 1, 1)
            and dilation == (1, 1, 1)):
        feats = feats.matmul(weight)
        if bias is not None:
            feats += bias
        outputs = SparseTensor(feats, coords, cur_stride)
        outputs.cmaps = inputs.cmaps
        outputs.cmaps.setdefault(outputs.stride, outputs.coords)
        outputs.kmaps = inputs.kmaps
    elif not transposed:
        kmap = inputs.kmaps.get((cur_stride, kernel_size, stride, dilation))

        if any(s > 1 for s in stride):
            offsets = get_kernel_offsets(kernel_size,
                                         stride=cur_stride,
                                         device=feats.device)
            new_coords = F.spdownsample(coords, stride, kernel_size, cur_stride)
            hash_query = F.sphash(new_coords, offsets)
            hash_target = F.sphash(coords)
            idx_query = F.sphashquery(hash_query, hash_target)
            idx_query = list(F.squeeze_nmap(idx_query))
            idx_query[1] = idx_query[1].to('cpu')
            sizes = (feats.shape[0], new_coords.shape[0])
            feats = ConvolutionFunction.apply(feats, weight, idx_query[0],
                                              idx_query[1], sizes, transposed)
            if bias is not None:
                feats += bias
            outputs = SparseTensor(
                feats, new_coords,
                tuple(a * b for a, b in zip(cur_stride, stride)))
            outputs.cmaps = inputs.cmaps
            outputs.cmaps.setdefault(outputs.stride, outputs.coords)
            outputs.kmaps = inputs.kmaps
            outputs.kmaps.setdefault(
                (cur_stride, kernel_size, stride, dilation),
                idx_query + [sizes])
        else:
            if kmap is None:
                offsets = get_kernel_offsets(kernel_size,
                                             stride=cur_stride,
                                             device=feats.device)
                hash_query = F.sphash(coords, offsets)
                hash_target = F.sphash(coords)
                idx_query = F.sphashquery(hash_query, hash_target)
                idx_query = list(F.squeeze_nmap(idx_query))
                idx_query[1] = idx_query[1].to('cpu')
                sizes = (feats.shape[0], feats.shape[0])
                feats = ConvolutionFunction.apply(feats, weight, idx_query[0],
                                                  idx_query[1], sizes,
                                                  transposed)
                if bias is not None:
                    feats += bias
                outputs = SparseTensor(feats, coords, cur_stride)
                outputs.cmaps = inputs.cmaps
                outputs.cmaps.setdefault(outputs.stride, outputs.coords)
                outputs.kmaps = inputs.kmaps
                outputs.kmaps.setdefault(
                    (cur_stride, kernel_size, stride, dilation),
                    idx_query + [sizes])
            else:
                feats = ConvolutionFunction.apply(feats, weight, kmap[0],
                                                  kmap[1], kmap[2], transposed)
                if bias is not None:
                    feats += bias
                outputs = SparseTensor(feats, coords, cur_stride)
                outputs.cmaps = inputs.cmaps
                outputs.cmaps.setdefault(outputs.stride, outputs.coords)
                outputs.kmaps = inputs.kmaps

    else:
        original_stride = tuple(int(a / b) for a, b in zip(cur_stride, stride))

        kmap = inputs.kmaps[(original_stride, kernel_size, stride, dilation)]
        assert kmap is not None
        feats = ConvolutionFunction.apply(feats, weight, kmap[0], kmap[1],
                                          kmap[2], transposed)
        if bias is not None:
            feats += bias

        coords = inputs.cmaps[original_stride]

        outputs = SparseTensor(feats, coords, original_stride)
        outputs.cmaps = inputs.cmaps
        outputs.cmaps.setdefault(outputs.stride, outputs.coords)
        outputs.kmaps = inputs.kmaps

    return outputs
