import copy

import torch
import torchsparse_backend
from torch.autograd import Function
from torchsparse import *
from torchsparse.nn.functional.convert_neighbor_map import *
from torchsparse.nn.functional.downsample import *
from torchsparse.nn.functional.hash import *
from torchsparse.nn.functional.query import *
from torchsparse.utils.kernel_region import *

__all__ = ['conv3d']


class SpConvolution(Function):
    @staticmethod
    def forward(ctx,
                features,
                kernel,
                neighbor_map,
                neighbor_offset,
                sizes,
                transpose=False):
        features = features.contiguous()
        kernel = kernel.contiguous()
        if not transpose:
            out = torch.zeros(sizes[1],
                              kernel.size(-1),
                              device=features.device)
        else:
            # tbd: ensure the original, upsampled size to be the same.
            out = torch.zeros(sizes[0],
                              kernel.size(-1),
                              device=features.device)

        if 'cuda' in str(features.device):
            torchsparse_backend.sparseconv_forward(features, out, kernel,
                                                   neighbor_map,
                                                   neighbor_offset, transpose)
        else:
            # use the native pytorch XLA APIs for the TPU.
            cur_st = 0
            for kernel_idx in range(kernel.shape[0]):
                cur_ed = cur_st + neighbor_offset[kernel_idx]
                in_map = neighbor_map[cur_st:cur_ed, 0].long()
                out_map = neighbor_map[cur_st:cur_ed, 1].long()
                cur_st += neighbor_offset[kernel_idx]

                if transpose:
                    in_map, out_map = out_map, in_map
                # gather
                cur_feat = features[in_map]
                # gemm
                cur_feat = torch.mm(cur_feat, kernel[kernel_idx])
                # scatter
                out[out_map] += cur_feat

        ctx.for_backwards = (features, kernel, neighbor_map, neighbor_offset,
                             transpose)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        features, kernel, neighbor_map, neighbor_offset, transpose = ctx.for_backwards
        K, c_in, c_out = kernel.size()
        N_in = features.size(0)
        grad_features = torch.zeros(N_in, c_in, device=features.device)
        grad_kernel = torch.zeros(K, c_in, c_out, device=kernel.device)

        if 'cuda' in str(features.device):
            torchsparse_backend.sparseconv_backward(features, grad_features,
                                                    grad_out.contiguous(),
                                                    kernel, grad_kernel,
                                                    neighbor_map,
                                                    neighbor_offset, transpose)
        else:
            raise NotImplementedError
        return grad_features, grad_kernel, None, None, None, None


sparseconv_op = SpConvolution.apply


def conv3d(inputs,
           kernel,
           kernel_size,
           bias=None,
           stride=1,
           dilation=1,
           transpose=False):
    features = inputs.F
    coords = inputs.C
    cur_stride = inputs.s

    if kernel_size == 1 and stride == 1 and dilation == 1:
        output_features = features.matmul(kernel)
        if bias is not None:
            output_features += bias
        output_tensor = SparseTensor(output_features, coords, cur_stride)
        output_tensor.coord_maps = inputs.coord_maps
        output_tensor.kernel_maps = inputs.kernel_maps
        output_tensor.check()
    elif not transpose:
        _stride = str(stride) if isinstance(stride, int) else \
                str(stride.cpu().numpy().tolist())
        kernel_map = inputs.kernel_maps.get(
            'k%s_os%d_s%s_d%d' % (kernel_size, cur_stride, _stride, dilation),
            None)

        if isinstance(stride, int):
            do_downsample = stride > 1
        else:
            do_downsample = torch.any(stride > 1).item()

        if do_downsample:
            # do downsample
            kRegion = KernelRegion(kernel_size=kernel_size,
                                   tensor_stride=cur_stride)
            kOffset = kRegion.get_kernel_offset().to(features.device)
            new_coords = spdownsample(coords, stride, kernel_size, cur_stride)
            hash_query = sphash(new_coords, kOffset)
            hash_target = sphash(coords)
            idx_query = sphashquery(hash_query, hash_target)
            idx_query = list(convert_neighbor_map_gpu(idx_query))
            idx_query[1] = idx_query[1].to('cpu')
            sizes = (features.shape[0], new_coords.shape[0])
            output_features = sparseconv_op(features, kernel, idx_query[0],
                                            idx_query[1], sizes, transpose)
            if bias is not None:
                output_features += bias
            output_tensor = SparseTensor(output_features, new_coords,
                                         cur_stride * stride)
            output_tensor.coord_maps = copy.deepcopy(inputs.coord_maps)
            output_tensor.check()
            _stride = str(stride) if isinstance(stride, int) else \
                str(stride[0].cpu().int().numpy().tolist())
            if isinstance(stride, torch.Tensor):
                _os = torch.ones_like(stride) * cur_stride
            _os = str(cur_stride) if isinstance(stride, int) else \
                str(_os[0].cpu().int().numpy().tolist())
            output_tensor.kernel_maps = copy.deepcopy(inputs.kernel_maps)
            output_tensor.kernel_maps['k%s_os%s_s%s_d%d' %
                                      (kernel_size, _os, _stride,
                                       dilation)] = idx_query + [sizes]

        else:
            if kernel_map is None:
                kRegion = KernelRegion(kernel_size=kernel_size,
                                       tensor_stride=cur_stride)
                try:
                    kOffset = kRegion.get_kernel_offset().to(features.device)
                except:
                    raise
                hash_query = sphash(coords, kOffset)
                hash_target = sphash(coords)
                idx_query = sphashquery(hash_query, hash_target)
                idx_query = list(convert_neighbor_map_gpu(idx_query))
                idx_query[1] = idx_query[1].to('cpu')
                sizes = (features.shape[0], features.shape[0])
                output_features = sparseconv_op(features, kernel, idx_query[0],
                                                idx_query[1], sizes, transpose)
                if bias is not None:
                    output_features += bias
                output_tensor = SparseTensor(output_features, coords,
                                             cur_stride)
                output_tensor.coord_maps = inputs.coord_maps
                output_tensor.check()
                output_tensor.kernel_maps = copy.deepcopy(inputs.kernel_maps)
                output_tensor.kernel_maps['k%s_os%d_s%d_d%d' %
                                          (kernel_size, cur_stride, stride,
                                           dilation)] = idx_query + [sizes]
            else:
                output_features = sparseconv_op(features, kernel,
                                                kernel_map[0], kernel_map[1],
                                                kernel_map[2], transpose)
                if bias is not None:
                    output_features += bias
                output_tensor = SparseTensor(output_features, coords,
                                             cur_stride)
                output_tensor.coord_maps = inputs.coord_maps
                output_tensor.check()
                output_tensor.kernel_maps = inputs.kernel_maps

    else:
        # do upsample
        if isinstance(stride, int):
            original_stride = int(cur_stride / stride)
            _original_stride = original_stride
            _stride = stride
        else:
            cur_stride_tensor = torch.ones_like(stride) * cur_stride
            original_stride = cur_stride_tensor / stride
            _original_stride = str(original_stride[0].cpu().int().numpy().tolist())
            _stride = str(stride[0].cpu().int().numpy().tolist())
        
        kernel_map = inputs.kernel_maps.get(
            'k%s_os%s_s%s_d%d' %
            (kernel_size, _original_stride, _stride, dilation), None)
        assert kernel_map is not None, 'kernel map k%s_os%s_s%s_d%d'%(
            kernel_size, _original_stride, _stride, dilation
        ) + ' does not exist.'
        output_features = sparseconv_op(features, kernel, kernel_map[0],
                                        kernel_map[1], kernel_map[2],
                                        transpose)
        if bias is not None:
            output_features += bias
        
        if _original_stride in inputs.coord_maps:
            cur_coords = inputs.coord_maps[_original_stride]
        else:
            cur_coords = inputs.coord_maps[original_stride[0][0].item()]
        output_tensor = SparseTensor(output_features,
                                     cur_coords,
                                     original_stride)
        output_tensor.coord_maps = inputs.coord_maps
        output_tensor.check()
        output_tensor.kernel_maps = inputs.kernel_maps

    return output_tensor