import torchsparse.backend

__all__ = ['spcount']


def spcount(coords, num):
    if coords.device.type == 'cuda':
        outs = torchsparse.backend.count_forward_cuda(coords.contiguous(), num)
    else:
        outs = torchsparse.backend.count_forward_cpu(coords.contiguous(), num)
    return outs
