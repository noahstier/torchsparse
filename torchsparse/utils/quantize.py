from collections import Sequence

import numpy as np
import torch

__all__ = ['sparse_quantize']


def ravel_hash_vec(arr):
    assert arr.ndim == 2
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def sparse_quantize(coords,
                    feats=None,
                    labels=None,
                    ignore_label=255,
                    return_index=False,
                    return_invs=False,
                    quantization_size=1):

    use_label = labels is not None
    use_feat = feats is not None
    if not use_label and not use_feat:
        return_index = True

    assert coords.ndim == 2
    if use_feat:
        assert feats.ndim == 2
        assert coords.shape[0] == feats.shape[0]
    if use_label:
        assert coords.shape[0] == len(labels)

    # Quantize the coordinates
    dimension = coords.shape[1]
    if isinstance(quantization_size, (Sequence, np.ndarray, torch.Tensor)):
        assert len(
            quantization_size
        ) == dimension, 'Quantization size and coordinates size mismatch.'
        quantization_size = list(quantization_size)
    elif np.isscalar(quantization_size):  # Assume that it is a scalar
        quantization_size = [int(quantization_size) for i in range(dimension)]
    else:
        raise ValueError('Not supported type for quantization_size.')
    discrete_coords = np.floor(coords / np.array(quantization_size))

    # Hash function type
    key = ravel_hash_vec(discrete_coords)
    if use_label:
        _, inds, invs, counts = np.unique(key,
                                          return_index=True,
                                          return_inverse=True,
                                          return_counts=True)
        filtered_labels = labels[inds]
        filtered_labels[counts > 1] = ignore_label
        if return_invs:
            if return_index:
                return inds, filtered_labels, invs
            else:
                return discrete_coords[inds], feats[inds], filtered_labels, invs
        else:
            if return_index:
                return inds, filtered_labels
            else:
                return discrete_coords[inds], feats[inds], filtered_labels

    else:
        _, inds, invs = np.unique(key, return_index=True, return_inverse=True)
        if return_invs:
            if return_index:
                return inds, invs
            else:
                if use_feat:
                    return discrete_coords[inds], feats[inds], invs
                else:
                    return discrete_coords[inds], invs
        else:
            if return_index:
                return inds
            else:
                if use_feat:
                    return discrete_coords[inds], feats[inds]
                else:
                    return discrete_coords[inds]
