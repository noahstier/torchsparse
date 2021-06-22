#ifndef _SPARSE_CONVOLUTION
#define _SPARSE_CONVOLUTION
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <torch/extension.h>

#include <algorithm>
#include <chrono>

void convolution_forward_kernel(const float *d_in_feat, int in_nchannel,
                                float *d_out_feat, int out_nchannel,
                                const float *d_kernel, const int *neighbor_map,
                                const int *neighbor_offset,
                                const int in_npoints, const int out_npoints,
                                const int n_neighbors, const bool transpose,
                                cublasHandle_t cuhandle, cudaStream_t stream);

void convolution_backward_kernel(const float *d_in_feat, float *d_grad_in_feat,
                                 int in_nchannel, const float *d_grad_out_feat,
                                 int out_nchannel, float *d_kernel,
                                 float *d_grad_kernel, const int *neighbor_map,
                                 const int *neighbor_offset,
                                 const int in_npoints, const int out_npoints,
                                 const int n_neighbors, const bool transpose,
                                 cublasHandle_t cuhandle, cudaStream_t stream);

void convolution_forward_cuda(at::Tensor in_feat, at::Tensor out_feat,
                              at::Tensor kernel, at::Tensor neighbor_map,
                              at::Tensor neighbor_offset, const bool transpose);

void convolution_backward_cuda(at::Tensor in_feat, at::Tensor grad_in_feat,
                               at::Tensor grad_out_feat, at::Tensor kernel,
                               at::Tensor grad_kernel, at::Tensor neighbor_map,
                               at::Tensor neighbor_offset,
                               const bool transpose);

#endif
