#include <torch/torch.h>

#include <vector>

#include "count_gpu.h"

// make sure indices is int type
// feat: (b,c,n) indices: (b,n) -> out: (b,c,s), out_indices: (b,n)
// (preprocessed indices)
at::Tensor count_forward_cuda(const at::Tensor idx, const int s) {
  // return group_point_forward_gpu(points, indices);
  int N = idx.size(0);
  at::Tensor out =
      torch::zeros({s}, at::device(idx.device()).dtype(at::ScalarType::Int));
  count_wrapper(N, idx.data_ptr<int>(), out.data_ptr<int>());
  return out;
}
