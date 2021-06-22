#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "convolution/convolution_cpu_header.h"
#include "convolution/convolution_gpu.h"
#include "devoxelize/devoxelize_cpu.h"
#include "devoxelize/devoxelize_cuda.h"
#include "hash/hash_cpu_header.h"
#include "hash/hash_gpu.h"
#include "others/count_cpu_header.h"
#include "others/count_gpu.h"
#include "others/insertion_cpu_header.h"
#include "others/insertion_gpu.h"
#include "others/query_cpu_header.h"
#include "others/query_gpu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparseconv_cpu_forward", &convolution_forward_cpu,
        "point cloud convolution forward (CPU)");
  m.def("sparseconv_cpu_backward", &convolution_backward_cpu,
        "point cloud convolution backward (CPU)");
  m.def("cpu_kernel_hash_forward", &cpu_kernel_hash_forward,
        "Kernel Hashing forward (CPU)");
  m.def("cpu_insertion_forward", &cpu_insertion_forward,
        "Insertion forward (CPU)");
  m.def("cpu_insertion_backward", &cpu_insertion_backward,
        "Insertion backward (CPU)");
  m.def("cpu_query_forward", &cpu_query_forward, "hash query forward (CPU)");
  m.def("convolution_forward_cuda", &convolution_forward_cuda,
        "point cloud convolution forward (CUDA)");
  m.def("convolution_backward_cuda", &convolution_backward_cuda,
        "point cloud convolution backward (CUDA)");
  m.def("hash_forward", &hash_forward, "Hashing forward (CUDA)");
  m.def("kernel_hash_forward", &kernel_hash_forward,
        "Kernel Hashing forward (CUDA)");
  m.def("cpu_hash_forward", &cpu_hash_forward, "Hashing forward (CPU)");
  m.def("devoxelize_forward_cuda", &devoxelize_forward_cuda,
        "Devoxelization forward (CUDA)");
  m.def("devoxelize_backward_cuda", &devoxelize_backward_cuda,
        "Devoxelization backward (CUDA)");
  m.def("devoxelize_forward_cpu", &devoxelize_forward_cpu,
        "Devoxelization forward (CPU)");
  m.def("devoxelize_backward_cpu", &devoxelize_backward_cpu,
        "Devoxelization backward (CPU)");
  m.def("count_forward_cuda", &count_forward_cuda, "Counting forward (CUDA)");
  m.def("count_forward_cpu", &count_forward_cpu, "count forward (CPU)");
  m.def("insertion_forward", &insertion_forward, "Insertion forward (CUDA)");
  m.def("insertion_backward", &insertion_backward, "Insertion backward (CUDA)");
  m.def("cpu_insertion_forward", &cpu_insertion_forward,
        "Insertion forward (CPU)");
  m.def("cpu_insertion_backward", &cpu_insertion_backward,
        "Insertion backward (CPU)");
  m.def("query_forward", &query_forward, "hash query forward (CUDA)");
}
