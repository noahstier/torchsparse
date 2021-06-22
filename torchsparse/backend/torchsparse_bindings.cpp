#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "convolution/convolution_cpu_header.h"
#include "devoxelize/devoxelize_cpu.h"
#include "hash/hash_cpu_header.h"
#include "others/count_cpu_header.h"
#include "others/insertion_cpu_header.h"
#include "others/query_cpu_header.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparseconv_cpu_forward", &convolution_forward_cpu,
        "point cloud convolution forward (CPU)");
  m.def("sparseconv_cpu_backward", &convolution_backward_cpu,
        "point cloud convolution backward (CPU)");
  m.def("cpu_hash_forward", &cpu_hash_forward, "Hashing forward (CPU)");
  m.def("cpu_kernel_hash_forward", &cpu_kernel_hash_forward,
        "Kernel Hashing forward (CPU)");
  m.def("cpu_insertion_forward", &cpu_insertion_forward,
        "Insertion forward (CPU)");
  m.def("cpu_insertion_backward", &cpu_insertion_backward,
        "Insertion backward (CPU)");
  m.def("devoxelize_forward_cpu", &devoxelize_forward_cpu,
        "Devoxelization forward (CPU)");
  m.def("devoxelize_backward_cpu", &devoxelize_backward_cpu,
        "Devoxelization backward (CPU)");
  m.def("cpu_query_forward", &cpu_query_forward, "hash query forward (CPU)");
  m.def("count_forward_cpu", &count_forward_cpu, "count forward (CPU)");
}
