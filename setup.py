import os

import torch
import torch.cuda
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (CUDA_HOME, BuildExtension, CppExtension,
                                       CUDAExtension)

from torchsparse import __version__

has_cuda = (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv(
    'FORCE_CUDA', '0') == '1'

# Notice that CUDA files, header files should not share names with CPP files.
# Otherwise, there will be "ninja: warning: multiple rules generate xxx.o",
# which leads to multiple definitions error!

file_lis = [
    'torchsparse/backend/torchsparse_bindings_gpu.cpp',
    'torchsparse/backend/convolution/convolution_cpu.cpp',
    'torchsparse/backend/convolution/convolution.cu',
    'torchsparse/backend/convolution/convolution_gpu.cu',
    'torchsparse/backend/hash/hash_cpu.cpp',
    'torchsparse/backend/hash/hash.cpp',
    'torchsparse/backend/hash/hash_gpu.cu',
    'torchsparse/backend/hashmap/hashmap.cu',
    'torchsparse/backend/hashmap/hashmap_cpu.cpp',
    'torchsparse/backend/devoxelize/devoxelize_cuda.cu',
    'torchsparse/backend/devoxelize/devoxelize_cpu.cpp',
    'torchsparse/backend/others/count.cpp',
    'torchsparse/backend/others/count_gpu.cu',
    'torchsparse/backend/others/count_cpu.cpp',
    'torchsparse/backend/others/insertion_gpu.cu',
    'torchsparse/backend/others/insertion_cpu.cpp',
    'torchsparse/backend/others/query.cpp',
    'torchsparse/backend/others/query_cpu.cpp',
] if has_cuda else [
    'torchsparse/backend/torchsparse_bindings.cpp',
    'torchsparse/backend/convolution/convolution_cpu.cpp',
    'torchsparse/backend/hash/hash_cpu.cpp',
    'torchsparse/backend/hashmap/hashmap_cpu.cpp',
    'torchsparse/backend/devoxelize/devoxelize_cpu.cpp',
    'torchsparse/backend/others/insertion_cpu.cpp',
    'torchsparse/backend/others/query_cpu.cpp',
    'torchsparse/backend/others/count_cpu.cpp'
]

extra_compile_args = {
    'cxx': ['-g', '-O3', '-fopenmp', '-lgomp'],
    'nvcc': ['-O3']
} if has_cuda else {
    'cxx': ['-g', '-O3', '-fopenmp', '-lgomp']
}

extension_type = CUDAExtension if has_cuda else CppExtension
setup(
    name='torchsparse',
    version=__version__,
    packages=find_packages(),
    ext_modules=[
        extension_type('torchsparse.backend',
                       file_lis,
                       extra_compile_args=extra_compile_args)
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
