import os
import torch
from torch.utils.cpp_extension import load_inline
import numpy as np

# Read CUDA kernel from file
def read_cuda_kernel(filename):
    with open(filename, 'r') as f:
        return f.read()

# Load the CUDA kernel source
cuda_source = read_cuda_kernel('kernel.cu')

# C++ function declarations
cpp_source = '''
torch::Tensor sinu_posemb_cuda(
    torch::Tensor t,
    int dim
);
'''

# Load the CUDA extension
sinu_posemb_forward = load_inline(
    name='sinu_posemb_forward',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['sinu_posemb_cuda'],# 'generate_anchors_single_level_cuda'],
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"],
    # build_directory='../load_inline_cuda',
)


# Example usage
if __name__ == "__main__":

    batch_size = 8
    dim = 256

    t = torch.randn([batch_size]).cuda()
    
    # Generate anchors using CUDA
    posembs_cuda = sinu_posemb_forward.sinu_posemb_cuda(
        t, dim
    )
    
    print(f"Generated {posembs_cuda.shape}")