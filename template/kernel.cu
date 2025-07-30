// C++ wrapper function for anchor generation CUDA kernel
// #include "cuAnchor.cuh"
#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <cmath>

int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}

__global__ void sinu_posemb_kernel(
    const float* t,        // Shape: (B)
    float* sin_out,        // Shape: (B, dim/2)
    float* cos_out,        // Shape: (B, dim/2)
    float  embed,
    int time_len,
    int half_dim
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < time_len && col < half_dim){
        float val = __expf(col * (-embed)) * t[row];
        sin_out[row * half_dim + col] = __sinf(val);
        cos_out[row * half_dim + col] = __cosf(val);
    }
}

// Host wrapper functions
void launch_sinu_posemb(
    const float* t,        // Shape: (B)
    float* sin_out,        // Shape: (B, dim/2)
    float* cos_out,        // Shape: (B, dim/2)
    int dim, int time_len
) {
    int block_size = 16;
    int half_dim = dim / 2;
    float embed = std::log(10000) / (half_dim - 1);

    dim3 blockDim(block_size, block_size);
    dim3 gridDim(
        ceildiv(half_dim, block_size),
        ceildiv(time_len, block_size)
    );

    sinu_posemb_kernel<<<blockDim, gridDim>>>(
        t, sin_out, cos_out, embed, time_len, half_dim
    );
}

torch::Tensor sinu_posemb_cuda(
    torch::Tensor t,           // Shape: (total_base_anchors, 4)
    int dim
) {
    // Check that tensors are on CUDA
    TORCH_CHECK(t.device().is_cuda(), "t must be a CUDA tensor");
    
    // Check tensor types
    TORCH_CHECK(t.dtype() == torch::kFloat32, "t must be float32");
    
    // Check tensor shapes
    TORCH_CHECK(t.dim() == 1, "t must be size of (BatchSize), dim = 1");
    
    // Create output tensor
    auto total_output_anchors = dim * t.size(0);
    auto options = torch::TensorOptions().dtype(t.dtype()).device(t.device());
    torch::Tensor sin_output_anchors = torch::empty({t.size(0), dim / 2}, options);
    torch::Tensor cos_output_anchors = torch::empty({t.size(0), dim / 2}, options);
    
    // Launch CUDA kernel
    launch_sinu_posemb(
        t.data_ptr<float>(),
        sin_output_anchors.data_ptr<float>(),
        cos_output_anchors.data_ptr<float>(),
        dim, t.size(0)
    );
    
    torch::Tensor output_anchors = \
        torch::cat({sin_output_anchors, cos_output_anchors}, 1);
    return output_anchors;
}
