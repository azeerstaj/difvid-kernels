// C++ wrapper function for anchor generation CUDA kernel
// #include "cuAnchor.cuh"
#include <torch/extension.h>
#include <vector>

// __global__ void sinu_posemb_kernel(
//     const float* t,    // Shape: (num_base_anchors, 4)
//     float* out        // Shape: (feat_h * feat_w * num_base_anchors, 4)
// ) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     return;
// }


// Host wrapper functions
void launch_sinu_posemb(
    const float* t,    // Shape: (B)
    float* out,        // Shape: (B, )
    int dim
) {
    int block_size = 256;
    // int grid_size = (feat_h * feat_w * num_base_anchors + block_size - 1) / block_size;
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
    torch::Tensor output_anchors = torch::cat({sin_output_anchors, cos_output_anchors}, 1);
    
    // Launch CUDA kernel
    // launch_sinu_posemb(
    //     t.data_ptr<float>(),
    //     output_anchors.data_ptr<float>(),
    //     dim
    // );
    
    return output_anchors;
}
