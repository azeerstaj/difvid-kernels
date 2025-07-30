// C++ wrapper function for GELU CUDA kernel

// Using the approximation from : \
// https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html

#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <cmath>

#define CONST_ 0.044715
#define SQRT_2_OVER_PI 0.797884560803

int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}

__global__ void gelu_kernel(
    const float* x,
    float* y,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        float val = x[idx];
        y[idx] = 0.5 * val * (1 + tanhf(SQRT_2_OVER_PI * (val + CONST_ * val * val * val)));
    }
}

// Host wrapper functions
void launch_gelu(
    const float* x,
    float* y,
    int n
) {
    int block_size = 256;

    dim3 blockDim(block_size);
    dim3 gridDim(ceildiv(n, block_size));

    gelu_kernel<<<blockDim, gridDim>>>(x, y, n);
}

torch::Tensor gelu_cuda(
    torch::Tensor x
) {
    // Check that tensors are on CUDA
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    
    // Check tensor types
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    
    // Create output tensor
    unsigned int total_elements = 1;
    for (int i = 0; i < x.dim(); ++i) {
        total_elements *= x.size(i);
    }

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor y = torch::empty(x.sizes(), options);
    
    // Launch CUDA kernel
    launch_gelu(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        total_elements
    );
    
    return y;
}
