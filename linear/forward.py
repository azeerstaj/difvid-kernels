import torch
from torch.utils.cpp_extension import load_inline

# Read CUDA kernel from file
def read_cuda_kernel(filename):
    with open(filename, 'r') as f:
        return f.read()

# Load the CUDA kernel source
cuda_source = read_cuda_kernel('kernel.cu')

# C++ function declarations
cpp_source = '''
torch::Tensor linear_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor bias
);
'''

# Load the CUDA extension
matmul_forward = load_inline(
    name='linear_forward',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['linear_cuda'],
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"],
)

if __name__ == "__main__":

    M, N, K = 2 ** 10, 2 ** 10, 2 ** 10  # Dimensions for the matrix multiplication
    A = torch.randn([M, K]).cuda()
    B = torch.randn([K, N]).cuda()
    bias = torch.randn([N]).cuda()
    gt = torch.nn.functional.gelu(A @ B + bias)

    torch.save(A, 'matmul_a_input.pt')
    torch.save(B, 'matmul_b_input.pt')
    torch.save(bias, 'matmul_bias_input.pt')

    # Generate anchors using CUDA
    out = matmul_forward.linear_cuda(
        A, B, bias
    )

    torch.save(out, 'matmul_output.pt')
    torch.save(gt, 'groundtruth.pt')
    
    print(f"A  [:5]: {A.view(-1)[:5]}")
    print(f"A [-5:]: {A.view(-1)[-5:]}")
    print("-" * 10)
    print(f"C  [:5]: {out.view(-1)[:5]}")
    print(f"C [-5:]: {out.view(-1)[-5:]}")
    print("-" * 10)
    print(f"Gt [:5]: {gt.view(-1)[:5]}")
    print(f"Gt[-5:]: {gt.view(-1)[-5:]}")