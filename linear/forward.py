import torch
from torch.utils.cpp_extension import load_inline
torch.manual_seed(0)

# Read CUDA kernel from file
def read_cuda_kernel(filename):
    with open(filename, 'r') as f:
        return f.read()

# Load the CUDA kernel source
cuda_source = read_cuda_kernel('kernel.cu')

# C++ function declarations
cpp_source = '''
torch::Tensor matmul_cuda(
    torch::Tensor A,
    torch::Tensor B
);
'''

# Load the CUDA extension
matmul_forward = load_inline(
    name='matmul_forward',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['matmul_cuda'],
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"],
)


if __name__ == "__main__":

    M, N, K = 2 ** 8, 2 ** 8, 2 ** 8  # Dimensions for the matrix multiplication
    A = torch.randn([M, K]).cuda()
    B = torch.randn([K, N]).cuda()
    gt = A @ B

    torch.save(A, 'matmul_a_input.pt')
    torch.save(B, 'matmul_b_input.pt')
    torch.save(gt, 'matmul_output.pt')

    # Generate anchors using CUDA
    out = matmul_forward.matmul_cuda(
        A, B
    )

    torch.save(out, 'matmul_output.pt')
    
    print(f"A  [:5]: {A.view(-1)[:5]}")
    print(f"A [-5:]: {A.view(-1)[-5:]}")
    print("-" * 10)
    print(f"C  [:5]: {out.view(-1)[:5]}")
    print(f"C [-5:]: {out.view(-1)[-5:]}")
    print("-" * 10)
    print(f"Gt [:5]: {gt.view(-1)[:5]}")
    print(f"Gt[-5:]: {gt.view(-1)[-5:]}")