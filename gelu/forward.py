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
torch::Tensor gelu_cuda(
    torch::Tensor x
);
'''

# Load the CUDA extension
gelu_forward = load_inline(
    name='gelu_forward',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['gelu_cuda'],# 'generate_anchors_single_level_cuda'],
    with_cuda=True,
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"],
    # build_directory='../load_inline_cuda',
)


if __name__ == "__main__":

    n_elements = 2 ** 8  # Number of elements in the tensor

    x = torch.randn([n_elements, n_elements]).cuda()
    # x = torch.arange(n_elements, dtype=torch.float32).cuda()
    # x = torch.arange(n_elements, dtype=torch.float32).cuda() + torch.randn([n_elements]).cuda()

    torch.save(x, 'gelu_input.pt')

    x = torch.load('gelu_input.pt').cuda()
    
    # Generate anchors using CUDA
    out = gelu_forward.gelu_cuda(x)
    gt = torch.nn.functional.gelu(x, approximate='tanh')

    torch.save(out, 'gelu_output.pt')
    torch.save(out, 'groundtruth.pt')
    
    print(f"Input   [:5]: {x.view(-1)[:5]}")
    print(f"Input  [-5:]: {x.view(-1)[-5:]}")
    print("-" * 10)
    print(f"Output  [:5]: {out.view(-1)[:5]}")
    print(f"Output [-5:]: {out.view(-1)[-5:]}")
    print("-" * 10)
    print(f"Gt      [:5]: {gt.view(-1)[:5]}")
    print(f"Gt     [-5:]: {gt.view(-1)[-5:]}")