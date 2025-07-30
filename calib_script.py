import torch
import argparse

def compare_tensors(a, b, top_k=100):
    if a.shape != b.shape:
        print(f"Tensors have different shapes: {a.shape} vs {b.shape}")
        print("Cannot compare element-wise")
        return

    diff_mask = a != b

    if not diff_mask.any():
        print("Tensors are identical")
        print("a:", a.view(-1)[:5])
        print("b:", b.view(-1)[:5])
        print("-" * 20)
        print("a:", a.view(-1)[-5:])
        print("b:", b.view(-1)[-5:])
        return

    diff_indices = torch.nonzero(diff_mask, as_tuple=False)

    # Compute absolute differences
    abs_diffs = torch.abs(a[diff_mask] - b[diff_mask])

    # Get top-k differences
    topk_values, topk_indices = torch.topk(abs_diffs, min(top_k, len(abs_diffs)))

    print(f"Found {len(diff_indices)} different elements. Showing top {len(topk_values)} by absolute difference:")
    print("-" * 60)

    for i in range(len(topk_values)):
        flat_idx = topk_indices[i]
        tensor_idx = tuple(diff_indices[flat_idx].tolist())

        a_val = a[tensor_idx]
        b_val = b[tensor_idx]
        diff = a_val - b_val

        print(f"Element {i+1} at index {tensor_idx}:")
        print(f"  a[{tensor_idx}] = {a_val}")
        print(f"  b[{tensor_idx}] = {b_val}")
        print(f"  Difference: {diff} (abs: {abs(diff)})")
        print()

def main():
    parser = argparse.ArgumentParser(description='Calibration script for comparing tensors.')
    parser.add_argument('--kernel_output', type=str, required=True,
                        help='Path to the kernel output tensor file.')
    parser.add_argument('--groundtruth', type=str, required=True,
                        help='Path to the ground truth tensor file.')

    args = parser.parse_args()

    a = torch.load(args.kernel_output)
    b = torch.load(args.groundtruth)

    compare_tensors(a, b)

if __name__ == "__main__":
    main()