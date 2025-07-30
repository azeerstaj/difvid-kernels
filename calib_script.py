import torch
import argparse

# Check if tensors have the same shape
def compare_tensors(a, b):
    if a.shape != b.shape:
        print(f"Tensors have different shapes: {a.shape} vs {b.shape}")
        print("Cannot compare element-wise")
        return

    # Find elements that are different
    diff_mask = a != b
    
    if not diff_mask.any():
        print("Tensors are identical")
        return
    
    # Find all different elements
    diff_indices = torch.nonzero(diff_mask, as_tuple=False)
    
    # Limit to maximum 5 elements
    max_elements = min(5, len(diff_indices))
    
    print(f"Found {len(diff_indices)} different elements. Showing first {max_elements}:")
    print("-" * 60)
    
    for i in range(max_elements):
        idx = diff_indices[i]
        idx_tuple = tuple(idx.tolist())
        
        print(f"Element {i+1} at index {idx_tuple}:")
        print(f"  a[{idx_tuple}] = {a[idx_tuple]}")
        print(f"  b[{idx_tuple}] = {b[idx_tuple]}")
        print(f"  Difference: {a[idx_tuple] - b[idx_tuple]}")
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