import torch

a = torch.load('gelu_output.pt')
b = torch.load('groundtruth.pt')

# Check if tensors have the same shape
if a.shape != b.shape:
    print(f"Tensors have different shapes: {a.shape} vs {b.shape}")
    print("Cannot compare element-wise")
else:
    # Find elements that are different
    diff_mask = a != b
    
    if not diff_mask.any():
        print("Tensors are identical")
    else:
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