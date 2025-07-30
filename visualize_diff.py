import torch
import argparse
from matplotlib.pyplot import imshow, show
import matplotlib.pyplot as plt
import numpy as np

# def visualize_diff(a, b):
#     if a.shape != b.shape:
#         raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

#     # Move to CPU and detach in case they are on GPU or require gradients
#     a, b = a.detach().cpu(), b.detach().cpu()

#     # Calculate difference and threshold
#     # diff_mask = (torch.abs(a - b) > 1).any(dim=-1) if a.ndim == 3 else (torch.abs(a - b) > 1)
#     diff_mask = (torch.abs(a - b)).any(dim=-1) if a.ndim == 3 else (torch.abs(a - b))
#     imshow(diff_mask.numpy(), cmap='gray')
#     show()

def visualize_diff(a, b):
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    # Move to CPU and detach if necessary
    a, b = a.detach().cpu(), b.detach().cpu()

    # Compute absolute difference
    diff = torch.abs(a - b)

    # Collapse channels if present (e.g., [H, W, C]) by averaging
    if diff.ndim == 3:
        diff = diff.mean(dim=-1)

    # Convert to numpy array
    diff_np = diff.numpy()

    # Get min and max for scaling
    diff_min = diff_np.min()
    diff_max = diff_np.max()

    # Plot with dynamic scaling
    plt.imshow(diff_np, cmap='gray', vmin=diff_min, vmax=diff_max)
    cbar = plt.colorbar()
    # cbar.set_label(f'Absolute Difference\nMin={diff_min:.2f}, Max={diff_max:.2f}')
    # plt.title("Visualized Differences")
    plt.axis('off')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Calibration script for comparing tensors.')
    parser.add_argument('--kernel_output', type=str, required=True,
                        help='Path to the kernel output tensor file.')
    parser.add_argument('--groundtruth', type=str, required=True,
                        help='Path to the ground truth tensor file.')
    args = parser.parse_args()

    a = torch.load(args.kernel_output)
    b = torch.load(args.groundtruth)

    visualize_diff(a, b)


if __name__ == "__main__":
    main()
