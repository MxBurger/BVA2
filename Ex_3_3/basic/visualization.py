import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from wiener_filter import calculate_psnr, pad_to_shape


def plot_kernel_comparison(img: np.ndarray, kernel_type: str, kernel_sizes: List[int], 
                          kernels_dict: Dict[int, np.ndarray], degraded_images: Dict[int, np.ndarray],
                          restored_images: Dict[int, Dict[float, np.ndarray]], K_values: List[float]):
    """Plot comparison of different kernel sizes for a specific kernel type."""
    fig, axes = plt.subplots(len(kernel_sizes), len(K_values) + 2, figsize=(20, 16))
    fig.suptitle(f'Wiener Filter - {kernel_type} kernel (varying radii)', fontsize=16)

    for size_idx, kernel_size in enumerate(kernel_sizes):
        degraded = degraded_images[kernel_size]
        degraded_psnr = calculate_psnr(img, degraded)

        # Show original and degraded
        axes[size_idx, 0].imshow(img, cmap='gray')
        axes[size_idx, 0].set_title(f'Original\n(Radius={kernel_size})')
        axes[size_idx, 0].axis('off')

        axes[size_idx, 1].imshow(degraded, cmap='gray')
        axes[size_idx, 1].set_title(f'Degraded\nRadius={kernel_size}\nPSNR={degraded_psnr:.1f} dB')
        axes[size_idx, 1].axis('off')

        # Test different K values
        for k_idx, K in enumerate(K_values):
            restored = restored_images[kernel_size][K]
            psnr = calculate_psnr(img, restored)

            axes[size_idx, k_idx + 2].imshow(restored, cmap='gray')
            axes[size_idx, k_idx + 2].set_title(f'K={K}\nPSNR={psnr:.1f} dB')
            axes[size_idx, k_idx + 2].axis('off')

    plt.tight_layout()
    plt.show()


def plot_noise_level_analysis(img: np.ndarray, kernel_type: str, kernel_size: int,
                             noise_variances: List[float], degraded_images: List[np.ndarray],
                             restored_images: Dict[float, List[np.ndarray]], K_values: List[float]):
    """Plot analysis of different noise levels for a specific kernel."""
    fig, axes = plt.subplots(len(noise_variances), len(K_values) + 2, figsize=(16, 12))
    fig.suptitle(f'Noise Level Analysis - {kernel_type} kernel (radius={kernel_size})', fontsize=14)

    for i, (noise_var, degraded) in enumerate(zip(noise_variances, degraded_images)):
        degraded_psnr = calculate_psnr(img, degraded)

        # Show original and degraded
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(degraded, cmap='gray')
        axes[i, 1].set_title(f'Degraded\nNoise var={noise_var}\nPSNR={degraded_psnr:.1f} dB')
        axes[i, 1].axis('off')

        # Test different K values
        for j, K in enumerate(K_values):
            restored = restored_images[K][i]
            psnr = calculate_psnr(img, restored)

            axes[i, j + 2].imshow(restored, cmap='gray')
            axes[i, j + 2].set_title(f'K={K}\nPSNR={psnr:.1f} dB')
            axes[i, j + 2].axis('off')

    plt.tight_layout()
    plt.show()


def plot_butterworth_comparison(img: np.ndarray, degraded: np.ndarray, 
                               restored_no_butter: np.ndarray, restored_butter: np.ndarray,
                               kernel_name: str = "Unknown"):
    """Plot comparison of Wiener filter with and without Butterworth enhancement."""
    degraded_psnr = calculate_psnr(img, degraded)
    psnr_no_butter = calculate_psnr(img, restored_no_butter)
    psnr_butter = calculate_psnr(img, restored_butter)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f'Wiener Filter with Butterworth Enhancement - {kernel_name}', fontsize=14)

    # Original and degraded
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(degraded, cmap='gray')
    axes[1].set_title(f'Degraded\nPSNR={degraded_psnr:.1f} dB')
    axes[1].axis('off')

    # Wiener without Butterworth
    axes[2].imshow(restored_no_butter, cmap='gray')
    axes[2].set_title(f'Wiener (no Butterworth)\nPSNR={psnr_no_butter:.1f} dB')
    axes[2].axis('off')

    # Wiener with Butterworth
    axes[3].imshow(restored_butter, cmap='gray')
    axes[3].set_title(f'Wiener + Butterworth\nPSNR={psnr_butter:.1f} dB')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

    print(f"PSNR Comparison for {kernel_name}:")
    print(f"  Degraded: {degraded_psnr:.2f} dB")
    print(f"  Wiener only: {psnr_no_butter:.2f} dB")
    print(f"  Wiener + Butterworth: {psnr_butter:.2f} dB")


def plot_frequency_response(kernel: np.ndarray, K_values: List[float]):
    """Visualize the frequency response of the Wiener filter."""
    # Pad kernel to a reasonable size for visualization
    size = 256
    kernel_padded = pad_to_shape(kernel, (size, size))

    # Compute frequency response
    H = np.fft.fft2(kernel_padded)
    H_mag = np.abs(H)

    fig, axes = plt.subplots(2, len(K_values) + 1, figsize=(20, 8))
    fig.suptitle('Wiener Filter Frequency Response', fontsize=14)

    # Show kernel frequency response
    axes[0, 0].imshow(np.log(H_mag + 1), cmap='gray')
    axes[0, 0].set_title('log|H(u,v)|')
    axes[0, 0].axis('off')

    axes[1, 0].imshow(kernel, cmap='gray')
    axes[1, 0].set_title('Kernel h(x,y)')
    axes[1, 0].axis('off')

    # Show Wiener filter for different K values
    for i, K in enumerate(K_values):
        H_conj = np.conj(H)
        H_squared = np.abs(H) ** 2
        W = H_conj / (H_squared + K + 1e-10)
        W_mag = np.abs(W)

        axes[0, i + 1].imshow(np.log(W_mag + 1), cmap='gray')
        axes[0, i + 1].set_title(f'log|W(u,v)| K={K}')
        axes[0, i + 1].axis('off')

        # Show inverse FFT of W (impulse response)
        w_spatial = np.fft.ifft2(W)
        w_spatial = np.fft.ifftshift(np.real(w_spatial))
        axes[1, i + 1].imshow(w_spatial, cmap='gray')
        axes[1, i + 1].set_title(f'w(x,y) K={K}')
        axes[1, i + 1].axis('off')

    plt.tight_layout()
    plt.show()