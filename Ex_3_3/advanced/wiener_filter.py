import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from scipy import signal
import matplotlib.pyplot as plt
import os


def load_reference_images(reference_paths: List[str]) -> List[np.ndarray]:
    """
    Load multiple natural reference images for kernel estimation.

    Args:
        reference_paths: List of paths to reference images

    Returns:
        List of loaded reference images as grayscale arrays
    """
    reference_images = []

    for path in reference_paths:
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                reference_images.append(img.astype(np.float64))
                print(f"Loaded reference image: {path}, shape: {img.shape}")
            else:
                print(f"Warning: Could not load image at {path}")
        else:
            print(f"Warning: File not found at {path}")

    return reference_images


def preprocess_images_for_comparison(ref_img: np.ndarray, degraded_img: np.ndarray,
                                     target_size: Tuple[int, int] = (256, 256)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess reference and degraded images for comparison.
    Resize to same dimensions and normalize.

    Args:
        ref_img: Reference natural image
        degraded_img: Degraded input image
        target_size: Target size for both images

    Returns:
        Tuple of preprocessed (reference, degraded) images
    """
    # Resize both images to same size
    ref_resized = cv2.resize(ref_img, target_size)
    deg_resized = cv2.resize(degraded_img, target_size)

    # Normalize to [0, 1] range
    ref_normalized = ref_resized / 255.0
    deg_normalized = deg_resized / 255.0

    return ref_normalized, deg_normalized


def estimate_kernel_frequency_domain(ref_img: np.ndarray, degraded_img: np.ndarray,
                                     kernel_size: int = 15) -> np.ndarray:
    """
    Estimate degradation kernel in frequency domain with proper fftshift handling.

    Args:
        ref_img: Reference image
        degraded_img: Degraded image
        kernel_size: Size of the extracted kernel

    Returns:
        Estimated normalized kernel
    """
    # Compute FFTs
    F = np.fft.fft2(ref_img)
    G = np.fft.fft2(degraded_img)

    # Estimate H = G / F with regularization
    epsilon = 1e-10
    H_estimate = G / (F + epsilon)

    # Convert back to spatial domain with proper shift
    h_estimate = np.fft.ifft2(H_estimate)
    h_estimate = np.fft.ifftshift(np.real(h_estimate))

    # Extract kernel from center
    center_row, center_col = h_estimate.shape[0] // 2, h_estimate.shape[1] // 2
    half_size = kernel_size // 2

    kernel = h_estimate[center_row - half_size:center_row + half_size + 1,
             center_col - half_size:center_col + half_size + 1]

    # Normalize kernel
    kernel = np.abs(kernel)
    if np.sum(kernel) > 0:
        kernel = kernel / np.sum(kernel)

    return kernel


def find_best_reference_match(degraded_img: np.ndarray,
                              reference_images: List[np.ndarray]) -> Tuple[int, float, np.ndarray]:
    """
    Find the best matching reference image based on cross-correlation.

    Args:
        degraded_img: Input degraded image
        reference_images: List of reference images

    Returns:
        Tuple of (best_index, similarity_score, best_reference_image)
    """
    best_similarity = -1
    best_index = 0
    best_reference = None

    print(f"\nComparing degraded image with {len(reference_images)} reference images...")

    for i, ref_img in enumerate(reference_images):
        # Preprocess images for comparison
        ref_processed, deg_processed = preprocess_images_for_comparison(ref_img, degraded_img)

        # Calculate cross-correlation
        correlation = signal.correlate2d(ref_processed, deg_processed, mode='full')
        max_correlation = np.max(correlation)

        print(f"  Reference {i}: max correlation = {max_correlation:.4f}")

        if max_correlation > best_similarity:
            best_similarity = max_correlation
            best_index = i
            best_reference = ref_img

    print(f"Best match: Reference {best_index} with correlation {best_similarity:.4f}")
    return best_index, best_similarity, best_reference


def wiener_deconvolution(degraded: np.ndarray, kernel: np.ndarray, K: float) -> np.ndarray:
    """
    Apply Wiener deconvolution with proper fftshift handling.

    Args:
        degraded: Degraded image [0, 1]
        kernel: Degradation kernel
        K: Regularization parameter

    Returns:
        Restored image [0, 1]
    """
    rows, cols = degraded.shape

    # Pad kernel to match degraded image size
    kernel_padded = np.zeros((rows, cols))
    k_rows, k_cols = kernel.shape
    start_row = (rows - k_rows) // 2
    start_col = (cols - k_cols) // 2
    kernel_padded[start_row:start_row + k_rows, start_col:start_col + k_cols] = kernel
    kernel_padded = np.fft.fftshift(kernel_padded)

    # Compute FFTs
    G = np.fft.fft2(degraded)
    H = np.fft.fft2(kernel_padded)

    # Apply Wiener filter
    H_conj = np.conj(H)
    H_squared = np.abs(H) ** 2

    epsilon = 1e-10
    W = H_conj / (H_squared + K + epsilon)

    # Restore image
    F_estimate = W * G
    restored = np.fft.ifft2(F_estimate)
    restored = np.real(restored)

    # Ensure output is in valid range [0, 1]
    restored = np.clip(restored, 0, 1)

    return restored


def calculate_psnr(original: np.ndarray, restored: np.ndarray) -> float:
    """
    Calculate PSNR for normalized float images [0, 1].

    Args:
        original: Original image [0, 1]
        restored: Restored image [0, 1]

    Returns:
        PSNR value in dB
    """
    mse = np.mean((original - restored) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def advanced_wiener_deconvolution(degraded_img: np.ndarray,
                                  reference_images: List[np.ndarray],
                                  K_values: List[float] = [0.001, 0.01, 0.1]) -> Dict:
    """
    Perform advanced Wiener deconvolution with kernel estimation from reference images.

    Args:
        degraded_img: Input degraded image
        reference_images: List of natural reference images
        K_values: List of regularization parameters to test

    Returns:
        Dictionary containing results and analysis
    """
    results = {
        'reference_match': {},
        'estimated_kernel': None,
        'restored_images': {},
        'psnr_scores': {},
        'best_K': None
    }

    # Find best matching reference image
    best_idx, similarity, best_ref = find_best_reference_match(degraded_img, reference_images)
    results['reference_match'] = {
        'best_index': best_idx,
        'similarity': similarity,
        'reference_image': best_ref
    }

    # Preprocess images for kernel estimation
    ref_processed, deg_processed = preprocess_images_for_comparison(best_ref, degraded_img)

    # Estimate kernel using frequency domain method
    print(f"\nEstimating kernel using frequency domain method...")
    estimated_kernel = estimate_kernel_frequency_domain(ref_processed, deg_processed)
    results['estimated_kernel'] = estimated_kernel

    print(f"Estimated kernel shape: {estimated_kernel.shape}")
    print(f"Kernel sum: {np.sum(estimated_kernel):.6f}")

    # Apply Wiener filter with estimated kernel for different K values
    restored_images = {}
    psnr_scores = {}

    for K in K_values:
        print(f"\nApplying Wiener filter with K = {K}...")
        restored = wiener_deconvolution(deg_processed, estimated_kernel, K)

        # Convert back to [0, 255] range
        restored_uint8 = (restored * 255).astype(np.uint8)
        restored_images[K] = restored_uint8

        # Calculate PSNR using best reference as approximation
        psnr = calculate_psnr(ref_processed, restored)
        psnr_scores[K] = psnr

        print(f"  PSNR with reference: {psnr:.2f} dB")

    results['restored_images'] = restored_images
    results['psnr_scores'] = psnr_scores
    results['best_K'] = max(psnr_scores.keys(), key=lambda k: psnr_scores[k])

    return results


def plot_wiener_results(results: Dict, degraded_img: np.ndarray):
    """
    Plot comprehensive results of Wiener filtering.

    Args:
        results: Results dictionary from advanced_wiener_deconvolution
        degraded_img: Original degraded image
    """
    estimated_kernel = results['estimated_kernel']
    restored_images = results['restored_images']
    psnr_scores = results['psnr_scores']
    best_K = results['best_K']
    best_ref = results['reference_match']['reference_image']

    # Create figure with subplots
    n_restored = len(restored_images)
    fig, axes = plt.subplots(2, max(3, n_restored + 1), figsize=(4 * max(3, n_restored + 1), 8))
    fig.suptitle('Wiener Filter Results with Kernel Estimation', fontsize=16)

    # Row 1: Images
    axes[0, 0].imshow(best_ref, cmap='gray')
    axes[0, 0].set_title('Best Reference')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(degraded_img, cmap='gray')
    axes[0, 1].set_title('Degraded Input')
    axes[0, 1].axis('off')

    for i, (K, restored) in enumerate(restored_images.items()):
        col_idx = i + 2
        if col_idx < axes.shape[1]:
            axes[0, col_idx].imshow(restored, cmap='gray')
            psnr = psnr_scores[K]
            title = f'K={K}\nPSNR={psnr:.1f} dB'
            if K == best_K:
                title += '\n(Best)'
            axes[0, col_idx].set_title(title)
            axes[0, col_idx].axis('off')

    # Clear unused axes in row 1
    for i in range(n_restored + 2, axes.shape[1]):
        axes[0, i].axis('off')

    # Row 2: Kernel and PSNR analysis
    axes[1, 0].imshow(estimated_kernel, cmap='hot')
    axes[1, 0].set_title(f'Estimated Kernel\n{estimated_kernel.shape}')
    axes[1, 0].axis('off')

    # PSNR plot
    K_list = list(psnr_scores.keys())
    psnr_list = list(psnr_scores.values())

    axes[1, 1].plot(K_list, psnr_list, 'bo-')
    axes[1, 1].set_xlabel('Regularization K')
    axes[1, 1].set_ylabel('PSNR (dB)')
    axes[1, 1].set_title('PSNR vs K')
    axes[1, 1].grid(True)
    axes[1, 1].set_xscale('log')

    # Highlight best K
    best_psnr = psnr_scores[best_K]
    axes[1, 1].plot(best_K, best_psnr, 'ro', markersize=8, label=f'Best: K={best_K}')
    axes[1, 1].legend()

    # Clear unused axes in row 2
    for i in range(2, axes.shape[1]):
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


def run_wiener_restoration(degraded_img: np.ndarray, reference_paths: List[str]):
    """
    Run Wiener filter restoration process.

    Args:
        degraded_img: Input degraded image
        reference_paths: List of paths to reference images
    """
    print("=== Wiener Filter Image Restoration ===")

    # Load reference images
    reference_images = load_reference_images(reference_paths)

    if not reference_images:
        print("Error: No reference images loaded!")
        return

    # Perform restoration
    results = advanced_wiener_deconvolution(
        degraded_img, reference_images,
        K_values=[0.001, 0.01, 0.1, 1.0]
    )

    # Display results
    plot_wiener_results(results, degraded_img)

    print(f"\nRestoration completed!")
    print(f"Best regularization parameter: K = {results['best_K']}")
    print(f"Best PSNR: {results['psnr_scores'][results['best_K']]:.2f} dB")


def main():
    """
    Main function demonstrating Wiener filter usage.
    """
    # Define paths
    reference_paths = [
        "ref/landscape_gray.png",
        "ref/lena_gray.tif",
        "ref/mandril_gray.tif",
        "ref/peppers_gray.tif",
        "ref/text.png",
    ]

    test_image_path = "bauhaus.png"

    # Load degraded image
    degraded_input = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    if degraded_input is None:
        print(f"Error: Could not load test image: {test_image_path}")
        return

    print(f"Loaded degraded image: {test_image_path}")

    # Run restoration
    run_wiener_restoration(degraded_input, reference_paths)


if __name__ == "__main__":
    main()