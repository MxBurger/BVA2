import numpy as np
import cv2
from typing import List, Tuple, Dict
from scipy import signal
import matplotlib.pyplot as plt


def add_controlled_noise(image: np.ndarray, noise_level: float, noise_type: str = 'gaussian') -> np.ndarray:
    """Add controlled noise to test Wiener filter robustness."""
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level, image.shape)
    elif noise_type == 'uniform':
        noise = np.random.uniform(-noise_level, noise_level, image.shape)
    else:
        raise ValueError("noise_type must be 'gaussian' or 'uniform'")

    noisy = image + noise
    return np.clip(noisy, 0, 1)


def load_reference_images(reference_paths: List[str]) -> List[np.ndarray]:
    """Load multiple natural reference images for kernel estimation."""
    reference_images = []
    for path in reference_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        reference_images.append(img.astype(np.float64))
        print(f"Loaded: {path}, shape: {img.shape}")
    return reference_images


def preprocess_images_for_comparison(ref_img: np.ndarray, degraded_img: np.ndarray,
                                     target_size: Tuple[int, int] = (256, 256)) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess reference and degraded images for comparison."""
    # Resize both images to same size
    ref_resized = cv2.resize(ref_img, target_size)
    deg_resized = cv2.resize(degraded_img, target_size)

    # Normalize to [0, 1] range
    ref_normalized = ref_resized / 255.0
    deg_normalized = deg_resized / 255.0

    return ref_normalized, deg_normalized


def estimate_kernel(ref_img: np.ndarray, degraded_img: np.ndarray,
                    kernel_size: int = 15, regularization: float = 1e-6) -> np.ndarray:
    """
    Estimate degradation kernel using Wiener-like regularization.

    Args:
        ref_img: Reference image [0, 1]
        degraded_img: Degraded image [0, 1]
        kernel_size: Size of the extracted kernel
        regularization: Regularization parameter for stable division

    Returns:
        Estimated normalized kernel
    """
    print(f"Estimating kernel {kernel_size}x{kernel_size}...")

    # Windowing to reduce edge effects
    window = np.outer(np.hanning(ref_img.shape[0]), np.hanning(ref_img.shape[1]))
    ref_windowed = ref_img * window
    deg_windowed = degraded_img * window

    # FFT with zero padding for better frequency resolution
    pad_shape = [2 * ref_img.shape[0], 2 * ref_img.shape[1]]
    F = np.fft.fft2(ref_windowed, s=pad_shape)
    G = np.fft.fft2(deg_windowed, s=pad_shape)

    # Kernel estimation in frequency domain: H = G / F with regularization
    F_magnitude = np.abs(F)
    regularization_term = regularization * np.max(F_magnitude)
    H_estimate = G * np.conj(F) / (F_magnitude ** 2 + regularization_term)

    # Convert back to spatial domain
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

    print(f"Kernel estimated. Sum: {np.sum(kernel):.6f}")
    return kernel


def calculate_quality_metrics(original: np.ndarray, restored: np.ndarray) -> Dict:
    """Calculate comprehensive image quality metrics."""
    mse = np.mean((original - restored) ** 2)
    psnr = float('inf') if mse == 0 else 10 * np.log10(1.0 / mse)

    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - restored) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

    # Edge preservation
    grad_orig = np.concatenate([np.gradient(original, axis=1).flatten(),
                                np.gradient(original, axis=0).flatten()])
    grad_rest = np.concatenate([np.gradient(restored, axis=1).flatten(),
                                np.gradient(restored, axis=0).flatten()])

    edge_preservation = 0.0
    if np.std(grad_orig) > 1e-10 and np.std(grad_rest) > 1e-10:
        edge_corr = np.corrcoef(grad_orig, grad_rest)[0, 1]
        edge_preservation = edge_corr if not np.isnan(edge_corr) else 0.0

    return {
        'psnr': psnr,
        'mse': mse,
        'snr': snr,
        'edge_preservation': edge_preservation
    }


def find_best_reference_match(degraded_img: np.ndarray,
                              reference_images: List[np.ndarray]) -> Tuple[int, float, np.ndarray]:
    """Find the best matching reference image based on cross-correlation."""
    best_similarity = -1
    best_index = 0
    best_reference = None

    print(f"\nComparing degraded image with {len(reference_images)} reference images...")

    for i, ref_img in enumerate(reference_images):
        ref_processed, deg_processed = preprocess_images_for_comparison(ref_img, degraded_img)
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
    """Apply Wiener deconvolution with proper regularization."""
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

    # Apply Wiener filter: W = H* / (|H|² + K)
    H_conj = np.conj(H)
    H_squared = np.abs(H) ** 2
    epsilon = 1e-10
    W = H_conj / (H_squared + K + epsilon)

    # Restore image
    F_estimate = W * G
    restored = np.fft.ifft2(F_estimate)
    restored = np.real(restored)

    return np.clip(restored, 0, 1)


def analyze_content_similarity_impact(degraded_img: np.ndarray,
                                      reference_images: List[np.ndarray],
                                      reference_labels: List[str]) -> Dict:
    """Analyze how content similarity affects restoration quality."""
    results = {}

    print(f"\n=== CONTENT SIMILARITY IMPACT ANALYSIS ===")
    print(f"Testing {len(reference_images)} reference images with different content types")

    for i, (ref_img, label) in enumerate(zip(reference_images, reference_labels)):
        print(f"\nAnalyzing reference '{label}'...")

        # Preprocess images
        ref_proc, deg_proc = preprocess_images_for_comparison(ref_img, degraded_img)
        estimated_kernel = estimate_kernel(ref_proc, deg_proc)

        # Test restoration with multiple K values
        best_restoration = None
        best_metrics = None
        best_psnr = -np.inf
        best_K = None

        K_values = [0.001, 0.01, 0.1, 1.0]
        for K in K_values:
            restored = wiener_deconvolution(deg_proc, estimated_kernel, K)
            metrics = calculate_quality_metrics(ref_proc, restored)

            if metrics['psnr'] > best_psnr:
                best_psnr = metrics['psnr']
                best_restoration = restored
                best_metrics = metrics
                best_K = K

        # Calculate similarity metrics
        correlation = signal.correlate2d(ref_proc, deg_proc, mode='same')
        max_spatial_corr = np.max(correlation)

        # Spectral similarity
        F_ref = np.fft.fft2(ref_proc)
        F_deg = np.fft.fft2(deg_proc)
        spectral_similarity = np.mean(np.abs(F_ref) * np.abs(F_deg)) / (
                np.sqrt(np.mean(np.abs(F_ref) ** 2) * np.mean(np.abs(F_deg) ** 2)) + 1e-10)

        # Store results
        results[label] = {
            'estimated_kernel': estimated_kernel,
            'best_restoration': best_restoration,
            'best_metrics': best_metrics,
            'best_K': best_K,
            'similarity_metrics': {
                'spatial_correlation': max_spatial_corr,
                'spectral_similarity': spectral_similarity
            }
        }

        print(f"  Best PSNR: {best_psnr:.2f} dB (K={best_K})")
        print(f"  Spatial correlation: {max_spatial_corr:.4f}")
        print(f"  Spectral similarity: {spectral_similarity:.4f}")

    return results


def advanced_wiener_deconvolution(degraded_img: np.ndarray,
                                  reference_images: List[np.ndarray],
                                  reference_labels: List[str] = None,
                                  K_values: List[float] = [0.001, 0.01, 0.1, 1.0],
                                  noise_level: float = 0.0,
                                  noise_type: str = 'gaussian') -> Dict:
    """
    Perform advanced Wiener deconvolution using kernel estimation.

    Args:
        degraded_img: Input degraded image
        reference_images: List of natural reference images
        reference_labels: Labels for reference images
        K_values: List of regularization parameters to test
        noise_level: Additional noise level for robustness testing
        noise_type: Type of noise ('gaussian' or 'uniform')

    Returns:
        Dictionary containing comprehensive results and analysis
    """
    results = {}

    # Normalize degraded image
    if degraded_img.max() > 1.0:
        degraded_img = degraded_img.astype(np.float64) / 255.0

    # Add controlled noise if specified
    if noise_level > 0:
        degraded_img = add_controlled_noise(degraded_img, noise_level, noise_type)

    # Perform content similarity analysis if labels provided
    if reference_labels and len(reference_labels) == len(reference_images):
        content_analysis = analyze_content_similarity_impact(
            degraded_img, reference_images, reference_labels)
        results['content_similarity_analysis'] = content_analysis

        # Use the reference that achieved the best PSNR
        best_label = max(content_analysis.keys(),
                         key=lambda x: content_analysis[x]['best_metrics']['psnr'])
        best_idx = reference_labels.index(best_label)
        best_ref = reference_images[best_idx]
        similarity_score = content_analysis[best_label]['similarity_metrics']['spatial_correlation']

        results['reference_match'] = {
            'best_index': best_idx,
            'similarity': similarity_score,
            'reference_image': best_ref,
            'selection_method': 'content_analysis'
        }
    else:
        # Fallback: Use cross-correlation method
        best_idx, similarity, best_ref = find_best_reference_match(degraded_img, reference_images)
        results['reference_match'] = {
            'best_index': best_idx,
            'similarity': similarity,
            'reference_image': best_ref,
            'selection_method': 'cross_correlation'
        }

    # Preprocess images for kernel estimation
    ref_processed, deg_processed = preprocess_images_for_comparison(best_ref, degraded_img)

    # Estimate kernel
    estimated_kernel = estimate_kernel(ref_processed, deg_processed)
    results['estimated_kernel'] = estimated_kernel

    # Apply Wiener filter for different K values
    restored_images = {}
    quality_metrics = {}

    for K in K_values:
        restored = wiener_deconvolution(deg_processed, estimated_kernel, K)
        restored_uint8 = (restored * 255).astype(np.uint8)
        restored_images[K] = restored_uint8
        quality_metrics[K] = calculate_quality_metrics(ref_processed, restored)

    results['restored_images'] = restored_images
    results['quality_metrics'] = quality_metrics

    # Find best K value
    best_K = max(K_values, key=lambda k: quality_metrics[k]['psnr'])
    results['best_K'] = best_K

    return results


def plot_results(results: Dict, degraded_img: np.ndarray):
    """Plot comprehensive results of advanced Wiener filtering."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    # Row 1: Original images and best restoration
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(results['reference_match']['reference_image'], cmap='gray')
    ax1.set_title('Best Reference Match')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(degraded_img, cmap='gray')
    ax2.set_title('Degraded Input')
    ax2.axis('off')

    # Best restoration
    best_K = results['best_K']
    ax3 = fig.add_subplot(gs[0, 2])
    best_restored = results['restored_images'][best_K]
    ax3.imshow(best_restored, cmap='gray')
    psnr_best = results['quality_metrics'][best_K]['psnr']
    ax3.set_title(f'Best Restoration\nK={best_K}, PSNR={psnr_best:.1f}dB')
    ax3.axis('off')

    # Estimated kernel
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(results['estimated_kernel'], cmap='hot')
    ax4.set_title('Estimated Kernel')
    ax4.axis('off')

    # PSNR comparison
    ax5 = fig.add_subplot(gs[1, 0:2])
    K_values = list(results['quality_metrics'].keys())
    psnr_values = [results['quality_metrics'][K]['psnr'] for K in K_values]

    ax5.semilogx(K_values, psnr_values, 'bo-', linewidth=2, markersize=8)
    ax5.set_xlabel('Regularization K')
    ax5.set_ylabel('PSNR (dB)')
    ax5.set_title('PSNR vs Regularization Parameter')
    ax5.grid(True, alpha=0.3)

    # Highlight best K
    best_psnr = results['quality_metrics'][best_K]['psnr']
    ax5.semilogx(best_K, best_psnr, 'ro', markersize=10, label=f'Best: K={best_K}')
    ax5.legend()

    # Content similarity analysis
    if 'content_similarity_analysis' in results and results['content_similarity_analysis']:
        content_analysis = results['content_similarity_analysis']

        ax6 = fig.add_subplot(gs[1, 2:4])
        labels = list(content_analysis.keys())
        spatial_corr = [content_analysis[label]['similarity_metrics']['spatial_correlation']
                        for label in labels]
        best_psnr_per_ref = [content_analysis[label]['best_metrics']['psnr']
                             for label in labels]

        ax6.scatter(spatial_corr, best_psnr_per_ref, s=100, alpha=0.7,
                    c=range(len(labels)), cmap='viridis')
        ax6.set_xlabel('Spatial Correlation with Degraded Image')
        ax6.set_ylabel('Best PSNR (dB)')
        ax6.set_title('Content Similarity vs Restoration Quality')
        ax6.grid(True, alpha=0.3)

        for i, label in enumerate(labels):
            ax6.annotate(label, (spatial_corr[i], best_psnr_per_ref[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

        if len(spatial_corr) > 2:
            correlation_coeff = np.corrcoef(spatial_corr, best_psnr_per_ref)[0, 1]
            ax6.text(0.05, 0.95, f'Correlation: {correlation_coeff:.3f}',
                     transform=ax6.transAxes, fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    fig.suptitle('Advanced Wiener Filter - Kernel Analysis', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.show()

    print_summary(results)


def print_summary(results: Dict):
    """Print detailed summary of the analysis results."""
    print(f"\n{'=' * 60}")
    print(f"ADVANCED WIENER FILTER - SUMMARY")
    print(f"{'=' * 60}")

    best_K = results['best_K']
    best_metrics = results['quality_metrics'][best_K]

    print(f"✓ Best Configuration: K = {best_K}")
    print(f"✓ Best PSNR: {best_metrics['psnr']:.2f} dB")
    print(f"✓ SNR: {best_metrics['snr']:.2f} dB")
    print(f"✓ Edge Preservation: {best_metrics['edge_preservation']:.3f}")

    # Content similarity insights
    if 'content_similarity_analysis' in results and results['content_similarity_analysis']:
        content_analysis = results['content_similarity_analysis']
        print(f"\n📊 CONTENT SIMILARITY INSIGHTS:")

        performance_ranking = []
        for label, data in content_analysis.items():
            best_psnr = data['best_metrics']['psnr']
            spatial_corr = data['similarity_metrics']['spatial_correlation']
            performance_ranking.append((label, best_psnr, spatial_corr))

        performance_ranking.sort(key=lambda x: x[1], reverse=True)

        print(f"🏆 Best performing reference: {performance_ranking[0][0]} "
              f"(PSNR: {performance_ranking[0][1]:.2f} dB, Correlation: {performance_ranking[0][2]:.4f})")
        print(f"📉 Worst performing reference: {performance_ranking[-1][0]} "
              f"(PSNR: {performance_ranking[-1][1]:.2f} dB, Correlation: {performance_ranking[-1][2]:.4f})")

        if len(performance_ranking) > 2:
            psnr_values = [x[1] for x in performance_ranking]
            corr_values = [x[2] for x in performance_ranking]
            correlation_coeff = np.corrcoef(psnr_values, corr_values)[0, 1]

            print(f"📈 Correlation between spatial similarity and restoration quality: {correlation_coeff:.3f}")

            if correlation_coeff > 0.5:
                print("✅ Strong positive correlation - Content similarity significantly helps!")
            elif correlation_coeff > 0.2:
                print("⚖️ Moderate correlation - Content similarity has some impact")
            else:
                print("❌ Weak correlation - Content similarity has limited impact")


def run_wiener_restoration(degraded_img: np.ndarray, reference_paths: List[str],
                           reference_labels: List[str] = None, noise_level: float = 0.0):
    """Run advanced Wiener filter restoration process."""
    print("=== ADVANCED WIENER FILTER ===")

    reference_images = load_reference_images(reference_paths)

    if reference_labels is None:
        reference_labels = [f"Reference_{i}" for i in range(len(reference_images))]

    results = advanced_wiener_deconvolution(
        degraded_img, reference_images, reference_labels,
        K_values=[0.001, 0.01, 0.1, 1.0],
        noise_level=noise_level
    )

    plot_results(results, degraded_img)
    print(f"\nOptimal Configuration: K = {results['best_K']}")


def main():
    """Main function demonstrating advanced Wiener filter with kernel estimation."""
    reference_paths = [
        "ref/landscape_gray.png",
        "ref/lena_gray.tif",
        "ref/mandril_gray.tif",
        "ref/peppers_gray.tif",
        "ref/text.png",
    ]

    reference_labels = ["landscape", "portrait", "animal", "food", "text"]

    degraded_input = cv2.imread("bauhaus.png", cv2.IMREAD_GRAYSCALE)
    print(f"Loaded degraded image: bauhaus.png")

    # Run restoration tests
    print(f"\n🔬 WITHOUT ADDITIONAL NOISE")
    run_wiener_restoration(degraded_input, reference_paths, reference_labels, 0.0)

    print(f"\n🔬 WITH LOW NOISE (σ=0.02)")
    run_wiener_restoration(degraded_input, reference_paths, reference_labels, 0.02)

    print(f"\n🔬 WITH MODERATE NOISE (σ=0.05)")
    run_wiener_restoration(degraded_input, reference_paths, reference_labels, 0.05)


if __name__ == "__main__":
    main()