import numpy as np
import cv2
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

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
    ref_resized = cv2.resize(ref_img, target_size)
    deg_resized = cv2.resize(degraded_img, target_size)

    ref_normalized = ref_resized / 255.0
    deg_normalized = deg_resized / 255.0

    return ref_normalized, deg_normalized


def estimate_kernel(original_img: np.ndarray, degraded_img: np.ndarray,
                           kernel_size: int = 15, regularization: float = 0.01) -> np.ndarray:
    """
    Estimate degradation kernel using stable Wiener formulation.
    """
    print(f"\n📐 Estimating kernel ({kernel_size}x{kernel_size})...")

    # Ensure same dimensions
    min_h = min(original_img.shape[0], degraded_img.shape[0])
    min_w = min(original_img.shape[1], degraded_img.shape[1])
    original_img = original_img[:min_h, :min_w]
    degraded_img = degraded_img[:min_h, :min_w]

    # Optional windowing (commented out)
    # window = np.outer(signal.windows.hann(min_h), signal.windows.hann(min_w))
    # original_img *= window
    # degraded_img *= window

    pad_shape = [min_h + kernel_size, min_w + kernel_size]

    # FFTs
    F = np.fft.fft2(original_img, s=pad_shape)
    G = np.fft.fft2(degraded_img, s=pad_shape)

    # Wiener-like estimation (classic form)
    F_conj = np.conj(F)
    F_magnitude_sq = np.abs(F) ** 2
    epsilon = np.mean(F_magnitude_sq) * regularization

    H_estimate = (G * F_conj) / (F_magnitude_sq + epsilon)

    # Inverse FFT
    h_estimate = np.fft.ifft2(H_estimate)
    h_estimate = np.real(np.fft.ifftshift(h_estimate))

    # Extract kernel centered around max energy
    max_pos = np.unravel_index(np.argmax(h_estimate), h_estimate.shape)
    center_row, center_col = max_pos
    half_k = kernel_size // 2

    start_r = max(center_row - half_k, 0)
    start_c = max(center_col - half_k, 0)
    end_r = start_r + kernel_size
    end_c = start_c + kernel_size

    # Handle boundary overflow
    if end_r > h_estimate.shape[0]:
        start_r = h_estimate.shape[0] - kernel_size
        end_r = h_estimate.shape[0]
    if end_c > h_estimate.shape[1]:
        start_c = h_estimate.shape[1] - kernel_size
        end_c = h_estimate.shape[1]

    kernel = h_estimate[start_r:end_r, start_c:end_c]

    # Normalize
    kernel_sum = np.sum(kernel)
    if np.abs(kernel_sum) < 1e-10:
        print("⚠️ WARNING: Estimated kernel nearly zero — replacing with delta")
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, kernel_size // 2] = 1.0
    else:
        kernel /= kernel_sum

    print(f"✅ Kernel estimated. Sum: {np.sum(kernel):.6f}, Max: {np.max(kernel):.6f}")

    # Debug plot
    plt.figure(figsize=(4, 4))
    plt.imshow(kernel, cmap='hot')
    plt.title("Estimated Kernel")
    plt.colorbar()
    plt.show()

    return kernel

def create_synthetic_reference(degraded_img: np.ndarray, blur_kernel: np.ndarray) -> np.ndarray:
    """
    Create a synthetic sharp reference by attempting basic deblurring.
    This provides a better starting point than an unrelated reference image.
    """
    kernel_padded = np.zeros_like(degraded_img)
    k_h, k_w = blur_kernel.shape
    start_h = (degraded_img.shape[0] - k_h) // 2
    start_w = (degraded_img.shape[1] - k_w) // 2
    kernel_padded[start_h:start_h + k_h, start_w:start_w + k_w] = blur_kernel
    kernel_padded = np.fft.fftshift(kernel_padded)

    G = np.fft.fft2(degraded_img)
    H = np.fft.fft2(kernel_padded)

    H_conj = np.conj(H)
    H_magnitude_sq = np.abs(H) ** 2

    inverse_filter = H_conj / (H_magnitude_sq + 0.1 * np.mean(H_magnitude_sq))
    F_estimate = inverse_filter * G

    synthetic_ref = np.fft.ifft2(F_estimate)
    synthetic_ref = np.real(synthetic_ref)

    return np.clip(synthetic_ref, 0, 1)


def calculate_quality_metrics(original: np.ndarray, restored: np.ndarray) -> Dict:
    """Calculate image quality metrics."""
    mse = np.mean((original - restored) ** 2)
    psnr = float('inf') if mse == 0 else 10 * np.log10(1.0 / mse)

    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - restored) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

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
    """Find the best matching reference image and create synthetic sharp version."""
    best_similarity = -1
    best_index = 0
    best_reference = None

    print(f"\nComparing degraded image with {len(reference_images)} reference images...")

    for i, ref_img in enumerate(reference_images):
        ref_processed, deg_processed = preprocess_images_for_comparison(ref_img, degraded_img)

        # Compute structural similarity
        ref_grad = np.gradient(ref_processed)
        deg_grad = np.gradient(deg_processed)

        grad_similarity = np.corrcoef(ref_grad[0].flatten(), deg_grad[0].flatten())[0, 1]
        grad_similarity += np.corrcoef(ref_grad[1].flatten(), deg_grad[1].flatten())[0, 1]
        grad_similarity /= 2

        if np.isnan(grad_similarity):
            grad_similarity = 0

        print(f"  Reference {i}: gradient similarity = {grad_similarity:.4f}")

        if grad_similarity > best_similarity:
            best_similarity = grad_similarity
            best_index = i
            best_reference = ref_img

    print(f"Best match: Reference {best_index} with similarity {best_similarity:.4f}")

    # Create synthetic sharp reference from the best match
    best_ref_processed, _ = preprocess_images_for_comparison(best_reference, degraded_img)
    motion_kernel = np.array([[0, 0, 0.2, 0.6, 0.2, 0, 0]])  # Simple motion blur estimate
    synthetic_ref = create_synthetic_reference(best_ref_processed, motion_kernel)

    return best_index, best_similarity, synthetic_ref


def wiener_deconvolution(degraded: np.ndarray, kernel: np.ndarray, K: float) -> np.ndarray:
    """Apply Wiener deconvolution with proper regularization."""
    rows, cols = degraded.shape

    kernel_padded = np.zeros((rows, cols))
    k_rows, k_cols = kernel.shape
    start_row = (rows - k_rows) // 2
    start_col = (cols - k_cols) // 2
    kernel_padded[start_row:start_row + k_rows, start_col:start_col + k_cols] = kernel
    kernel_padded = np.fft.fftshift(kernel_padded)

    G = np.fft.fft2(degraded)
    H = np.fft.fft2(kernel_padded)

    H_conj = np.conj(H)
    H_squared = np.abs(H) ** 2

    signal_power = np.mean(H_squared)
    regularization = K * signal_power

    W = H_conj / (H_squared + regularization)

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

        ref_proc, deg_proc = preprocess_images_for_comparison(ref_img, degraded_img)

        # Create better synthetic reference
        motion_kernel = np.array([[0.1, 0.2, 0.4, 0.2, 0.1]])
        synthetic_ref = create_synthetic_reference(ref_proc, motion_kernel)

        estimated_kernel = estimate_kernel(synthetic_ref, deg_proc)

        best_restoration = None
        best_metrics = None
        best_psnr = -np.inf
        best_K = None

        K_values = [0.01, 0.05, 0.1, 0.5]
        for K in K_values:
            restored = wiener_deconvolution(deg_proc, estimated_kernel, K)
            metrics = calculate_quality_metrics(synthetic_ref, restored)

            if metrics['psnr'] > best_psnr:
                best_psnr = metrics['psnr']
                best_restoration = restored
                best_metrics = metrics
                best_K = K

        # Calculate similarity metrics
        ref_grad = np.gradient(ref_proc)
        deg_grad = np.gradient(deg_proc)

        spatial_similarity = np.corrcoef(ref_grad[0].flatten(), deg_grad[0].flatten())[0, 1]
        if np.isnan(spatial_similarity):
            spatial_similarity = 0

        F_ref = np.fft.fft2(ref_proc)
        F_deg = np.fft.fft2(deg_proc)
        spectral_similarity = np.mean(np.abs(F_ref) * np.abs(F_deg)) / (
                np.sqrt(np.mean(np.abs(F_ref) ** 2) * np.mean(np.abs(F_deg) ** 2)) + 1e-10)

        results[label] = {
            'estimated_kernel': estimated_kernel,
            'best_restoration': best_restoration,
            'best_metrics': best_metrics,
            'best_K': best_K,
            'similarity_metrics': {
                'spatial_correlation': spatial_similarity,
                'spectral_similarity': spectral_similarity
            }
        }

        print(f"  Best PSNR: {best_psnr:.2f} dB (K={best_K})")
        print(f"  Spatial correlation: {spatial_similarity:.4f}")
        print(f"  Spectral similarity: {spectral_similarity:.4f}")

    return results


def advanced_wiener_deconvolution(degraded_img: np.ndarray,
                                  reference_images: List[np.ndarray],
                                  reference_labels: List[str] = None,
                                  K_values: List[float] = [0.01, 0.05, 0.1, 0.5],
                                  noise_type: str = 'gaussian') -> Dict:
    """
    Perform advanced Wiener deconvolution using kernel estimation.
    """
    results = {}

    if degraded_img.max() > 1.0:
        degraded_img = degraded_img.astype(np.float64) / 255.0

    if reference_labels and len(reference_labels) == len(reference_images):
        content_analysis = analyze_content_similarity_impact(
            degraded_img, reference_images, reference_labels)
        results['content_similarity_analysis'] = content_analysis

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
        best_idx, similarity, synthetic_ref = find_best_reference_match(degraded_img, reference_images)
        results['reference_match'] = {
            'best_index': best_idx,
            'similarity': similarity,
            'reference_image': synthetic_ref,
            'selection_method': 'gradient_correlation'
        }

    ref_for_kernel = results['reference_match']['reference_image']
    if len(ref_for_kernel.shape) == 3:
        ref_for_kernel = cv2.cvtColor(ref_for_kernel, cv2.COLOR_BGR2GRAY)

    if ref_for_kernel.max() > 1.0:
        ref_for_kernel = ref_for_kernel / 255.0

    estimated_kernel = estimate_kernel(ref_for_kernel, degraded_img)
    results['estimated_kernel'] = estimated_kernel

    restored_images = {}
    quality_metrics = {}

    for K in K_values:
        restored = wiener_deconvolution(degraded_img, estimated_kernel, K)
        restored_uint8 = (restored * 255).astype(np.uint8)
        restored_images[K] = restored_uint8
        quality_metrics[K] = calculate_quality_metrics(ref_for_kernel, restored)

    results['restored_images'] = restored_images
    results['quality_metrics'] = quality_metrics

    best_K = max(K_values, key=lambda k: quality_metrics[k]['psnr'])
    results['best_K'] = best_K

    return results


def plot_results(results: Dict, degraded_img: np.ndarray):
    """Plot results of advanced Wiener filtering."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(results['reference_match']['reference_image'], cmap='gray')
    ax1.set_title('Best Reference Match')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(degraded_img, cmap='gray')
    ax2.set_title('Degraded Input')
    ax2.axis('off')

    best_K = results['best_K']
    ax3 = fig.add_subplot(gs[0, 2])
    best_restored = results['restored_images'][best_K]
    ax3.imshow(best_restored, cmap='gray')
    psnr_best = results['quality_metrics'][best_K]['psnr']
    ax3.set_title(f'Best Restoration\nK={best_K}, PSNR={psnr_best:.1f}dB')
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(results['estimated_kernel'], cmap='hot')
    ax4.set_title('Estimated Kernel')
    ax4.axis('off')

    ax5 = fig.add_subplot(gs[1, 0:2])
    K_values = list(results['quality_metrics'].keys())
    psnr_values = [results['quality_metrics'][K]['psnr'] for K in K_values]

    ax5.semilogx(K_values, psnr_values, 'bo-', linewidth=2, markersize=8)
    ax5.set_xlabel('Regularization K')
    ax5.set_ylabel('PSNR (dB)')
    ax5.set_title('PSNR vs Regularization Parameter')
    ax5.grid(True, alpha=0.3)

    best_psnr = results['quality_metrics'][best_K]['psnr']
    ax5.semilogx(best_K, best_psnr, 'ro', markersize=10, label=f'Best: K={best_K}')
    ax5.legend()

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
                           reference_labels: List[str] = None):
    """Run advanced Wiener filter restoration process."""
    print("=== ADVANCED WIENER FILTER ===")

    reference_images = load_reference_images(reference_paths)

    if reference_labels is None:
        reference_labels = [f"Reference_{i}" for i in range(len(reference_images))]

    results = advanced_wiener_deconvolution(
        degraded_img, reference_images, reference_labels,
        K_values=[0.01, 0.05, 0.1, 0.5]
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

    degraded_input = cv2.imread("simple_5px.png", cv2.IMREAD_GRAYSCALE)
    run_wiener_restoration(degraded_input, reference_paths, reference_labels)



if __name__ == "__main__":
    main()