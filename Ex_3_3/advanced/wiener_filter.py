import numpy as np
import cv2
from typing import List, Dict
from image_utils import preprocess_images_for_comparison, find_best_reference_match, calculate_quality_metrics


def estimate_kernel(original_img: np.ndarray, degraded_img: np.ndarray,
                    kernel_size: int = 15, regularization: float = 0.01) -> np.ndarray:
    """Estimate degradation kernel using stable Wiener formulation."""
    min_h = min(original_img.shape[0], degraded_img.shape[0])
    min_w = min(original_img.shape[1], degraded_img.shape[1])
    original_img = original_img[:min_h, :min_w]
    degraded_img = degraded_img[:min_h, :min_w]

    pad_shape = [min_h + kernel_size, min_w + kernel_size]

    F = np.fft.fft2(original_img, s=pad_shape)
    G = np.fft.fft2(degraded_img, s=pad_shape)

    F_conj = np.conj(F)
    F_magnitude_sq = np.abs(F) ** 2
    epsilon = np.mean(F_magnitude_sq) * regularization

    H_estimate = (G * F_conj) / (F_magnitude_sq + epsilon)

    h_estimate = np.fft.ifft2(H_estimate)
    h_estimate = np.real(np.fft.ifftshift(h_estimate))

    max_pos = np.unravel_index(np.argmax(h_estimate), h_estimate.shape)
    center_row, center_col = max_pos
    half_k = kernel_size // 2

    start_r = max(center_row - half_k, 0)
    start_c = max(center_col - half_k, 0)
    end_r = start_r + kernel_size
    end_c = start_c + kernel_size

    if end_r > h_estimate.shape[0]:
        start_r = h_estimate.shape[0] - kernel_size
        end_r = h_estimate.shape[0]
    if end_c > h_estimate.shape[1]:
        start_c = h_estimate.shape[1] - kernel_size
        end_c = h_estimate.shape[1]

    kernel = h_estimate[start_r:end_r, start_c:end_c]

    kernel_sum = np.sum(kernel)
    if np.abs(kernel_sum) < 1e-10:
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, kernel_size // 2] = 1.0
    else:
        kernel /= kernel_sum

    return kernel


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


def create_synthetic_reference(degraded_img: np.ndarray, blur_kernel: np.ndarray) -> np.ndarray:
    """Create a synthetic sharp reference by attempting basic deblurring."""
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
    """Perform advanced Wiener deconvolution using kernel estimation."""
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


