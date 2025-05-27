import numpy as np
import cv2
from typing import List, Dict
from filters import estimate_kernel, wiener_deconvolution, create_synthetic_reference
from image_utils import preprocess_images_for_comparison, find_best_reference_match, calculate_quality_metrics


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