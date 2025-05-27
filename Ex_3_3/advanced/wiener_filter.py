import numpy as np
import cv2
from typing import List, Dict

from image_utilities import find_best_reference_match, calculate_quality_metrics


def estimate_kernel_size(degraded_img: np.ndarray) -> int:
    min_dim = min(degraded_img.shape[:2])
    size = max(5, min(25, min_dim // 15))
    return size if size % 2 == 1 else size + 1


def estimate_kernel(original_img: np.ndarray, degraded_img: np.ndarray, kernel_size: int = None) -> np.ndarray:
    if kernel_size is None:
        kernel_size = estimate_kernel_size(degraded_img)

    min_h = min(original_img.shape[0], degraded_img.shape[0])
    min_w = min(original_img.shape[1], degraded_img.shape[1])
    original_img = original_img[:min_h, :min_w]
    degraded_img = degraded_img[:min_h, :min_w]

    pad_shape = [min_h + kernel_size, min_w + kernel_size]

    F = np.fft.fft2(original_img, s=pad_shape)
    G = np.fft.fft2(degraded_img, s=pad_shape)

    F_conj = np.conj(F)
    F_magnitude_sq = np.abs(F) ** 2
    epsilon = np.mean(F_magnitude_sq) * 0.01

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


def advanced_wiener_deconvolution(degraded_img: np.ndarray, reference_images: List[np.ndarray],
                                  reference_labels: List[str] = None,
                                  K_values: List[float] = [0.01, 0.05, 0.1, 0.5]) -> Dict:
    results = {}

    if degraded_img.max() > 1.0:
        degraded_img = degraded_img.astype(np.float64) / 255.0

    best_idx, similarity, reference = find_best_reference_match(degraded_img, reference_images)
    results['reference_match'] = {
        'best_index': best_idx,
        'similarity': similarity,
        'reference_image': reference
    }

    ref_for_kernel = reference
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