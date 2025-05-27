import numpy as np
import cv2
from typing import Tuple


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