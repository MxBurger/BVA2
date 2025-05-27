import numpy as np
import cv2
from typing import List, Tuple, Dict


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


def find_best_reference_match(degraded_img: np.ndarray,
                              reference_images: List[np.ndarray]) -> Tuple[int, float, np.ndarray]:
    """Find the best matching reference image and create synthetic sharp version."""
    best_similarity = -1
    best_index = 0
    best_reference = None

    print(f"\nComparing degraded image with {len(reference_images)} reference images...")

    for i, ref_img in enumerate(reference_images):
        ref_processed, deg_processed = preprocess_images_for_comparison(ref_img, degraded_img)

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

    best_ref_processed, _ = preprocess_images_for_comparison(best_reference, degraded_img)
    motion_kernel = np.array([[0, 0, 0.2, 0.6, 0.2, 0, 0]])
    synthetic_ref = create_synthetic_reference(best_ref_processed, motion_kernel)

    return best_index, best_similarity, synthetic_ref


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