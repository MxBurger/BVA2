import numpy as np
import cv2
from typing import List, Tuple


def find_best_reference_match(degraded_img: np.ndarray, reference_images: List[np.ndarray]) -> Tuple[
    int, float, np.ndarray]:
    """Find the best matching reference image for a degraded image using multiple similarity metrics."""
    best_similarity = -1
    best_index = 0
    best_reference = None

    # Normalize degraded image
    if degraded_img.max() > 1.0:
        degraded_img = degraded_img.astype(np.float64) / 255.0
    else:
        degraded_img = degraded_img.astype(np.float64)

    for i, ref_img in enumerate(reference_images):
        # Normalize reference image
        if ref_img.max() > 1.0:
            ref_processed = ref_img.astype(np.float64) / 255.0
        else:
            ref_processed = ref_img.astype(np.float64)

        # Ensure same dimensions (assuming they are same size as mentioned)
        if ref_processed.shape != degraded_img.shape:
            ref_processed = cv2.resize(ref_processed, (degraded_img.shape[1], degraded_img.shape[0]))

        # 1. Histogram correlation
        hist_ref = cv2.calcHist([ref_processed.astype(np.float32)], [0], None, [256], [0, 1])
        hist_deg = cv2.calcHist([degraded_img.astype(np.float32)], [0], None, [256], [0, 1])
        hist_corr = cv2.compareHist(hist_ref, hist_deg, cv2.HISTCMP_CORREL)

        # 2. Structural similarity (simplified SSIM)
        ssim_score = structural_similarity_simple(ref_processed, degraded_img)

        # 3. Edge correlation
        edges_ref = cv2.Canny((ref_processed * 255).astype(np.uint8), 50, 150)
        edges_deg = cv2.Canny((degraded_img * 255).astype(np.uint8), 50, 150)

        # Normalize edge maps
        edges_ref = edges_ref.astype(np.float64) / 255.0
        edges_deg = edges_deg.astype(np.float64) / 255.0

        edge_corr = np.corrcoef(edges_ref.flatten(), edges_deg.flatten())[0, 1]
        if np.isnan(edge_corr):
            edge_corr = 0

        # 4. Texture similarity using Local Binary Patterns
        texture_corr = texture_similarity(ref_processed, degraded_img)

        # 5. Frequency domain correlation
        freq_corr = frequency_correlation(ref_processed, degraded_img)

        # Combined similarity score with weights
        combined_score = (0.25 * hist_corr +
                          0.30 * ssim_score +
                          0.20 * edge_corr +
                          0.15 * texture_corr +
                          0.10 * freq_corr)

        if combined_score > best_similarity:
            best_similarity = combined_score
            best_index = i
            best_reference = ref_processed

    return best_index, best_similarity, best_reference


def structural_similarity_simple(img1: np.ndarray, img2: np.ndarray) -> float:
    """Simplified SSIM calculation"""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / (denominator + 1e-10)
    return np.mean(ssim_map)


def texture_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate texture similarity using statistical moments"""
    # Convert to uint8 for texture analysis
    img1_uint8 = (img1 * 255).astype(np.uint8)
    img2_uint8 = (img2 * 255).astype(np.uint8)

    # Calculate local variance as texture measure
    kernel = np.ones((9, 9), np.float32) / 81

    mean1 = cv2.filter2D(img1_uint8.astype(np.float32), -1, kernel)
    mean2 = cv2.filter2D(img2_uint8.astype(np.float32), -1, kernel)

    sqr1 = cv2.filter2D((img1_uint8.astype(np.float32)) ** 2, -1, kernel)
    sqr2 = cv2.filter2D((img2_uint8.astype(np.float32)) ** 2, -1, kernel)

    var1 = sqr1 - mean1 ** 2
    var2 = sqr2 - mean2 ** 2

    # Correlation between variance maps
    corr = np.corrcoef(var1.flatten(), var2.flatten())[0, 1]
    return 0 if np.isnan(corr) else corr


def frequency_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate correlation in frequency domain"""
    # FFT magnitude spectra
    fft1 = np.fft.fft2(img1)
    fft2 = np.fft.fft2(img2)

    mag1 = np.abs(fft1)
    mag2 = np.abs(fft2)

    # Log transform to compress dynamic range
    log_mag1 = np.log(mag1 + 1)
    log_mag2 = np.log(mag2 + 1)

    # Correlation of log magnitudes
    corr = np.corrcoef(log_mag1.flatten(), log_mag2.flatten())[0, 1]
    return 0 if np.isnan(corr) else corr