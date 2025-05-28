import numpy as np
import cv2
from typing import List, Tuple, Dict


def structural_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    min_h = min(img1.shape[0], img2.shape[0])
    min_w = min(img1.shape[1], img2.shape[1])
    img1 = img1[:min_h, :min_w]
    img2 = img2[:min_h, :min_w]

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu1 = cv2.GaussianBlur(img1, (7, 7), 1.5)
    mu2 = cv2.GaussianBlur(img2, (7, 7), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (7, 7), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (7, 7), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (7, 7), 1.5) - mu1_mu2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / (denominator + 1e-10)
    return np.mean(ssim_map)


def load_reference_images(reference_paths: List[str]) -> List[np.ndarray]:
    reference_images = []
    for path in reference_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        reference_images.append(img.astype(np.float64))
    return reference_images


def preprocess_images_for_comparison(ref_img: np.ndarray, degraded_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    target_size = (256, 256)
    ref_resized = cv2.resize(ref_img, target_size)
    deg_resized = cv2.resize(degraded_img, target_size)
    return ref_resized / 255.0, deg_resized / 255.0

def calculate_quality_metrics(original: np.ndarray, restored: np.ndarray) -> Dict:
    mse = np.mean((original - restored) ** 2)
    psnr = float('inf') if mse == 0 else 10 * np.log10(1.0 / mse)

    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - restored) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

    grad_orig = np.concatenate([np.gradient(original, axis=1).flatten(), np.gradient(original, axis=0).flatten()])
    grad_rest = np.concatenate([np.gradient(restored, axis=1).flatten(), np.gradient(restored, axis=0).flatten()])

    edge_preservation = 0.0
    if np.std(grad_orig) > 1e-10 and np.std(grad_rest) > 1e-10:
        edge_corr = np.corrcoef(grad_orig, grad_rest)[0, 1]
        edge_preservation = edge_corr if not np.isnan(edge_corr) else 0.0

    return {'psnr': psnr, 'mse': mse, 'snr': snr, 'edge_preservation': edge_preservation}


