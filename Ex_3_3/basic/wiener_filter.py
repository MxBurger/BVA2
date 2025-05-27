import numpy as np
import cv2
from typing import Tuple, Dict


def create_kernels(kernel_size: int) -> Dict[str, np.ndarray]:
    """
    Create various blur kernels for testing.
    """
    kernels = {}

    # Mean filter (box filter)
    kernels['mean'] = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)

    # Gaussian filter
    sigma = kernel_size / 3.0
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernels['gaussian'] = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernels['gaussian'] /= kernels['gaussian'].sum()

    # Horizontal motion blur
    kernels['horizontal'] = np.zeros((kernel_size, kernel_size))
    kernels['horizontal'][kernel_size // 2, :] = 1.0 / kernel_size

    # Vertical motion blur
    kernels['vertical'] = np.zeros((kernel_size, kernel_size))
    kernels['vertical'][:, kernel_size // 2] = 1.0 / kernel_size

    # Diagonal motion blur
    kernels['diagonal'] = np.zeros((kernel_size, kernel_size))
    np.fill_diagonal(kernels['diagonal'], 1.0 / kernel_size)

    return kernels

def add_noise(image: np.ndarray, noise_variance: float) -> np.ndarray:
    """
    Add Gaussian noise to an image.

    Returns:
        Noisy image
    """
    noise = np.random.normal(0, np.sqrt(noise_variance), image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def degrade_image(image: np.ndarray, kernel: np.ndarray, noise_variance: float) -> np.ndarray:
    """
    Degrade an image by applying convolution and adding noise.
    g(x,y) = f(x,y) * h(x,y) + n(x,y)

    Returns:
        Degraded image
    """
    # Apply convolution in spatial domain
    blurred = cv2.filter2D(image.astype(np.float64), -1, kernel)
    
    # Add noise
    degraded = add_noise(blurred, noise_variance)
    
    return degraded

def pad_to_shape(array: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Pad array to target shape for FFT operations.
    """
    pad_h = target_shape[0] - array.shape[0]
    pad_w = target_shape[1] - array.shape[1]
    
    return np.pad(array, ((0, pad_h), (0, pad_w)), mode='constant')

def wiener_deconvolution(degraded: np.ndarray, kernel: np.ndarray, K: float) -> np.ndarray:
    """
    Perform Wiener deconvolution to restore the image.

    Formula: F'(u,v) = [H*(u,v) / (|H(u,v)|² + K)] * G(u,v)

    Args:
        degraded: Degraded image G(x,y)
        kernel: Degradation kernel h(x,y)
        K: Noise-to-signal power ratio (regularization parameter)

    Returns:
        Restored image
    """
    # Ensure image and kernel have the same dimensions for FFT
    rows, cols = degraded.shape
    kernel_padded = pad_to_shape(kernel, (rows, cols))

    # Step 1: Compute FFT of degraded image and kernel
    G = np.fft.fft2(degraded.astype(np.float64))
    H = np.fft.fft2(kernel_padded)

    # Step 2: Compute H* (complex conjugate of H)
    H_conj = np.conj(H)

    # Step 3: Compute |H|² = H * H*
    H_squared = np.abs(H) ** 2

    # Step 4: Apply Wiener filter formula
    # W(u,v) = H*(u,v) / (|H(u,v)|² + K)
    # F'(u,v) = W(u,v) * G(u,v)

    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    W = H_conj / (H_squared + K + epsilon)

    # Apply Wiener filter
    F_estimate = W * G

    # Step 5: Inverse FFT to get restored image
    restored = np.fft.ifft2(F_estimate)
    restored = np.real(restored)

    # Ensure output is in valid range
    restored = np.clip(restored, 0, 255)

    return restored.astype(np.uint8)


def butterworth_filter(shape: Tuple[int, int], cutoff: float, order: int = 2) -> np.ndarray:
    """
    Create a Butterworth low-pass filter for noise suppression.

    Returns:
        Butterworth filter in frequency domain
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    # Create coordinate arrays
    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol

    # Calculate distance from center
    D = np.sqrt(u ** 2 + v ** 2)

    # Normalize by image dimensions
    D0 = cutoff * min(rows, cols) / 2

    # Butterworth filter formula: 1 / (1 + (D/D0)^(2n))
    H = 1 / (1 + (D / D0) ** (2 * order))

    # Shift zero frequency to corners for FFT
    return np.fft.ifftshift(H)


def wiener_with_butterworth(degraded: np.ndarray, kernel: np.ndarray, 
                           K: float, cutoff: float = 0.5) -> np.ndarray:
    """
    Wiener deconvolution with Butterworth filtering for noise suppression.
    """
    rows, cols = degraded.shape
    kernel_padded = pad_to_shape(kernel, (rows, cols))

    # Compute FFTs
    G = np.fft.fft2(degraded.astype(np.float64))
    H = np.fft.fft2(kernel_padded)

    # Wiener filter
    H_conj = np.conj(H)
    H_squared = np.abs(H) ** 2
    W = H_conj / (H_squared + K + 1e-10)

    # Apply Butterworth filter to suppress high-frequency noise
    B = butterworth_filter((rows, cols), cutoff)
    W = W * B

    # Restore image
    F_estimate = W * G
    restored = np.fft.ifft2(F_estimate)
    restored = np.real(restored)

    return np.clip(restored, 0, 255).astype(np.uint8)


def calculate_psnr(original: np.ndarray, restored: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between original and restored image.

    Returns:
        PSNR value in dB
    """
    mse = np.mean((original.astype(float) - restored.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255 ** 2 / mse)