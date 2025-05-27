from typing import Dict
import cv2

import numpy as np


def create_kernels(kernel_size: int) -> Dict[str, np.ndarray]:
    kernels = {}

    kernels['mean'] = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)

    sigma = kernel_size / 3.0
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernels['gaussian'] = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernels['gaussian'] /= kernels['gaussian'].sum()

    kernels['horizontal'] = np.zeros((kernel_size, kernel_size))
    kernels['horizontal'][kernel_size // 2, :] = 1.0 / kernel_size

    kernels['vertical'] = np.zeros((kernel_size, kernel_size))
    kernels['vertical'][:, kernel_size // 2] = 1.0 / kernel_size

    kernels['diagonal'] = np.zeros((kernel_size, kernel_size))
    np.fill_diagonal(kernels['diagonal'], 1.0 / kernel_size)

    return kernels


def add_noise(image: np.ndarray, noise_variance: float) -> np.ndarray:
    noise = np.random.normal(0, np.sqrt(noise_variance), image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def degrade_image(image: np.ndarray, kernel: np.ndarray, noise_variance: float = 0.0) -> np.ndarray:
    blurred = cv2.filter2D(image.astype(np.float64), -1, kernel)
    degraded = add_noise(blurred, noise_variance)
    return degraded