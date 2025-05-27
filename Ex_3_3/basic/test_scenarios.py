import numpy as np
from typing import List, Dict
from wiener_filter import (
    create_kernels, degrade_image, wiener_deconvolution, 
    wiener_with_butterworth, calculate_psnr
)
from visualization import (
    plot_kernel_comparison, plot_noise_level_analysis, 
    plot_butterworth_comparison, plot_frequency_response
)


def test_kernel_variations(img: np.ndarray, kernel_type: str, kernel_sizes: List[int], 
                          noise_variance: float, K_values: List[float]):
    """
    Test Wiener filter performance with different kernel sizes for a specific kernel type.
    """
    print(f"\n=== Testing {kernel_type} kernel with different radii ===")
    
    kernels_dict = {}
    degraded_images = {}
    restored_images = {}
    
    # Generate data for all kernel sizes
    for kernel_size in kernel_sizes:
        print(f"\nKernel size: {kernel_size}")
        
        # Create kernels for this size
        kernels = create_kernels(kernel_size)
        kernel = kernels[kernel_type]
        kernels_dict[kernel_size] = kernel
        
        # Degrade image
        degraded = degrade_image(img, kernel, noise_variance)
        degraded_images[kernel_size] = degraded
        
        # Test different K values
        restored_images[kernel_size] = {}
        for K in K_values:
            restored = wiener_deconvolution(degraded, kernel, K)
            restored_images[kernel_size][K] = restored
            psnr = calculate_psnr(img, restored)
            print(f"  K={K}: PSNR={psnr:.2f} dB")
    
    # Plot results
    plot_kernel_comparison(img, kernel_type, kernel_sizes, kernels_dict, 
                          degraded_images, restored_images, K_values)


def test_noise_levels(img: np.ndarray, kernel_type: str, kernel_size: int, 
                     noise_variances: List[float], K_values: List[float]):
    """
    Test different noise levels with a specific kernel radius.
    """
    kernels = create_kernels(kernel_size)
    kernel = kernels[kernel_type]
    
    print(f"\n--- Noise level analysis for {kernel_type} kernel (radius={kernel_size}) ---")
    
    degraded_images = []
    restored_images = {K: [] for K in K_values}
    
    for noise_var in noise_variances:
        # Degrade image
        degraded = degrade_image(img, kernel, noise_var)
        degraded_images.append(degraded)
        
        # Test different K values
        for K in K_values:
            restored = wiener_deconvolution(degraded, kernel, K)
            restored_images[K].append(restored)
            psnr = calculate_psnr(img, restored)
            print(f"  Noise var={noise_var}, K={K}: PSNR={psnr:.2f} dB")
    
    # Plot results
    plot_noise_level_analysis(img, kernel_type, kernel_size, noise_variances, 
                             degraded_images, restored_images, K_values)


def test_butterworth_enhancement(img: np.ndarray, kernel: np.ndarray, 
                                noise_var: float, kernel_name: str = "Unknown"):
    """
    Test Wiener filter with and without Butterworth filtering.
    """
    degraded = degrade_image(img, kernel, noise_var)
    
    # Wiener without Butterworth
    K = 0.01
    restored_no_butter = wiener_deconvolution(degraded, kernel, K)
    
    # Wiener with Butterworth
    restored_butter = wiener_with_butterworth(degraded, kernel, K, cutoff=0.3)
    
    # Plot comparison
    plot_butterworth_comparison(img, degraded, restored_no_butter, 
                               restored_butter, kernel_name)


def run_tests(img: np.ndarray):
    """
    Run all tests for the Wiener filter.
    """
    # Test parameters
    kernel_sizes = [5, 15, 25, 35]
    noise_variances = [10, 50, 250]
    K_values = [0.0001, 0.001, 0.01, 0.1]
    kernel_types = ['mean', 'gaussian', 'horizontal', 'vertical', 'diagonal']
    
    # Test different kernels with varying radii
    for kernel_type in kernel_types:
        test_kernel_variations(img, kernel_type, kernel_sizes, 
                              noise_variances[1], K_values)  # Use medium noise level
        
        # Test noise levels for medium kernel size
        test_noise_levels(img, kernel_type, kernel_sizes[1], 
                         noise_variances, K_values[:3])  # Use first 3 K values
        
        # Test Butterworth enhancement
        kernels = create_kernels(kernel_sizes[1])
        test_butterworth_enhancement(img, kernels[kernel_type], noise_variances[1],
                                    f"{kernel_type} (radius={kernel_sizes[1]})")


def run_frequency_analysis(kernel_sizes: List[int], K_values: List[float]):
    """
    Run frequency response analysis for different kernel sizes.
    """
    print("\n=== Frequency Response Analysis ===")
    
    for size in kernel_sizes:
        kernels = create_kernels(size)
        for kernel_name in ['mean', 'gaussian']:
            print(f"\nFrequency response for {kernel_name} kernel (radius={size}):")
            plot_frequency_response(kernels[kernel_name], K_values)


def evaluate_restoration_quality(original: np.ndarray, degraded: np.ndarray, 
                                restored: np.ndarray) -> Dict[str, float]:
    """
    Evaluate restoration quality using multiple metrics.
    """
    metrics = {}
    
    # PSNR
    metrics['psnr'] = calculate_psnr(original, restored)
    
    # MSE
    mse = np.mean((original.astype(float) - restored.astype(float)) ** 2)
    metrics['mse'] = mse
    
    # Improvement over degraded image
    degraded_psnr = calculate_psnr(original, degraded)
    metrics['psnr_improvement'] = metrics['psnr'] - degraded_psnr
    
    return metrics