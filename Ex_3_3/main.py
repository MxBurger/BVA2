import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple

from Samples import create_synthetic_image


class WienerFilter:
    """
    Implementation of Wiener Filter for image restoration.
    This class handles image degradation and restoration using the Wiener filter approach.
    """

    def __init__(self):
        self.degraded_image = None
        self.kernel = None
        self.noise_level = None

    def create_kernels(self, kernel_size: int) -> dict:
        """
        Create various blur kernels for testing.

        Args:
            kernel_size: Size of the kernel (must be odd)

        Returns:
            Dictionary containing different kernels
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

    def add_noise(self, image: np.ndarray, noise_variance: float) -> np.ndarray:
        """
        Add Gaussian noise to an image.

        Args:
            image: Input image
            noise_variance: Variance of the Gaussian noise

        Returns:
            Noisy image
        """
        noise = np.random.normal(0, np.sqrt(noise_variance), image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def degrade_image(self, image: np.ndarray, kernel: np.ndarray,
                      noise_variance: float) -> np.ndarray:
        """
        Degrade an image by applying convolution and adding noise.
        g(x,y) = f(x,y) * h(x,y) + n(x,y)

        Args:
            image: Original image
            kernel: Blur kernel
            noise_variance: Variance of additive Gaussian noise

        Returns:
            Degraded image
        """
        # Apply convolution in spatial domain
        blurred = cv2.filter2D(image.astype(np.float64), -1, kernel)

        # Add noise
        degraded = self.add_noise(blurred, noise_variance)

        return degraded

    def pad_to_shape(self, array: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Pad array to target shape for FFT operations.
        """
        pad_h = target_shape[0] - array.shape[0]
        pad_w = target_shape[1] - array.shape[1]

        return np.pad(array, ((0, pad_h), (0, pad_w)), mode='constant')

    def wiener_deconvolution(self, degraded: np.ndarray, kernel: np.ndarray,
                             K: float) -> np.ndarray:
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
        kernel_padded = self.pad_to_shape(kernel, (rows, cols))

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

    def butterworth_filter(self, shape: Tuple[int, int], cutoff: float,
                           order: int = 2) -> np.ndarray:
        """
        Create a Butterworth low-pass filter for noise suppression.

        Args:
            shape: Shape of the filter
            cutoff: Cutoff frequency (0 to 1)
            order: Order of the filter

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

    def wiener_with_butterworth(self, degraded: np.ndarray, kernel: np.ndarray,
                                K: float, cutoff: float = 0.5) -> np.ndarray:
        """
        Wiener deconvolution with Butterworth filtering for noise suppression.
        """
        rows, cols = degraded.shape
        kernel_padded = self.pad_to_shape(kernel, (rows, cols))

        # Compute FFTs
        G = np.fft.fft2(degraded.astype(np.float64))
        H = np.fft.fft2(kernel_padded)

        # Wiener filter
        H_conj = np.conj(H)
        H_squared = np.abs(H) ** 2
        W = H_conj / (H_squared + K + 1e-10)

        # Apply Butterworth filter to suppress high-frequency noise
        B = self.butterworth_filter((rows, cols), cutoff)
        W = W * B

        # Restore image
        F_estimate = W * G
        restored = np.fft.ifft2(F_estimate)
        restored = np.real(restored)

        return np.clip(restored, 0, 255).astype(np.uint8)

    def calculate_psnr(self, original: np.ndarray, restored: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR) between original and restored image.

        Args:
            original: Original image
            restored: Restored image

        Returns:
            PSNR value in dB
        """
        mse = np.mean((original.astype(float) - restored.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10(255 ** 2 / mse)

    def demonstrate_restoration(self, image_path: str):
        """
        Demonstrate the complete Wiener filter restoration process with varying kernel radii.
        """
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Create synthetic image if file not found
            img = create_synthetic_image()

        # Test different kernel sizes (radii)
        kernel_sizes = [5, 15, 25, 35]
        noise_variances = [10, 50, 250]
        K_values = [0.0001, 0.001, 0.01, 0.1]

        # Test with different kernels AND different radii
        kernel_types = ['mean', 'gaussian', 'horizontal', 'vertical', 'diagonal']

        for kernel_type in kernel_types:
            print(f"\n=== Testing {kernel_type} kernel with different radii ===")

            # Create a figure showing all kernel sizes
            fig, axes = plt.subplots(len(kernel_sizes), len(K_values) + 2, figsize=(20, 16))
            fig.suptitle(f'Wiener Filter - {kernel_type} kernel (varying radii)', fontsize=16)

            for size_idx, kernel_size in enumerate(kernel_sizes):
                print(f"\nKernel size: {kernel_size}")

                # Create kernels for this size
                kernels = self.create_kernels(kernel_size)
                kernel = kernels[kernel_type]

                # Use medium noise level for this comparison
                noise_var = noise_variances[1]  # noise_var = 50

                # Degrade image
                degraded = self.degrade_image(img, kernel, noise_var)
                degraded_psnr = self.calculate_psnr(img, degraded)

                # Show original and degraded
                axes[size_idx, 0].imshow(img, cmap='gray')
                axes[size_idx, 0].set_title(f'Original\n(Radius={kernel_size})')
                axes[size_idx, 0].axis('off')

                axes[size_idx, 1].imshow(degraded, cmap='gray')
                axes[size_idx, 1].set_title(f'Degraded\nRadius={kernel_size}\nPSNR={degraded_psnr:.1f} dB')
                axes[size_idx, 1].axis('off')

                # Test different K values
                for k_idx, K in enumerate(K_values):
                    restored = self.wiener_deconvolution(degraded, kernel, K)
                    psnr = self.calculate_psnr(img, restored)

                    axes[size_idx, k_idx + 2].imshow(restored, cmap='gray')
                    axes[size_idx, k_idx + 2].set_title(f'K={K}\nPSNR={psnr:.1f} dB')
                    axes[size_idx, k_idx + 2].axis('off')

                    print(f"  K={K}: PSNR={psnr:.2f} dB")

            plt.tight_layout()
            plt.show()

            # Test effect of different noise levels for medium kernel size
            self.test_noise_levels_with_radius(img, kernel_types, kernel_sizes[1])  # Use kernel_size=15

            # Test Butterworth enhancement for this kernel type
            self.test_butterworth_enhancement(img, kernels[kernel_type], noise_variances[1],
                                              f"{kernel_type} (radius={kernel_sizes[1]})")

    def test_noise_levels_with_radius(self, img: np.ndarray, kernel_types: list, kernel_size: int):
        """
        Test different noise levels with a specific kernel radius.
        """
        noise_variances = [10, 50, 250]
        K_values = [0.001, 0.01, 0.1]

        for kernel_type in kernel_types:  # Test only first 2 kernel types to avoid too many plots
            kernels = self.create_kernels(kernel_size)
            kernel = kernels[kernel_type]

            print(f"\n--- Noise level analysis for {kernel_type} kernel (radius={kernel_size}) ---")

            fig, axes = plt.subplots(len(noise_variances), len(K_values) + 2, figsize=(16, 12))
            fig.suptitle(f'Noise Level Analysis - {kernel_type} kernel (radius={kernel_size})', fontsize=14)

            for i, noise_var in enumerate(noise_variances):
                # Degrade image
                degraded = self.degrade_image(img, kernel, noise_var)
                degraded_psnr = self.calculate_psnr(img, degraded)

                # Show original and degraded
                axes[i, 0].imshow(img, cmap='gray')
                axes[i, 0].set_title('Original')
                axes[i, 0].axis('off')

                axes[i, 1].imshow(degraded, cmap='gray')
                axes[i, 1].set_title(f'Degraded\nNoise var={noise_var}\nPSNR={degraded_psnr:.1f} dB')
                axes[i, 1].axis('off')

                # Test different K values
                for j, K in enumerate(K_values):
                    restored = self.wiener_deconvolution(degraded, kernel, K)
                    psnr = self.calculate_psnr(img, restored)

                    axes[i, j + 2].imshow(restored, cmap='gray')
                    axes[i, j + 2].set_title(f'K={K}\nPSNR={psnr:.1f} dB')
                    axes[i, j + 2].axis('off')

                    print(f"  Noise var={noise_var}, K={K}: PSNR={psnr:.2f} dB")

            plt.tight_layout()
            plt.show()

    def test_butterworth_enhancement(self, img: np.ndarray, kernel: np.ndarray,
                                     noise_var: float, kernel_name: str = "Unknown"):
        """
        Test Wiener filter with and without Butterworth filtering.

        Args:
            img: Original image
            kernel: Blur kernel
            noise_var: Noise variance
            kernel_name: Name of the kernel for display purposes
        """
        degraded = self.degrade_image(img, kernel, noise_var)
        degraded_psnr = self.calculate_psnr(img, degraded)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f'Wiener Filter with Butterworth Enhancement - {kernel_name}', fontsize=14)

        # Original and degraded
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(degraded, cmap='gray')
        axes[1].set_title(f'Degraded\nPSNR={degraded_psnr:.1f} dB')
        axes[1].axis('off')

        # Wiener without Butterworth
        K = 0.01
        restored_no_butter = self.wiener_deconvolution(degraded, kernel, K)
        psnr_no_butter = self.calculate_psnr(img, restored_no_butter)
        axes[2].imshow(restored_no_butter, cmap='gray')
        axes[2].set_title(f'Wiener (no Butterworth)\nPSNR={psnr_no_butter:.1f} dB')
        axes[2].axis('off')

        # Wiener with Butterworth
        restored_butter = self.wiener_with_butterworth(degraded, kernel, K, cutoff=0.3)
        psnr_butter = self.calculate_psnr(img, restored_butter)
        axes[3].imshow(restored_butter, cmap='gray')
        axes[3].set_title(f'Wiener + Butterworth\nPSNR={psnr_butter:.1f} dB')
        axes[3].axis('off')

        plt.tight_layout()
        plt.show()

        print(f"PSNR Comparison for {kernel_name}:")
        print(f"  Degraded: {degraded_psnr:.2f} dB")
        print(f"  Wiener only: {psnr_no_butter:.2f} dB")
        print(f"  Wiener + Butterworth: {psnr_butter:.2f} dB")

    def visualize_frequency_response(self, kernel: np.ndarray, K_values: list):
        """
        Visualize the frequency response of the Wiener filter.
        """
        # Pad kernel to a reasonable size for visualization
        size = 256
        kernel_padded = self.pad_to_shape(kernel, (size, size))

        # Compute frequency response
        H = np.fft.fft2(kernel_padded)
        H_mag = np.abs(H)

        fig, axes = plt.subplots(2, len(K_values) + 1, figsize=(20, 8))
        fig.suptitle('Wiener Filter Frequency Response', fontsize=14)

        # Show kernel frequency response
        axes[0, 0].imshow(np.log(H_mag + 1), cmap='gray')
        axes[0, 0].set_title('log|H(u,v)|')
        axes[0, 0].axis('off')

        axes[1, 0].imshow(kernel, cmap='gray')
        axes[1, 0].set_title('Kernel h(x,y)')
        axes[1, 0].axis('off')

        # Show Wiener filter for different K values
        for i, K in enumerate(K_values):
            H_conj = np.conj(H)
            H_squared = np.abs(H) ** 2
            W = H_conj / (H_squared + K + 1e-10)
            W_mag = np.abs(W)

            axes[0, i + 1].imshow(np.log(W_mag + 1), cmap='gray')
            axes[0, i + 1].set_title(f'log|W(u,v)| K={K}')
            axes[0, i + 1].axis('off')

            # Show inverse FFT of W (impulse response)
            w_spatial = np.fft.ifft2(W)
            w_spatial = np.fft.ifftshift(np.real(w_spatial))
            axes[1, i + 1].imshow(w_spatial, cmap='gray')
            axes[1, i + 1].set_title(f'w(x,y) K={K}')
            axes[1, i + 1].axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Create Wiener filter instance
    wf = WienerFilter()

    image_path = ""

    # Main demonstration with varying kernel radii
    wf.demonstrate_restoration(image_path)

    # Visualize frequency responses for different kernel sizes
    print("\n=== Frequency Response Analysis ===")
    kernel_sizes = [5, 15, 25]
    K_values = [0.0001, 0.001, 0.01, 0.1]

    for size in kernel_sizes:
        kernels = wf.create_kernels(size)
        for kernel_name in ['mean', 'gaussian']:
            print(f"\nFrequency response for {kernel_name} kernel (radius={size}):")
            wf.visualize_frequency_response(kernels[kernel_name], K_values)