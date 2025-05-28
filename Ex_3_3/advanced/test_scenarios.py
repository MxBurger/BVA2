import cv2
from typing import List

from degradation import create_kernels, degrade_image
from image_utilities import load_reference_images
from visualization import plot_results, print_summary
from wiener_filter import advanced_wiener_deconvolution


def run_wiener_restoration(degraded_img, reference_paths: List[str], reference_labels: List[str] = None,
                           true_kernel=None, kernel_info=None):
    reference_images = load_reference_images(reference_paths)

    if reference_labels is None:
        reference_labels = [f"Reference_{i}" for i in range(len(reference_images))]

    results = advanced_wiener_deconvolution(degraded_img, reference_images, reference_labels,
                                            K_values=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.5])

    plot_results(results, degraded_img, true_kernel, kernel_info)
    print_summary(results)


def test_different_kernels():
    reference_paths = ["ref/landscape_gray.png", "ref/lena_gray.tif", "ref/mandril_gray.tif",
                       "ref/peppers_gray.tif", "ref/text.png", "ref/boat.png"]
    reference_labels = ["landscape", "portrait", "animal", "food", "text", "boat"]

    clean_input = cv2.imread("simple.png", cv2.IMREAD_GRAYSCALE)

    kernel_sizes = [5, 15, 25]
    for kernel_size in kernel_sizes:
        kernels = create_kernels(kernel_size)

        for kernel_name, kernel in kernels.items():
            degraded_input = degrade_image(clean_input, kernel)
            kernel_info = f"{kernel_name.capitalize()} {kernel_size}x{kernel_size}"
            run_wiener_restoration(degraded_input, reference_paths, reference_labels, kernel, kernel_info)


def test_noise_levels():
    reference_paths = ["ref/landscape_gray.png", "ref/lena_gray.tif", "ref/mandril_gray.tif",
                       "ref/peppers_gray.tif", "ref/text.png", "ref/boat.png"]
    reference_labels = ["landscape", "portrait", "animal", "food", "text", "boat"]

    clean_input = cv2.imread("simple.png", cv2.IMREAD_GRAYSCALE)
    kernels = create_kernels(15)
    gaussian_kernel = kernels['gaussian']

    noise_levels = [0.0, 1.0, 5.0, 10.0]
    for noise_var in noise_levels:
        degraded_input = degrade_image(clean_input, gaussian_kernel, noise_var)
        kernel_info = f"Gaussian {gaussian_kernel.shape[0]}x{gaussian_kernel.shape[1]}"
        run_wiener_restoration(degraded_input, reference_paths, reference_labels, gaussian_kernel, kernel_info)