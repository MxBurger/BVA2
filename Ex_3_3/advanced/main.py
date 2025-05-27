
import cv2
from test_wiener import run_wiener_restoration, test_different_kernels, test_noise_levels


def main():
    """Main function demonstrating advanced Wiener filter with kernel estimation."""

    # Example 1: Single restoration
    print("Example 1: Single Image Restoration")
    print("=" * 50)

    reference_paths = [
        "ref/landscape_gray.png",
        "ref/lena_gray.tif",
        "ref/mandril_gray.tif",
        "ref/peppers_gray.tif",
        "ref/text.png",
    ]

    reference_labels = ["landscape", "portrait", "animal", "food", "text"]

    degraded_image = cv2.imread("degraded_input.png", cv2.IMREAD_GRAYSCALE)

    if degraded_image is not None:
        run_wiener_restoration(degraded_image, reference_paths, reference_labels)
    else:
        print("Warning: degraded_input.png not found, skipping single restoration example")

    # Example 2: Comprehensive testing
    print("\n\nExample 2: Comprehensive Kernel Testing")
    print("=" * 50)
    test_different_kernels()

    # Example 3: Noise level testing
    print("\n\nExample 3: Noise Level Testing")
    print("=" * 50)
    test_noise_levels()


if __name__ == "__main__":
    main()