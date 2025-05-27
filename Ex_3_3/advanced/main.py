
import cv2
from test_scenarios import test_different_kernels, test_noise_levels


def main():
    """Main function demonstrating advanced Wiener filter with kernel estimation."""

    print("\n\n Kernel Testing")
    print("=" * 50)
    test_different_kernels()

    print("\n\nNoise Level Testing")
    print("=" * 50)
    #test_noise_levels()


if __name__ == "__main__":
    main()