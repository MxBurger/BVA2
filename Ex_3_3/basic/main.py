import cv2
from sample_generator import create_synthetic_image
from test_scenarios import run_tests, run_frequency_analysis


def load_image(image_path: str):
    """
    Load image from path or create synthetic image if file not found.
    """
    if image_path:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    
    # Create synthetic image if file not found or path is empty
    return create_synthetic_image()


def main():
    """
    Main demonstration function.
    """
    image_path = ""  # Enter image path here or leave empty for synthetic image
    img = load_image(image_path)
    
    print("Starting Wiener Filter Demonstration...")
    print(f"Image shape: {img.shape}")

    run_tests(img)
    
    # Run frequency analysis
    kernel_sizes = [5, 15, 25]
    K_values = [0.0001, 0.001, 0.01, 0.1]
    run_frequency_analysis(kernel_sizes, K_values)
    
    print("\nDemonstration completed!")


if __name__ == "__main__":
    main()