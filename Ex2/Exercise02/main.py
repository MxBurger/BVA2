import cv2
import numpy as np
import matplotlib.pyplot as plt

import visualization as visu

# Calculate Euclidean distance between two colors in RGB space
def colour_dist(ref_color, curr_color):
    return np.sqrt(np.sum((ref_color - curr_color) ** 2))

# Calculate Gaussian weight based on distance and bandwidth
def gaussian_weight(dist, bandwidth):
    return np.exp(-0.5 * (dist**2) / (bandwidth**2))

# Perform Mean Shift clustering on pixels
def mean_shift_color_pixel(in_pixels, bandwidth, max_iterations=50, epsilon=0.005):
    # Convert input to numpy array if it's not already
    pixels = np.array(in_pixels)
    n_samples = pixels.shape[0]

    # Initialize shifted pixels with input pixels
    shifted_pixels = pixels.copy()

    # To track movement history for visualization
    iterations_history = [shifted_pixels.copy()]

    # Iterate until convergence or max iterations
    for it in range(max_iterations):
        max_shift = 0

        # For each point
        for i in range(n_samples):
            # Current point
            current = shifted_pixels[i]

            # Calculate weights for all points
            distances = np.array([colour_dist(current, shifted_pixels[j]) for j in range(n_samples)])
            weights = np.array([gaussian_weight(dist, bandwidth) for dist in distances])

            # Calculate weighted mean
            weighted_sum = np.zeros(3)
            weight_sum = 0

            for j in range(n_samples):
                weighted_sum += weights[j] * shifted_pixels[j]
                weight_sum += weights[j]

            # New position
            if weight_sum > 0:
                new_position = weighted_sum / weight_sum
                # Calculate shift size
                shift = colour_dist(current, new_position)
                max_shift = max(max_shift, shift)
                # Update position
                shifted_pixels[i] = new_position

        # Save current state for visualization
        iterations_history.append(shifted_pixels.copy())

        print(f"current max shift: {max_shift:4f}")
        # Check for convergence
        if max_shift < epsilon:
            print(f"Converged after {it + 1} iterations")
            break

    return shifted_pixels, iterations_history

# Apply Mean Shift clustering to an image
def process_image(image_path, bandwidth, sampling_ratio=0.1):
    # Load image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    height, width, _ = original_image.shape

    # Reshape image to a list of pixels
    pixels = original_image.reshape(-1, 3).astype(np.float64) / 255.0

    # Sample pixels to reduce computation time
    n_pixels = pixels.shape[0]
    n_samples = int(n_pixels * sampling_ratio)
    sample_indices = np.random.choice(n_pixels, n_samples, replace=False)
    sampled_pixels = pixels[sample_indices]

    # Apply Mean Shift clustering
    shifted_pixels, iterations_history = mean_shift_color_pixel(sampled_pixels, bandwidth)

    # Find unique clusters (approximately)
    rounded_shifted = np.round(shifted_pixels * 100) / 100
    _, unique_indices = np.unique(rounded_shifted, axis=0, return_index=True)
    unique_colors = shifted_pixels[unique_indices]
    n_clusters = len(unique_colors)
    print(f"Number of clusters: {n_clusters}")

    # Map all pixels to their nearest cluster center
    clustered_image = np.zeros_like(original_image, dtype=np.float64)
    flat_clustered = clustered_image.reshape(-1, 3)

    # For each pixel in the original image
    for i in range(n_pixels):
        # Find the nearest cluster center
        distances = [colour_dist(pixels[i], cluster) for cluster in unique_colors]
        nearest_cluster = np.argmin(distances)
        flat_clustered[i] = unique_colors[nearest_cluster]

    # Convert back to uint8
    clustered_image = (clustered_image * 255).astype(np.uint8)

    return original_image, clustered_image, sampled_pixels, shifted_pixels, iterations_history


def main():
    # Parameters
    image_path = 'Tanke.jpg'  # Replace with your image path
    bandwidth = 0.1  # Mean Shift bandwidth parameter
    sampling_ratio = 0.05  # Ratio of pixels to sample

    # Process the image
    original, clustered, sampled_colors, shifted_colors, iterations_history = process_image(
        image_path, bandwidth, sampling_ratio
    )

    # Display original and clustered images
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(clustered)
    plt.title(f'Mean Shift Clustering (Bandwidth={bandwidth})')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('mean_shift_result.png')

    # Create visualizations
    visu.visualize_3d_color_space(sampled_colors, shifted_colors)
    plt.savefig('color_space_clustering.png')

    visu.visualize_3d_color_space(sampled_colors, shifted_colors, iterations_history, 'color_space_animation.gif')

    visu.visualize_color_density(sampled_colors, bandwidth)
    plt.savefig('color_density_topography.png')

    plt.show()

if __name__ == "__main__":
    main()