import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
import time

import visualization as visu


# Calculate Euclidean distance between two colors in RGB space
def colour_dist(ref_color, curr_color):
    return np.sqrt(np.sum((ref_color - curr_color) ** 2))


# Apply Mean Shift clustering to an image using scikit-learn
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

    # Store original sampled pixels for visualization
    original_sampled = sampled_pixels.copy()

    # Start timing
    start_time = time.time()

    # Apply scikit-learn's MeanShift clustering
    mean_shift = MeanShift(
        bandwidth=bandwidth,
        bin_seeding=False,
        cluster_all=True,
        max_iter=50,
        n_jobs=-1,
    )

    # Fit the model to sample pixels
    mean_shift.fit(sampled_pixels)

    # Get cluster centers and labels
    cluster_centers = mean_shift.cluster_centers_
    labels = mean_shift.labels_

    # Map sampled pixels to their cluster centers for visualization
    shifted_pixels = cluster_centers[labels]

    # MODIFICATION: Post-process cluster centers to mimic the custom implementation
    # Round to 2 decimal places (similar to the custom implementation)
    rounded_centers = np.round(cluster_centers * 100) / 100

    # Find unique rounded centers
    _, unique_indices = np.unique(rounded_centers, axis=0, return_index=True)
    unique_colors = cluster_centers[unique_indices]

    # Re-map original labels to the merged clusters
    new_labels = np.zeros_like(labels)
    for i, center in enumerate(cluster_centers):
        # Find the nearest unique center for each original center
        distances = [colour_dist(center, unique) for unique in unique_colors]
        nearest_unique = np.argmin(distances)

        # Map all pixels with the original label to the new merged label
        new_labels[labels == i] = nearest_unique

    n_clusters = len(unique_colors)
    print(f"Original number of clusters: {len(cluster_centers)}")
    print(f"Number of clusters after merging: {n_clusters}")
    print(f"Clustering completed in {time.time() - start_time:.2f} seconds")

    # Start timing segmentation
    start_time = time.time()

    # Map all pixels to their nearest cluster center
    clustered_image = np.zeros_like(original_image, dtype=np.float64)
    flat_clustered = clustered_image.reshape(-1, 3)

    # Predict labels for all pixels
    all_labels = mean_shift.predict(pixels)

    # Map the original labels to new merged labels
    merged_all_labels = np.zeros_like(all_labels)
    for i, original_label in enumerate(range(len(cluster_centers))):
        merged_label = 0
        # Find the nearest unique center
        if original_label < len(cluster_centers):
            distances = [colour_dist(cluster_centers[original_label], unique) for unique in unique_colors]
            merged_label = np.argmin(distances)

        # Assign the new label to all pixels with the original label
        merged_all_labels[all_labels == original_label] = merged_label

    # Use the unique colors with the merged labels
    flat_clustered[:] = unique_colors[merged_all_labels]

    print(f"Image segmentation completed in {time.time() - start_time:.2f} seconds")

    # Convert back to uint8
    clustered_image = (clustered_image * 255).astype(np.uint8)

    # Create a simple iterations history for visualization
    # (before and after since we don't have detailed iteration history)
    iterations_history = [original_sampled, shifted_pixels]

    return original_image, clustered_image, original_sampled, shifted_pixels, iterations_history


def main():
    # Parameters
    image_path = 'Tanke.jpg'
    bandwidth = 0.1
    sampling_ratio = 0.05

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
    plt.title(f'Modified Mean Shift Clustering (Bandwidth={bandwidth})')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('modified_mean_shift_result.png')

    # Create visualizations
    visu.visualize_3d_color_space(sampled_colors, shifted_colors)
    plt.savefig('modified_color_space_clustering.png')

    visu.visualize_3d_color_space(sampled_colors, shifted_colors, iterations_history,
                                  'modified_color_space_animation.gif')

    visu.visualize_color_density(sampled_colors, bandwidth)
    plt.savefig('modified_color_density_topography.png')

    plt.show()


if __name__ == "__main__":
    main()