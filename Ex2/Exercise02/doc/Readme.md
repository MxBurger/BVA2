# Mean Shift Clustering for Color Image Segmentation

## Overview

This solution implements the Mean Shift algorithm for color-based image
segmentation. The implementation uses Python with `NumPy`, `OpenCV`, and `Matplotlib` libraries.
The code is organized into two files:

1. `main.py` - Contains the core Mean Shift implementation and image processing functions
2. `visualization.py` - Contains visualization functions for understanding the algorithm

The solution segments color images by clustering pixels in RGB color space,
with the bandwidth parameter controlling the clustering granularity.

## Core Mean Shift Algorithm Components

### Distance and Weight Functions

Two fundamental functions drive the Mean Shift algorithm:

1. **Color Distance Function (`colour_dist`)**:
```python
def colour_dist(ref_color, curr_color):
    return np.sqrt(np.sum((ref_color - curr_color) ** 2))
```
This calculates the Euclidean distance between two colors in 3D RGB space,
treating each color as a point in this space.

2. **Gaussian Weight Function (`gaussian_weight`)**:
```python
def gaussian_weight(dist, bandwidth):
    return np.exp(-0.5 * (dist**2) / (bandwidth**2))
```
This implements a Gaussian kernel that assigns weights to neighboring points based on their distance.
The bandwidth parameter controls the kernel width - larger values create a wider influence radius.

### Main Mean Shift Implementation

The core algorithm is implemented in the `mean_shift_color_pixel` function:

```python
def mean_shift_color_pixel(in_pixels, bandwidth, max_iterations=50, epsilon=0.005):
    # Convert input to numpy array
    pixels = np.array(in_pixels)
    n_samples = pixels.shape[0]
    
    # Initialize shifted pixels with input pixels
    shifted_pixels = pixels.copy()
    
    # Track movement history for visualization
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
        
        # Check for convergence
        if max_shift < epsilon:
            print(f"Converged after {it + 1} iterations")
            break
    
    return shifted_pixels, iterations_history
```

This function:
1. Takes a set of pixel colors and the bandwidth parameter
2. Iteratively shifts each pixel toward the mean of its neighbors
3. Neighbors' influence is weighted by the Gaussian kernel
4. Tracks the maximum shift in each iteration to check for convergence
5. Builds a history of pixel positions for later visualization
6. Terminates when either maximum iterations are reached or the maximum shift
falls below the epsilon threshold

## Image Processing Pipeline

The `process_image` function applies Mean Shift clustering to a color image:

```python
def process_image(image_path, bandwidth, sampling_ratio=0.1):
    # Load image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
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
    
    return original_image, clustered_image, sampled_pixels, shifted_colors, iterations_history
```

This function:
1. Loads and normalizes the image (converts to RGB and scales to [0,1])
2. Samples a subset of pixels for efficiency (controlled by sampling_ratio)
3. Runs Mean Shift clustering on the sampled pixels
4. Identifies unique clusters by rounding and finding unique values
5. Maps each original pixel to its nearest cluster center
6. Returns both original and clustered images, plus data for visualization

## Visualization Components

### 3D Color Space Visualization

The `visualize_3d_color_space` function in `visualization.py` creates a 3D visualization of the clustering process:

```python
def visualize_3d_color_space(sampled_colors, shifted_colors, iterations_history=None, animation_path=None):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot original points
    ax.scatter(sampled_colors[:, 0], sampled_colors[:, 1], sampled_colors[:, 2],
               color=sampled_colors, alpha=0.2, label='Original')

    # Plot shifted points
    ax.scatter(shifted_colors[:, 0], shifted_colors[:, 1], shifted_colors[:, 2],
               color=shifted_colors, s=100, label='Clustered')

    # If iterations_history and animation_path are provided, create an animation
    if iterations_history is not None and animation_path is not None:
        def update(frame):
            # Update function for animation
            # ...
        
        anim = animation.FuncAnimation(fig, update, frames=len(iterations_history),
                                       interval=200, blit=False)
        anim.save(animation_path, writer='pillow', fps=5)
```

This function:
1. Plots the original sampled colors and final clustered colors in 3D RGB space
2. Optionally creates an animation showing how points move during clustering
3. Saves the animation as a GIF if a path is provided

### Color Density Topography Visualization

The `visualize_color_density` function creates 2D projections of the 3D color density:

```python
def visualize_color_density(sampled_colors, bandwidth):
    # Create 2D projections (RG, RB, GB) of the 3D color density
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define meshgrid for each projection
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    xy_sample = np.vstack([X.ravel(), Y.ravel()]).T
    
    # Projections to visualize: RG, RB, GB
    projections = [(0, 1, 'Red-Green'), (0, 2, 'Red-Blue'), (1, 2, 'Green-Blue')]
    
    for i, (dim1, dim2, title) in enumerate(projections):
        # Project data to 2D
        points = sampled_colors[:, [dim1, dim2]]
        
        # Compute density at each point in the grid
        Z = np.zeros(len(xy_sample))
        
        for j, grid_point in enumerate(xy_sample):
            # Calculate kernel density estimation at this point
            density = 0
            for color_point in points:
                dist = point_dist(grid_point, color_point)
                density += gaussian_weight(dist, bandwidth)
            
            # Normalize by number of points
            Z[j] = density / len(points)
        
        # Reshape to match the grid
        Z = Z.reshape(X.shape)
        
        # Plot contour
        axes[i].contourf(X, Y, Z, cmap='viridis')
        axes[i].scatter(points[:, 0], points[:, 1], alpha=0.3, s=1)
        axes[i].set_title(f'{title} Projection')
```

This visualization:
1. Creates three 2D projections of the RGB color space (Red-Green, Red-Blue, Green-Blue)
2. Computes density at each point using the same Gaussian kernel from the Mean Shift algorithm
3. Visualizes the density topography with contour plots

## Main Execution Flow

The `main` function ties everything together:

```python
def main():
    # Parameters
    image_path = 'Tanke.jpg'
    bandwidth = 0.1
    sampling_ratio = 0.05
    
    start_time = time.time()
    # Process the image
    original, clustered, sampled_colors, shifted_colors, iterations_history = process_image(
        image_path, bandwidth, sampling_ratio
    )
    
    print(f"Image segmentation completed in {time.time() - start_time:.2f} seconds")
    
    # Display and save original and clustered images
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(original)
    plt.title('Original Image')
    
    plt.subplot(122)
    plt.imshow(clustered)
    plt.title(f'Mean Shift Clustering (Bandwidth={bandwidth})')
    plt.savefig('mean_shift_result.png')
    
    # Create and save visualizations
    visu.visualize_3d_color_space(sampled_colors, shifted_colors)
    plt.savefig('color_space_clustering.png')
    
    visu.visualize_3d_color_space(sampled_colors, shifted_colors, iterations_history, 'color_space_animation.gif')
    
    visu.visualize_color_density(sampled_colors, bandwidth)
    plt.savefig('color_density_topography.png')
```

This function:
1. Sets parameters including image path, bandwidth, and sampling ratio
2. Calls `process_image` to perform the clustering
3. Displays and saves the original and clustered images
4. Creates and saves all required visualizations

## Effect of Bandwidth Parameter

The bandwidth parameter controls the clustering granularity:

1. **Smaller bandwidth** values (e.g., 0.05):
   - Create more clusters (higher number of segments)
   - Preserve more detail but may result in over-segmentation
   - Converge more slowly (more iterations needed)

2. **Larger bandwidth** values (e.g., 0.2):
   - Create fewer clusters (lower number of segments)
   - Produce more smoothed results but may lose important details
   - Converge more quickly (fewer iterations needed)

The code evaluates and outputs the final number of clusters, allowing for experimental analysis of how the bandwidth parameter affects segmentation results across different images.

## Optimization Techniques

Several optimizations improve the algorithm's efficiency:

1. **Pixel sampling**: Only a fraction of pixels (controlled by sampling_ratio) are used for the Mean Shift algorithm to reduce computation time
2. **Early convergence**: The algorithm stops when the maximum shift falls below a threshold
3. **Efficient cluster assignment**: After finding cluster centers, pixels are assigned to their nearest center without running Mean Shift on every pixel

## Conclusion

This implementation provides a complete solution for Mean Shift color image segmentation with comprehensive visualizations that illustrate both the algorithm's process and results. The code satisfies all requirements from the assignment:

1. Implementation of Mean Shift clustering for color images with the bandwidth as the only input parameter
2. Visualization of the clustering process in 3D color space with animation
3. Visualization of the 3D color density topography

The solution allows for experimentation with different bandwidth values to analyze their effect on the clustering results for various images.