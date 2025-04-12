import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from sklearn.neighbors import KernelDensity

# Visualize original and shifted colors in 3D RGB space
def visualize_3d_color_space(sampled_colors, shifted_colors, iterations_history=None, animation_path=None):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot original points
    ax.scatter(sampled_colors[:, 0], sampled_colors[:, 1], sampled_colors[:, 2],
               color=sampled_colors, alpha=0.2, label='Original')

    # Plot shifted points
    ax.scatter(shifted_colors[:, 0], shifted_colors[:, 1], shifted_colors[:, 2],
               color=shifted_colors, s=100, label='Clustered')

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_title('Mean Shift Clustering in RGB Color Space')
    ax.legend()

    plt.tight_layout()

    # If we have iteration history and animation path, create animation
    if iterations_history is not None and animation_path is not None:
        def update(frame):
            ax.clear()
            current_points = iterations_history[frame]

            # Plot original points (static)
            ax.scatter(sampled_colors[:, 0], sampled_colors[:, 1], sampled_colors[:, 2],
                       color=sampled_colors, alpha=0.2)

            # Plot current iteration points
            ax.scatter(current_points[:, 0], current_points[:, 1], current_points[:, 2],
                       color=current_points, s=100)

            ax.set_xlabel('Red')
            ax.set_ylabel('Green')
            ax.set_zlabel('Blue')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)
            ax.set_title(f'Mean Shift Iteration {frame}')

            return ax,

        anim = animation.FuncAnimation(fig, update, frames=len(iterations_history),
                                       interval=200, blit=False)
        anim.save(animation_path, writer='pillow', fps=5)

    return fig

# Visualize the 3D color density topography
def visualize_color_density(sampled_colors, bandwidth):
    # 2D projections of the 3D density
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Define meshgrid for each projection
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)

    # Projections to visualize: RG, RB, GB
    projections = [(0, 1, 'Red-Green'), (0, 2, 'Red-Blue'), (1, 2, 'Green-Blue')]

    for i, (dim1, dim2, title) in enumerate(projections):
        # Project data to 2D
        points = sampled_colors[:, [dim1, dim2]]

        # Fit kernel density
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(points)

        # Evaluate density on the grid
        xy_sample = np.vstack([X.ravel(), Y.ravel()]).T
        log_density = kde.score_samples(xy_sample)
        Z = np.exp(log_density).reshape(X.shape)

        # Plot contour
        axes[i].contourf(X, Y, Z, cmap='viridis')
        axes[i].scatter(points[:, 0], points[:, 1], alpha=0.3, s=1)
        axes[i].set_xlabel(['Red', 'Red', 'Green'][i])
        axes[i].set_ylabel(['Green', 'Blue', 'Blue'][i])
        axes[i].set_title(f'{title} Projection')

    plt.tight_layout()
    return fig