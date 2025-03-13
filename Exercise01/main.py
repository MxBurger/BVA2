import numpy as np
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt

# Original points P
P = np.array([
    [1, 4],
    [-4, -2],
    [0.1, 5],
    [-1, 2],
    [3, 3],
    [7, -2],
    [5, 5],
    [-6, 3.3]
])

# Transformed points P'
P_prime = np.array([
    [-1.26546, 3.222386],
    [-4.53286, 0.459128],
    [-1.64771, 3.831308],
    [-2.57985, 2.283247],
    [-0.28072, 2.44692],
    [1.322025, -0.69344],
    [1.021729, 3.299737],
    [-5.10871, 3.523542]
])


# Function to apply transformation and calculate error
def transformation_error(params):
    # Extract parameters
    theta, s, tx, ty = params

    # Create transformation matrix
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    # Apply transformation to each point
    transformed_points = []
    for p in P:
        x, y = p
        # Apply rotation and scale
        x_rot = s * (x * cos_theta - y * sin_theta)
        y_rot = s * (x * sin_theta + y * cos_theta)
        # Apply translation
        x_final = x_rot + tx
        y_final = y_rot + ty
        transformed_points.append([x_final, y_final])

    transformed_points = np.array(transformed_points)

    # Calculate mean squared error
    error = np.mean(np.sum((transformed_points - P_prime) ** 2, axis=1)) # axis=1 row
    return error


# Initial guess for parameters: [theta, scale, tx, ty]
initial_guess = [0, 1, 0, 0]

# Run optimization
result = minimize(transformation_error, initial_guess, method='Nelder-Mead')

# Get optimal parameters
opt_theta, opt_scale, opt_tx, opt_ty = result.x


# Calculate residual error (approximation of noise level)
residual_error = result.fun
noise = np.sqrt(residual_error)

print(f"Optimal scale: {opt_scale:.4f}")
print(f"Optimal translation Tx: {opt_tx:.4f}")
print(f"Optimal translation Ty: {opt_ty:.4f}")
print(f"Estimated noise level: {noise:.4f}")

# Calculate transformed points with optimal parameters
transformed_P = []
for p in P:
    x, y = p
    # Apply optimal transformation
    x_rot = opt_scale * (x * math.cos(opt_theta) - y * math.sin(opt_theta))
    y_rot = opt_scale * (x * math.sin(opt_theta) + y * math.cos(opt_theta))
    x_final = x_rot + opt_tx
    y_final = y_rot + opt_ty
    transformed_P.append([x_final, y_final])

transformed_P = np.array(transformed_P)

# Calculate error for each point
point_errors = np.sqrt(np.sum((transformed_P - P_prime) ** 2, axis=1))
print("\nPoint-wise errors:")
for i, err in enumerate(point_errors):
    print(f"Point {i + 1}: {err:.4f}")

# Create visualization
plt.figure(figsize=(10, 8))

# Plot original points
plt.scatter(P[:, 0], P[:, 1], color='blue', label='Original Points (P)', s=50)

# Plot target points (P')
plt.scatter(P_prime[:, 0], P_prime[:, 1], color='red', label='Target Points (P\')', s=50)

# Plot transformed points
plt.scatter(transformed_P[:, 0], transformed_P[:, 1], color='green', label='Transformed Points', s=50, alpha=0.7)

# Draw lines from transformed points to target points to show errors
for i in range(len(P)):
    plt.plot([transformed_P[i, 0], P_prime[i, 0]],
             [transformed_P[i, 1], P_prime[i, 1]],
             'k--', alpha=0.3)

    # Annotate points with their indices
    plt.annotate(str(i + 1), (P[i, 0], P[i, 1]), xytext=(-10, 10),
                 textcoords='offset points', color='blue')
    plt.annotate(str(i + 1), (P_prime[i, 0], P_prime[i, 1]), xytext=(5, 5),
                 textcoords='offset points', color='red')

# Add text box
param_text = f"Scale: {opt_scale:.3f}\nTx: {opt_tx:.3f}\nTy: {opt_ty:.3f}\nNoise: {noise:.3f}"
plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.axis('equal') # Fix Ratio

plt.grid(True, alpha=0.3)
plt.legend()
plt.title('Point Transformation Visualization')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')

plt.tight_layout()
plt.show()