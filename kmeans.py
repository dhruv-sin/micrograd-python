import numpy as np

# 1. DATA SETUP
# 100 points, 2D coordinates (x, y)
X = np.random.rand(100, 2) 
# 3 Centroids, 2D coordinates
centroids = np.random.rand(3, 2)

# --- THE CHALLENGE ---
# Calculate the distance from EVERY point to EVERY centroid.
# Input Shapes: X is (100, 2). Centroids is (3, 2).
# Target Shape: Distances should be (100, 3).

# HINT: Use Broadcasting. 
# Reshape X to (100, 1, 2) and Centroids to (1, 3, 2).
distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

print("Distances Shape (Should be 100, 3):", distances.shape)

# 2. ASSIGNMENT STEP (Masking/Index)
# Find the index of the closest centroid for each point (0, 1, or 2)
labels = np.argmin(distances, axis=1)

print("First 5 labels:", labels[:5])
# import numpy as np

# # 1. Generate a random array of 10,000 2D points using np.random.rand
# # The shape is (10000, 2), representing 10000 points with x and y coordinates
# points = np.random.rand(10000, 2)
# print(f"Shape of the points array: {points.shape}")

# # 2. Compute the "center of mass" (average x and y coordinates)
# # np.mean with axis=0 calculates the mean of each column (all x-coords, all y-coords)
# center_of_mass = np.mean(points, axis=0)
# print(f"Center of mass (average coordinates): {center_of_mass}")

# # 3. Use broadcasting to compute the position of the points relative to the center of mass
# # Subtracting the 1D center_of_mass array from the 2D points array
# # uses broadcasting to apply the subtraction to each row (point)
# relative_positions = points - center_of_mass
# print(f"Shape of the relative positions array: {relative_positions.shape}")
# print(f"First 3 relative positions:\n{relative_positions[:3]}")

# # Optional: Verification
# # The average of the relative positions should be very close to zero
# mean_relative_positions = np.mean(relative_positions, axis=0)
# print(f"Mean of relative positions (should be near zero): {mean_relative_positions}")
