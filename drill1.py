# import numpy as np

# # Data Setup
# A = np.array([[1, 2], [3, 4]])          # Shape (2, 2)
# B = np.array([[5, 6], [7, 8]])          # Shape (2, 2)
# v = np.array([1, 2])                    # Shape (2,)
# Batch = np.random.rand(10, 5, 5)        # Shape (10, 5, 5) - "10 images of 5x5 pixels"

# print("--- Level 1: Dot Product ---")
# # Classic Matrix Multiplication: rows of A * columns of B
# # Math: C_ik = sum_j (A_ij * B_jk)
# # Task: Fill in the string '...'
# level_1 = np.einsum('ij,jk->ik', A, B)
# print("MatMul (Should be [[19, 22], [43, 50]]):\n", level_1)

# print("\n--- Level 2: The 'Diagonal' ---")
# # Extract the diagonal elements of A (1 and 4) and sum them (Trace).
# # Hint: If you use 'ii', it picks elements where row=col. If you omit ->, it sums.
# level_2 = np.einsum('ii->', A)
# print("Trace (Should be 5):", level_2)

# print("\n--- Level 3: The 'Batch' Operation (Transformer Style) ---")
# # You want to sum up all the pixels in each image.
# # Input: (batch, height, width) -> (10, 5, 5)
# # Output: (batch,) -> (10,)
# # Hint: Keep 'b', sum over 'h' and 'w'.
# level_3 = np.einsum('bhw->b', Batch)
# print("Batch Sum Shape (Should be (10,)):", level_3.shape)

