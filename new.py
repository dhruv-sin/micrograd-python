import numpy as np

np.random.seed(42)

# 1. THE DATA
# Shape (4, 5): 4 examples, 5 neurons each
X = np.random.randint(0, 20, size=(4, 5))
print("Original Input:\n", X)

# --- STEP 1: BROADCASTING (The "Friday" Debt) ---
# Calculate mean of each neuron (column). Shape should be (5,)
means = X.mean(axis=0) 
print(means.shape)
print(X.shape)

# Subtract mean from X. 
# X is (4,5), means is (5,). NumPy broadcasts the subtraction.
X_centered = X - means

print("\nCentered Data (Mean should be near 0):\n", X_centered)

# --- STEP 2: MASKING (The "Saturday" Target) ---
# Implement ReLU: Set all negative values in X_centered to 0
# Hint: Use a boolean mask
mask_neg = X_centered < 0
X_centered[mask_neg<0] = 0

print("\nAfter ReLU (No negatives):\n", X_centered)

# --- STEP 3: THE DROPOUT (Bonus Research Skill) ---
# Create a mask of random noise same shape as X
# If random value < 0.2, set the neuron to 0 (20% dropout)
dropout_mask = np.random.rand(*X_centered.shape) < 0.2
X_centered[dropout_mask] = 0

print("\nFinal Output (With Dropout):\n", X_centered)