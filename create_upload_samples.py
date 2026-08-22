import numpy as np
import os

# Load full dataset
X = np.load("data/raw/inputs.npy")

# Create folder
os.makedirs("upload_samples", exist_ok=True)

# Save a few example samples
for i in [0, 5, 10, 20, 50]:
    np.save(f"upload_samples/sample_{i}.npy", X[i])

print("Done!")